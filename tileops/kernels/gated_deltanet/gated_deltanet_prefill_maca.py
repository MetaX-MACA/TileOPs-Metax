"""MACA Gated DeltaNet prefill (v2 file) — self-contained drop-in for maca.py.

Same public API as ``gated_deltanet_prefill_maca`` (``GatedDeltaNetPrefillFwdMACAKernel``).
Uses decomposed blocksolve + recompute + h/o only (no ``gdn_prefill`` / fused_gdr).
Swap test: point ``kernels/gated_deltanet/__init__.py`` import at this module.
"""

import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from .gated_deltanet_fwd import _LOG2E, _chunk_local_cumsum

__all__ = [
    "GatedDeltaNetPrefillFwdMACAKernel",
    "_prefill_blocksolve_A_bthd_tl_maca",
    "_prefill_h_recurrence_bhtd_tl_maca",
    "_prefill_output_o_bhtd_tl_maca",
    "_prefill_recompute_w_u_from_A_bthd_tl_maca",
]


@functools.lru_cache(maxsize=32)
def _prefill_h_recurrence_bhtd_tl_maca(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
):
    """MACA h_recurrence (bhtd) with K-tiling to stay within 64KB smem."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = dim_v if block_v <= 0 else block_v
    # MACA MMA requires gemm N (BV) divisible by 16.
    if BV % 16 != 0:
        for b in (32, 16):
            if dim_v % b == 0:
                BV = b
                break
        else:
            BV = dim_v
    num_v_tiles = dim_v // BV
    BK = 64 if dim_k >= 64 and dim_k % 64 == 0 else dim_k
    num_k_tiles = dim_k // BK

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128):
        @T.prim_func
        def h_recurrence_bhtd_maca(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S_0: T.Tensor([batch, head, dim_k, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                kw_c = T.alloc_shared([block_C, BK], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                h_tile = T.alloc_shared([BK, BV], dtype)
                h_next_s = T.alloc_shared([dim_k, BV], accum_dtype)
                h_k_s = T.alloc_shared([BK, BV], accum_dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)

                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                h_k_frag = T.alloc_fragment([BK, BV], accum_dtype)

                v_offset = vid * BV

                T.copy(
                    S_0[bid, hid, :, v_offset : v_offset + BV],
                    h_c,
                    disable_tma=True,
                )
                for i, j in T.Parallel(dim_k, BV):
                    S[bid, hid, 0, i, v_offset + j] = h_c[i, j]

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    base = t * block_C
                    T.copy(
                        g[bid, hid, base : base + block_C],
                        g_c,
                        disable_tma=True,
                    )
                    T.copy(
                        u[
                            bid,
                            hid,
                            base : base + block_C,
                            v_offset : v_offset + BV,
                        ],
                        u_c,
                        disable_tma=True,
                    )

                    T.clear(ws_frag)
                    for kt in T.Serial(num_k_tiles):
                        koff = kt * BK
                        T.copy(
                            w[
                                bid,
                                hid,
                                base : base + block_C,
                                koff : koff + BK,
                            ],
                            kw_c,
                            disable_tma=True,
                        )
                        for i, d in T.Parallel(BK, BV):
                            h_tile[i, d] = h_c[koff + i, d]
                        T.gemm(kw_c, h_tile, ws_frag)

                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_c[block_C - 1]) * _LOG2E
                        )

                    T.copy(
                        v_new_c,
                        v_new[
                            bid,
                            hid,
                            base : base + block_C,
                            v_offset : v_offset + BV,
                        ],
                        disable_tma=True,
                    )

                    for n, j in T.Parallel(block_C, BV):
                        v_new_c[n, j] = v_new_c[n, j] * T.exp2(
                            (g_c[block_C - 1] - g_c[n]) * _LOG2E
                        )
                    for i, j in T.Parallel(dim_k, BV):
                        h_next_s[i, j] = T.cast(h_c[i, j], accum_dtype) * T.exp2(
                            g_c[block_C - 1] * _LOG2E
                        )
                    for kt in T.Serial(num_k_tiles):
                        koff = kt * BK
                        T.copy(
                            k[
                                bid,
                                hid,
                                base : base + block_C,
                                koff : koff + BK,
                            ],
                            kw_c,
                            disable_tma=True,
                        )
                        # MACA gemm C must be a full fragment; accumulate via shared.
                        T.clear(h_k_frag)
                        T.gemm(
                            kw_c,
                            v_new_c,
                            h_k_frag,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.copy(h_k_frag, h_k_s)
                        for i, j in T.Parallel(BK, BV):
                            h_next_s[koff + i, j] = h_next_s[koff + i, j] + h_k_s[i, j]
                    for i, j in T.Parallel(dim_k, BV):
                        h_c[i, j] = T.cast(h_next_s[i, j], dtype)
                    for i, j in T.Parallel(dim_k, BV):
                        S[bid, hid, t + 1, i, v_offset + j] = h_c[i, j]

        return h_recurrence_bhtd_maca

    return _func


@functools.lru_cache(maxsize=32)
def _prefill_output_o_bhtd_tl_maca(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """MACA output_o (bhtd) with V-tiling to stay within 64KB smem."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = 32 if dim_v >= 32 and dim_v % 32 == 0 else dim_v
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def output_o_bhtd_maca(
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            o: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            # V tiles are mapped to grid (not T.Serial) so MACA gemm layout
            # inference keeps a valid thread_bounds / num_warps.
            with T.Kernel(
                num_chunks, num_v_tiles, batch * head, threads=threads
            ) as (tid, vt, bhid):
                bid = bhid // head
                hid = bhid % head
                base = tid * block_C
                v_offset = vt * BV

                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], accum_dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                # Keep attn/v_new in dtype so A/B gemm dtypes match (MACA MMA).
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)

                o_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)

                T.copy(q[bid, hid, base : base + block_C, :], q_c, disable_tma=True)
                T.copy(k[bid, hid, base : base + block_C, :], k_c, disable_tma=True)
                T.copy(g[bid, hid, base : base + block_C], g_c, disable_tma=True)
                T.copy(
                    S[bid, hid, tid, :, v_offset : v_offset + BV],
                    h_c,
                    disable_tma=True,
                )
                T.copy(
                    v_new[
                        bid,
                        hid,
                        base : base + block_C,
                        v_offset : v_offset + BV,
                    ],
                    v_new_c,
                    disable_tma=True,
                )

                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(
                        i >= j,
                        T.cast(
                            attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                            dtype,
                        ),
                        T.cast(0, dtype),
                    )

                T.clear(o_frag)
                T.gemm(q_c, h_c, o_frag)
                for i, j in T.Parallel(block_C, BV):
                    o_frag[i, j] = o_frag[i, j] * T.exp2(g_c[i] * _LOG2E)

                T.gemm(attn, v_new_c, o_frag)
                T.copy(
                    o_frag,
                    o[
                        bid,
                        hid,
                        base : base + block_C,
                        v_offset : v_offset + BV,
                    ],
                    disable_tma=True,
                )

        return output_o_bhtd_maca

    return _func


@functools.lru_cache(maxsize=32)
def _prefill_blocksolve_A_bthd_tl_maca(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dtype: str,
):
    """MACA blocksolve-A: sync copy + 64 threads (warp size)."""
    if chunk_size != 64 or dim_k != 128:
        raise ValueError("TileLang blocksolve-A currently expects chunk64 and K=128")

    block_t = 64
    block_c = 16
    block_k = 64
    accum_dtype = "float32"
    solve_dtype = dtype

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=64):
        @T.prim_func
        def prefill_blocksolve_A_bthd_maca(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            g: T.Tensor([batch, seq_len, head], dtype),
            beta: T.Tensor([batch, seq_len, head], dtype),
            A: T.Tensor([batch, seq_len, head, chunk_size], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_t, threads=threads) as (
                bid,
                hid,
                cid,
            ):
                base = cid * block_t
                k0 = T.alloc_shared([block_c, block_k], dtype)
                k1 = T.alloc_shared([block_c, block_k], dtype)
                k2 = T.alloc_shared([block_c, block_k], dtype)
                k3 = T.alloc_shared([block_c, block_k], dtype)
                g_s = T.alloc_shared([block_t], dtype)
                beta_s = T.alloc_shared([block_t], dtype)
                gate0_s = T.alloc_shared([block_t], dtype)
                gate1_s = T.alloc_shared([block_t], dtype)
                gate2_s = T.alloc_shared([block_t], dtype)
                gate3_s = T.alloc_shared([block_t], dtype)
                beta_f_s = T.alloc_shared([block_t], dtype)
                a_s = T.alloc_shared([10, block_c, block_c], solve_dtype)
                i_s = T.alloc_shared([4, block_c, block_c], solve_dtype)
                work_s = T.alloc_shared([1, block_c, block_c], solve_dtype)

                G00 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G10 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G11 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G20 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G21 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G22 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G30 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G31 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G32 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G33 = T.alloc_fragment([block_c, block_c], accum_dtype)
                tmp = T.alloc_fragment([block_c, block_c], accum_dtype)

                T.annotate_layout({
                    k0: tilelang.layout.make_swizzled_layout(k0),
                    k1: tilelang.layout.make_swizzled_layout(k1),
                    k2: tilelang.layout.make_swizzled_layout(k2),
                    k3: tilelang.layout.make_swizzled_layout(k3),
                    a_s: tilelang.layout.make_swizzled_layout(a_s),
                    i_s: tilelang.layout.make_swizzled_layout(i_s),
                    work_s: tilelang.layout.make_swizzled_layout(work_s),
                })

                T.copy(g[bid, base : base + block_t, hid], g_s, disable_tma=True)
                T.copy(
                    beta[bid, base : base + block_t, hid],
                    beta_s,
                    disable_tma=True,
                )

                T.clear(G00)
                T.clear(G10)
                T.clear(G11)
                T.clear(G20)
                T.clear(G21)
                T.clear(G22)
                T.clear(G30)
                T.clear(G31)
                T.clear(G32)
                T.clear(G33)

                for kt in T.Serial(dim_k // block_k):
                    koff = kt * block_k
                    T.copy(
                        k[bid, base : base + block_c, hid, koff : koff + block_k],
                        k0,
                        disable_tma=True,
                    )
                    T.copy(
                        k[
                            bid,
                            base + block_c : base + 2 * block_c,
                            hid,
                            koff : koff + block_k,
                        ],
                        k1,
                        disable_tma=True,
                    )
                    T.copy(
                        k[
                            bid,
                            base + 2 * block_c : base + 3 * block_c,
                            hid,
                            koff : koff + block_k,
                        ],
                        k2,
                        disable_tma=True,
                    )
                    T.copy(
                        k[
                            bid,
                            base + 3 * block_c : base + 4 * block_c,
                            hid,
                            koff : koff + block_k,
                        ],
                        k3,
                        disable_tma=True,
                    )
                    T.sync_threads()

                    T.gemm(k0, k0, G00, transpose_B=True)
                    T.gemm(k1, k0, G10, transpose_B=True)
                    T.gemm(k1, k1, G11, transpose_B=True)
                    T.gemm(k2, k0, G20, transpose_B=True)
                    T.gemm(k2, k1, G21, transpose_B=True)
                    T.gemm(k2, k2, G22, transpose_B=True)
                    T.gemm(k3, k0, G30, transpose_B=True)
                    T.gemm(k3, k1, G31, transpose_B=True)
                    T.gemm(k3, k2, G32, transpose_B=True)
                    T.gemm(k3, k3, G33, transpose_B=True)

                for t in T.Parallel(block_t):
                    g_val = T.cast(g_s[t], accum_dtype)
                    gate0_s[t] = T.exp2(
                        (g_val - T.cast(g_s[0], accum_dtype)) * _LOG2E
                    )
                    gate1_s[t] = T.exp2(
                        (g_val - T.cast(g_s[block_c], accum_dtype)) * _LOG2E
                    )
                    gate2_s[t] = T.exp2(
                        (g_val - T.cast(g_s[2 * block_c], accum_dtype)) * _LOG2E
                    )
                    gate3_s[t] = T.exp2(
                        (g_val - T.cast(g_s[3 * block_c], accum_dtype)) * _LOG2E
                    )
                    beta_f_s[t] = beta_s[t]
                T.sync_threads()

                for i, j in T.Parallel(block_c, block_c):
                    s00 = (
                        T.cast(beta_f_s[i], accum_dtype)
                        * T.cast(gate0_s[i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s11 = (
                        T.cast(beta_f_s[block_c + i], accum_dtype)
                        * T.cast(gate1_s[block_c + i], accum_dtype)
                        / T.cast(gate1_s[block_c + j], accum_dtype)
                    )
                    s22 = (
                        T.cast(beta_f_s[2 * block_c + i], accum_dtype)
                        * T.cast(gate2_s[2 * block_c + i], accum_dtype)
                        / T.cast(gate2_s[2 * block_c + j], accum_dtype)
                    )
                    s33 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate3_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate3_s[3 * block_c + j], accum_dtype)
                    )
                    s10 = (
                        T.cast(beta_f_s[block_c + i], accum_dtype)
                        * T.cast(gate0_s[block_c + i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s20 = (
                        T.cast(beta_f_s[2 * block_c + i], accum_dtype)
                        * T.cast(gate0_s[2 * block_c + i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s21 = (
                        T.cast(beta_f_s[2 * block_c + i], accum_dtype)
                        * T.cast(gate1_s[2 * block_c + i], accum_dtype)
                        / T.cast(gate1_s[block_c + j], accum_dtype)
                    )
                    s30 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate0_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s31 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate1_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate1_s[block_c + j], accum_dtype)
                    )
                    s32 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate2_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate2_s[2 * block_c + j], accum_dtype)
                    )
                    a_s[0, i, j] = T.if_then_else(
                        i > j,
                        -G00[i, j] * s00,
                        T.float32(0.0),
                    )
                    a_s[2, i, j] = T.if_then_else(
                        i > j,
                        -G11[i, j] * s11,
                        T.float32(0.0),
                    )
                    a_s[5, i, j] = T.if_then_else(
                        i > j,
                        -G22[i, j] * s22,
                        T.float32(0.0),
                    )
                    a_s[9, i, j] = T.if_then_else(
                        i > j,
                        -G33[i, j] * s33,
                        T.float32(0.0),
                    )
                    a_s[1, i, j] = G10[i, j] * s10
                    a_s[3, i, j] = G20[i, j] * s20
                    a_s[4, i, j] = G21[i, j] * s21
                    a_s[6, i, j] = G30[i, j] * s30
                    a_s[7, i, j] = G31[i, j] * s31
                    a_s[8, i, j] = G32[i, j] * s32
                    i_s[0, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                    i_s[1, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                    i_s[2, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                    i_s[3, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                T.sync_threads()

                for _r in T.Serial(1):
                    T.clear(tmp)
                    T.gemm(a_s[0, :, :], i_s[0, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[0, i, j] = i_s[0, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[0, :, :], a_s[0, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[0, i, j] = tmp[i, j]

                    T.clear(tmp)
                    T.gemm(a_s[2, :, :], i_s[1, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[1, i, j] = i_s[1, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[2, :, :], a_s[2, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[2, i, j] = tmp[i, j]

                    T.clear(tmp)
                    T.gemm(a_s[5, :, :], i_s[2, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[2, i, j] = i_s[2, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[5, :, :], a_s[5, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[5, i, j] = tmp[i, j]

                    T.clear(tmp)
                    T.gemm(a_s[9, :, :], i_s[3, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[3, i, j] = i_s[3, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[9, :, :], a_s[9, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[9, i, j] = tmp[i, j]
                T.sync_threads()

                T.clear(tmp)
                T.gemm(i_s[1, :, :], a_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(work_s[0, :, :], i_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[1, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(i_s[2, :, :], a_s[4, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(work_s[0, :, :], i_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[4, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(a_s[3, :, :], i_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[4, :, :], a_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(i_s[2, :, :], work_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[3, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(a_s[6, :, :], i_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[7, :, :], a_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[8, :, :], a_s[3, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(i_s[3, :, :], work_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[6, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(a_s[7, :, :], i_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[8, :, :], a_s[4, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(i_s[3, :, :], work_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[7, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(i_s[3, :, :], a_s[8, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(work_s[0, :, :], i_s[2, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[8, i, j] = -tmp[i, j]
                T.sync_threads()

                for i, j in T.Parallel(block_c, block_c):
                    A[bid, base + i, hid, j] = T.cast(i_s[0, i, j], dtype)
                    A[bid, base + block_c + i, hid, j] = T.cast(a_s[1, i, j], dtype)
                    A[bid, base + block_c + i, hid, block_c + j] = T.cast(i_s[1, i, j], dtype)
                    A[bid, base + 2 * block_c + i, hid, j] = T.cast(a_s[3, i, j], dtype)
                    A[bid, base + 2 * block_c + i, hid, block_c + j] = T.cast(a_s[4, i, j], dtype)
                    A[bid, base + 2 * block_c + i, hid, 2 * block_c + j] = T.cast(i_s[2, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, j] = T.cast(a_s[6, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, block_c + j] = T.cast(a_s[7, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, 2 * block_c + j] = T.cast(a_s[8, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, 3 * block_c + j] = T.cast(i_s[3, i, j], dtype)

        return prefill_blocksolve_A_bthd_maca

    return _func(64)


def _prefill_blocksolve_A_bthd_maca(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch, seq_len, head, dim_k = k.shape
    A = torch.zeros(batch, seq_len, head, chunk_size, dtype=k.dtype, device=k.device)
    kernel = _prefill_blocksolve_A_bthd_tl_maca(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        str(k.dtype).split(".")[-1],
    )
    kernel(k, g, beta, A)
    return A


@functools.lru_cache(maxsize=32)
def _prefill_recompute_w_u_from_A_bthd_tl_maca(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
):
    """MACA recompute: sync T.copy (no async_copy / ptx_wait_group)."""
    block_c = chunk_size
    block_d = 64
    num_k_tiles = dim_k // block_d
    num_v_tiles = dim_v // block_d
    accum_dtype = "float32"

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def recompute_w_u_from_A_bthd_maca(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            v: T.Tensor([batch, seq_len, head, dim_v], dtype),
            beta: T.Tensor([batch, seq_len, head], dtype),
            A: T.Tensor([batch, seq_len, head, chunk_size], dtype),
            w: T.Tensor([batch, seq_len, head, dim_k], dtype),
            u: T.Tensor([batch, seq_len, head, dim_v], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_c, threads=threads) as (
                bid,
                hid,
                by,
            ):
                base = by * block_c
                A_s = T.alloc_shared([block_c, block_c], dtype)
                beta_s = T.alloc_shared([block_c], dtype)
                x_s = T.alloc_shared([block_c, block_d], dtype)
                out_s = T.alloc_shared([block_c, block_d], dtype)
                out_frag = T.alloc_fragment([block_c, block_d], accum_dtype)

                T.copy(
                    A[bid, base : base + block_c, hid, 0:block_c],
                    A_s,
                    disable_tma=True,
                )
                T.copy(
                    beta[bid, base : base + block_c, hid],
                    beta_s,
                    disable_tma=True,
                )
                T.sync_threads()

                for vt in T.Serial(num_v_tiles):
                    v_offset = vt * block_d
                    T.copy(
                        v[
                            bid,
                            base : base + block_c,
                            hid,
                            v_offset : v_offset + block_d,
                        ],
                        x_s,
                        disable_tma=True,
                    )
                    T.sync_threads()
                    for i, d in T.Parallel(block_c, block_d):
                        x_s[i, d] = x_s[i, d] * beta_s[i]
                    T.clear(out_frag)
                    T.gemm(A_s, x_s, out_frag)
                    T.copy(out_frag, out_s)
                    T.copy(
                        out_s,
                        u[
                            bid,
                            base : base + block_c,
                            hid,
                            v_offset : v_offset + block_d,
                        ],
                        disable_tma=True,
                    )

                for kt in T.Serial(num_k_tiles):
                    k_offset = kt * block_d
                    T.copy(
                        k[
                            bid,
                            base : base + block_c,
                            hid,
                            k_offset : k_offset + block_d,
                        ],
                        x_s,
                        disable_tma=True,
                    )
                    T.sync_threads()
                    for i, d in T.Parallel(block_c, block_d):
                        x_s[i, d] = x_s[i, d] * beta_s[i]
                    T.clear(out_frag)
                    T.gemm(A_s, x_s, out_frag)
                    T.copy(out_frag, out_s)
                    T.copy(
                        out_s,
                        w[
                            bid,
                            base : base + block_c,
                            hid,
                            k_offset : k_offset + block_d,
                        ],
                        disable_tma=True,
                    )

        return recompute_w_u_from_A_bthd_maca

    return _func


def _run_bhtd_prefill_maca(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    fused_num_stages: int,
    fused_threads: int,
    h_num_stages: int,
    h_threads: int,
    h_block_v: int,
    o_threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from .gated_deltanet_prefill import _prefill_prepare_w_u_bhtd_tl

    if chunk_size == 64 and batch * head > 64:
        from .gated_deltanet_prefill import _prefill_chunk_local_cumsum_bhtd_tl

        g_cum = _prefill_chunk_local_cumsum_bhtd_tl(
            batch, head, seq_len, chunk_size, dtype
        )(g)
    else:
        g_cum = _chunk_local_cumsum(g.float(), chunk_size).to(g.dtype)

    prepare_fn = _prefill_prepare_w_u_bhtd_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype
    )(fused_num_stages, fused_threads)
    h_fn = _prefill_h_recurrence_bhtd_tl_maca(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        block_v=h_block_v,
    )(h_num_stages, h_threads)
    o_fn = _prefill_output_o_bhtd_tl_maca(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype
    )(o_threads)
    S_0 = torch.zeros(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    w, u = prepare_fn(k, v, g_cum, beta)
    states, v_new = h_fn(k, g_cum, w, u, S_0)
    o = o_fn(q, k, g_cum, states, v_new)
    final_state = states[:, :, -1, :, :].contiguous()
    return o, final_state


def _run_bthd_prefill_maca(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    fused_num_stages: int,
    fused_threads: int,
    h_num_stages: int,
    h_threads: int,
    h_block_v: int,
    o_threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decomposed bthd path (no fused_gdr / gdn_prefill)."""
    from .gated_deltanet_prefill import (
        _prefill_chunk_local_cumsum_bthd_tl,
        _prefill_prepare_w_u_bthd_tl,
    )

    g_cum = _prefill_chunk_local_cumsum_bthd_tl(
        batch, head, seq_len, chunk_size, dtype
    )(g)
    use_blocksolve_prepare = (
        chunk_size == 64
        and dim_k == 128
        and dim_v == 128
        and dtype == "float16"
    )

    if use_blocksolve_prepare:
        A = _prefill_blocksolve_A_bthd_maca(k, g_cum, beta, chunk_size)
        recompute_fn = _prefill_recompute_w_u_from_A_bthd_tl_maca(
            batch, head, seq_len, chunk_size, dim_k, dim_v, dtype
        )(128)
        w, u = recompute_fn(k, v, beta, A)
    else:
        prepare_fn = _prefill_prepare_w_u_bthd_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, dtype
        )(fused_num_stages, fused_threads)
        w, u = prepare_fn(k, v, g_cum, beta)

    q_h = q.permute(0, 2, 1, 3).contiguous()
    k_h = k.permute(0, 2, 1, 3).contiguous()
    g_h = g_cum.permute(0, 2, 1).contiguous()
    w_h = w.permute(0, 2, 1, 3).contiguous()
    u_h = u.permute(0, 2, 1, 3).contiguous()
    h_fn = _prefill_h_recurrence_bhtd_tl_maca(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        block_v=h_block_v,
    )(h_num_stages, h_threads)
    o_fn = _prefill_output_o_bhtd_tl_maca(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype
    )(o_threads)
    S_0 = torch.zeros(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    states, v_new_h = h_fn(k_h, g_h, w_h, u_h, S_0)
    o_h = o_fn(q_h, k_h, g_h, states, v_new_h)
    o = o_h.permute(0, 2, 1, 3).contiguous()
    final_state = states[:, :, -1, :, :].contiguous()
    return o, final_state


class GatedDeltaNetPrefillFwdMACAKernel(Kernel):
    """MACA Gated DeltaNet zero-state prefill (drop-in; decomposed path)."""

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        head: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: str = "float32",
        config: Optional[dict] = None,
        layout: str = "bhtd",
        tune: bool = False,
    ):
        super().__init__()
        layout = layout.lower()
        if layout == "bhsd":
            layout = "bhtd"
        if layout not in ("bhtd", "bthd"):
            raise ValueError(f"Unsupported layout: {layout}")
        self.batch = batch
        self.head = head
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.layout = layout
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """MACA-safe tiling: BV N-dim must be multiple of 16; keep smem under 64KB."""
        streams = self.batch * self.head
        dim_v = self.dim_v

        def _safe_bv(raw: int) -> int:
            # MACA gemm: N % k_n_per_warp(16) == 0. Prefer <=32 for smem.
            cand = dim_v if raw <= 0 else min(raw, dim_v)
            if cand % 16 != 0:
                cand = max(16, (cand // 16) * 16)
            if cand > 32 and dim_v % 32 == 0:
                cand = 32
            elif cand > 32 and dim_v % 16 == 0:
                cand = 16
            if cand > dim_v or dim_v % cand != 0:
                for b in (32, 16):
                    if dim_v % b == 0:
                        return b
                return dim_v
            return cand

        if self.layout == "bthd":
            if self.chunk_size == 64 and streams >= 64:
                h_block_v, h_num_stages, h_threads = 32, 1, 256
            else:
                h_block_v, h_num_stages, h_threads = 16, 1, 128
            return {
                "fused_num_stages": 1,
                "fused_threads": 128,
                "h_num_stages": h_num_stages,
                "h_threads": h_threads,
                "h_block_v": _safe_bv(h_block_v),
                "o_threads": 256,
            }

        # Start from CUDA-ish heuristics, then clamp for MACA.
        if self.chunk_size >= 128 and 1 < streams <= 8:
            h_block_v, h_num_stages, h_threads = 16, 1, 64
        elif self.chunk_size >= 64 and streams <= 48:
            h_block_v, h_num_stages, h_threads = 16, 1, 128
        elif self.chunk_size == 64 and streams <= 66:
            h_block_v, h_num_stages, h_threads = 32, 1, 256
        elif self.chunk_size >= 64:
            h_block_v, h_num_stages, h_threads = 16, 1, 128
        else:
            # chunk_size < 64 (e.g. 32): full-DV + stages=2 easily exceeds 64KB smem.
            h_block_v, h_num_stages, h_threads = 16, 1, 128

        fused_threads = 128 if self.chunk_size >= 64 else 256
        o_threads = 128 if self.chunk_size == 64 and streams > 64 else 256
        return {
            "fused_num_stages": 1,
            "fused_threads": fused_threads,
            "h_num_stages": h_num_stages,
            "h_threads": h_threads,
            "h_block_v": _safe_bv(h_block_v),
            "o_threads": o_threads,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        args = (
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            cfg["fused_num_stages"],
            cfg["fused_threads"],
            cfg["h_num_stages"],
            cfg["h_threads"],
            cfg.get("h_block_v", 0),
            cfg["o_threads"],
            q,
            k,
            v,
            g,
            beta,
        )
        if self.layout == "bthd":
            return _run_bthd_prefill_maca(*args)
        return _run_bhtd_prefill_maca(*args)
