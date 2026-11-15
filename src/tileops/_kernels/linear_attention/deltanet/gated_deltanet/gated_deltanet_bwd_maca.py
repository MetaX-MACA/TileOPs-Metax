"""
Gated DeltaNet backward: given dL/do, compute dL/d(q, k, v, g, beta).

Backward (split for SM utilisation):
  1. fused_prepare_compute_w_u: recompute w, u from forward
  2. bwd_parallel:    per-chunk gradients (grid: num_chunks x B x H)
  3. dh_recurrence_bwd: sequential dh propagation + corrections (grid: B x H)
  4. compute_w_u_bwd: dw, du -> dk_wu, dv, dbeta
  5. merge: dk = dk_parallel + dk_correction + dk_wu
"""

import functools
import math
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.kernel_base import Kernel

__all__ = [
    "GatedDeltaNetBwdMACAKernel",
]

_LOG2E = 1.4426950408889634
_MACA_SMEM_CAP = 65536
_MACA_SMEM_SLACK = 2048
_TILE_CANDIDATES = (64, 32, 16)


def _dtype_nbytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


def _maca_safe_threads(m: int, n: int, preferred: int, warp_size: int = 64) -> int:
    """Cap threads so MACA MMA can partition gemm ``(M, N)`` across warps."""
    cells = max(1, (m // 16) * (n // 16))
    max_threads = cells * warp_size
    t = min(int(preferred), int(max_threads))
    t = max(warp_size, t - (t % warp_size))
    return t


def _require_tile(dim: int, tile: int, axis: str) -> None:
    if tile <= 0:
        raise ValueError(f"{axis} tile must be positive, got {tile}")
    if dim % tile != 0:
        raise ValueError(f"{axis}={dim} is not divisible by tile={tile}")
    if tile % 16 != 0:
        raise ValueError(f"{axis} tile={tile} is not a multiple of 16 (MACA MMA)")


def _chunk_local_cumsum(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    B, H, S = g.shape
    return g.reshape(B, H, S // chunk_size, chunk_size).cumsum(-1).reshape(B, H, S)


def _fused_smem(chunk_size: int, bk: int, bv: int, dtype: str) -> int:
    elem = _dtype_nbytes(dtype)
    kv_w = bk if bk >= bv else bv
    return (
        chunk_size * bk * elem
        + chunk_size * bv * elem
        + chunk_size * kv_w * 4
        + chunk_size * 4 * 2
        + chunk_size * chunk_size * 4 * 2
    )


def _bwd_parallel_smem(chunk_size: int, bk: int, bv: int, dtype: str) -> int:
    elem = _dtype_nbytes(dtype)
    # g_c is float32 (Gamma = exp(g_i-g_j) must not overflow in fp16).
    # attn/d_attn stay dtype after fp32 fragment scaling (bounded products).
    return (
        4 * chunk_size * bk * elem
        + 5 * chunk_size * bv * elem
        + bk * bv * elem
        + 2 * chunk_size * chunk_size * elem
        + 5 * chunk_size * elem
        + chunk_size * 4  # g_c fp32
        + chunk_size * 4  # dg_step5_acc fp32
        + 4
    )


def _recurrence_resident_smem(chunk_size: int, dim_k: int, bk: int, bv: int, dtype: str) -> int:
    elem = _dtype_nbytes(dtype)
    return (
        chunk_size * elem * 2
        + chunk_size * bk * elem * 3
        + chunk_size * bv * elem * 2
        + 3 * dim_k * bv * elem
        + bk * bv * elem
    )


def _recurrence_kvtiled_smem(chunk_size: int, bk: int, bv: int, dtype: str) -> int:
    elem = _dtype_nbytes(dtype)
    return (
        chunk_size * elem * 2
        + chunk_size * bk * elem * 3
        + chunk_size * bv * elem * 2
        + 4 * bk * bv * elem
    )


def _segment_summary_smem(chunk_size: int, bk: int, bv: int, dtype: str) -> int:
    # g_c is float32; dh_loc stays input dtype; summary carry tile is float32.
    elem = _dtype_nbytes(dtype)
    return chunk_size * 4 + bk * bv * elem + bk * bv * 4


def _segment_boundary_smem(bk: int, bv: int, dtype: str) -> int:
    del dtype
    # local + carry tiles both float32.
    return 2 * bk * bv * 4


def _segment_local_carry_smem(chunk_size: int, bk: int, bv: int, dtype: str) -> int:
    # g_c float32; dh_loc input dtype; carry tile float32.
    elem = _dtype_nbytes(dtype)
    return chunk_size * 4 + bk * bv * elem + bk * bv * 4


def _correction_from_carry_smem(chunk_size: int, bk: int, bv: int, dtype: str) -> int:
    # g_c float32; dh_buf dtype for GEMM (fp32 carry downcast via fragment).
    elem = _dtype_nbytes(dtype)
    return (
        chunk_size * 4
        + chunk_size * elem
        + chunk_size * bk * elem * 3
        + chunk_size * bv * elem * 2
        + 2 * bk * bv * elem
    )


def _segmented_path_smem(chunk_size: int, bk: int, bv: int, dtype: str) -> int:
    return max(
        _segment_summary_smem(chunk_size, bk, bv, dtype),
        _segment_boundary_smem(bk, bv, dtype),
        _segment_local_carry_smem(chunk_size, bk, bv, dtype),
        _correction_from_carry_smem(chunk_size, bk, bv, dtype),
    )


def _pick_segment_chunks(num_chunks: int) -> int:
    if num_chunks <= 0:
        return 1
    if num_chunks % 8 == 0:
        return 8
    if num_chunks % 4 == 0:
        return 4
    if num_chunks % 2 == 0:
        return 2
    return 1


def _resolve_recurrence_for_bv(
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    block_v: int,
    *,
    for_segmented: bool = False,
) -> Tuple[int, int, int]:
    """Honor forced BV; pick largest BK (and k_tiled) that fits under 64KiB.

    Returns ``(block_k, block_v, recurrence_k_tiled)``.
    """
    bv = dim_v if block_v <= 0 else block_v
    _require_tile(dim_v, bv, "dim_v")
    if not for_segmented:
        for bk in _TILE_CANDIDATES:
            if dim_k % bk != 0:
                continue
            used = _recurrence_resident_smem(chunk_size, dim_k, bk, bv, dtype)
            if used + _MACA_SMEM_SLACK <= _MACA_SMEM_CAP:
                return bk, bv, 0
        for bk in _TILE_CANDIDATES:
            if dim_k % bk != 0:
                continue
            used = _recurrence_kvtiled_smem(chunk_size, bk, bv, dtype)
            if used + _MACA_SMEM_SLACK <= _MACA_SMEM_CAP:
                return bk, bv, 1
    else:
        for bk in _TILE_CANDIDATES:
            if dim_k % bk != 0:
                continue
            used = _segmented_path_smem(chunk_size, bk, bv, dtype)
            if used + _MACA_SMEM_SLACK <= _MACA_SMEM_CAP:
                # Segmented always uses [BK,BV] state; k_tiled marks BK < DK.
                return bk, bv, int(bk < dim_k)
    raise ValueError(
        f"MACA recurrence: no BK in {_TILE_CANDIDATES} fits under "
        f"{_MACA_SMEM_CAP} bytes for dim_k={dim_k} dim_v={dim_v} "
        f"block_v={bv} for_segmented={for_segmented}"
    )


def _wu_bwd_smem(chunk_size: int, bk: int, bv: int, dtype: str) -> int:
    elem = _dtype_nbytes(dtype)
    return (
        2 * chunk_size * chunk_size * elem
        + 3 * chunk_size * bk * elem
        + 3 * chunk_size * bv * elem
        + 2 * chunk_size * elem
        + 5 * chunk_size * 4
    )


def _pick_stage_tiles(dim_k: int, dim_v: int, smem_fn, stage: str) -> Tuple[int, int]:
    best = None
    for bk in _TILE_CANDIDATES:
        if dim_k % bk != 0:
            continue
        for bv in _TILE_CANDIDATES:
            if dim_v % bv != 0:
                continue
            used = smem_fn(bk, bv)
            if used + _MACA_SMEM_SLACK > _MACA_SMEM_CAP:
                continue
            score = (bk * bv, bk + bv)
            if best is None or score > best[0]:
                best = (score, bk, bv)
    if best is None:
        raise ValueError(
            f"MACA {stage}: no BK/BV in {_TILE_CANDIDATES} fits under "
            f"{_MACA_SMEM_CAP} bytes for dim_k={dim_k} dim_v={dim_v}"
        )
    return best[1], best[2]


def _plan_bwd_config(
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    seq_len: int = 0,
) -> dict:
    if chunk_size % 16 != 0:
        raise ValueError(f"chunk_size={chunk_size} must be a multiple of 16")
    preferred = 256 if chunk_size >= 64 else 128
    fused_bk, fused_bv = _pick_stage_tiles(
        dim_k,
        dim_v,
        lambda bk, bv: _fused_smem(chunk_size, bk, bv, dtype),
        "fused_prepare",
    )
    par_bk, par_bv = _pick_stage_tiles(
        dim_k,
        dim_v,
        lambda bk, bv: _bwd_parallel_smem(chunk_size, bk, bv, dtype),
        "bwd_parallel",
    )
    wu_bk, wu_bv = _pick_stage_tiles(
        dim_k,
        dim_v,
        lambda bk, bv: _wu_bwd_smem(chunk_size, bk, bv, dtype),
        "wu_bwd",
    )
    try:
        rec_bk, rec_bv = _pick_stage_tiles(
            dim_k,
            dim_v,
            lambda bk, bv: _recurrence_resident_smem(chunk_size, dim_k, bk, bv, dtype),
            "dh_recurrence_resident",
        )
        rec_k_tiled = 0
    except ValueError:
        rec_bk, rec_bv = _pick_stage_tiles(
            dim_k,
            dim_v,
            lambda bk, bv: _recurrence_kvtiled_smem(chunk_size, bk, bv, dtype),
            "dh_recurrence_kvtiled",
        )
        rec_k_tiled = 1

    num_chunks = seq_len // chunk_size if seq_len > 0 else 0
    segment_chunks = _pick_segment_chunks(num_chunks)
    use_segmented = 0
    if (
        chunk_size >= 64
        and dim_v > 64
        and dim_v % 64 == 0
        and num_chunks >= 32
        and segment_chunks > 1
    ):
        try:
            seg_bk, seg_bv = _pick_stage_tiles(
                dim_k,
                dim_v,
                lambda bk, bv: _segmented_path_smem(chunk_size, bk, bv, dtype),
                "dh_segmented_carry",
            )
            rec_bk, rec_bv = seg_bk, seg_bv
            rec_k_tiled = int(seg_bk < dim_k)
            use_segmented = 1
        except ValueError:
            use_segmented = 0

    kv_w = fused_bk if fused_bk >= fused_bv else fused_bv
    fused_threads = _maca_safe_threads(chunk_size, min(chunk_size, kv_w), preferred)
    parallel_threads = _maca_safe_threads(par_bk, par_bv, preferred)
    recurrence_threads = _maca_safe_threads(chunk_size, min(rec_bk, rec_bv), preferred)
    wu_threads = min(
        _maca_safe_threads(chunk_size, wu_bk, preferred),
        _maca_safe_threads(chunk_size, wu_bv, preferred),
    )
    return {
        "num_stages": 1,
        "threads": fused_threads,
        "wu_threads": wu_threads,
        "parallel_threads": parallel_threads,
        "recurrence_threads": recurrence_threads,
        "block_k": fused_bk,
        "block_v": fused_bv,
        "fused_block_k": fused_bk,
        "fused_block_v": fused_bv,
        "parallel_block_k": par_bk,
        "parallel_block_v": par_bv,
        "wu_block_k": wu_bk,
        "wu_block_v": wu_bv,
        "recurrence_block_k": rec_bk,
        "recurrence_block_v": rec_bv,
        "recurrence_k_tiled": rec_k_tiled,
        "recurrence_segmented_carry": use_segmented,
        "recurrence_segment_chunks": segment_chunks,
    }


# =============================================================================
# MACA fused prepare: K/V-tiled (64KB smem)
# =============================================================================


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _fused_prepare_compute_w_u_maca_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_k: int = 0,
    block_v: int = 0,
):
    """Fused WY + w/u recompute with K/V tiles instead of full DK/DV shared.

    ``S``/``P`` stay ``[BC, BC]`` fp32 for the Neumann solve. ``k``/``v`` are
    loaded as ``[BC, BK]`` / ``[BC, BV]`` tiles so there is no extra fp32
    ``k_beta[BC, DK]`` buffer (the d128 64KB hole).
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    KV_W = BK if BK >= BV else BV
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _fused_func(num_stages, threads=128):
        @T.prim_func
        def fused_prepare_compute_w_u_maca(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                k_tile = T.alloc_shared([block_C, BK], dtype)
                v_tile = T.alloc_shared([block_C, BV], dtype)
                kv_beta = T.alloc_shared([block_C, KV_W], accum_dtype)
                g_shared = T.alloc_shared([block_C], accum_dtype)
                beta_shared = T.alloc_shared([block_C], accum_dtype)
                S_shared = T.alloc_shared([block_C, block_C], accum_dtype)
                P_shared = T.alloc_shared([block_C, block_C], accum_dtype)
                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                kv_frag = T.alloc_fragment([block_C, KV_W], accum_dtype)

                T.copy(
                    g[bid, hid, by * block_C : (by + 1) * block_C],
                    g_shared,
                    disable_tma=True,
                )
                T.copy(
                    beta[bid, hid, by * block_C : (by + 1) * block_C],
                    beta_shared,
                    disable_tma=True,
                )

                T.clear(gram_frag)
                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.copy(
                        k[
                            bid,
                            hid,
                            by * block_C : (by + 1) * block_C,
                            koff : koff + BK,
                        ],
                        k_tile,
                        disable_tma=True,
                    )
                    T.gemm(k_tile, k_tile, gram_frag, transpose_B=True)

                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(
                        i > j,
                        -gram_frag[i, j]
                        * beta_shared[i]
                        * T.exp2((g_shared[i] - g_shared[j]) * _LOG2E),
                        T.float32(0.0),
                    )
                for i, j in T.Parallel(block_C, block_C):
                    S_shared[i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))

                for _r in T.serial(0, num_rounds):
                    T.clear(temp_frag)
                    T.gemm(P_shared, S_shared, temp_frag)
                    for i, j in T.Parallel(block_C, block_C):
                        S_shared[i, j] = S_shared[i, j] + temp_frag[i, j]
                    T.clear(temp_frag)
                    T.gemm(P_shared, P_shared, temp_frag)
                    T.copy(temp_frag, P_shared)

                T.copy(S_shared, temp_frag)
                T.copy(
                    temp_frag,
                    Aw[bid, hid, by * block_C : (by + 1) * block_C, :],
                    disable_tma=True,
                )
                T.copy(
                    temp_frag,
                    Au[bid, hid, by * block_C : (by + 1) * block_C, :],
                    disable_tma=True,
                )

                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.copy(
                        k[
                            bid,
                            hid,
                            by * block_C : (by + 1) * block_C,
                            koff : koff + BK,
                        ],
                        k_tile,
                        disable_tma=True,
                    )
                    for i, j in T.Parallel(block_C, BK):
                        kv_beta[i, j] = k_tile[i, j] * beta_shared[i]
                    if BK < KV_W:
                        for i, j in T.Parallel(block_C, KV_W - BK):
                            kv_beta[i, BK + j] = T.float32(0.0)
                    T.clear(kv_frag)
                    T.gemm(S_shared, kv_beta, kv_frag)
                    for i, j in T.Parallel(block_C, BK):
                        w[
                            bid,
                            hid,
                            by * block_C + i,
                            koff + j,
                        ] = kv_frag[i, j]

                for vt in T.serial(0, num_v_tiles):
                    voff = vt * BV
                    T.copy(
                        v[
                            bid,
                            hid,
                            by * block_C : (by + 1) * block_C,
                            voff : voff + BV,
                        ],
                        v_tile,
                        disable_tma=True,
                    )
                    for i, j in T.Parallel(block_C, BV):
                        kv_beta[i, j] = v_tile[i, j] * beta_shared[i]
                    if BV < KV_W:
                        for i, j in T.Parallel(block_C, KV_W - BV):
                            kv_beta[i, BV + j] = T.float32(0.0)
                    T.clear(kv_frag)
                    T.gemm(S_shared, kv_beta, kv_frag)
                    for i, j in T.Parallel(block_C, BV):
                        u[
                            bid,
                            hid,
                            by * block_C + i,
                            voff + j,
                        ] = kv_frag[i, j]

        return fused_prepare_compute_w_u_maca

    return _fused_func


# =============================================================================
# Split kernel: bwd_parallel (fully parallel over chunks)
# =============================================================================


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _bwd_parallel_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_k: int = 0,
    block_v: int = 0,
):
    """Parallel per-chunk backward gradients.

    Grid: (num_chunks, batch, head) — fully parallel across chunks.
    Two V scans: first builds ``v_new``/``du``/``d_attn``/``dg3``; second
    accumulates ``dq``/``dP`` in fp32 fragments and writes ``dh_local``.
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-7, -6, -5, -4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=256):
        @T.prim_func
        def bwd_parallel_kernel(
            do: T.Tensor([batch, head, seq_len, dim_v], dtype),
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            # Outputs
            dq: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dk_partial: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dg_partial: T.Tensor([batch, head, seq_len], dtype),
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_partial: T.Tensor([batch, head, seq_len, dim_v], dtype),
            v_new_out: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
        ):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                q_c = T.alloc_shared([block_C, BK], dtype)
                k_c = T.alloc_shared([block_C, BK], dtype)
                # fp32: Gamma=exp(g_i-g_j) can be ~1e15; fp16 exp2 overflows to Inf/NaN.
                g_c = T.alloc_shared([block_C], accum_dtype)
                w_c = T.alloc_shared([block_C, BK], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                do_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([BK, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                o_part = T.alloc_shared([block_C, BV], dtype)
                d_v_new_c = T.alloc_shared([block_C, BV], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                d_attn = T.alloc_shared([block_C, block_C], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                exp_g = T.alloc_shared([block_C], dtype)
                P = T.alloc_shared([block_C, BK], dtype)
                dg_step4_row = T.alloc_shared([block_C], dtype)
                dg_step4_col = T.alloc_shared([block_C], dtype)
                dg_step5_tmp = T.alloc_shared([block_C], dtype)
                dg_step5_acc = T.alloc_shared([block_C], accum_dtype)
                dg_step5_total = T.alloc_shared([1], accum_dtype)

                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_v_new_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                d_attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_q_acc = T.alloc_fragment([block_C, BK], accum_dtype)
                dP_acc = T.alloc_fragment([block_C, BK], accum_dtype)
                gemm_tmp = T.alloc_fragment([block_C, BK], accum_dtype)
                d_k_tile = T.alloc_fragment([block_C, BK], accum_dtype)
                dw_tile = T.alloc_fragment([block_C, BK], accum_dtype)
                dh_tile_frag = T.alloc_fragment([BK, BV], accum_dtype)
                dh_sub_frag = T.alloc_fragment([BK, BV], accum_dtype)
                dg_row = T.alloc_fragment([block_C], accum_dtype)
                dg3_acc = T.alloc_fragment([block_C], accum_dtype)

                T.copy(g[bid, hid, tid * block_C : (tid + 1) * block_C], g_c, disable_tma=True)
                for i in T.Parallel(block_C):
                    exp_g[i] = T.exp2(g_c[i] * _LOG2E)

                # attn = causal(q @ k^T) * Gamma  (accumulate over K tiles)
                # Scale in fp32 fragment first, then store bounded values to dtype attn
                # for same-dtype GEMM with do_c (avoids fp16 Gamma Inf * score -> NaN).
                T.clear(attn_frag)
                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.copy(
                        q[
                            bid,
                            hid,
                            tid * block_C : (tid + 1) * block_C,
                            koff : koff + BK,
                        ],
                        q_c,
                        disable_tma=True,
                    )
                    T.copy(
                        k[
                            bid,
                            hid,
                            tid * block_C : (tid + 1) * block_C,
                            koff : koff + BK,
                        ],
                        k_c,
                        disable_tma=True,
                    )
                    T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn_frag[i, j] = T.if_then_else(
                        i >= j,
                        attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        T.float32(0.0),
                    )
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = attn_frag[i, j]

                T.clear(d_attn_frag)
                for i in T.Parallel(block_C):
                    dg3_acc[i] = T.float32(0.0)

                for vt in T.serial(0, num_v_tiles):
                    v_off = vt * BV
                    T.copy(
                        u[bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV],
                        u_c,
                        disable_tma=True,
                    )
                    T.copy(
                        do[bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV],
                        do_c,
                        disable_tma=True,
                    )

                    # v_new = u - (w @ h) * exp(g + g_last)  (reduce over K)
                    T.clear(ws_frag)
                    for kt in T.serial(0, num_k_tiles):
                        koff = kt * BK
                        T.copy(
                            w[
                                bid,
                                hid,
                                tid * block_C : (tid + 1) * block_C,
                                koff : koff + BK,
                            ],
                            w_c,
                            disable_tma=True,
                        )
                        T.copy(
                            S[bid, hid, tid, koff : koff + BK, v_off : v_off + BV],
                            h_c,
                            disable_tma=True,
                        )
                        T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_c[block_C - 1]) * _LOG2E
                        )
                    T.copy(
                        v_new_c,
                        v_new_out[
                            bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV
                        ],
                        disable_tma=True,
                    )

                    # o_part = (q @ h) * exp_g
                    T.clear(ws_frag)
                    for kt in T.serial(0, num_k_tiles):
                        koff = kt * BK
                        T.copy(
                            q[
                                bid,
                                hid,
                                tid * block_C : (tid + 1) * block_C,
                                koff : koff + BK,
                            ],
                            q_c,
                            disable_tma=True,
                        )
                        T.copy(
                            S[bid, hid, tid, koff : koff + BK, v_off : v_off + BV],
                            h_c,
                            disable_tma=True,
                        )
                        T.gemm(q_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        o_part[i, j] = ws_frag[i, j] * exp_g[i]

                    # Step 2: d_v_new = attn^T @ do -> du_partial
                    T.clear(d_v_new_frag)
                    T.gemm(attn, do_c, d_v_new_frag, transpose_A=True)
                    T.copy(d_v_new_frag, d_v_new_c)
                    T.copy(
                        d_v_new_c,
                        du_partial[
                            bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV
                        ],
                        disable_tma=True,
                    )

                    # d_attn += do @ v_new^T
                    T.gemm(do_c, v_new_c, d_attn_frag, transpose_B=True)

                    # Step 3: dg3 += rowsum(do * o_part)
                    for i, j in T.Parallel(block_C, BV):
                        o_part[i, j] = do_c[i, j] * o_part[i, j]
                    T.reduce_sum(o_part, dg_row, dim=1)
                    for i in T.Parallel(block_C):
                        dg3_acc[i] += dg_row[i]

                # ---- V-independent finalisation ----
                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(i >= j, d_attn_frag[i, j], T.float32(0.0))
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = d_attn[i, j] * attn[i, j]
                T.reduce_sum(attn, dg_step4_row, dim=1)
                T.reduce_sum(attn, dg_step4_col, dim=0)
                for i in T.Parallel(block_C):
                    dg_c[i] = dg3_acc[i] + dg_step4_row[i] - dg_step4_col[i]

                # Second Gamma scale in fp32 fragment (raw d_attn still in d_attn_frag).
                for i, j in T.Parallel(block_C, block_C):
                    d_attn_frag[i, j] = T.if_then_else(
                        i >= j,
                        d_attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        T.float32(0.0),
                    )
                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = d_attn_frag[i, j]

                for i in T.Parallel(block_C):
                    dg_step5_acc[i] = T.float32(0.0)
                T.clear(dg_step5_total)

                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.clear(d_q_acc)
                    T.clear(dP_acc)
                    for vt in T.serial(0, num_v_tiles):
                        v_off = vt * BV
                        T.copy(
                            do[
                                bid,
                                hid,
                                tid * block_C : (tid + 1) * block_C,
                                v_off : v_off + BV,
                            ],
                            do_c,
                            disable_tma=True,
                        )
                        T.copy(
                            du_partial[
                                bid,
                                hid,
                                tid * block_C : (tid + 1) * block_C,
                                v_off : v_off + BV,
                            ],
                            d_v_new_c,
                            disable_tma=True,
                        )
                        T.copy(
                            q[
                                bid,
                                hid,
                                tid * block_C : (tid + 1) * block_C,
                                koff : koff + BK,
                            ],
                            q_c,
                            disable_tma=True,
                        )
                        T.copy(
                            w[
                                bid,
                                hid,
                                tid * block_C : (tid + 1) * block_C,
                                koff : koff + BK,
                            ],
                            w_c,
                            disable_tma=True,
                        )
                        T.copy(
                            S[bid, hid, tid, koff : koff + BK, v_off : v_off + BV],
                            h_c,
                            disable_tma=True,
                        )

                        for i, j in T.Parallel(block_C, BV):
                            o_part[i, j] = do_c[i, j] * exp_g[i]
                        T.clear(gemm_tmp)
                        T.gemm(o_part, h_c, gemm_tmp, transpose_B=True)
                        for i, j in T.Parallel(block_C, BK):
                            d_q_acc[i, j] = d_q_acc[i, j] + gemm_tmp[i, j]

                        T.clear(gemm_tmp)
                        T.gemm(d_v_new_c, h_c, gemm_tmp, transpose_B=True)
                        for i, j in T.Parallel(block_C, BK):
                            dP_acc[i, j] = dP_acc[i, j] - gemm_tmp[i, j]

                        for i, j in T.Parallel(block_C, BK):
                            P[i, j] = q_c[i, j] * exp_g[i]
                        T.clear(dh_tile_frag)
                        T.gemm(P, do_c, dh_tile_frag, transpose_A=True)
                        for i, j in T.Parallel(block_C, BK):
                            P[i, j] = w_c[i, j] * T.exp2((g_c[i] + g_c[block_C - 1]) * _LOG2E)
                        T.clear(dh_sub_frag)
                        T.gemm(P, d_v_new_c, dh_sub_frag, transpose_A=True)
                        for i, j in T.Parallel(BK, BV):
                            dh_tile_frag[i, j] = dh_tile_frag[i, j] - dh_sub_frag[i, j]
                        T.copy(
                            dh_tile_frag,
                            dh_local[
                                bid,
                                hid,
                                tid,
                                koff : koff + BK,
                                v_off : v_off + BV,
                            ],
                            disable_tma=True,
                        )

                    T.copy(
                        k[
                            bid,
                            hid,
                            tid * block_C : (tid + 1) * block_C,
                            koff : koff + BK,
                        ],
                        k_c,
                        disable_tma=True,
                    )
                    T.copy(
                        q[
                            bid,
                            hid,
                            tid * block_C : (tid + 1) * block_C,
                            koff : koff + BK,
                        ],
                        q_c,
                        disable_tma=True,
                    )
                    T.copy(
                        w[
                            bid,
                            hid,
                            tid * block_C : (tid + 1) * block_C,
                            koff : koff + BK,
                        ],
                        w_c,
                        disable_tma=True,
                    )

                    T.clear(gemm_tmp)
                    T.gemm(d_attn, k_c, gemm_tmp)
                    for i, j in T.Parallel(block_C, BK):
                        gemm_tmp[i, j] = gemm_tmp[i, j] + d_q_acc[i, j]
                    T.copy(
                        gemm_tmp,
                        dq[
                            bid,
                            hid,
                            tid * block_C : (tid + 1) * block_C,
                            koff : koff + BK,
                        ],
                        disable_tma=True,
                    )

                    T.clear(d_k_tile)
                    T.gemm(d_attn, q_c, d_k_tile, transpose_A=True)
                    T.copy(
                        d_k_tile,
                        dk_partial[
                            bid,
                            hid,
                            tid * block_C : (tid + 1) * block_C,
                            koff : koff + BK,
                        ],
                        disable_tma=True,
                    )

                    for i, j in T.Parallel(block_C, BK):
                        dw_tile[i, j] = dP_acc[i, j] * T.exp2((g_c[i] + g_c[block_C - 1]) * _LOG2E)
                    for i, j in T.Parallel(block_C, BK):
                        P[i, j] = (
                            w_c[i, j] * T.exp2((g_c[i] + g_c[block_C - 1]) * _LOG2E) * dP_acc[i, j]
                        )
                    T.copy(
                        dw_tile,
                        dw[
                            bid,
                            hid,
                            tid * block_C : (tid + 1) * block_C,
                            koff : koff + BK,
                        ],
                        disable_tma=True,
                    )
                    T.reduce_sum(P, dg_step5_tmp, dim=1)
                    for i in T.Parallel(block_C):
                        dg_step5_acc[i] = dg_step5_acc[i] + dg_step5_tmp[i]

                T.reduce_sum(dg_step5_acc, dg_step5_total, dim=0)
                for i in T.Parallel(block_C):
                    dg_c[i] = dg_c[i] + dg_step5_acc[i]
                dg_c[block_C - 1] = dg_c[block_C - 1] + dg_step5_total[0]

                for i in T.Parallel(block_C):
                    dg_partial[bid, hid, tid * block_C + i] = dg_c[i]

        return bwd_parallel_kernel

    return _func


# =============================================================================
# Split kernel: dh_recurrence_bwd (sequential backward over chunks)
# =============================================================================


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _dh_recurrence_bwd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_k: int = 0,
    block_v: int = 0,
):
    """Sequential backward dh recurrence with corrections.

    Grid: (num_v_tiles, batch, head) — sequential over chunks (backward),
    parallel/independent over V-tiles.

    K and V are tiled for MACA's 64KB smem: ``k``/``k_scaled``/``dP`` are
    ``[BC, BK]``; state tiles ``h``/``dh_loc``/``dh_buf`` stay ``[DK, BV]``
    (separable in V, carried across chunks in K).
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dh_recurrence_bwd_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            dk_corr: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dg_corr: T.Tensor([batch, head, num_v_tiles, seq_len], dtype),
            dw_corr: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                v_off = vid * BV
                g_c = T.alloc_shared([block_C], dtype)
                k_c = T.alloc_shared([block_C, BK], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                k_scaled = T.alloc_shared([block_C, BK], dtype)
                dP = T.alloc_shared([block_C, BK], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                dh_buf = T.alloc_shared([dim_k, BV], dtype)
                dh_buf_k = T.alloc_shared([BK, BV], dtype)
                du_corr_c = T.alloc_shared([block_C, BV], dtype)

                dh_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                du_corr_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                dh_h_tmp = T.alloc_fragment([dim_k, BV], accum_dtype)
                d_g_pos = T.alloc_fragment([block_C], accum_dtype)
                d_g_pos_tmp = T.alloc_fragment([block_C], accum_dtype)
                d_g_last_partial = T.alloc_fragment([dim_k], accum_dtype)
                d_g_last_scalar1 = T.alloc_fragment([1], accum_dtype)
                d_g_last_scalar2 = T.alloc_fragment([1], accum_dtype)

                for i, j in T.Parallel(dim_k, BV):
                    dh_buf[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    T.copy(
                        g[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C],
                        g_c,
                        disable_tma=True,
                    )
                    T.copy(
                        v_new[
                            bid,
                            hid,
                            t_bwd * block_C : (t_bwd + 1) * block_C,
                            v_off : v_off + BV,
                        ],
                        v_new_c,
                        disable_tma=True,
                    )
                    T.copy(S[bid, hid, t_bwd, :, v_off : v_off + BV], h_c, disable_tma=True)
                    T.copy(
                        dh_local[bid, hid, t_bwd, :, v_off : v_off + BV],
                        dh_loc,
                        disable_tma=True,
                    )

                    for i, j in T.Parallel(dim_k, BV):
                        dh_frag[i, j] = dh_loc[i, j] + dh_buf[i, j] * T.exp2(
                            g_c[block_C - 1] * _LOG2E
                        )

                    # du_corr = k_scaled @ dh_buf  (accumulate over K)
                    T.clear(du_corr_frag)
                    for kt in T.serial(0, num_k_tiles):
                        koff = kt * BK
                        T.copy(
                            k[
                                bid,
                                hid,
                                t_bwd * block_C : (t_bwd + 1) * block_C,
                                koff : koff + BK,
                            ],
                            k_c,
                            disable_tma=True,
                        )
                        for pn, sk in T.Parallel(block_C, BK):
                            k_scaled[pn, sk] = k_c[pn, sk] * T.exp2(
                                (g_c[block_C - 1] - g_c[pn]) * _LOG2E
                            )
                        for i, j in T.Parallel(BK, BV):
                            dh_buf_k[i, j] = dh_buf[koff + i, j]
                        T.gemm(k_scaled, dh_buf_k, du_corr_frag)
                    T.copy(du_corr_frag, du_corr_c)
                    T.copy(
                        du_corr_c,
                        du_corr[
                            bid,
                            hid,
                            t_bwd * block_C : (t_bwd + 1) * block_C,
                            v_off : v_off + BV,
                        ],
                        disable_tma=True,
                    )

                    # dw_corr = -(du_corr @ h^T) * exp(g + g_last)
                    for kt in T.serial(0, num_k_tiles):
                        koff = kt * BK
                        for i, j in T.Parallel(BK, BV):
                            dh_buf_k[i, j] = h_c[koff + i, j]
                        T.clear(dP_frag)
                        T.gemm(du_corr_c, dh_buf_k, dP_frag, transpose_B=True)
                        for n, kk in T.Parallel(block_C, BK):
                            dw_corr[bid, hid, vid, t_bwd * block_C + n, koff + kk] = -dP_frag[
                                n, kk
                            ] * T.exp2((g_c[n] + g_c[block_C - 1]) * _LOG2E)

                    # dk_corr / dg over K tiles
                    for i in T.Parallel(block_C):
                        d_g_pos[i] = T.float32(0.0)
                    for kt in T.serial(0, num_k_tiles):
                        koff = kt * BK
                        T.copy(
                            k[
                                bid,
                                hid,
                                t_bwd * block_C : (t_bwd + 1) * block_C,
                                koff : koff + BK,
                            ],
                            k_c,
                            disable_tma=True,
                        )
                        for pn, sk in T.Parallel(block_C, BK):
                            k_scaled[pn, sk] = k_c[pn, sk] * T.exp2(
                                (g_c[block_C - 1] - g_c[pn]) * _LOG2E
                            )
                        for i, j in T.Parallel(BK, BV):
                            dh_buf_k[i, j] = dh_buf[koff + i, j]

                        T.clear(dP_frag)
                        T.gemm(v_new_c, dh_buf_k, dP_frag, transpose_B=True)
                        T.copy(dP_frag, dP)
                        for n, kk in T.Parallel(block_C, BK):
                            dk_corr[bid, hid, vid, t_bwd * block_C + n, koff + kk] = dP[
                                n, kk
                            ] * T.exp2((g_c[block_C - 1] - g_c[n]) * _LOG2E)

                        for n, kk in T.Parallel(block_C, BK):
                            dP[n, kk] = dP[n, kk] * k_scaled[n, kk]
                        T.reduce_sum(dP, d_g_pos_tmp, dim=1)
                        for n in T.Parallel(block_C):
                            d_g_pos[n] = d_g_pos[n] + d_g_pos_tmp[n]

                    for n in T.Parallel(block_C):
                        dg_c[n] = -d_g_pos[n]

                    for i, j in T.Parallel(dim_k, BV):
                        dh_h_tmp[i, j] = dh_buf[i, j] * h_c[i, j]
                    T.reduce_sum(dh_h_tmp, d_g_last_partial, dim=1)
                    T.reduce_sum(d_g_last_partial, d_g_last_scalar1, dim=0)
                    T.reduce_sum(d_g_pos, d_g_last_scalar2, dim=0)
                    dg_c[block_C - 1] = (
                        dg_c[block_C - 1]
                        + d_g_last_scalar1[0] * T.exp2(g_c[block_C - 1] * _LOG2E)
                        + d_g_last_scalar2[0]
                    )

                    for i in T.Parallel(block_C):
                        dg_corr[bid, hid, vid, t_bwd * block_C + i] = dg_c[i]

                    T.copy(dh_frag, dh_buf)

        return dh_recurrence_bwd_kernel

    return _func


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _dh_recurrence_bwd_kvtile_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_k: int = 0,
    block_v: int = 0,
):
    """KxV-tiled sequential dh recurrence for oversized DK.

    Grid: (num_k_tiles, num_v_tiles, batch * head). Each block carries only
    ``[BK, BV]`` state. ``du``/``dg`` are K-partials and must be reduced.
    ``dw_corr`` is computed afterwards from the reduced ``du_corr``.
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dh_recurrence_bwd_kvtile(
            g: T.Tensor([batch, head, seq_len], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            dk_corr: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, num_k_tiles, seq_len, dim_v], dtype),
            dg_corr: T.Tensor([batch, head, num_k_tiles, num_v_tiles, seq_len], dtype),
        ):
            with T.Kernel(num_k_tiles, num_v_tiles, batch * head, threads=threads) as (
                kid,
                vid,
                bhid,
            ):
                bid = bhid // head
                hid = bhid % head
                koff = kid * BK
                v_off = vid * BV
                g_c = T.alloc_shared([block_C], dtype)
                k_c = T.alloc_shared([block_C, BK], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([BK, BV], dtype)
                dh_loc = T.alloc_shared([BK, BV], dtype)
                k_scaled = T.alloc_shared([block_C, BK], dtype)
                dP = T.alloc_shared([block_C, BK], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                dh_buf = T.alloc_shared([BK, BV], dtype)
                du_corr_c = T.alloc_shared([block_C, BV], dtype)

                dh_frag = T.alloc_fragment([BK, BV], accum_dtype)
                du_corr_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                dh_h_tmp = T.alloc_fragment([BK, BV], accum_dtype)
                d_g_pos = T.alloc_fragment([block_C], accum_dtype)
                d_g_last_partial = T.alloc_fragment([BK], accum_dtype)
                d_g_last_scalar1 = T.alloc_fragment([1], accum_dtype)
                d_g_last_scalar2 = T.alloc_fragment([1], accum_dtype)

                for i, j in T.Parallel(BK, BV):
                    dh_buf[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    T.copy(
                        g[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C],
                        g_c,
                        disable_tma=True,
                    )
                    T.copy(
                        v_new[
                            bid,
                            hid,
                            t_bwd * block_C : (t_bwd + 1) * block_C,
                            v_off : v_off + BV,
                        ],
                        v_new_c,
                        disable_tma=True,
                    )
                    T.copy(
                        S[bid, hid, t_bwd, koff : koff + BK, v_off : v_off + BV],
                        h_c,
                        disable_tma=True,
                    )
                    T.copy(
                        dh_local[bid, hid, t_bwd, koff : koff + BK, v_off : v_off + BV],
                        dh_loc,
                        disable_tma=True,
                    )
                    T.copy(
                        k[
                            bid,
                            hid,
                            t_bwd * block_C : (t_bwd + 1) * block_C,
                            koff : koff + BK,
                        ],
                        k_c,
                        disable_tma=True,
                    )
                    for pn, sk in T.Parallel(block_C, BK):
                        k_scaled[pn, sk] = k_c[pn, sk] * T.exp2(
                            (g_c[block_C - 1] - g_c[pn]) * _LOG2E
                        )

                    for i, j in T.Parallel(BK, BV):
                        dh_frag[i, j] = dh_loc[i, j] + dh_buf[i, j] * T.exp2(
                            g_c[block_C - 1] * _LOG2E
                        )

                    T.clear(du_corr_frag)
                    T.gemm(k_scaled, dh_buf, du_corr_frag)
                    T.copy(du_corr_frag, du_corr_c)
                    T.copy(
                        du_corr_c,
                        du_corr[
                            bid,
                            hid,
                            kid,
                            t_bwd * block_C : (t_bwd + 1) * block_C,
                            v_off : v_off + BV,
                        ],
                        disable_tma=True,
                    )

                    T.clear(dP_frag)
                    T.gemm(v_new_c, dh_buf, dP_frag, transpose_B=True)
                    T.copy(dP_frag, dP)
                    for n, kk in T.Parallel(block_C, BK):
                        dk_corr[bid, hid, vid, t_bwd * block_C + n, koff + kk] = dP[n, kk] * T.exp2(
                            (g_c[block_C - 1] - g_c[n]) * _LOG2E
                        )

                    for n, kk in T.Parallel(block_C, BK):
                        dP[n, kk] = dP[n, kk] * k_scaled[n, kk]
                    T.reduce_sum(dP, d_g_pos, dim=1)
                    for n in T.Parallel(block_C):
                        dg_c[n] = -d_g_pos[n]

                    for i, j in T.Parallel(BK, BV):
                        dh_h_tmp[i, j] = dh_buf[i, j] * h_c[i, j]
                    T.reduce_sum(dh_h_tmp, d_g_last_partial, dim=1)
                    T.reduce_sum(d_g_last_partial, d_g_last_scalar1, dim=0)
                    T.reduce_sum(d_g_pos, d_g_last_scalar2, dim=0)
                    dg_c[block_C - 1] = (
                        dg_c[block_C - 1]
                        + d_g_last_scalar1[0] * T.exp2(g_c[block_C - 1] * _LOG2E)
                        + d_g_last_scalar2[0]
                    )
                    for i in T.Parallel(block_C):
                        dg_corr[bid, hid, kid, vid, t_bwd * block_C + i] = dg_c[i]

                    T.copy(dh_frag, dh_buf)

        return dh_recurrence_bwd_kvtile

    return _func


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _dw_corr_from_du_maca_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_k: int = 0,
    block_v: int = 0,
):
    """Parallel-over-chunks ``dw_corr = -(du_corr @ h^T) * exp(g+g_last)``."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def dw_corr_from_du(
            g: T.Tensor([batch, head, seq_len], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dw_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
        ):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                g_c = T.alloc_shared([block_C], dtype)
                du_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([BK, BV], dtype)
                dw_acc = T.alloc_fragment([block_C, BK], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, BK], accum_dtype)

                T.copy(g[bid, hid, tid * block_C : (tid + 1) * block_C], g_c, disable_tma=True)
                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.clear(dw_acc)
                    for vt in T.serial(0, num_v_tiles):
                        v_off = vt * BV
                        T.copy(
                            du_corr[
                                bid,
                                hid,
                                tid * block_C : (tid + 1) * block_C,
                                v_off : v_off + BV,
                            ],
                            du_c,
                            disable_tma=True,
                        )
                        T.copy(
                            S[bid, hid, tid, koff : koff + BK, v_off : v_off + BV],
                            h_c,
                            disable_tma=True,
                        )
                        T.clear(dP_frag)
                        T.gemm(du_c, h_c, dP_frag, transpose_B=True)
                        for n, kk in T.Parallel(block_C, BK):
                            dw_acc[n, kk] = dw_acc[n, kk] - dP_frag[n, kk]
                    for n, kk in T.Parallel(block_C, BK):
                        dw_corr[bid, hid, tid * block_C + n, koff + kk] = dw_acc[n, kk] * T.exp2(
                            (g_c[n] + g_c[block_C - 1]) * _LOG2E
                        )

        return dw_corr_from_du

    return _func


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _compute_w_u_bwd_maca_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_k: int = 0,
    block_v: int = 0,
):
    """Prepare backward with K/V tiles instead of full DK/DV shared.

    Math matches ``compute_w_u_bwd_full_tl``. ``A`` stays resident
    ``[BC, BC]`` (dtype). ``dw``/``k``/``k_work`` and ``du``/``v``/``v_work``
    are ``[BC, BK]`` / ``[BC, BV]``. ``matrix_b_s`` aliases ``matrix_a_s``.
    GEMM operands are dtype; accumulation is fp32.
    """
    accum_dtype = "float32"
    block_C = chunk_size
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _kernel_func(num_stages, threads=128):
        @T.prim_func
        def compute_w_u_bwd_maca(
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dw_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_partial: T.Tensor([batch, head, seq_len, dim_v], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            A: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dv: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dbeta: T.Tensor([batch, head, seq_len], dtype),
            dg: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                offset = by * block_C
                A_s = T.alloc_shared([block_C, block_C], dtype)
                dw_s = T.alloc_shared([block_C, BK], dtype)
                k_s = T.alloc_shared([block_C, BK], dtype)
                k_work_s = T.alloc_shared([block_C, BK], dtype)
                du_s = T.alloc_shared([block_C, BV], dtype)
                v_s = T.alloc_shared([block_C, BV], dtype)
                v_work_s = T.alloc_shared([block_C, BV], dtype)
                g_s = T.alloc_shared([block_C], dtype)
                beta_s = T.alloc_shared([block_C], dtype)
                matrix_a_s = T.alloc_shared([block_C, block_C], dtype)
                # matrix_b_s aliases matrix_a_s (no second [BC, BC] workspace).
                dbeta_A_s = T.alloc_shared([block_C], accum_dtype)
                dg_row_s = T.alloc_shared([block_C], accum_dtype)
                dg_col_s = T.alloc_shared([block_C], accum_dtype)
                row_acc_s = T.alloc_shared([block_C], accum_dtype)
                dbeta_v_s = T.alloc_shared([block_C], accum_dtype)

                matrix_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                vector_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                dk_A_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                dv_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                # BK and BV may differ; keep K/V reduce dests and accs apart.
                row_tmp_k = T.alloc_fragment([block_C], accum_dtype)
                row_tmp_v = T.alloc_fragment([block_C], accum_dtype)

                T.copy(A[bid, hid, offset : offset + block_C, :], A_s, disable_tma=True)
                T.copy(g[bid, hid, offset : offset + block_C], g_s, disable_tma=True)
                T.copy(beta[bid, hid, offset : offset + block_C], beta_s, disable_tma=True)

                T.clear(matrix_frag)
                T.clear(gram_frag)
                for i in T.Parallel(block_C):
                    row_acc_s[i] = T.float32(0.0)
                    dbeta_v_s[i] = T.float32(0.0)

                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.copy(
                        k[bid, hid, offset : offset + block_C, koff : koff + BK],
                        k_s,
                        disable_tma=True,
                    )
                    for i, j in T.Parallel(block_C, BK):
                        dw_s[i, j] = (
                            dw[bid, hid, offset + i, koff + j]
                            + dw_corr[bid, hid, offset + i, koff + j]
                        )
                        k_work_s[i, j] = k_s[i, j] * beta_s[i]
                    T.gemm(dw_s, k_work_s, matrix_frag, transpose_B=True)
                    T.gemm(k_s, k_s, gram_frag, transpose_B=True)
                    T.clear(vector_frag)
                    T.gemm(A_s, dw_s, vector_frag, transpose_A=True)
                    T.copy(vector_frag, k_work_s)
                    for i, j in T.Parallel(block_C, BK):
                        dw_s[i, j] = k_work_s[i, j] * k_s[i, j]
                    T.reduce_sum(dw_s, row_tmp_k, dim=1)
                    for i in T.Parallel(block_C):
                        row_acc_s[i] = row_acc_s[i] + row_tmp_k[i]

                for vt in T.serial(0, num_v_tiles):
                    voff = vt * BV
                    T.copy(
                        v[bid, hid, offset : offset + block_C, voff : voff + BV],
                        v_s,
                        disable_tma=True,
                    )
                    for i, j in T.Parallel(block_C, BV):
                        du_s[i, j] = (
                            du_partial[bid, hid, offset + i, voff + j]
                            + du_corr[bid, hid, offset + i, voff + j]
                        )
                        v_work_s[i, j] = v_s[i, j] * beta_s[i]
                    T.gemm(du_s, v_work_s, matrix_frag, transpose_B=True)
                    T.clear(dv_frag)
                    T.gemm(A_s, du_s, dv_frag, transpose_A=True)
                    T.copy(dv_frag, v_work_s)
                    for i, j in T.Parallel(block_C, BV):
                        du_s[i, j] = v_work_s[i, j] * v_s[i, j]
                        dv[bid, hid, offset + i, voff + j] = v_work_s[i, j] * beta_s[i]
                    T.reduce_sum(du_s, row_tmp_v, dim=1)
                    for i in T.Parallel(block_C):
                        dbeta_v_s[i] = dbeta_v_s[i] + row_tmp_v[i]

                T.copy(matrix_frag, matrix_a_s)

                # If A = (I + L)^-1, then dL = -A^T @ dA @ A^T.
                T.clear(matrix_frag)
                T.gemm(matrix_a_s, A_s, matrix_frag, transpose_B=True)
                T.copy(matrix_frag, matrix_a_s)
                T.clear(matrix_frag)
                T.gemm(A_s, matrix_a_s, matrix_frag, transpose_A=True)
                for i, j in T.Parallel(block_C, block_C):
                    matrix_a_s[i, j] = T.if_then_else(i > j, -matrix_frag[i, j], T.float32(0.0))
                # Snapshot dL in matrix_frag; scaled dL*beta*exp reuses matrix_a_s.
                T.copy(matrix_a_s, matrix_frag)
                for i, j in T.Parallel(block_C, block_C):
                    matrix_a_s[i, j] = (
                        matrix_a_s[i, j] * beta_s[i] * T.exp2((g_s[i] - g_s[j]) * _LOG2E)
                    )

                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.copy(
                        k[bid, hid, offset : offset + block_C, koff : koff + BK],
                        k_s,
                        disable_tma=True,
                    )
                    for i, j in T.Parallel(block_C, BK):
                        dw_s[i, j] = (
                            dw[bid, hid, offset + i, koff + j]
                            + dw_corr[bid, hid, offset + i, koff + j]
                        )
                    T.clear(vector_frag)
                    T.gemm(A_s, dw_s, vector_frag, transpose_A=True)
                    T.copy(vector_frag, k_work_s)
                    T.clear(dk_A_frag)
                    T.gemm(matrix_a_s, k_s, dk_A_frag)
                    T.clear(vector_frag)
                    T.gemm(matrix_a_s, k_s, vector_frag, transpose_A=True)
                    for i, j in T.Parallel(block_C, BK):
                        dk[bid, hid, offset + i, koff + j] = (
                            k_work_s[i, j] * beta_s[i] + dk_A_frag[i, j] + vector_frag[i, j]
                        )

                # dbeta_A and dg_A use dL * exp(g_i-g_j) * <k_i,k_j>.
                for i, j in T.Parallel(block_C, block_C):
                    matrix_a_s[i, j] = (
                        matrix_frag[i, j] * T.exp2((g_s[i] - g_s[j]) * _LOG2E) * gram_frag[i, j]
                    )
                T.reduce_sum(matrix_a_s, dbeta_A_s, dim=1)
                for i, j in T.Parallel(block_C, block_C):
                    matrix_a_s[i, j] = matrix_a_s[i, j] * beta_s[i]
                T.reduce_sum(matrix_a_s, dg_row_s, dim=1)
                T.reduce_sum(matrix_a_s, dg_col_s, dim=0)
                for i in T.Parallel(block_C):
                    dbeta[bid, hid, offset + i] = row_acc_s[i] + dbeta_v_s[i] + dbeta_A_s[i]
                    dg[bid, hid, offset + i] = dg_row_s[i] - dg_col_s[i]

        return compute_w_u_bwd_maca

    return _kernel_func


# =============================================================================
# Segmented affine dh carry (MACA, K/V-tiled for 64KB)
# =============================================================================


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _dh_segment_summary_maca_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_k: int = 0,
    block_v: int = 0,
    segment_chunks: int = 8,
):
    """Summarize a reverse chunk segment as X[left] = B + A * X[right]."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    if num_chunks % segment_chunks != 0:
        raise ValueError("num_chunks must be divisible by segment_chunks")
    num_segments = num_chunks // segment_chunks
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        del num_stages

        @T.prim_func
        def dh_segment_summary_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            segment_alpha: T.Tensor([batch, head, num_segments], "float32"),
            segment_local: T.Tensor(
                [batch, head, num_k_tiles, num_v_tiles, num_segments, BK, BV],
                "float32",
            ),
        ):
            with T.Kernel(
                num_k_tiles, num_v_tiles, num_segments * batch * head, threads=threads
            ) as (kid, vid, sbhid):
                sid = sbhid // (batch * head)
                bhid = sbhid - sid * batch * head
                bid = bhid // head
                hid = bhid - bid * head
                koff = kid * BK
                v_offset = vid * BV
                g_c = T.alloc_shared([block_C], "float32")
                dh_loc = T.alloc_shared([BK, BV], dtype)
                summary = T.alloc_shared([BK, BV], "float32")
                summary_frag = T.alloc_fragment([BK, BV], accum_dtype)
                alpha_acc = T.alloc_var(T.float32, init=1.0)

                for i, j in T.Parallel(BK, BV):
                    summary[i, j] = T.float32(0.0)

                for step in T.Serial(segment_chunks):
                    cid = sid * segment_chunks + (segment_chunks - 1 - step)
                    T.copy(
                        g[bid, hid, cid * block_C : (cid + 1) * block_C],
                        g_c,
                        disable_tma=True,
                    )
                    T.copy(
                        dh_local[bid, hid, cid, koff : koff + BK, v_offset : v_offset + BV],
                        dh_loc,
                        disable_tma=True,
                    )
                    alpha = T.exp2(g_c[block_C - 1] * _LOG2E)
                    for i, j in T.Parallel(BK, BV):
                        summary_frag[i, j] = dh_loc[i, j] + summary[i, j] * alpha
                    T.copy(summary_frag, summary)
                    alpha_acc = alpha * alpha_acc

                if kid == 0 and vid == 0:
                    segment_alpha[bid, hid, sid] = alpha_acc
                T.copy(
                    summary,
                    segment_local[bid, hid, kid, vid, sid, :, :],
                    disable_tma=True,
                )

        return dh_segment_summary_kernel

    return _func


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _dh_segment_boundary_scan_maca_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_k: int = 0,
    block_v: int = 0,
    segment_chunks: int = 8,
):
    """Reverse scan segment summaries to produce each segment's successor carry."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    if num_chunks % segment_chunks != 0:
        raise ValueError("num_chunks must be divisible by segment_chunks")
    num_segments = num_chunks // segment_chunks
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        del num_stages

        @T.prim_func
        def dh_segment_boundary_scan_kernel(
            segment_alpha: T.Tensor([batch, head, num_segments], "float32"),
            segment_local: T.Tensor(
                [batch, head, num_k_tiles, num_v_tiles, num_segments, BK, BV],
                "float32",
            ),
            segment_carry_after: T.Tensor(
                [batch, head, num_k_tiles, num_v_tiles, num_segments, BK, BV],
                "float32",
            ),
        ):
            with T.Kernel(num_k_tiles, num_v_tiles, batch * head, threads=threads) as (
                kid,
                vid,
                bhid,
            ):
                bid = bhid // head
                hid = bhid - bid * head
                local = T.alloc_shared([BK, BV], "float32")
                carry = T.alloc_shared([BK, BV], "float32")
                carry_frag = T.alloc_fragment([BK, BV], accum_dtype)

                for i, j in T.Parallel(BK, BV):
                    carry[i, j] = T.float32(0.0)

                # Serial: carry has a true cross-iteration dependence.
                for step in T.Serial(num_segments):
                    sid = num_segments - 1 - step
                    T.copy(
                        carry,
                        segment_carry_after[bid, hid, kid, vid, sid, :, :],
                        disable_tma=True,
                    )
                    T.copy(
                        segment_local[bid, hid, kid, vid, sid, :, :],
                        local,
                        disable_tma=True,
                    )
                    alpha = segment_alpha[bid, hid, sid]
                    for i, j in T.Parallel(BK, BV):
                        carry_frag[i, j] = local[i, j] + carry[i, j] * alpha
                    T.copy(carry_frag, carry)

        return dh_segment_boundary_scan_kernel

    return _func


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _dh_segment_local_carry_maca_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_k: int = 0,
    block_v: int = 0,
    segment_chunks: int = 8,
):
    """Expand segment boundary carries into per-chunk successor carries."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    if num_chunks % segment_chunks != 0:
        raise ValueError("num_chunks must be divisible by segment_chunks")
    num_segments = num_chunks // segment_chunks
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        del num_stages

        @T.prim_func
        def dh_segment_local_carry_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            segment_carry_after: T.Tensor(
                [batch, head, num_k_tiles, num_v_tiles, num_segments, BK, BV],
                "float32",
            ),
            dh_carry_after: T.Tensor(
                [batch, head, num_k_tiles, num_v_tiles, num_chunks, BK, BV],
                "float32",
            ),
        ):
            with T.Kernel(
                num_k_tiles, num_v_tiles, num_segments * batch * head, threads=threads
            ) as (kid, vid, sbhid):
                sid = sbhid // (batch * head)
                bhid = sbhid - sid * batch * head
                bid = bhid // head
                hid = bhid - bid * head
                koff = kid * BK
                v_offset = vid * BV
                g_c = T.alloc_shared([block_C], "float32")
                dh_loc = T.alloc_shared([BK, BV], dtype)
                carry = T.alloc_shared([BK, BV], "float32")
                carry_frag = T.alloc_fragment([BK, BV], accum_dtype)

                T.copy(
                    segment_carry_after[bid, hid, kid, vid, sid, :, :],
                    carry,
                    disable_tma=True,
                )

                for step in T.Serial(segment_chunks):
                    local_idx = segment_chunks - 1 - step
                    cid = sid * segment_chunks + local_idx
                    T.copy(
                        carry,
                        dh_carry_after[bid, hid, kid, vid, cid, :, :],
                        disable_tma=True,
                    )
                    T.copy(
                        g[bid, hid, cid * block_C : (cid + 1) * block_C],
                        g_c,
                        disable_tma=True,
                    )
                    T.copy(
                        dh_local[bid, hid, cid, koff : koff + BK, v_offset : v_offset + BV],
                        dh_loc,
                        disable_tma=True,
                    )
                    alpha = T.exp2(g_c[block_C - 1] * _LOG2E)
                    for i, j in T.Parallel(BK, BV):
                        carry_frag[i, j] = dh_loc[i, j] + carry[i, j] * alpha
                    T.copy(carry_frag, carry)

        return dh_segment_local_carry_kernel

    return _func


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _dh_correction_from_carry_maca_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_k: int = 0,
    block_v: int = 0,
):
    """Per-chunk corrections from precomputed successor carries ([BK,BV] state)."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=256):
        @T.prim_func
        def dh_correction_from_carry_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            dh_carry_after: T.Tensor(
                [batch, head, num_k_tiles, num_v_tiles, num_chunks, BK, BV],
                "float32",
            ),
            dk_corr: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, num_k_tiles, seq_len, dim_v], dtype),
            dg_corr: T.Tensor([batch, head, num_k_tiles, num_v_tiles, seq_len], dtype),
        ):
            with T.Kernel(num_k_tiles, num_v_tiles, num_chunks * batch * head, threads=threads) as (
                kid,
                vid,
                cbhid,
            ):
                cid = cbhid // (batch * head)
                bhid = cbhid - cid * batch * head
                bid = bhid // head
                hid = bhid - bid * head
                koff = kid * BK
                v_off = vid * BV
                g_c = T.alloc_shared([block_C], "float32")
                k_c = T.alloc_shared([block_C, BK], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([BK, BV], dtype)
                k_scaled = T.alloc_shared([block_C, BK], dtype)
                dP = T.alloc_shared([block_C, BK], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                # Carry stays fp32 in HBM; fragment downcast keeps GEMM same-dtype
                # without a second BK*BV shared tile (smem-critical for BV=128).
                dh_buf = T.alloc_shared([BK, BV], dtype)
                du_corr_c = T.alloc_shared([block_C, BV], dtype)

                du_corr_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                dh_carry_frag = T.alloc_fragment([BK, BV], accum_dtype)
                dh_h_tmp = T.alloc_fragment([BK, BV], accum_dtype)
                d_g_pos = T.alloc_fragment([block_C], accum_dtype)
                d_g_last_partial = T.alloc_fragment([BK], accum_dtype)
                d_g_last_scalar1 = T.alloc_fragment([1], accum_dtype)
                d_g_last_scalar2 = T.alloc_fragment([1], accum_dtype)

                T.copy(
                    g[bid, hid, cid * block_C : (cid + 1) * block_C],
                    g_c,
                    disable_tma=True,
                )
                T.copy(
                    v_new[
                        bid,
                        hid,
                        cid * block_C : (cid + 1) * block_C,
                        v_off : v_off + BV,
                    ],
                    v_new_c,
                    disable_tma=True,
                )
                T.copy(
                    S[bid, hid, cid, koff : koff + BK, v_off : v_off + BV],
                    h_c,
                    disable_tma=True,
                )
                T.copy(
                    dh_carry_after[bid, hid, kid, vid, cid, :, :],
                    dh_carry_frag,
                    disable_tma=True,
                )
                for i, j in T.Parallel(BK, BV):
                    dh_buf[i, j] = dh_carry_frag[i, j]
                T.copy(
                    k[
                        bid,
                        hid,
                        cid * block_C : (cid + 1) * block_C,
                        koff : koff + BK,
                    ],
                    k_c,
                    disable_tma=True,
                )
                for pn, sk in T.Parallel(block_C, BK):
                    k_scaled[pn, sk] = k_c[pn, sk] * T.exp2((g_c[block_C - 1] - g_c[pn]) * _LOG2E)

                T.clear(du_corr_frag)
                T.gemm(k_scaled, dh_buf, du_corr_frag)
                T.copy(du_corr_frag, du_corr_c)
                T.copy(
                    du_corr_c,
                    du_corr[
                        bid,
                        hid,
                        kid,
                        cid * block_C : (cid + 1) * block_C,
                        v_off : v_off + BV,
                    ],
                    disable_tma=True,
                )

                T.clear(dP_frag)
                T.gemm(v_new_c, dh_buf, dP_frag, transpose_B=True)
                T.copy(dP_frag, dP)
                for n, kk in T.Parallel(block_C, BK):
                    dk_corr[bid, hid, vid, cid * block_C + n, koff + kk] = dP[n, kk] * T.exp2(
                        (g_c[block_C - 1] - g_c[n]) * _LOG2E
                    )

                for n, kk in T.Parallel(block_C, BK):
                    dP[n, kk] = dP[n, kk] * k_scaled[n, kk]
                T.reduce_sum(dP, d_g_pos, dim=1)
                for n in T.Parallel(block_C):
                    dg_c[n] = -d_g_pos[n]

                for i, j in T.Parallel(BK, BV):
                    dh_h_tmp[i, j] = dh_buf[i, j] * h_c[i, j]
                T.reduce_sum(dh_h_tmp, d_g_last_partial, dim=1)
                T.reduce_sum(d_g_last_partial, d_g_last_scalar1, dim=0)
                T.reduce_sum(d_g_pos, d_g_last_scalar2, dim=0)
                dg_c[block_C - 1] = (
                    dg_c[block_C - 1]
                    + d_g_last_scalar1[0] * T.exp2(g_c[block_C - 1] * _LOG2E)
                    + d_g_last_scalar2[0]
                )
                for i in T.Parallel(block_C):
                    dg_corr[bid, hid, kid, vid, cid * block_C + i] = dg_c[i]

        return dh_correction_from_carry_kernel

    return _func


@torch.library.custom_op("tileops::gated_deltanet_bwd_kernel_maca", mutates_args=())
def _gated_deltanet_bwd_wrapped_kernel(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    num_stages: int,
    threads: int,
    wu_threads: int,
    parallel_threads: int,
    recurrence_threads: int,
    fused_block_k: int,
    fused_block_v: int,
    parallel_block_k: int,
    parallel_block_v: int,
    wu_block_k: int,
    wu_block_v: int,
    recurrence_block_k: int,
    recurrence_block_v: int,
    recurrence_k_tiled: int,
    recurrence_segmented_carry: int,
    recurrence_segment_chunks: int,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    S: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g_cum = _chunk_local_cumsum(g.float(), chunk_size).to(g.dtype)

    fused_fn = _fused_prepare_compute_w_u_maca_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        block_k=fused_block_k,
        block_v=fused_block_v,
    )(1, threads)
    bwd_parallel_fn = _bwd_parallel_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        block_k=parallel_block_k,
        block_v=parallel_block_v,
    )(parallel_threads)
    wu_bwd_fn = _compute_w_u_bwd_maca_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        block_k=wu_block_k,
        block_v=wu_block_v,
    )(1, wu_threads)

    Aw, _Au, w, u = fused_fn(k, v, g_cum, beta)
    dq, dk_partial, dg_partial, dw, du_partial, v_new, dh_local = bwd_parallel_fn(
        do, q, k, g_cum, w, u, S
    )

    rec_bk, rec_bv, rec_k_tiled = _resolve_recurrence_for_bv(
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        recurrence_block_v,
        for_segmented=(recurrence_segmented_carry != 0),
    )
    recurrence_block_k = rec_bk
    recurrence_block_v = rec_bv
    recurrence_k_tiled = rec_k_tiled
    recurrence_threads = _maca_safe_threads(
        chunk_size, min(recurrence_block_k, recurrence_block_v), recurrence_threads
    )

    if recurrence_segmented_carry != 0:
        seg_kwargs = dict(
            batch=batch,
            head=head,
            seq_len=seq_len,
            chunk_size=chunk_size,
            dim_k=dim_k,
            dim_v=dim_v,
            dtype=dtype,
            block_k=recurrence_block_k,
            block_v=recurrence_block_v,
            segment_chunks=recurrence_segment_chunks,
        )
        summary_fn = _dh_segment_summary_maca_tl(**seg_kwargs)(num_stages, recurrence_threads)
        boundary_fn = _dh_segment_boundary_scan_maca_tl(**seg_kwargs)(
            num_stages, recurrence_threads
        )
        local_fn = _dh_segment_local_carry_maca_tl(**seg_kwargs)(num_stages, recurrence_threads)
        corr_fn = _dh_correction_from_carry_maca_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
            block_k=recurrence_block_k,
            block_v=recurrence_block_v,
        )(recurrence_threads)
        segment_alpha, segment_local = summary_fn(g_cum, dh_local)

        segment_carry_after = boundary_fn(segment_alpha, segment_local)

        dh_carry_after = local_fn(g_cum, dh_local, segment_carry_after)

        dk_corr, du_corr, dg_corr = corr_fn(g_cum, k, v_new, S, dh_carry_after)

        if du_corr.dim() == 5:
            du_corr = du_corr.float().sum(dim=2).to(du_partial.dtype)
        if dg_corr.dim() >= 5:
            dg_corr = dg_corr.float().sum(dim=(2, 3)).to(dg_partial.dtype)
        if dk_corr.dim() == 5:
            dk_corr = dk_corr.float().sum(dim=2).to(dk_partial.dtype)

        dw_corr_fn = _dw_corr_from_du_maca_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
            block_k=recurrence_block_k,
            block_v=recurrence_block_v,
        )(recurrence_threads)
        dw_corr = dw_corr_fn(g_cum, S, du_corr)

    elif recurrence_k_tiled:
        dh_fn = _dh_recurrence_bwd_kvtile_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
            block_k=recurrence_block_k,
            block_v=recurrence_block_v,
        )(num_stages, recurrence_threads)
        dk_corr, du_corr, dg_corr = dh_fn(g_cum, k, v_new, S, dh_local)

        if du_corr.dim() == 5:
            du_corr = du_corr.float().sum(dim=2).to(du_partial.dtype)
        if dg_corr.dim() >= 5:
            dg_corr = dg_corr.float().sum(dim=(2, 3)).to(dg_partial.dtype)
        if dk_corr.dim() == 5:
            dk_corr = dk_corr.float().sum(dim=2).to(dk_partial.dtype)

        dw_corr_fn = _dw_corr_from_du_maca_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
            block_k=recurrence_block_k,
            block_v=recurrence_block_v,
        )(recurrence_threads)
        dw_corr = dw_corr_fn(g_cum, S, du_corr)

    else:
        dh_fn = _dh_recurrence_bwd_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
            block_k=recurrence_block_k,
            block_v=recurrence_block_v,
        )(num_stages, recurrence_threads)
        dk_corr, du_corr, dg_corr, dw_corr = dh_fn(g_cum, k, v_new, S, dh_local)

        if dk_corr.dim() == 5:
            dk_corr = dk_corr.float().sum(dim=2).to(dk_partial.dtype)
            dg_corr = dg_corr.float().sum(dim=2).to(dg_partial.dtype)
            dw_corr = dw_corr.float().sum(dim=2).to(dw.dtype)

    dk_wu, dv, dbeta, dg_prepare = wu_bwd_fn(
        dw,
        dw_corr,
        du_partial,
        du_corr,
        Aw,
        k,
        v,
        g_cum,
        beta,
    )

    dk = dk_partial + dk_corr + dk_wu
    dg_cum = dg_partial + dg_corr + dg_prepare

    B, H, SL = g.shape
    dg = dg_cum.float().reshape(B, H, SL // chunk_size, chunk_size)
    dg = dg.flip(-1).cumsum(-1).flip(-1).reshape(B, H, SL).to(g.dtype)

    return dq, dk, dv, dg, dbeta


@_gated_deltanet_bwd_wrapped_kernel.register_fake
def _gated_deltanet_bwd_wrapped_kernel_fake(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    num_stages: int,
    threads: int,
    wu_threads: int,
    parallel_threads: int,
    recurrence_threads: int,
    fused_block_k: int,
    fused_block_v: int,
    parallel_block_k: int,
    parallel_block_v: int,
    wu_block_k: int,
    wu_block_v: int,
    recurrence_block_k: int,
    recurrence_block_v: int,
    recurrence_k_tiled: int,
    recurrence_segmented_carry: int,
    recurrence_segment_chunks: int,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    S: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        dtype,
        num_stages,
        threads,
        wu_threads,
        parallel_threads,
        recurrence_threads,
        fused_block_k,
        fused_block_v,
        parallel_block_k,
        parallel_block_v,
        wu_block_k,
        wu_block_v,
        recurrence_block_k,
        recurrence_block_v,
        recurrence_k_tiled,
        recurrence_segmented_carry,
        recurrence_segment_chunks,
        do,
        k,
        v,
        g,
        beta,
        S,
    )
    dq = torch.empty(batch, head, seq_len, dim_k, dtype=q.dtype, device=q.device)
    dk = torch.empty_like(dq)
    dv = torch.empty(batch, head, seq_len, dim_v, dtype=q.dtype, device=q.device)
    dg = torch.empty(batch, head, seq_len, dtype=q.dtype, device=q.device)
    dbeta = torch.empty(batch, head, seq_len, dtype=q.dtype, device=q.device)
    return dq, dk, dv, dg, dbeta


class GatedDeltaNetBwdMACAKernel(Kernel):
    """Gated DeltaNet backward kernel.

    Full backward: do -> (dq, dk, dv, dg, dbeta).

    Split pipeline (Phase 2 optimisation):
      1. fused_prepare_compute_w_u: recompute w, u
      2. bwd_parallel: per-chunk gradients (grid: num_chunks x B x H)
      3. dh carry: sequential or segmented affine carry + corrections
      4. compute_w_u_bwd: dw, du -> dk_wu, dv, dbeta
      5. merge: dk = dk_partial + dk_correction + dk_wu, etc.
    """

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
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.head = head
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.init_config(config)
        if tune:
            self.request_tune()

    @property
    def default_config(self) -> dict:
        return _plan_bwd_config(
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            seq_len=self.seq_len,
        )

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        defaults = self.default_config
        return _gated_deltanet_bwd_wrapped_kernel(
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            cfg.get("num_stages", defaults["num_stages"]),
            cfg.get("threads", defaults["threads"]),
            cfg.get("wu_threads", defaults["wu_threads"]),
            cfg.get("parallel_threads", defaults["parallel_threads"]),
            cfg.get("recurrence_threads", defaults["recurrence_threads"]),
            cfg.get("fused_block_k", defaults["fused_block_k"]),
            cfg.get("fused_block_v", cfg.get("block_v", defaults["fused_block_v"])),
            cfg.get("parallel_block_k", defaults["parallel_block_k"]),
            cfg.get("parallel_block_v", cfg.get("block_v", defaults["parallel_block_v"])),
            cfg.get("wu_block_k", defaults["wu_block_k"]),
            cfg.get("wu_block_v", defaults["wu_block_v"]),
            cfg.get("recurrence_block_k", defaults["recurrence_block_k"]),
            cfg.get("recurrence_block_v", cfg.get("block_v", defaults["recurrence_block_v"])),
            cfg.get("recurrence_k_tiled", defaults["recurrence_k_tiled"]),
            cfg.get(
                "recurrence_segmented_carry",
                defaults["recurrence_segmented_carry"],
            ),
            cfg.get(
                "recurrence_segment_chunks",
                defaults["recurrence_segment_chunks"],
            ),
            do,
            q,
            k,
            v,
            g,
            beta,
            S,
        )
