"""
Gated DeltaNet forward for MACA (64KB smem).

Self-contained: does not import CUDA generic fused_prepare / fwd kernels.

Pipeline (same ABI as GatedDeltaNetFwdKernel):
  1. fused_prepare (K/V-tiled): (k, v, g, beta) -> (Aw, Au, w, u)
  2. h_recurrence (V-tiled):     (k, g, w, u, S_0) -> (S, v_new)
  3. output_o (V-tiled):         (q, k, g, S, v_new) -> o
"""

import functools
import itertools
import math
from typing import Dict, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["GatedDeltaNetFwdBTHDMACAKernel", "GatedDeltaNetFwdMACAKernel"]

_LOG2E = 1.4426950408889634
_MACA_SMEM_CAP = 65536
_MACA_SMEM_SLACK = 2048
_TILE_CANDIDATES = (64, 32, 16)
# output_o can reuse a single low-precision attn tile across all V columns.
# Unlike fused_prepare / h_recurrence, a 64x128 output tile fits in 64KB.
_OUTPUT_TILE_CANDIDATES = (128, 64, 32, 16)
_BLOCKWISE_INVERSE_CHUNK = 64
_BLOCKWISE_INVERSE_TILE = 32
_BLOCKWISE_INVERSE_STAGING_BYTES = 2 * _BLOCKWISE_INVERSE_TILE**2 * 4


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
    # ``P_shared`` is dead after the Neumann inverse has been formed.  When
    # its [C, C] shape matches the widest K/V tile, it can subsequently hold
    # k*beta or v*beta.  Count only the peak live allocation, not both buffers.
    kv_beta_bytes = 0 if kv_w == chunk_size else chunk_size * kv_w * 4
    # MACA GEMM does not accept a first-dimension-offset shared view.  The
    # C=64 blockwise inverse stages its 32x32 operands through two zero-offset
    # shared buffers before issuing each GEMM.
    inverse_workspace_bytes = (
        _BLOCKWISE_INVERSE_STAGING_BYTES if chunk_size == _BLOCKWISE_INVERSE_CHUNK else 0
    )
    return (
        chunk_size * bk * elem
        + chunk_size * bv * elem
        + kv_beta_bytes
        + chunk_size * 4 * 2
        + chunk_size * chunk_size * 4 * 2
        + inverse_workspace_bytes
    )


def _h_recurrence_smem(chunk_size: int, dim_k: int, bk: int, bv: int, dtype: str) -> int:
    elem = _dtype_nbytes(dtype)
    return (
        chunk_size * bk * elem * 2
        + chunk_size * 4
        + chunk_size * bv * elem * 2
        + dim_k * bv * elem
        + bk * bv * elem
    )


def _output_o_smem(chunk_size: int, bk: int, bv: int, dtype: str) -> int:
    elem = _dtype_nbytes(dtype)
    return (
        chunk_size * bk * elem * 2
        + chunk_size * 4
        + bk * bv * elem
        # The final attn @ v_new GEMM uses MACA low-precision MMA inputs.
        # Its fp32 accumulation remains in o_frag, not these shared buffers.
        + chunk_size * bv * elem
        + chunk_size * chunk_size * elem
    )


def _select_h_block_v(
    dim_k: int,
    dim_v: int,
    streams: int,
    candidate_bv: int,
) -> int:
    """Split V only when h recurrence would otherwise underfill MACA."""
    programs = int(streams) * (dim_v // candidate_bv)
    min_programs = 64 if dim_k >= 128 and dim_v >= 128 else 32
    if programs < min_programs and candidate_bv > 16 and dim_v % 16 == 0:
        return 16
    return candidate_bv


def _pick_stage_tiles(
    dim_k: int,
    dim_v: int,
    smem_fn,
    stage: str,
    candidates: Tuple[int, ...] = _TILE_CANDIDATES,
) -> Tuple[int, int]:
    best = None
    for bk in candidates:
        if dim_k % bk != 0:
            continue
        for bv in candidates:
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
            f"MACA {stage}: no BK/BV in {candidates} fits under "
            f"{_MACA_SMEM_CAP} bytes for dim_k={dim_k} dim_v={dim_v}"
        )
    return best[1], best[2]


def _plan_fwd_config(
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    streams: int = 1,
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
    h_bk, h_bv = _pick_stage_tiles(
        dim_k,
        dim_v,
        lambda bk, bv: _h_recurrence_smem(chunk_size, dim_k, bk, bv, dtype),
        "h_recurrence",
    )
    h_bv = _select_h_block_v(dim_k, dim_v, streams, h_bv)
    o_bk, o_bv = _pick_stage_tiles(
        dim_k,
        dim_v,
        lambda bk, bv: _output_o_smem(chunk_size, bk, bv, dtype),
        "output_o",
        candidates=_OUTPUT_TILE_CANDIDATES,
    )
    kv_w = fused_bk if fused_bk >= fused_bv else fused_bv
    fused_threads = _maca_safe_threads(chunk_size, min(chunk_size, kv_w), preferred)
    h_threads = _maca_safe_threads(min(chunk_size, h_bk), h_bv, preferred)
    o_threads = _maca_safe_threads(chunk_size, o_bv, preferred)
    return {
        "fused_num_stages": 1,
        "fused_threads": fused_threads,
        "fused_block_k": fused_bk,
        "fused_block_v": fused_bv,
        "h_num_stages": 0,
        "h_threads": h_threads,
        "h_block_k": h_bk,
        "h_block_v": h_bv,
        "o_threads": o_threads,
        "o_block_k": o_bk,
        "o_block_v": o_bv,
        "block_k": fused_bk,
    }


def _maca_thread_candidates(m: int, n: int) -> Tuple[int, ...]:
    """Return the distinct safe launch widths worth timing for one MMA shape."""
    candidates = []
    for requested in (64, 128, 256):
        threads = _maca_safe_threads(m, n, requested)
        if threads not in candidates:
            candidates.append(threads)
    return tuple(candidates)


def _maca_fwd_autotune_configs(
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    streams: int = 1,
) -> list[dict]:
    """Return the reachable MACA forward configs for one shape.

    ``BK`` and ``BV`` select the generated kernel layout, rather than a runtime
    launch parameter. The planner chooses the largest shared-memory-safe pair
    for each stage, and tuning only sweeps the launch parameters of those three
    generated kernels. This keeps a tune request bounded while guaranteeing
    every candidate has already passed the MACA 64KB shared-memory budget.
    """
    default = _plan_fwd_config(chunk_size, dim_k, dim_v, dtype, streams)
    fused_threads = _maca_thread_candidates(
        chunk_size,
        min(chunk_size, max(default["fused_block_k"], default["fused_block_v"])),
    )
    h_threads = _maca_thread_candidates(min(chunk_size, default["h_block_k"]), default["h_block_v"])
    o_threads = _maca_thread_candidates(chunk_size, default["o_block_v"])

    return [
        {
            "fused_num_stages": fused["num_stages"],
            "fused_threads": fused["threads"],
            "fused_block_k": default["fused_block_k"],
            "fused_block_v": default["fused_block_v"],
            "h_num_stages": recurrence["num_stages"],
            "h_threads": recurrence["threads"],
            "h_block_k": default["h_block_k"],
            "h_block_v": default["h_block_v"],
            "o_threads": output["threads"],
            "o_block_k": default["o_block_k"],
            "o_block_v": default["o_block_v"],
            "block_k": default["block_k"],
        }
        for fused, recurrence, output in itertools.product(
            ({"num_stages": 1, "threads": threads} for threads in fused_threads),
            (
                {"num_stages": num_stages, "threads": threads}
                for num_stages in (0, 1)
                for threads in h_threads
            ),
            ({"threads": threads} for threads in o_threads),
        )
    ]


# =============================================================================
# MACA fused prepare: K/V-tiled (64KB smem)
# =============================================================================


@functools.lru_cache(maxsize=32)
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
    """Fused WY + w/u with K/V tiles instead of full DK/DV shared."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0
    BK = dim_k if block_k <= 0 else block_k
    BV = dim_v if block_v <= 0 else block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    # The C=64 inverse is the dominant work in the original forward path:
    # six squaring rounds issue twelve 64x64 GEMMs per chunk.  ``P`` is
    # strictly lower triangular, so invert its two 32x32 diagonal blocks
    # independently, then use forward block substitution for the remaining
    # off-diagonal block.  This is algebraically the same inverse of
    # ``I - P`` but avoids the full-matrix powers.  Keep the old generic
    # method for other chunk sizes.
    use_blockwise_inverse = block_C == _BLOCKWISE_INVERSE_CHUNK
    # 32x32 keeps each local MMA sufficiently large for the 256-thread fused
    # kernel.  16x16 creates mostly-idle MACA warps in this mixed-size kernel.
    inverse_block = _BLOCKWISE_INVERSE_TILE
    inverse_parts = block_C // inverse_block if use_blockwise_inverse else 0
    inverse_rounds = int(math.ceil(math.log2(inverse_block)))
    KV_W = BK if BK >= BV else BV
    # Once the inverse is complete, P_shared is no longer live.  Reusing it
    # for the widest K/V tile avoids a second [C, C] fp32 shared allocation.
    # The alias is shape-safe only when KV_W equals C.
    reuse_p_for_kv_beta = block_C == KV_W
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
        del num_stages

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
                g_shared = T.alloc_shared([block_C], accum_dtype)
                beta_shared = T.alloc_shared([block_C], accum_dtype)
                S_shared = T.alloc_shared([block_C, block_C], accum_dtype)
                P_shared = T.alloc_shared([block_C, block_C], accum_dtype)
                kv_beta = (
                    P_shared
                    if reuse_p_for_kv_beta
                    else T.alloc_shared([block_C, KV_W], accum_dtype)
                )
                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                kv_frag = T.alloc_fragment([block_C, KV_W], accum_dtype)
                if use_blockwise_inverse:
                    inverse_frag = T.alloc_fragment([inverse_block, inverse_block], accum_dtype)
                    inverse_p = T.alloc_shared([inverse_block, inverse_block], accum_dtype)
                    inverse_s = T.alloc_shared([inverse_block, inverse_block], accum_dtype)

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

                if use_blockwise_inverse:
                    # Diagonal blocks: I + P_ii + ... + P_ii^31.  P_ii is
                    # copied into zero-offset staging buffers because MACA's
                    # GEMM lowering rejects a first-dimension-offset view.
                    for bi in T.serial(0, inverse_parts):
                        boff = bi * inverse_block
                        T.copy(
                            P_shared[
                                boff : boff + inverse_block,
                                boff : boff + inverse_block,
                            ],
                            inverse_p,
                        )
                        for _r in T.serial(0, inverse_rounds):
                            T.copy(
                                S_shared[
                                    boff : boff + inverse_block,
                                    boff : boff + inverse_block,
                                ],
                                inverse_s,
                            )
                            T.clear(inverse_frag)
                            T.gemm(
                                inverse_p,
                                inverse_s,
                                inverse_frag,
                            )
                            for i, j in T.Parallel(inverse_block, inverse_block):
                                S_shared[boff + i, boff + j] = (
                                    S_shared[boff + i, boff + j] + inverse_frag[i, j]
                                )
                            T.clear(inverse_frag)
                            T.gemm(
                                inverse_p,
                                inverse_p,
                                inverse_frag,
                            )
                            T.copy(inverse_frag, inverse_p)

                    # Block forward substitution for S = (I - P)^-1:
                    # S_ij = S_ii * sum(k=j..i-1) P_ik * S_kj.
                    for bi in T.serial(1, inverse_parts):
                        boff = bi * inverse_block
                        for bj in T.serial(0, bi):
                            joff = bj * inverse_block
                            T.clear(inverse_frag)
                            for bk in T.serial(bj, bi):
                                koff = bk * inverse_block
                                T.copy(
                                    P_shared[
                                        boff : boff + inverse_block,
                                        koff : koff + inverse_block,
                                    ],
                                    inverse_p,
                                )
                                T.copy(
                                    S_shared[
                                        koff : koff + inverse_block,
                                        joff : joff + inverse_block,
                                    ],
                                    inverse_s,
                                )
                                T.gemm(
                                    inverse_p,
                                    inverse_s,
                                    inverse_frag,
                                )
                            T.copy(inverse_frag, inverse_p)
                            T.copy(
                                S_shared[
                                    boff : boff + inverse_block,
                                    boff : boff + inverse_block,
                                ],
                                inverse_s,
                            )
                            T.clear(inverse_frag)
                            T.gemm(
                                inverse_s,
                                inverse_p,
                                inverse_frag,
                            )
                            T.copy(
                                inverse_frag,
                                S_shared[
                                    boff : boff + inverse_block,
                                    joff : joff + inverse_block,
                                ],
                            )
                else:
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
# h_recurrence (V-tiled)
# =============================================================================


@functools.lru_cache(maxsize=32)
def _h_recurrence_maca_tl(
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
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128):
        @T.prim_func
        def h_recurrence_maca(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S_0: T.Tensor([batch, head, dim_k, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                k_c = T.alloc_shared([block_C, BK], dtype)
                g_c = T.alloc_shared([block_C], accum_dtype)
                w_c = T.alloc_shared([block_C, BK], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                h_tile = T.alloc_shared([BK, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)

                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                h_next_frag = T.alloc_fragment([BK, BV], accum_dtype)

                v_offset = vid * BV

                T.copy(
                    S_0[bid, hid, :, v_offset : v_offset + BV],
                    h_c,
                    disable_tma=True,
                )
                for i, j in T.Parallel(dim_k, BV):
                    S[bid, hid, 0, i, v_offset + j] = h_c[i, j]

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(
                        g[bid, hid, t * block_C : (t + 1) * block_C],
                        g_c,
                        disable_tma=True,
                    )
                    T.copy(
                        u[
                            bid,
                            hid,
                            t * block_C : (t + 1) * block_C,
                            v_offset : v_offset + BV,
                        ],
                        u_c,
                        disable_tma=True,
                    )

                    T.clear(ws_frag)
                    for kt in T.serial(0, num_k_tiles):
                        koff = kt * BK
                        T.copy(
                            w[
                                bid,
                                hid,
                                t * block_C : (t + 1) * block_C,
                                koff : koff + BK,
                            ],
                            w_c,
                            disable_tma=True,
                        )
                        for i, j in T.Parallel(BK, BV):
                            h_tile[i, j] = h_c[koff + i, j]
                        T.gemm(w_c, h_tile, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_c[block_C - 1]) * _LOG2E
                        )

                    T.copy(
                        v_new_c,
                        v_new[
                            bid,
                            hid,
                            t * block_C : (t + 1) * block_C,
                            v_offset : v_offset + BV,
                        ],
                        disable_tma=True,
                    )

                    for n, j in T.Parallel(block_C, BV):
                        v_new_c[n, j] = v_new_c[n, j] * T.exp2((g_c[block_C - 1] - g_c[n]) * _LOG2E)
                    for kt in T.serial(0, num_k_tiles):
                        koff = kt * BK
                        T.copy(
                            k[
                                bid,
                                hid,
                                t * block_C : (t + 1) * block_C,
                                koff : koff + BK,
                            ],
                            k_c,
                            disable_tma=True,
                        )
                        for i, j in T.Parallel(BK, BV):
                            h_next_frag[i, j] = h_c[koff + i, j] * T.exp2(g_c[block_C - 1] * _LOG2E)
                        T.gemm(
                            k_c,
                            v_new_c,
                            h_next_frag,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(BK, BV):
                            h_c[koff + i, j] = h_next_frag[i, j]
                            S[bid, hid, t + 1, koff + i, v_offset + j] = h_next_frag[i, j]

        return h_recurrence_maca

    return _func


# =============================================================================
# output_o (V-tiled; grid maps V tiles for MACA MMA layout)
# =============================================================================


@functools.lru_cache(maxsize=32)
def _output_o_maca_tl(
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
    """MACA output_o with K/V-tiling to stay within 64KB smem."""
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
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def output_o_maca(
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            o: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(num_chunks, num_v_tiles, batch * head, threads=threads) as (
                tid,
                vt,
                bhid,
            ):
                bid = bhid // head
                hid = bhid % head
                base = tid * block_C
                v_offset = vt * BV

                q_c = T.alloc_shared([block_C, BK], dtype)
                k_c = T.alloc_shared([block_C, BK], dtype)
                g_c = T.alloc_shared([block_C], accum_dtype)
                h_c = T.alloc_shared([BK, BV], dtype)
                # Keep GEMM inputs in the storage dtype so MACA lowers
                # attn @ v_new to low-precision MMA; o_frag stays fp32.
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)

                o_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)

                T.copy(g[bid, hid, base : base + block_C], g_c, disable_tma=True)
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
                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.copy(
                        q[bid, hid, base : base + block_C, koff : koff + BK],
                        q_c,
                        disable_tma=True,
                    )
                    T.copy(
                        k[bid, hid, base : base + block_C, koff : koff + BK],
                        k_c,
                        disable_tma=True,
                    )
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
                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.copy(
                        q[bid, hid, base : base + block_C, koff : koff + BK],
                        q_c,
                        disable_tma=True,
                    )
                    T.copy(
                        S[bid, hid, tid, koff : koff + BK, v_offset : v_offset + BV],
                        h_c,
                        disable_tma=True,
                    )
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

        return output_o_maca

    return _func


@torch.library.custom_op("tileops::gated_deltanet_fwd_kernel_maca", mutates_args=())
def _gated_deltanet_fwd_wrapped_kernel_maca(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    fused_num_stages: int,
    fused_threads: int,
    fused_block_k: int,
    fused_block_v: int,
    h_num_stages: int,
    h_threads: int,
    h_block_k: int,
    h_block_v: int,
    o_threads: int,
    o_block_k: int,
    o_block_v: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del fused_num_stages
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
    )(1, fused_threads)
    h_fn = _h_recurrence_maca_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        block_k=h_block_k,
        block_v=h_block_v,
    )(h_num_stages, h_threads)
    o_fn = _output_o_maca_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        block_k=o_block_k,
        block_v=o_block_v,
    )(o_threads)

    S_0 = torch.zeros(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    Aw, Au, w, u = fused_fn(k, v, g_cum, beta)
    S_buf, v_new = h_fn(k, g_cum, w, u, S_0)
    o = o_fn(q, k, g_cum, S_buf, v_new)
    return o, S_buf, Aw, Au


@_gated_deltanet_fwd_wrapped_kernel_maca.register_fake
def _gated_deltanet_fwd_wrapped_kernel_maca_fake(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    fused_num_stages: int,
    fused_threads: int,
    fused_block_k: int,
    fused_block_v: int,
    h_num_stages: int,
    h_threads: int,
    h_block_k: int,
    h_block_v: int,
    o_threads: int,
    o_block_k: int,
    o_block_v: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        dtype,
        fused_num_stages,
        fused_threads,
        fused_block_k,
        fused_block_v,
        h_num_stages,
        h_threads,
        h_block_k,
        h_block_v,
        o_threads,
        o_block_k,
        o_block_v,
        k,
        v,
        g,
        beta,
    )
    num_chunks = seq_len // chunk_size
    o = torch.empty(batch, head, seq_len, dim_v, dtype=q.dtype, device=q.device)
    S = torch.empty(batch, head, num_chunks + 1, dim_k, dim_v, dtype=q.dtype, device=q.device)
    Aw = torch.empty(batch, head, seq_len, chunk_size, dtype=q.dtype, device=q.device)
    Au = torch.empty_like(Aw)
    return o, S, Aw, Au


class GatedDeltaNetFwdMACAKernel(Kernel):
    """MACA Gated DeltaNet forward: (q, k, v, g, beta) -> (o, S, Aw, Au)."""

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
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return _plan_fwd_config(
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            self.batch * self.head,
        )

    @property
    def autotune_configs(self) -> list[dict]:
        return _maca_fwd_autotune_configs(
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            self.batch * self.head,
        )

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Tune the launch parameters of the three MACA forward stages.

        The stage tile sizes remain those selected by ``_plan_fwd_config``:
        changing one changes the generated TileLang program and rapidly turns a
        small launch sweep into a compile-heavy cross product. Every result is
        flattened back into the public config ABI and is therefore a member of
        ``autotune_configs``.
        """
        default = self.default_config
        shape = (
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
        )

        fused_candidates = [
            {"num_stages": 1, "threads": threads}
            for threads in _maca_thread_candidates(
                self.chunk_size,
                min(
                    self.chunk_size,
                    max(default["fused_block_k"], default["fused_block_v"]),
                ),
            )
        ]
        recurrence_candidates = [
            {"num_stages": num_stages, "threads": threads}
            for num_stages in (0, 1)
            for threads in _maca_thread_candidates(
                min(self.chunk_size, default["h_block_k"]), default["h_block_v"]
            )
        ]
        output_candidates = [
            {"threads": threads}
            for threads in _maca_thread_candidates(self.chunk_size, default["o_block_v"])
        ]

        def tune_stage(label: str, jit_kernel, candidates: list[dict]) -> Dict[str, int]:
            print(f"Autotuning {label} ({len(candidates)} configs)...")
            tuned = self.tune_jit_kernel(
                jit_kernel,
                candidates,
                warmup=warmup,
                rep=rep,
                seed_config=candidates[0],
                supply_prog=None,
            )
            config = getattr(tuned, "config", None)
            print(f"  Best: {config}")
            return config or {}

        print(f"Start autotuning {self.__class__.__name__}...")
        fused = tune_stage(
            "fused_prepare_compute_w_u",
            _fused_prepare_compute_w_u_maca_tl(
                *shape,
                block_k=default["fused_block_k"],
                block_v=default["fused_block_v"],
            ),
            fused_candidates,
        )
        recurrence = tune_stage(
            "h_recurrence",
            _h_recurrence_maca_tl(
                *shape,
                block_k=default["h_block_k"],
                block_v=default["h_block_v"],
            ),
            recurrence_candidates,
        )
        output = tune_stage(
            "output_o",
            _output_o_maca_tl(
                *shape,
                block_k=default["o_block_k"],
                block_v=default["o_block_v"],
            ),
            output_candidates,
        )
        self.config = {
            **default,
            "fused_num_stages": fused.get("num_stages", default["fused_num_stages"]),
            "fused_threads": fused.get("threads", default["fused_threads"]),
            "h_num_stages": recurrence.get("num_stages", default["h_num_stages"]),
            "h_threads": recurrence.get("threads", default["h_threads"]),
            "o_threads": output.get("threads", default["o_threads"]),
        }
        print(f"{self.__class__.__name__} autotuned config: {self.config}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        defaults = self.default_config
        return _gated_deltanet_fwd_wrapped_kernel_maca(
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            cfg.get("fused_num_stages", defaults["fused_num_stages"]),
            cfg.get("fused_threads", defaults["fused_threads"]),
            cfg.get("fused_block_k", defaults["fused_block_k"]),
            cfg.get("fused_block_v", defaults["fused_block_v"]),
            cfg.get("h_num_stages", defaults["h_num_stages"]),
            cfg.get("h_threads", defaults["h_threads"]),
            cfg.get("h_block_k", defaults["h_block_k"]),
            cfg.get("h_block_v", defaults["h_block_v"]),
            cfg.get("o_threads", defaults["o_threads"]),
            cfg.get("o_block_k", defaults["o_block_k"]),
            cfg.get("o_block_v", defaults["o_block_v"]),
            q,
            k,
            v,
            g,
            beta,
        )


class GatedDeltaNetFwdBTHDMACAKernel(GatedDeltaNetFwdMACAKernel):
    """MACA Gated DeltaNet forward with the public BTHD ABI."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        o, S, Aw, Au = super().forward(
            q.permute(0, 2, 1, 3).contiguous(),
            k.permute(0, 2, 1, 3).contiguous(),
            v.permute(0, 2, 1, 3).contiguous(),
            g.permute(0, 2, 1).contiguous(),
            beta.permute(0, 2, 1).contiguous(),
        )
        return (
            o.permute(0, 2, 1, 3).contiguous(),
            S,
            Aw.permute(0, 2, 1, 3).contiguous(),
            Au.permute(0, 2, 1, 3).contiguous(),
        )
