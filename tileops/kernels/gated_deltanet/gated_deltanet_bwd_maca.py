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
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from .gated_deltanet_fwd import _LOG2E

__all__ = [
    "GatedDeltaNetBwdMACAKernel",
]


def _maca_block_v(chunk_size: int, dim_v: int, dtype: str) -> int:
    """Pick a value-tile size that keeps both backward kernels under MACA's
    64KB shared-memory cap.  ``0`` means "no tiling" (used on non-MACA archs).

    fp32 (4 bytes/elem) needs a smaller tile than fp16/bf16 (2 bytes/elem).
    """
    if dtype == "float32":
        # fp32 doubles every buffer; use the smallest sensible V-tile.
        if dim_v % 16 == 0:
            return 16
        return 32 if dim_v % 32 == 0 else dim_v
    # fp16 / bf16: BV=32 already fits comfortably.
    return 32 if dim_v % 32 == 0 else dim_v


# =============================================================================
# Split kernel: bwd_parallel (fully parallel over chunks)
# =============================================================================

@functools.lru_cache(maxsize=32)
def _bwd_parallel_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
):
    """Parallel per-chunk backward gradients.

    Grid: (num_chunks, batch, head) — fully parallel across chunks.
    Computes everything that does NOT depend on dh_buf from other chunks.

    Outputs: dq, dk_partial, dg_partial, dw, du_partial, v_new, dh_local

    Shared-memory-safe variant (for MACA's 64KB SMEM cap): the value dimension
    ``dim_v`` is tiled by ``block_v``.  All V-independent quantities (``attn``,
    ``dk``, and the step-4/step-5 K-space work) are computed once; every
    quantity that reduces over V (``d_attn``, step-3 ``dg``/``dq_h``, step-5
    ``dP``) is accumulated across V-tiles in fp32 fragments; every quantity that
    is separable along V (``v_new``, ``o_part``, ``d_v_new``, the step-3/step-5
    ``dh`` contributions) is produced per V-tile and written to its own V-slice.
    ``block_v <= 0`` means no tiling (BV = dim_v), reproducing the original.
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = dim_v if block_v <= 0 else block_v
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
                # Shared buffers (V-tiled where they span dim_v)
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                do_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                o_part = T.alloc_shared([block_C, BV], dtype)
                d_v_new_c = T.alloc_shared([block_C, BV], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                d_attn = T.alloc_shared([block_C, block_C], dtype)
                # Gradients / working (dq/dk/dw are written straight from
                # fragments; no shared staging buffers for them)
                dg_c = T.alloc_shared([block_C], dtype)
                exp_g = T.alloc_shared([block_C], dtype)
                P = T.alloc_shared([block_C, dim_k], dtype)
                # Fragments (fp32 accumulators)
                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_v_new_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                d_attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_q_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                d_k_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dP_acc = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dw_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dh_tile_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                dh_sub_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                dg_row = T.alloc_fragment([block_C], accum_dtype)
                dg3_acc = T.alloc_fragment([block_C], accum_dtype)

                # Load V-independent chunk data
                T.copy(q[bid, hid, tid * block_C : (tid + 1) * block_C, :], q_c, disable_tma=True)
                T.copy(k[bid, hid, tid * block_C : (tid + 1) * block_C, :], k_c, disable_tma=True)
                T.copy(g[bid, hid, tid * block_C : (tid + 1) * block_C], g_c, disable_tma=True)
                T.copy(w[bid, hid, tid * block_C : (tid + 1) * block_C, :], w_c, disable_tma=True)

                for i in T.Parallel(block_C):
                    exp_g[i] = T.exp2(g_c[i] * _LOG2E)

                # attn = causal(q @ k^T) * Gamma   (V-independent, computed once)
                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(
                        i >= j,
                        attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        T.float32(0.0))

                # V-reducing accumulators
                T.clear(d_attn_frag)   # d_attn = sum_vt do @ v_new^T
                T.clear(d_q_c_frag)    # dq: step-3 part accumulates here
                T.clear(dP_acc)        # dP = -sum_vt d_v_new @ h^T
                for i in T.Parallel(block_C):
                    dg3_acc[i] = T.float32(0.0)

                for vt in T.serial(0, num_v_tiles):
                    v_off = vt * BV
                    T.copy(u[bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV], u_c, disable_tma=True)
                    T.copy(do[bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV], do_c, disable_tma=True)
                    T.copy(S[bid, hid, tid, :, v_off : v_off + BV], h_c, disable_tma=True)

                    # v_new_c = u - (w @ h) * exp(g + g_last)   (separable in V)
                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_c[block_C - 1]) * _LOG2E)
                    T.copy(v_new_c, v_new_out[bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV], disable_tma=True)

                    # o_part = (q @ h) * exp_g   (separable in V)
                    T.clear(ws_frag)
                    T.gemm(q_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        o_part[i, j] = ws_frag[i, j] * exp_g[i]

                    # Step 2: d_v_new = attn^T @ do  (separable in V) -> du_partial
                    T.clear(d_v_new_frag)
                    T.gemm(attn, do_c, d_v_new_frag, transpose_A=True)
                    T.copy(d_v_new_frag, d_v_new_c)
                    T.copy(d_v_new_c, du_partial[bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV], disable_tma=True)

                    # d_attn += do @ v_new^T   (reduces over V -> accumulate)
                    T.gemm(do_c, v_new_c, d_attn_frag, transpose_B=True)

                    # Step 3: dg3 += rowsum_j(do * o_part)   (reduces over V)
                    for i, j in T.Parallel(block_C, BV):
                        o_part[i, j] = do_c[i, j] * o_part[i, j]
                    T.reduce_sum(o_part, dg_row, dim=1)
                    for i in T.Parallel(block_C):
                        dg3_acc[i] += dg_row[i]

                    # Step 3: dq += (do * exp_g) @ h^T   (reduces over V)
                    for i, j in T.Parallel(block_C, BV):
                        o_part[i, j] = do_c[i, j] * exp_g[i]
                    T.gemm(o_part, h_c, d_q_c_frag, transpose_B=True)

                    # Step 3: dh_local tile = (q * exp_g)^T @ do   (separable in V)
                    for i, j in T.Parallel(block_C, dim_k):
                        P[i, j] = q_c[i, j] * exp_g[i]
                    T.clear(dh_tile_frag)
                    T.gemm(P, do_c, dh_tile_frag, transpose_A=True)

                    # Step 5 (V-reducing / separable parts):
                    #   dP_acc += -(d_v_new @ h^T)   (reduces over V)
                    #   dh_local tile -= (w*exp(g+g_last))^T @ d_v_new  (separable)
                    T.clear(dP_frag)
                    T.gemm(d_v_new_c, h_c, dP_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, dim_k):
                        dP_acc[i, j] -= dP_frag[i, j]
                    for i, j in T.Parallel(block_C, dim_k):
                        P[i, j] = w_c[i, j] * T.exp2(
                            (g_c[i] + g_c[block_C - 1]) * _LOG2E)
                    T.clear(dh_sub_frag)
                    T.gemm(P, d_v_new_c, dh_sub_frag, transpose_A=True)
                    for i, j in T.Parallel(dim_k, BV):
                        dh_tile_frag[i, j] -= dh_sub_frag[i, j]
                    T.copy(dh_tile_frag, dh_local[bid, hid, tid, :, v_off : v_off + BV], disable_tma=True)

                # ---- V-independent finalisation ----
                # Step 4: d_attn causal mask, dg4 = rowsum - colsum of (d_attn*attn),
                #         dq += (d_attn*Gamma) @ k, dk = (d_attn*Gamma)^T @ q
                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(i >= j, d_attn_frag[i, j], T.float32(0.0))
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = d_attn[i, j] * attn[i, j]
                dg_step4_row = T.alloc_shared([block_C], dtype)
                T.reduce_sum(attn, dg_step4_row, dim=1)
                dg_step4_col = T.alloc_shared([block_C], dtype)
                T.reduce_sum(attn, dg_step4_col, dim=0)
                for i in T.Parallel(block_C):
                    dg_c[i] = dg3_acc[i] + dg_step4_row[i] - dg_step4_col[i]

                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(
                        i >= j,
                        d_attn[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        T.float32(0.0))
                T.gemm(d_attn, k_c, d_q_c_frag)
                T.copy(d_q_c_frag, dq[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)
                T.clear(d_k_c_frag)
                T.gemm(d_attn, q_c, d_k_c_frag, transpose_A=True)
                T.copy(d_k_c_frag, dk_partial[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)

                # Step 5 finalise from full dP_acc:
                #   dw = dP * exp(g + g_last);  dg5[i] = rowsum_k(P5*dP); dg5[last] += sum
                for i, j in T.Parallel(block_C, dim_k):
                    dw_frag[i, j] = dP_acc[i, j] * T.exp2((g_c[i] + g_c[block_C - 1]) * _LOG2E)
                T.copy(dw_frag, dw[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)
                for i, j in T.Parallel(block_C, dim_k):
                    P[i, j] = w_c[i, j] * T.exp2((g_c[i] + g_c[block_C - 1]) * _LOG2E)
                for i, j in T.Parallel(block_C, dim_k):
                    P[i, j] = P[i, j] * dP_acc[i, j]
                dg_step5_tmp = T.alloc_shared([block_C], dtype)
                T.reduce_sum(P, dg_step5_tmp, dim=1)
                dg_step5_total = T.alloc_shared([1], accum_dtype)
                T.reduce_sum(dg_step5_tmp, dg_step5_total, dim=0)
                for i in T.Parallel(block_C):
                    dg_c[i] += dg_step5_tmp[i]
                dg_c[block_C - 1] = dg_c[block_C - 1] + dg_step5_total[0]

                # Write remaining outputs
                for i in T.Parallel(block_C):
                    dg_partial[bid, hid, tid * block_C + i] = dg_c[i]

        return bwd_parallel_kernel

    return _func


# =============================================================================
# Split kernel: dh_recurrence_bwd (sequential backward over chunks)
# =============================================================================

@functools.lru_cache(maxsize=32)
def _dh_recurrence_bwd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
):
    """Sequential backward dh recurrence with corrections.

    Grid: (num_v_tiles, batch, head) — sequential over chunks (backward),
    parallel/independent over V-tiles.

    Shared-memory-safe variant: the dh recurrence is *separable along the value
    dimension* (every state element (i, j) only depends on the same value
    column j), so we tile ``dim_v`` by ``block_v`` and give each V-tile its own
    ``dh_buf`` state.  This shrinks the three DK x DV state tiles
    (``h_c``/``dh_loc``/``dh_buf``) to DK x BV.

    ``du_corr`` reduces over DK, so each V-tile writes a final, disjoint V-slice.
    ``dk_corr`` and ``dg_corr`` reduce over V, so each V-tile emits a *partial*
    into a ``[num_v_tiles, ...]`` buffer that the wrapper sums over the tile
    axis.  ``block_v <= 0`` means no tiling (BV = dim_v).

    Outputs: dk_correction (partial per V-tile), du_correction, dg_correction
             (partial per V-tile)
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = dim_v if block_v <= 0 else block_v
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
        def dh_recurrence_bwd_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            # Outputs (dk_corr / dg_corr are partial per V-tile)
            dk_corr: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dg_corr: T.Tensor([batch, head, num_v_tiles, seq_len], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                v_off = vid * BV
                # Shared buffers (state tiles shrunk to DK x BV)
                g_c = T.alloc_shared([block_C], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                k_scaled = T.alloc_shared([block_C, dim_k], dtype)
                dP = T.alloc_shared([block_C, dim_k], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                # dh_buf carries gradient from the next chunk (backward)
                dh_buf = T.alloc_shared([dim_k, BV], dtype)
                # Fragments
                dh_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                du_corr_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dh_h_tmp = T.alloc_fragment([dim_k, BV], accum_dtype)
                d_g_pos = T.alloc_fragment([block_C], accum_dtype)
                d_g_last_partial = T.alloc_fragment([dim_k], accum_dtype)
                d_g_last_scalar1 = T.alloc_fragment([1], accum_dtype)
                d_g_last_scalar2 = T.alloc_fragment([1], accum_dtype)

                # Zero dh_buf (last chunk has no successor)
                for i, j in T.Parallel(dim_k, BV):
                    dh_buf[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    # Load data (V-sliced)
                    T.copy(g[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C], g_c, disable_tma=True)
                    T.copy(k[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], k_c, disable_tma=True)
                    T.copy(v_new[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, v_off : v_off + BV], v_new_c, disable_tma=True)
                    T.copy(S[bid, hid, t_bwd, :, v_off : v_off + BV], h_c, disable_tma=True)
                    T.copy(dh_local[bid, hid, t_bwd, :, v_off : v_off + BV], dh_loc, disable_tma=True)

                    # k_scaled = k * exp(g_last - g)   (V-independent)
                    for pn, sk in T.Parallel(block_C, dim_k):
                        k_scaled[pn, sk] = k_c[pn, sk] * T.exp2(
                            (g_c[block_C - 1] - g_c[pn]) * _LOG2E)

                    # dh = dh_local + dh_buf * exp(g_last)
                    for i, j in T.Parallel(dim_k, BV):
                        dh_frag[i, j] = dh_loc[i, j] + dh_buf[i, j] * T.exp2(
                            g_c[block_C - 1] * _LOG2E)

                    # du_correction = k_scaled @ dh_buf   (reduces over DK -> final V-slice)
                    T.clear(du_corr_frag)
                    T.gemm(k_scaled, dh_buf, du_corr_frag)
                    T.copy(du_corr_frag, du_corr[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, v_off : v_off + BV], disable_tma=True)

                    # dk_correction(partial) = (v_new @ dh_buf^T) * exp(g_last - g)
                    T.clear(dP_frag)
                    T.gemm(v_new_c, dh_buf, dP_frag, transpose_B=True)
                    T.copy(dP_frag, dP)
                    for n, kk in T.Parallel(block_C, dim_k):
                        dk_corr[bid, hid, vid, t_bwd * block_C + n, kk] = dP[n, kk] * T.exp2(
                            (g_c[block_C - 1] - g_c[n]) * _LOG2E)

                    # dg_correction(partial): per-position and g_last terms
                    # Per-position: -sum_k(dP * k_scaled) per row n
                    for n, kk in T.Parallel(block_C, dim_k):
                        dP[n, kk] = dP[n, kk] * k_scaled[n, kk]
                    T.reduce_sum(dP, d_g_pos, dim=1)
                    for n in T.Parallel(block_C):
                        dg_c[n] = -d_g_pos[n]

                    # g_last term 1: sum(dh_buf * h_c) * exp_g_last  (partial over V)
                    for i, j in T.Parallel(dim_k, BV):
                        dh_h_tmp[i, j] = dh_buf[i, j] * h_c[i, j]
                    T.reduce_sum(dh_h_tmp, d_g_last_partial, dim=1)
                    T.reduce_sum(d_g_last_partial, d_g_last_scalar1, dim=0)

                    # g_last term 2: sum_n(d_g_pos)
                    T.reduce_sum(d_g_pos, d_g_last_scalar2, dim=0)
                    dg_c[block_C - 1] = dg_c[block_C - 1] + d_g_last_scalar1[0] * T.exp2(
                        g_c[block_C - 1] * _LOG2E) + d_g_last_scalar2[0]

                    # Write dg_correction (partial per V-tile)
                    for i in T.Parallel(block_C):
                        dg_corr[bid, hid, vid, t_bwd * block_C + i] = dg_c[i]

                    # Carry dh to next iteration
                    T.copy(dh_frag, dh_buf)

        return dh_recurrence_bwd_kernel

    return _func


@torch.library.custom_op("tileops::gated_deltanet_bwd_maca_kernel", mutates_args=())
def _gated_deltanet_bwd_wrapped_kernel(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    num_stages: int, threads: int,
    parallel_threads: int, recurrence_threads: int,
    block_v: int,
    do: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    S: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .compute_w_u_bwd import compute_w_u_bwd_tl
    from .fused_prepare_compute_w_u import fused_prepare_compute_w_u_tl
    from .gated_deltanet_fwd import _chunk_local_cumsum

    g_cum = _chunk_local_cumsum(g.float(), chunk_size).to(g.dtype)

    fused_fn = fused_prepare_compute_w_u_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(num_stages, threads)
    bwd_parallel_fn = _bwd_parallel_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype, block_v=block_v,
    )(parallel_threads)
    dh_recurrence_bwd_fn = _dh_recurrence_bwd_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype, block_v=block_v,
    )(num_stages, recurrence_threads)
    wu_bwd_fn = compute_w_u_bwd_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(num_stages, threads)

    Aw, Au, w, u = fused_fn(k, v, g_cum, beta)
    dq, dk_partial, dg_partial, dw, du_partial, v_new, dh_local = \
        bwd_parallel_fn(do, q, k, g_cum, w, u, S)
    dk_corr, du_corr, dg_corr = \
        dh_recurrence_bwd_fn(g_cum, k, v_new, S, dh_local)

    # When block_v > 0 the recurrence tiles dim_v and emits per-V-tile partials
    # for dk_corr / dg_corr; sum them (in fp32) over the tile axis.
    if dk_corr.dim() == 5:
        dk_corr = dk_corr.float().sum(dim=2).to(dk_partial.dtype)
        dg_corr = dg_corr.float().sum(dim=2).to(dg_partial.dtype)

    du = du_partial + du_corr
    _dAw, _dAu, dk_wu, dv, dbeta = wu_bwd_fn(dw, du, Aw, Au, k, v, beta)

    dk = dk_partial + dk_corr + dk_wu
    dg_cum = dg_partial + dg_corr

    B, H, SL = g.shape
    dg = dg_cum.float().reshape(B, H, SL // chunk_size, chunk_size)
    dg = dg.flip(-1).cumsum(-1).flip(-1).reshape(B, H, SL).to(g.dtype)
    return dq, dk, dv, dg, dbeta


@_gated_deltanet_bwd_wrapped_kernel.register_fake
def _gated_deltanet_bwd_wrapped_kernel_fake(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    num_stages: int, threads: int,
    parallel_threads: int, recurrence_threads: int,
    block_v: int,
    do: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    S: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty(batch, head, seq_len, dim_k, dtype=q.dtype, device=q.device)
    dk = torch.empty_like(dq)
    dv = torch.empty(batch, head, seq_len, dim_v, dtype=v.dtype, device=v.device)
    dg = torch.empty(batch, head, seq_len, dtype=g.dtype, device=g.device)
    dbeta = torch.empty(batch, head, seq_len, dtype=beta.dtype, device=beta.device)
    return dq, dk, dv, dg, dbeta


class GatedDeltaNetBwdMACAKernel(Kernel):
    """Gated DeltaNet backward kernel.

    Full backward: do -> (dq, dk, dv, dg, dbeta).

    Split pipeline (Phase 2 optimisation):
      1. fused_prepare_compute_w_u: recompute w, u
      2. bwd_parallel: per-chunk gradients (grid: num_chunks x B x H)
      3. dh_recurrence_bwd: sequential dh propagation + corrections (grid: B x H)
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
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        threads = 256 if self.chunk_size >= 64 else 128
        # On MACA the per-block shared memory is 64KB, which the un-tiled
        # backward kernels exceed. Tile the value dimension there; block_v=0
        # (no tiling) keeps the original behaviour on other archs.
        block_v = _maca_block_v(self.chunk_size, self.dim_v, self.dtype_str)
        return {
            "num_stages": 2,
            "threads": threads,
            "parallel_threads": threads,
            "recurrence_threads": threads,
            "block_v": block_v,
        }

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
        return _gated_deltanet_bwd_wrapped_kernel(
            self.batch, self.head, self.seq_len, self.chunk_size,
            self.dim_k, self.dim_v, self.dtype_str,
            self.config.get("num_stages", 2), self.config.get("threads", 256),
            self.config.get("parallel_threads", 256),
            self.config.get("recurrence_threads", 256),
            self.config.get("block_v", 0),
            do, q, k, v, g, beta, S,
        )
