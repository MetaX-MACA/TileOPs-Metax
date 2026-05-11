# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

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
    "GatedDeltaNetBwdKernel",
]


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
):
    """Parallel per-chunk backward gradients.

    Grid: (num_chunks, batch, head) — fully parallel across chunks.
    Computes everything that does NOT depend on dh_buf from other chunks.

    Outputs: dq, dk_partial, dg_partial, dw, du_partial, v_new, dh_local
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

    BK = 32
    BV = 32
    sub_dim_k = dim_k // BK
    sub_dim_v = dim_v // BV

    assert dim_k % BK == 0, "dim_k must be divisible by BK"
    assert dim_v % BV == 0, "dim_v must be divisible by BV"

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
                # Shared buffers and working state (chunk-invariant for this kernel instance)
                # chunk-invariant buffers
                g_c = T.alloc_shared([block_C], dtype)
                exp_g = T.alloc_shared([block_C], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                d_attn = T.alloc_shared([block_C, block_C], dtype)
                dg_c = T.alloc_shared([block_C], dtype)

                # Gradients / outputs initialisation
                # accumulators on global tensors
                for i, j in T.Parallel(block_C, dim_k):
                    dq[bid, hid, tid * block_C + i, j] = T.float32(0.0)
                    dk_partial[bid, hid, tid * block_C + i, j] = T.float32(0.0)
                    dw[bid, hid, tid * block_C + i, j] = T.float32(0.0)
                for i, j in T.Parallel(block_C, dim_v):
                    du_partial[bid, hid, tid * block_C + i, j] = T.float32(0.0)
                    v_new_out[bid, hid, tid * block_C + i, j] = T.float32(0.0)
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_local[bid, hid, tid, i, j] = T.float32(0.0)
                for i in T.Parallel(block_C):
                    dg_c[i] = T.float32(0.0)

                T.copy(g[bid, hid, tid * block_C : (tid + 1) * block_C], g_c, disable_tma=True)
                g_last = g_c[block_C - 1]
                for i in T.Parallel(block_C):
                    exp_g[i] = T.exp2(g_c[i] * _LOG2E)

                # Recompute forward: attention logits Γ (attn) accumulated across k-tiles
                # attn / d_attn are full-[C,C], accumulated across k/v tiles
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.float32(0.0)
                    d_attn[i, j] = T.float32(0.0)

                # Step 1: attn += q @ k^T over k-tiles
                q_k_attn = T.alloc_shared([block_C, BK], dtype)
                k_k_attn = T.alloc_shared([block_C, BK], dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                for k0 in T.serial(0, sub_dim_k):
                    T.copy(q[bid, hid, tid * block_C : (tid + 1) * block_C, k0 * BK : (k0 + 1) * BK], q_k_attn, disable_tma=True)
                    T.copy(k[bid, hid, tid * block_C : (tid + 1) * block_C, k0 * BK : (k0 + 1) * BK], k_k_attn, disable_tma=True)
                    T.clear(attn_frag)
                    T.gemm(q_k_attn, k_k_attn, attn_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        attn[i, j] += attn_frag[i, j]

                # apply causal + gate scaling
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(
                        i >= j,
                        attn[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        T.float32(0.0),
                    )

                # 2) v_new / o_part / d_attn accumulation over v-tiles (and k-tiles inside)
                u_v = T.alloc_shared([block_C, BV], dtype)
                do_v = T.alloc_shared([block_C, BV], dtype)
                vnew_v = T.alloc_shared([block_C, BV], dtype)
                d_vnew_v = T.alloc_shared([block_C, BV], dtype)
                o_part_v = T.alloc_shared([block_C, BV], dtype)
                w_k_v = T.alloc_shared([block_C, BK], dtype)
                q_k_v = T.alloc_shared([block_C, BK], dtype)
                h_kv_v = T.alloc_shared([BK, BV], dtype)
                ws_frag_v = T.alloc_fragment([block_C, BV], accum_dtype)
                ws_tmp_v = T.alloc_fragment([block_C, BV], accum_dtype)
                d_vnew_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                d_attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                o_mul = T.alloc_shared([block_C, BV], dtype)
                dg_v = T.alloc_shared([block_C], dtype)
                for v0 in T.serial(0, sub_dim_v):
                    # Recompute forward: v_new_v, and partial outputs o_part_v for this v-tile
                    # v_new_v = u_v - (sum_k w_k @ h_kv) * exp(g+g_last)
                    T.clear(ws_frag_v)
                    T.copy(u[bid, hid, tid * block_C : (tid + 1) * block_C, v0 * BV : (v0 + 1) * BV], u_v, disable_tma=True)
                    T.copy(do[bid, hid, tid * block_C : (tid + 1) * block_C, v0 * BV : (v0 + 1) * BV], do_v, disable_tma=True)
                    for k0 in T.serial(0, sub_dim_k):
                        T.copy(w[bid, hid, tid * block_C : (tid + 1) * block_C, k0 * BK : (k0 + 1) * BK], w_k_v, disable_tma=True)
                        T.copy(S[bid, hid, tid, k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], h_kv_v, disable_tma=True)
                        T.clear(ws_tmp_v)
                        T.gemm(w_k_v, h_kv_v, ws_tmp_v)
                        for i, j in T.Parallel(block_C, BV):
                            ws_frag_v[i, j] += ws_tmp_v[i, j]

                    for i, j in T.Parallel(block_C, BV):
                        vnew_v[i, j] = u_v[i, j] - ws_frag_v[i, j] * T.exp2((g_c[i] + g_last) * _LOG2E)
                        v_new_out[bid, hid, tid * block_C + i, v0 * BV + j] = vnew_v[i, j]

                    # o_part_v = (sum_k q_k @ h_kv) * exp_g
                    T.clear(ws_frag_v)
                    for k0 in T.serial(0, sub_dim_k):
                        T.copy(q[bid, hid, tid * block_C : (tid + 1) * block_C, k0 * BK : (k0 + 1) * BK], q_k_v, disable_tma=True)
                        T.copy(S[bid, hid, tid, k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], h_kv_v, disable_tma=True)
                        T.clear(ws_tmp_v)
                        T.gemm(q_k_v, h_kv_v, ws_tmp_v)
                        for i, j in T.Parallel(block_C, BV):
                            ws_frag_v[i, j] += ws_tmp_v[i, j]
                    for i, j in T.Parallel(block_C, BV):
                        o_part_v[i, j] = ws_frag_v[i, j] * exp_g[i]

                    # Step 2: d_v_new_c = attn^T @ do (partial du)
                    # d_vnew_v = attn^T @ do_v
                    T.clear(d_vnew_frag)
                    T.gemm(attn, do_v, d_vnew_frag, transpose_A=True)
                    T.copy(d_vnew_frag, d_vnew_v)
                    for i, j in T.Parallel(block_C, BV):
                        du_partial[bid, hid, tid * block_C + i, v0 * BV + j] = d_vnew_v[i, j]

                    # d_attn = do @ v_new^T (causal masked)
                    # d_attn += do_v @ vnew_v^T
                    T.clear(d_attn_frag)
                    T.gemm(do_v, vnew_v, d_attn_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        d_attn[i, j] += T.if_then_else(i >= j, d_attn_frag[i, j], T.float32(0.0))

                    # Step 3 (part 1): dg from o_part
                    # dg step3 += sum_v do_v * o_part_v
                    for i, j in T.Parallel(block_C, BV):
                        o_mul[i, j] = do_v[i, j] * o_part_v[i, j]
                    T.reduce_sum(o_mul, dg_v, dim=1)
                    for i in T.Parallel(block_C):
                        dg_c[i] += dg_v[i]

                # Step 4: dg from Γ using d_attn * attn
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = d_attn[i, j] * attn[i, j]
                dg_step4_row = T.alloc_shared([block_C], dtype)
                dg_step4_col = T.alloc_shared([block_C], dtype)
                T.reduce_sum(attn, dg_step4_row, dim=1)
                T.reduce_sum(attn, dg_step4_col, dim=0)
                for i in T.Parallel(block_C):
                    dg_c[i] = dg_c[i] + dg_step4_row[i] - dg_step4_col[i]

                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(
                        i >= j,
                        d_attn[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        T.float32(0.0),
                    )

                # Step 4 (cont.): dq / dk and
                # Step 5: dh from w/v_new, dw, dg from P in k-v blocked form
                q_k = T.alloc_shared([block_C, BK], dtype)
                k_k = T.alloc_shared([block_C, BK], dtype)
                w_k = T.alloc_shared([block_C, BK], dtype)
                d_q_k = T.alloc_shared([block_C, BK], dtype)
                d_k_k = T.alloc_shared([block_C, BK], dtype)
                P_k = T.alloc_shared([block_C, BK], dtype)
                dP_k = T.alloc_shared([block_C, BK], dtype)
                d_q_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                d_k_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                dq_add_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                do_v_k = T.alloc_shared([block_C, BV], dtype)
                h_kv_k = T.alloc_shared([BK, BV], dtype)
                o_scaled = T.alloc_shared([block_C, BV], dtype)
                tmp = T.alloc_fragment([block_C, BK], accum_dtype)
                d_vnew_v_k = T.alloc_shared([block_C, BV], dtype)
                dP_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                dh_sub = T.alloc_fragment([BK, BV], accum_dtype)
                p_mul = T.alloc_shared([block_C, BK], dtype)
                dg_k = T.alloc_shared([block_C], dtype)
                dg_k_total = T.alloc_shared([1], accum_dtype)
                for k0 in T.serial(0, sub_dim_k):
                    T.copy(q[bid, hid, tid * block_C : (tid + 1) * block_C, k0 * BK : (k0 + 1) * BK], q_k, disable_tma=True)
                    T.copy(k[bid, hid, tid * block_C : (tid + 1) * block_C, k0 * BK : (k0 + 1) * BK], k_k, disable_tma=True)
                    T.copy(w[bid, hid, tid * block_C : (tid + 1) * block_C, k0 * BK : (k0 + 1) * BK], w_k, disable_tma=True)

                    T.clear(d_q_frag)
                    T.clear(d_k_frag)
                    T.gemm(d_attn, k_k, d_q_frag)
                    T.gemm(d_attn, q_k, d_k_frag, transpose_A=True)
                    T.copy(d_q_frag, d_q_k)
                    T.copy(d_k_frag, d_k_k)

                    # dq / dk from d_attn * Γ
                    for i, j in T.Parallel(block_C, BK):
                        dq[bid, hid, tid * block_C + i, k0 * BK + j] = d_q_k[i, j]
                        dk_partial[bid, hid, tid * block_C + i, k0 * BK + j] = d_k_k[i, j]
                        P_k[i, j] = q_k[i, j] * exp_g[i]
                        dP_k[i, j] = T.float32(0.0)

                    # Step 3 (part 2): dq from h via (do * exp_g) @ H^T over v-tiles
                    # step3 dq += (do*exp_g) @ h^T over v-tiles
                    T.clear(dq_add_frag)
                    for v0 in T.serial(0, sub_dim_v):
                        T.copy(do[bid, hid, tid * block_C : (tid + 1) * block_C, v0 * BV : (v0 + 1) * BV], do_v_k, disable_tma=True)
                        T.copy(S[bid, hid, tid, k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], h_kv_k, disable_tma=True)
                        for i, j in T.Parallel(block_C, BV):
                            o_scaled[i, j] = do_v_k[i, j] * exp_g[i]
                        T.clear(tmp)
                        T.gemm(o_scaled, h_kv_k, tmp, transpose_B=True)
                        for i, j in T.Parallel(block_C, BK):
                            dq_add_frag[i, j] += tmp[i, j]

                    for i, j in T.Parallel(block_C, BK):
                        dq[bid, hid, tid * block_C + i, k0 * BK + j] += dq_add_frag[i, j]

                    # Step 5: dh from w/v_new, dw, dg from P (k-v tiled)
                    # step5: dP_k and dh_local contributions over v-tiles
                    for i, j in T.Parallel(block_C, BK):
                        P_k[i, j] = w_k[i, j] * T.exp2((g_c[i] + g_last) * _LOG2E)

                    for v0 in T.serial(0, sub_dim_v):
                        T.copy(S[bid, hid, tid, k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], h_kv_k, disable_tma=True)
                        T.copy(du_partial[bid, hid, tid * block_C : (tid + 1) * block_C, v0 * BV : (v0 + 1) * BV], d_vnew_v_k, disable_tma=True)

                        T.clear(dP_frag)
                        T.gemm(d_vnew_v_k, h_kv_k, dP_frag, transpose_B=True)
                        for i, j in T.Parallel(block_C, BK):
                            dP_k[i, j] += -dP_frag[i, j]

                        # dh from P and d_vnew (update dh_local)
                        T.clear(dh_sub)
                        T.gemm(P_k, d_vnew_v_k, dh_sub, transpose_A=True)
                        for i, j in T.Parallel(BK, BV):
                            dh_local[bid, hid, tid, k0 * BK + i, v0 * BV + j] -= dh_sub[i, j]

                    # dw from dP, and dg from P * dP
                    for i, j in T.Parallel(block_C, BK):
                        dw[bid, hid, tid * block_C + i, k0 * BK + j] = dP_k[i, j] * T.exp2((g_c[i] + g_last) * _LOG2E)

                    # dg step5
                    for i, j in T.Parallel(block_C, BK):
                        p_mul[i, j] = P_k[i, j] * dP_k[i, j]
                    T.reduce_sum(p_mul, dg_k, dim=1)
                    T.reduce_sum(dg_k, dg_k_total, dim=0)
                    for i in T.Parallel(block_C):
                        dg_c[i] += dg_k[i]
                    dg_c[block_C - 1] = dg_c[block_C - 1] + dg_k_total[0]

                # dh_local += step3 dh from P=q*exp_g, do over all k-v tiles
                Pk = T.alloc_shared([block_C, BK], dtype)
                do_v_dh = T.alloc_shared([block_C, BV], dtype)
                dh_add = T.alloc_fragment([BK, BV], accum_dtype)
                for k0 in T.serial(0, sub_dim_k):
                    for i, j in T.Parallel(block_C, BK):
                        Pk[i, j] = q[bid, hid, tid * block_C + i, k0 * BK + j] * exp_g[i]
                    for v0 in T.serial(0, sub_dim_v):
                        T.copy(do[bid, hid, tid * block_C : (tid + 1) * block_C, v0 * BV : (v0 + 1) * BV], do_v_dh, disable_tma=True)
                        T.clear(dh_add)
                        T.gemm(Pk, do_v_dh, dh_add, transpose_A=True)
                        for i, j in T.Parallel(BK, BV):
                            dh_local[bid, hid, tid, k0 * BK + i, v0 * BV + j] += dh_add[i, j]

                # Write dg_partial and dh_local for recurrence kernel
                # write dg
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
):
    """Sequential backward dh recurrence with corrections.

    Grid: (batch, head) — sequential over chunks (backward).
    Reads dh_local from bwd_parallel, propagates dh backward, and computes
    corrections for dk, du, dg that depend on dh_buf from other chunks.

    Outputs: dk_correction, du_correction, dg_correction
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

    BK = 32
    BV = 32
    sub_dim_k = dim_k // BK
    sub_dim_v = dim_v // BV

    assert dim_k % BK == 0, "dim_k must be divisible by BK"
    assert dim_v % BV == 0, "dim_v must be divisible by BV"

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
            # Outputs
            dk_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dg_corr: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                # Shared buffers for per-chunk recurrence
                g_c = T.alloc_shared([block_C], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                d_g_pos = T.alloc_shared([block_C], dtype)

                # dh_buf carries gradient from the next chunk (backward)
                # Carry state across chunks (backward): dH from successor chunk
                dh_buf = T.alloc_shared([dim_k, dim_v], dtype)

                # Working buffers (tiled in k/v)
                k_k = T.alloc_shared([block_C, BK], dtype)
                v_v = T.alloc_shared([block_C, BV], dtype)
                h_kv = T.alloc_shared([BK, BV], dtype)
                dh_buf_kv = T.alloc_shared([BK, BV], dtype)
                dh_loc_kv = T.alloc_shared([BK, BV], dtype)
                dh_new_kv = T.alloc_shared([BK, BV], dtype)
                p_mul = T.alloc_shared([block_C, BK], dtype)

                du_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                du_tmp_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                dP_tmp_frag = T.alloc_fragment([block_C, BK], accum_dtype)
                d_g_pos_tmp = T.alloc_shared([block_C], dtype)
                dh_row_tmp = T.alloc_shared([BK], dtype)
                dg_last_scalar1 = T.alloc_shared([1], accum_dtype)
                dg_last_scalar2 = T.alloc_shared([1], accum_dtype)

                # Zero dh_buf (last chunk has no successor)
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_buf[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    # Load data for chunk t_bwd
                    T.copy(g[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C], g_c, disable_tma=True)

                    g_last = g_c[block_C - 1]
                    exp_g_last = T.exp2(g_last * _LOG2E)

                    for i in T.Parallel(block_C):
                        d_g_pos[i] = T.float32(0.0)

                    # du_correction = (k * exp(g_last - g)) @ dh_buf, tiled on k/v
                    # (same as k_scaled @ dh_buf in the untiled version)
                    for v0 in T.serial(0, sub_dim_v):
                        T.clear(du_frag)
                        for k0 in T.serial(0, sub_dim_k):
                            T.copy(k[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, k0 * BK : (k0 + 1) * BK], k_k, disable_tma=True)
                            T.copy(dh_buf[k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], dh_buf_kv, disable_tma=True)
                            for n, kk in T.Parallel(block_C, BK):
                                k_k[n, kk] = k_k[n, kk] * T.exp2((g_last - g_c[n]) * _LOG2E)
                            T.clear(du_tmp_frag)
                            T.gemm(k_k, dh_buf_kv, du_tmp_frag)
                            for i, j in T.Parallel(block_C, BV):
                                du_frag[i, j] += du_tmp_frag[i, j]
                        T.copy(du_frag, du_corr[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, v0 * BV : (v0 + 1) * BV], disable_tma=True)

                    # dk_correction = (v_new @ dh_buf^T) * exp(g_last - g)
                    # plus dg_correction per-position, both tiled on k/v
                    # dk_correction + dg per-position, tiled on k/v
                    for k0 in T.serial(0, sub_dim_k):
                        T.clear(dP_frag)
                        for v0 in T.serial(0, sub_dim_v):
                            T.copy(v_new[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, v0 * BV : (v0 + 1) * BV], v_v, disable_tma=True)
                            T.copy(dh_buf[k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], dh_buf_kv, disable_tma=True)
                            T.clear(dP_tmp_frag)
                            T.gemm(v_v, dh_buf_kv, dP_tmp_frag, transpose_B=True)
                            for n, kk in T.Parallel(block_C, BK):
                                dP_frag[n, kk] += dP_tmp_frag[n, kk]

                        T.copy(k[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, k0 * BK : (k0 + 1) * BK], k_k, disable_tma=True)
                        for n, kk in T.Parallel(block_C, BK):
                            dk_corr[bid, hid, t_bwd * block_C + n, k0 * BK + kk] = dP_frag[n, kk] * T.exp2((g_last - g_c[n]) * _LOG2E)
                            p_mul[n, kk] = dP_frag[n, kk] * k_k[n, kk] * T.exp2((g_last - g_c[n]) * _LOG2E)

                        T.reduce_sum(p_mul, d_g_pos_tmp, dim=1)
                        for n in T.Parallel(block_C):
                            d_g_pos[n] += d_g_pos_tmp[n]

                    for n in T.Parallel(block_C):
                        dg_c[n] = -d_g_pos[n]

                    # g_last term1: sum(dh_buf * H) * exp(g_last)
                    dg_last_scalar1[0] = T.float32(0.0)
                    for k0 in T.serial(0, sub_dim_k):
                        for v0 in T.serial(0, sub_dim_v):
                            T.copy(dh_buf[k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], dh_buf_kv, disable_tma=True)
                            T.copy(S[bid, hid, t_bwd, k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], h_kv, disable_tma=True)
                            for i, j in T.Parallel(BK, BV):
                                dh_new_kv[i, j] = dh_buf_kv[i, j] * h_kv[i, j]
                            T.reduce_sum(dh_new_kv, dh_row_tmp, dim=1)
                            T.reduce_sum(dh_row_tmp, dg_last_scalar2, dim=0)
                            dg_last_scalar1[0] += dg_last_scalar2[0]

                    # g_last term2: sum_n(d_g_pos)
                    T.reduce_sum(d_g_pos, dg_last_scalar2, dim=0)
                    dg_c[block_C - 1] = dg_c[block_C - 1] + dg_last_scalar1[0] * exp_g_last + dg_last_scalar2[0]

                    # Write dg_correction for this chunk
                    for i in T.Parallel(block_C):
                        dg_corr[bid, hid, t_bwd * block_C + i] = dg_c[i]

                    # Carry dh to next iteration:
                    # dh_buf = dh_local + dh_buf * exp(g_last), tiled on k/v
                    for k0 in T.serial(0, sub_dim_k):
                        for v0 in T.serial(0, sub_dim_v):
                            T.copy(dh_local[bid, hid, t_bwd, k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], dh_loc_kv, disable_tma=True)
                            T.copy(dh_buf[k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], dh_buf_kv, disable_tma=True)
                            for i, j in T.Parallel(BK, BV):
                                dh_new_kv[i, j] = dh_loc_kv[i, j] + dh_buf_kv[i, j] * exp_g_last
                            T.copy(dh_new_kv, dh_buf[k0 * BK : (k0 + 1) * BK, v0 * BV : (v0 + 1) * BV], disable_tma=True)

        return dh_recurrence_bwd_kernel

    return _func


@torch.library.custom_op("tileops::gated_deltanet_bwd_kernel", mutates_args=())
def _gated_deltanet_bwd_wrapped_kernel(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    num_stages: int, threads: int,
    parallel_threads: int, recurrence_threads: int,
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
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(parallel_threads)
    dh_recurrence_bwd_fn = _dh_recurrence_bwd_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(num_stages, recurrence_threads)
    wu_bwd_fn = compute_w_u_bwd_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(num_stages, threads)

    Aw, Au, w, u = fused_fn(k, v, g_cum, beta)
    dq, dk_partial, dg_partial, dw, du_partial, v_new, dh_local = \
        bwd_parallel_fn(do, q, k, g_cum, w, u, S)
    dk_corr, du_corr, dg_corr = \
        dh_recurrence_bwd_fn(g_cum, k, v_new, S, dh_local)

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


class GatedDeltaNetBwdKernel(Kernel):
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
        return {
            "num_stages": 2,
            "threads": threads,
            "parallel_threads": threads,
            "recurrence_threads": threads,
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
            do, q, k, v, g, beta, S,
        )
