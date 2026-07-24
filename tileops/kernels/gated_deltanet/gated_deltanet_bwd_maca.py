"""MACA Gated DeltaNet backward kernels for the 64 KiB smem limit."""

import functools
from typing import Tuple

import tilelang
import tilelang.language as T
import torch

from .gated_deltanet_bwd import GatedDeltaNetBwdKernel
from .gated_deltanet_fwd import _LOG2E

__all__ = ["GatedDeltaNetBwdMACAKernel"]


@functools.lru_cache(maxsize=32)
def _bwd_parallel_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 16,
):
    """Parallel per-chunk backward gradients with a tiled V dimension."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = block_v
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
            dq: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dk_partial: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dg_partial: T.Tensor([batch, head, seq_len], dtype),
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_partial: T.Tensor([batch, head, seq_len, dim_v], dtype),
            v_new_out: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
        ):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_fragment([block_C], dtype)
                g_last = T.alloc_fragment([1], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                do_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                o_part = T.alloc_shared([block_C, BV], dtype)
                d_v_new_c = T.alloc_shared([block_C, BV], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                d_attn = T.alloc_shared([block_C, block_C], dtype)
                P = T.alloc_shared([block_C, dim_k], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                exp_g = T.alloc_fragment([block_C], accum_dtype)
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

                T.copy(q[bid, hid, tid * block_C : (tid + 1) * block_C, :], q_c, disable_tma=True)
                T.copy(k[bid, hid, tid * block_C : (tid + 1) * block_C, :], k_c, disable_tma=True)
                T.copy(g[bid, hid, tid * block_C : (tid + 1) * block_C], g_c, disable_tma=True)
                g_last[0] = g[bid, hid, (tid + 1) * block_C - 1]
                T.copy(w[bid, hid, tid * block_C : (tid + 1) * block_C, :], w_c, disable_tma=True)

                for i in T.Parallel(block_C):
                    exp_g[i] = T.exp2(g_c[i] * _LOG2E)

                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(
                        i >= j,
                        attn_frag[i, j] * T.exp2(
                            (g_c[i] - g[bid, hid, tid * block_C + j]) * _LOG2E),
                        T.float32(0.0))

                T.clear(d_attn_frag)
                T.clear(d_q_c_frag)
                T.clear(dP_acc)
                for i in T.Parallel(block_C):
                    dg3_acc[i] = T.float32(0.0)

                for vt in T.serial(0, num_v_tiles):
                    v_off = vt * BV
                    T.copy(u[bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV], u_c, disable_tma=True)
                    T.copy(do[bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV], do_c, disable_tma=True)
                    T.copy(S[bid, hid, tid, :, v_off : v_off + BV], h_c, disable_tma=True)

                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_last[0]) * _LOG2E)
                    T.copy(v_new_c, v_new_out[bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV], disable_tma=True)

                    T.clear(ws_frag)
                    T.gemm(q_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        o_part[i, j] = ws_frag[i, j] * exp_g[i]

                    T.clear(d_v_new_frag)
                    T.gemm(attn, do_c, d_v_new_frag, transpose_A=True)
                    T.copy(d_v_new_frag, d_v_new_c)
                    T.copy(d_v_new_c, du_partial[bid, hid, tid * block_C : (tid + 1) * block_C, v_off : v_off + BV], disable_tma=True)

                    T.gemm(do_c, v_new_c, d_attn_frag, transpose_B=True)

                    for i, j in T.Parallel(block_C, BV):
                        o_part[i, j] = do_c[i, j] * o_part[i, j]
                    T.reduce_sum(o_part, dg_row, dim=1)
                    for i in T.Parallel(block_C):
                        dg3_acc[i] += dg_row[i]

                    for i, j in T.Parallel(block_C, BV):
                        o_part[i, j] = do_c[i, j] * exp_g[i]
                    T.gemm(o_part, h_c, d_q_c_frag, transpose_B=True)

                    for i, j in T.Parallel(block_C, dim_k):
                        P[i, j] = q_c[i, j] * exp_g[i]
                    T.clear(dh_tile_frag)
                    T.gemm(P, do_c, dh_tile_frag, transpose_A=True)

                    T.clear(dP_frag)
                    T.gemm(d_v_new_c, h_c, dP_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, dim_k):
                        dP_acc[i, j] -= dP_frag[i, j]
                    for i, j in T.Parallel(block_C, dim_k):
                        P[i, j] = w_c[i, j] * T.exp2(
                            (g_c[i] + g_last[0]) * _LOG2E)
                    T.clear(dh_sub_frag)
                    T.gemm(P, d_v_new_c, dh_sub_frag, transpose_A=True)
                    for i, j in T.Parallel(dim_k, BV):
                        dh_tile_frag[i, j] -= dh_sub_frag[i, j]
                    T.copy(dh_tile_frag, dh_local[bid, hid, tid, :, v_off : v_off + BV], disable_tma=True)

                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(i >= j, d_attn_frag[i, j], T.float32(0.0))
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = d_attn[i, j] * attn[i, j]
                dg_work = T.alloc_shared([block_C], dtype)
                T.reduce_sum(attn, dg_c, dim=1)
                T.reduce_sum(attn, dg_work, dim=0)
                for i in T.Parallel(block_C):
                    dg_c[i] = dg3_acc[i] + dg_c[i] - dg_work[i]

                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(
                        i >= j,
                        d_attn[i, j] * T.exp2(
                            (g_c[i] - g[bid, hid, tid * block_C + j]) * _LOG2E),
                        T.float32(0.0))
                T.gemm(d_attn, k_c, d_q_c_frag)
                T.copy(d_q_c_frag, dq[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)
                T.clear(d_k_c_frag)
                T.gemm(d_attn, q_c, d_k_c_frag, transpose_A=True)
                T.copy(d_k_c_frag, dk_partial[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)

                for i, j in T.Parallel(block_C, dim_k):
                    dw_frag[i, j] = dP_acc[i, j] * T.exp2((g_c[i] + g_last[0]) * _LOG2E)
                T.copy(dw_frag, dw[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)
                for i, j in T.Parallel(block_C, dim_k):
                    P[i, j] = w_c[i, j] * T.exp2((g_c[i] + g_last[0]) * _LOG2E)
                for i, j in T.Parallel(block_C, dim_k):
                    P[i, j] = P[i, j] * dP_acc[i, j]
                T.reduce_sum(P, dg_work, dim=1)
                dg_step5_total = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(dg_work, dg_step5_total, dim=0)
                for i in T.Parallel(block_C):
                    dg_c[i] += dg_work[i]
                dg_c[block_C - 1] = dg_c[block_C - 1] + dg_step5_total[0]

                for i in T.Parallel(block_C):
                    dg_partial[bid, hid, tid * block_C + i] = dg_c[i]

        return bwd_parallel_kernel

    return _func


@functools.lru_cache(maxsize=32)
def _dh_recurrence_bwd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 16,
):
    """Traverse V tiles serially in one CTA per batch/head recurrence."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = block_v
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
            dk_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dg_corr: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                g_c = T.alloc_shared([block_C], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                k_scaled = T.alloc_shared([block_C, dim_k], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                dh_carry = T.alloc_shared([dim_k, dim_v], dtype)
                dh_buf = T.alloc_shared([dim_k, BV], dtype)
                dh_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                du_corr_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_tile = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dP_acc = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dh_h_tmp = T.alloc_fragment([dim_k, BV], accum_dtype)
                d_g_last_tile = T.alloc_fragment([dim_k], accum_dtype)
                d_g_last_scalar = T.alloc_fragment([1], accum_dtype)
                d_g_last_acc = T.alloc_fragment([1], accum_dtype)
                d_g_pos = T.alloc_fragment([block_C], accum_dtype)
                d_g_last_scalar2 = T.alloc_fragment([1], accum_dtype)

                for i, j in T.Parallel(dim_k, dim_v):
                    dh_carry[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    T.copy(g[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C], g_c, disable_tma=True)
                    T.copy(k[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], k_c, disable_tma=True)

                    for pn, sk in T.Parallel(block_C, dim_k):
                        k_scaled[pn, sk] = k_c[pn, sk] * T.exp2(
                            (g_c[block_C - 1] - g_c[pn]) * _LOG2E)

                    T.clear(dP_acc)
                    T.clear(d_g_last_acc)
                    for vt in T.serial(0, num_v_tiles):
                        v_off = vt * BV
                        T.copy(
                            v_new[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, v_off : v_off + BV],
                            v_new_c,
                            disable_tma=True,
                        )
                        T.copy(S[bid, hid, t_bwd, :, v_off : v_off + BV], h_c, disable_tma=True)
                        T.copy(
                            dh_local[bid, hid, t_bwd, :, v_off : v_off + BV],
                            dh_loc,
                            disable_tma=True,
                        )
                        T.copy(dh_carry[:, v_off : v_off + BV], dh_buf, disable_tma=True)

                        T.clear(du_corr_frag)
                        T.gemm(k_scaled, dh_buf, du_corr_frag)
                        T.copy(
                            du_corr_frag,
                            du_corr[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, v_off : v_off + BV],
                            disable_tma=True,
                        )

                        T.clear(dP_tile)
                        T.gemm(v_new_c, dh_buf, dP_tile, transpose_B=True)
                        for n, kk in T.Parallel(block_C, dim_k):
                            dP_acc[n, kk] += dP_tile[n, kk]

                        for i, j in T.Parallel(dim_k, BV):
                            dh_h_tmp[i, j] = dh_buf[i, j] * h_c[i, j]
                        T.reduce_sum(dh_h_tmp, d_g_last_tile, dim=1)
                        T.reduce_sum(d_g_last_tile, d_g_last_scalar, dim=0)
                        d_g_last_acc[0] += d_g_last_scalar[0]

                        for i, j in T.Parallel(dim_k, BV):
                            dh_frag[i, j] = dh_loc[i, j] + dh_buf[i, j] * T.exp2(
                                g_c[block_C - 1] * _LOG2E)
                        T.copy(dh_frag, dh_carry[:, v_off : v_off + BV], disable_tma=True)

                    for n, kk in T.Parallel(block_C, dim_k):
                        dk_corr[bid, hid, t_bwd * block_C + n, kk] = dP_acc[n, kk] * T.exp2(
                            (g_c[block_C - 1] - g_c[n]) * _LOG2E)

                    for n, kk in T.Parallel(block_C, dim_k):
                        dP_acc[n, kk] = dP_acc[n, kk] * k_scaled[n, kk]
                    T.reduce_sum(dP_acc, d_g_pos, dim=1)
                    for n in T.Parallel(block_C):
                        dg_c[n] = -d_g_pos[n]

                    T.reduce_sum(d_g_pos, d_g_last_scalar2, dim=0)
                    dg_c[block_C - 1] = dg_c[block_C - 1] + d_g_last_acc[0] * T.exp2(
                        g_c[block_C - 1] * _LOG2E) + d_g_last_scalar2[0]

                    for i in T.Parallel(block_C):
                        dg_corr[bid, hid, t_bwd * block_C + i] = dg_c[i]

        return dh_recurrence_bwd_kernel

    return _func


@torch.library.custom_op("tileops_maca::gated_deltanet_bwd_kernel", mutates_args=())
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


def _safe_block_v(dim_v: int, dtype: str) -> int:
    """Select a MACA MMA-compatible V tile with shared-memory headroom."""
    preferred = 16 if dtype == "float32" else 32
    for block_v in (preferred, 16):
        if dim_v % block_v == 0:
            return block_v
    return dim_v


class GatedDeltaNetBwdMACAKernel(GatedDeltaNetBwdKernel):
    """Gated DeltaNet backward using V-tiling under MACA's smem cap."""

    @property
    def default_config(self) -> dict:
        threads = 256 if self.chunk_size >= 64 else 128
        return {
            "num_stages": 1,
            "threads": threads,
            "parallel_threads": threads,
            "recurrence_threads": threads,
            "block_v": _safe_block_v(self.dim_v, self.dtype_str),
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
            self.config.get("num_stages", 1), self.config.get("threads", 256),
            self.config.get("parallel_threads", 256),
            self.config.get("recurrence_threads", 256),
            self.config.get("block_v", _safe_block_v(self.dim_v, self.dtype_str)),
            do, q, k, v, g, beta, S,
        )
