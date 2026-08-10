"""GLA backward MACA path: smem-safe sub-chunk tiled kernels.

Keeps CUDA ``GLABwdKernel`` unchanged. Owns Pass-1 dh (FLA store→decay→add)
and Stage-2 dq/dkdv/dg under MACA's 64 KiB/block budget by streaming one
output band of size ``sub_chunk_size`` at a time.
"""

import functools
from typing import Callable, Optional, Tuple

import tilelang
import torch
from tilelang import language as T
from tilelang.profiler import do_bench

from tileops.kernels.kernel_base import Kernel

from .gla_bwd import LOG2_E
from .gla_fwd import _gla_precompute_g_kernel

__all__ = ["GLABwdMACAKernel"]


@functools.lru_cache(maxsize=32)
def _gla_bwd_dh_kernel_maca(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    has_initial_state: bool,
    dtype: str,
    num_v_partitions: int = 1,
) -> Callable:
    """MACA Pass-1 dh: FLA order store(∂L/∂h_{c+1}) → decay → add local.

    Independent of CUDA ``_gla_bwd_dh_kernel`` so MACA can fix inter-chunk
    semantics without touching ``gla_bwd.py``.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    dim_v_part = dim_v // num_v_partitions

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _dh_func(num_stages, threads=128):
        q_shape = [batch, seq_len, heads, dim_k]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        do_shape = [batch, seq_len, heads, dim_v]
        dht_shape = [batch, heads, dim_k, dim_v]
        dh_out_shape = [batch, num_chunks, heads, dim_k, dim_v]
        dh0_shape = [batch, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            do: T.Tensor(do_shape, dtype),
            dht: T.Tensor(dht_shape, accum_dtype),
            dh_out: T.Tensor(dh_out_shape, accum_dtype),
            dh0: T.Tensor(dh0_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_v_partitions,
                          threads=threads) as bx:
                i_b = bx // (heads * num_v_partitions)
                i_h = (bx // num_v_partitions) % heads
                i_vp = bx % num_v_partitions
                v_offset = i_vp * dim_v_part

                dh_s = T.alloc_shared([dim_k, dim_v_part], accum_dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                q_s = T.alloc_shared([chunk_size, dim_k], dtype)
                do_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                q_gated_s = T.alloc_shared([chunk_size, dim_k], dtype)

                for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                    dh_s[i_k, i_v] = dht[i_b, i_h, i_k, v_offset + i_v]

                for t in T.Serial(num_chunks):
                    i_c = num_chunks - 1 - t
                    chunk_start = i_c * chunk_size

                    T.copy(q[i_b, chunk_start:chunk_start + chunk_size,
                             i_h, :],
                           q_s, disable_tma=True)
                    T.copy(do[i_b, chunk_start:chunk_start + chunk_size,
                              i_h, v_offset:v_offset + dim_v_part],
                           do_s, disable_tma=True)
                    T.copy(g_cumsum[i_b,
                                    chunk_start:chunk_start + chunk_size,
                                    i_h, :],
                           g_cumsum_s, disable_tma=True)

                    g_last = T.alloc_fragment([dim_k], accum_dtype)
                    for i_k in T.Parallel(dim_k):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

                    # Store incoming ∂L/∂h_{c+1} (FLA; used by dk/dv/dg inter).
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh_out[i_b, i_c, i_h, i_k,
                               v_offset + i_v] = dh_s[i_k, i_v]

                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh_s[i_k, i_v] = (dh_s[i_k, i_v]
                                          * T.exp2(g_last[i_k] * LOG2_E))

                    for i_t, i_k in T.Parallel(chunk_size, dim_k):
                        q_gated_s[i_t, i_k] = T.cast(
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.exp2(g_cumsum_s[i_t, i_k] * LOG2_E),
                            dtype)

                    dh_delta = T.alloc_fragment([dim_k, dim_v_part],
                                               accum_dtype)
                    T.fill(dh_delta, 0.0)
                    T.gemm(q_gated_s, do_s, dh_delta, transpose_A=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh_s[i_k, i_v] = (dh_s[i_k, i_v]
                                          + scale * dh_delta[i_k, i_v])

                if has_initial_state:
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh0[i_b, i_h, i_k, v_offset + i_v] = dh_s[i_k, i_v]
                else:
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh0[i_b, i_h, i_k, v_offset + i_v] = 0.0

        return _main

    return _dh_func


@functools.lru_cache(maxsize=32)
def _gla_bwd_dq_kernel_maca(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    sub_chunk_size: int = 16,
) -> Callable:
    """One CTA computes one query sub-chunk of dq (intra + inter)."""
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    BT = chunk_size
    BC = sub_chunk_size
    NS = BT // BC

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _dq_func(num_stages, threads=64):
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        do_shape = [batch, seq_len, heads, dim_v]
        h_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]
        dq_shape = [batch, seq_len, heads, dim_k]

        @T.prim_func
        def _main(
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            do: T.Tensor(do_shape, dtype),
            h: T.Tensor(h_shape, accum_dtype),
            dq_out: T.Tensor(dq_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks * NS, threads=threads) as bx:
                i_b = bx // (heads * num_chunks * NS)
                rem = bx % (heads * num_chunks * NS)
                i_h = rem // (num_chunks * NS)
                rem2 = rem % (num_chunks * NS)
                i_c = rem2 // NS
                s_i = rem2 % NS
                chunk_start = i_c * BT
                q_start = chunk_start + s_i * BC

                do_s = T.alloc_shared([BC, dim_v], dtype)
                g_i_s = T.alloc_shared([BC, dim_k], accum_dtype)
                k_s = T.alloc_shared([BC, dim_k], dtype)
                v_s = T.alloc_shared([BC, dim_v], dtype)
                g_j_s = T.alloc_shared([BC, dim_k], accum_dtype)
                dA_s = T.alloc_shared([BC, BC], dtype)
                k_shifted = T.alloc_shared([BC, dim_k], dtype)
                h_s = T.alloc_shared([dim_k, dim_v], dtype)

                T.copy(do[i_b, q_start:q_start + BC, i_h, :], do_s, disable_tma=True)
                T.copy(
                    g_cumsum[i_b, q_start:q_start + BC, i_h, :],
                    g_i_s,
                    disable_tma=True,
                )

                dq_frag = T.alloc_fragment([BC, dim_k], accum_dtype)
                T.fill(dq_frag, 0.0)

                # Off-diagonal key bands: s_j < s_i
                for s_j in T.Serial(s_i):
                    k_start = chunk_start + s_j * BC
                    T.copy(
                        k[i_b, k_start:k_start + BC, i_h, :],
                        k_s,
                        disable_tma=True,
                    )
                    T.copy(
                        v[i_b, k_start:k_start + BC, i_h, :],
                        v_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[i_b, k_start:k_start + BC, i_h, :],
                        g_j_s,
                        disable_tma=True,
                    )

                    dA_frag = T.alloc_fragment([BC, BC], accum_dtype)
                    T.fill(dA_frag, 0.0)
                    T.gemm(
                        do_s,
                        v_s,
                        dA_frag,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i_t, i_j in T.Parallel(BC, BC):
                        dA_s[i_t, i_j] = T.cast(scale * dA_frag[i_t, i_j], dtype)

                    for i_t, i_k in T.Parallel(BC, dim_k):
                        k_shifted[i_t, i_k] = T.cast(
                            T.cast(k_s[i_t, i_k], accum_dtype)
                            * T.exp2(
                                (g_i_s[0, i_k] - g_j_s[i_t, i_k]) * LOG2_E),
                            dtype,
                        )
                    dq_sub = T.alloc_fragment([BC, dim_k], accum_dtype)
                    T.fill(dq_sub, 0.0)
                    T.gemm(
                        dA_s,
                        k_shifted,
                        dq_sub,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i_t, i_k in T.Parallel(BC, dim_k):
                        dq_frag[i_t, i_k] = (
                            dq_frag[i_t, i_k]
                            + dq_sub[i_t, i_k]
                            * T.exp2(
                                (g_i_s[i_t, i_k] - g_i_s[0, i_k]) * LOG2_E))

                # Diagonal band: fragment path (matches CUDA bf16-safe form)
                k_start = chunk_start + s_i * BC
                T.copy(k[i_b, k_start:k_start + BC, i_h, :], k_s, disable_tma=True)
                T.copy(v[i_b, k_start:k_start + BC, i_h, :], v_s, disable_tma=True)
                T.copy(
                    g_cumsum[i_b, k_start:k_start + BC, i_h, :],
                    g_j_s,
                    disable_tma=True,
                )
                dA_frag = T.alloc_fragment([BC, BC], accum_dtype)
                T.fill(dA_frag, 0.0)
                T.gemm(
                    do_s,
                    v_s,
                    dA_frag,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i_t, i_j in T.Parallel(BC, BC):
                    dA_s[i_t, i_j] = T.cast(
                        T.if_then_else(
                            i_j <= i_t,
                            scale * dA_frag[i_t, i_j],
                            T.float32(0.0),
                        ),
                        dtype,
                    )
                for j_local in T.Serial(BC):
                    dA_col = T.alloc_fragment([BC], accum_dtype)
                    k_row = T.alloc_fragment([dim_k], accum_dtype)
                    g_j = T.alloc_fragment([dim_k], accum_dtype)
                    for i_local in T.Parallel(BC):
                        dA_col[i_local] = T.if_then_else(
                            j_local <= i_local,
                            T.cast(dA_s[i_local, j_local], accum_dtype),
                            T.float32(0.0),
                        )
                    for i_k in T.Parallel(dim_k):
                        k_row[i_k] = T.cast(k_s[j_local, i_k], accum_dtype)
                        g_j[i_k] = g_j_s[j_local, i_k]
                    for i_local, i_k in T.Parallel(BC, dim_k):
                        dq_frag[i_local, i_k] = (
                            dq_frag[i_local, i_k]
                            + dA_col[i_local]
                            * k_row[i_k]
                            * T.exp2(
                                (g_i_s[i_local, i_k] - g_j[i_k]) * LOG2_E))

                # Inter-chunk: dq += scale * (do @ h.T) * exp(g)
                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    h_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, i_v], dtype)
                dq_inter = T.alloc_fragment([BC, dim_k], accum_dtype)
                T.fill(dq_inter, 0.0)
                T.gemm(
                    do_s,
                    h_s,
                    dq_inter,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i_t, i_k in T.Parallel(BC, dim_k):
                    dq_frag[i_t, i_k] = (
                        dq_frag[i_t, i_k]
                        + scale
                        * dq_inter[i_t, i_k]
                        * T.exp2(g_i_s[i_t, i_k] * LOG2_E))

                for i_t, i_k in T.Parallel(BC, dim_k):
                    dq_out[i_b, q_start + i_t, i_h, i_k] = dq_frag[i_t, i_k]

        return _main

    return _dq_func


@functools.lru_cache(maxsize=32)
def _gla_bwd_dkdv_kernel_maca(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    sub_chunk_size: int = 16,
) -> Callable:
    """One CTA computes one key/value sub-chunk of dk and dv (intra + inter)."""
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    BT = chunk_size
    BC = sub_chunk_size
    NS = BT // BC

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _dkdv_func(num_stages, threads=64):
        q_shape = [batch, seq_len, heads, dim_k]
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        do_shape = [batch, seq_len, heads, dim_v]
        dh_shape = [batch, num_chunks, heads, dim_k, dim_v]
        dk_shape = [batch, seq_len, heads, dim_k]
        dv_shape = [batch, seq_len, heads, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            do: T.Tensor(do_shape, dtype),
            dh: T.Tensor(dh_shape, accum_dtype),
            dk_out: T.Tensor(dk_shape, accum_dtype),
            dv_out: T.Tensor(dv_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks * NS, threads=threads) as bx:
                i_b = bx // (heads * num_chunks * NS)
                rem = bx % (heads * num_chunks * NS)
                i_h = rem // (num_chunks * NS)
                rem2 = rem % (num_chunks * NS)
                i_c = rem2 // NS
                s_j = rem2 % NS
                chunk_start = i_c * BT
                k_start = chunk_start + s_j * BC

                k_s = T.alloc_shared([BC, dim_k], dtype)
                v_s = T.alloc_shared([BC, dim_v], dtype)
                g_j_s = T.alloc_shared([BC, dim_k], accum_dtype)
                q_s = T.alloc_shared([BC, dim_k], dtype)
                do_s = T.alloc_shared([BC, dim_v], dtype)
                do_acc = T.alloc_shared([BC, dim_v], accum_dtype)
                g_i_s = T.alloc_shared([BC, dim_k], accum_dtype)
                dA_s = T.alloc_shared([BC, BC], accum_dtype)
                A_s = T.alloc_shared([BC, BC], accum_dtype)
                q_shifted = T.alloc_shared([BC, dim_k], accum_dtype)
                k_gated = T.alloc_shared([BC, dim_k], accum_dtype)
                v_acc = T.alloc_shared([BC, dim_v], accum_dtype)
                dh_s = T.alloc_shared([dim_k, dim_v], accum_dtype)
                g_last = T.alloc_fragment([dim_k], accum_dtype)

                T.copy(k[i_b, k_start:k_start + BC, i_h, :], k_s, disable_tma=True)
                T.copy(v[i_b, k_start:k_start + BC, i_h, :], v_s, disable_tma=True)
                T.copy(
                    g_cumsum[i_b, k_start:k_start + BC, i_h, :],
                    g_j_s,
                    disable_tma=True,
                )
                for i_k in T.Parallel(dim_k):
                    g_last[i_k] = g_cumsum[i_b, chunk_start + BT - 1, i_h, i_k]
                for i_t, i_v in T.Parallel(BC, dim_v):
                    v_acc[i_t, i_v] = T.cast(v_s[i_t, i_v], accum_dtype)

                dk_frag = T.alloc_fragment([BC, dim_k], accum_dtype)
                dv_frag = T.alloc_fragment([BC, dim_v], accum_dtype)
                T.fill(dk_frag, 0.0)
                T.fill(dv_frag, 0.0)

                # Off-diagonal query bands: s_i > s_j
                for s_off in T.Serial(NS - s_j - 1):
                    s_i = s_j + 1 + s_off
                    q_start = chunk_start + s_i * BC
                    T.copy(
                        q[i_b, q_start:q_start + BC, i_h, :],
                        q_s,
                        disable_tma=True,
                    )
                    T.copy(
                        do[i_b, q_start:q_start + BC, i_h, :],
                        do_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[i_b, q_start:q_start + BC, i_h, :],
                        g_i_s,
                        disable_tma=True,
                    )
                    for i_t, i_v in T.Parallel(BC, dim_v):
                        do_acc[i_t, i_v] = T.cast(
                            do_s[i_t, i_v], accum_dtype)

                    dA_frag = T.alloc_fragment([BC, BC], accum_dtype)
                    T.fill(dA_frag, 0.0)
                    T.gemm(
                        do_s,
                        v_s,
                        dA_frag,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i_t, i_j in T.Parallel(BC, BC):
                        dA_s[i_t, i_j] = scale * dA_frag[i_t, i_j]

                    for i_t, i_k in T.Parallel(BC, dim_k):
                        q_shifted[i_t, i_k] = (
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.exp2(
                                (g_i_s[i_t, i_k] - g_j_s[BC - 1, i_k])
                                * LOG2_E))
                    dk_sub = T.alloc_fragment([BC, dim_k], accum_dtype)
                    T.fill(dk_sub, 0.0)
                    T.gemm(
                        dA_s,
                        q_shifted,
                        dk_sub,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for j_t, i_k in T.Parallel(BC, dim_k):
                        dk_frag[j_t, i_k] = (
                            dk_frag[j_t, i_k]
                            + dk_sub[j_t, i_k]
                            * T.exp2(
                                (g_j_s[BC - 1, i_k] - g_j_s[j_t, i_k])
                                * LOG2_E))

                    A_frag = T.alloc_fragment([BC, BC], accum_dtype)
                    T.fill(A_frag, 0.0)
                    for i_k in T.Serial(dim_k):
                        for i_t, i_j in T.Parallel(BC, BC):
                            A_frag[i_t, i_j] = A_frag[i_t, i_j] + (
                                T.cast(q_s[i_t, i_k], accum_dtype)
                                * T.cast(k_s[i_j, i_k], accum_dtype)
                                * T.exp2(
                                    (g_i_s[i_t, i_k] - g_j_s[i_j, i_k])
                                    * LOG2_E))
                    for i_t, i_j in T.Parallel(BC, BC):
                        A_s[i_t, i_j] = A_frag[i_t, i_j] * scale
                    T.gemm(
                        A_s,
                        do_acc,
                        dv_frag,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                # Diagonal band
                q_start = k_start
                T.copy(
                    q[i_b, q_start:q_start + BC, i_h, :],
                    q_s,
                    disable_tma=True,
                )
                T.copy(
                    do[i_b, q_start:q_start + BC, i_h, :],
                    do_s,
                    disable_tma=True,
                )
                T.copy(
                    g_cumsum[i_b, q_start:q_start + BC, i_h, :],
                    g_i_s,
                    disable_tma=True,
                )
                for i_t, i_v in T.Parallel(BC, dim_v):
                    do_acc[i_t, i_v] = T.cast(do_s[i_t, i_v], accum_dtype)

                dA_frag = T.alloc_fragment([BC, BC], accum_dtype)
                T.fill(dA_frag, 0.0)
                T.gemm(
                    do_s,
                    v_s,
                    dA_frag,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i_t, i_j in T.Parallel(BC, BC):
                    dA_s[i_t, i_j] = T.if_then_else(
                        i_j <= i_t,
                        scale * dA_frag[i_t, i_j],
                        T.float32(0.0),
                    )
                for i_local in T.Serial(BC):
                    dA_row = T.alloc_fragment([BC], accum_dtype)
                    q_row = T.alloc_fragment([dim_k], accum_dtype)
                    g_i = T.alloc_fragment([dim_k], accum_dtype)
                    for j_local in T.Parallel(BC):
                        dA_row[j_local] = T.if_then_else(
                            j_local <= i_local,
                            dA_s[i_local, j_local],
                            T.float32(0.0),
                        )
                    for i_k in T.Parallel(dim_k):
                        q_row[i_k] = T.cast(
                            q_s[i_local, i_k], accum_dtype)
                        g_i[i_k] = g_i_s[i_local, i_k]
                    for j_local, i_k in T.Parallel(BC, dim_k):
                        dk_frag[j_local, i_k] = (
                            dk_frag[j_local, i_k]
                            + dA_row[j_local]
                            * q_row[i_k]
                            * T.exp2(
                                (g_i[i_k] - g_j_s[j_local, i_k])
                                * LOG2_E))

                A_frag = T.alloc_fragment([BC, BC], accum_dtype)
                T.fill(A_frag, 0.0)
                for i_k in T.Serial(dim_k):
                    for i_t, i_j in T.Parallel(BC, BC):
                        A_frag[i_t, i_j] = A_frag[i_t, i_j] + (
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.cast(k_s[i_j, i_k], accum_dtype)
                            * T.exp2(
                                (g_i_s[i_t, i_k] - g_j_s[i_j, i_k])
                                * LOG2_E))
                for i_t, i_j in T.Parallel(BC, BC):
                    A_s[i_t, i_j] = T.if_then_else(
                        i_j <= i_t,
                        A_frag[i_t, i_j] * scale,
                        T.float32(0.0),
                    )
                T.gemm(
                    A_s,
                    do_acc,
                    dv_frag,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    dh_s[i_k, i_v] = dh[i_b, i_c, i_h, i_k, i_v]

                for i_t, i_k in T.Parallel(BC, dim_k):
                    k_gated[i_t, i_k] = (
                        T.cast(k_s[i_t, i_k], accum_dtype)
                        * T.exp2(
                            (g_last[i_k] - g_j_s[i_t, i_k]) * LOG2_E))
                T.gemm(
                    k_gated,
                    dh_s,
                    dv_frag,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                dk_inter = T.alloc_fragment([BC, dim_k], accum_dtype)
                T.fill(dk_inter, 0.0)
                T.gemm(
                    v_acc,
                    dh_s,
                    dk_inter,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i_t, i_k in T.Parallel(BC, dim_k):
                    dk_frag[i_t, i_k] = (
                        dk_frag[i_t, i_k]
                        + dk_inter[i_t, i_k]
                        * T.exp2(
                            (g_last[i_k] - g_j_s[i_t, i_k]) * LOG2_E))

                for i_t, i_k in T.Parallel(BC, dim_k):
                    dk_out[i_b, k_start + i_t, i_h, i_k] = dk_frag[i_t, i_k]
                for i_t, i_v in T.Parallel(BC, dim_v):
                    dv_out[i_b, k_start + i_t, i_h, i_v] = dv_frag[i_t, i_v]

        return _main

    return _dkdv_func


@functools.lru_cache(maxsize=32)
def _gla_bwd_dg_kernel_maca(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    sub_chunk_size: int = 16,
) -> Callable:
    """Per-chunk dg: reverse-cumsum(q*dq - k*dk) + inter correction."""
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    BT = chunk_size
    BC = sub_chunk_size
    NS = BT // BC

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _dg_func(num_stages, threads=64):
        q_shape = [batch, seq_len, heads, dim_k]
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        dq_shape = [batch, seq_len, heads, dim_k]
        dk_shape = [batch, seq_len, heads, dim_k]
        h_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]
        dh_shape = [batch, num_chunks, heads, dim_k, dim_v]
        dg_shape = [batch, seq_len, heads, dim_k]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            dq: T.Tensor(dq_shape, accum_dtype),
            dk: T.Tensor(dk_shape, accum_dtype),
            h: T.Tensor(h_shape, accum_dtype),
            dh: T.Tensor(dh_shape, accum_dtype),
            dg_out: T.Tensor(dg_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                chunk_start = i_c * BT

                dg_s = T.alloc_shared([BT, dim_k], accum_dtype)
                q_s = T.alloc_shared([BC, dim_k], dtype)
                k_s = T.alloc_shared([BC, dim_k], dtype)
                dq_s = T.alloc_shared([BC, dim_k], accum_dtype)
                dk_s = T.alloc_shared([BC, dim_k], accum_dtype)
                g_s = T.alloc_shared([BC, dim_k], accum_dtype)
                v_s = T.alloc_shared([BC, dim_v], dtype)
                dh_s = T.alloc_shared([dim_k, dim_v], dtype)
                g_last = T.alloc_fragment([dim_k], accum_dtype)
                dg_inter = T.alloc_shared([dim_k], accum_dtype)

                for i_k in T.Parallel(dim_k):
                    g_last[i_k] = g_cumsum[i_b, chunk_start + BT - 1, i_h, i_k]
                    dg_inter[i_k] = T.float32(0.0)

                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    dh_s[i_k, i_v] = T.cast(dh[i_b, i_c, i_h, i_k, i_v], dtype)

                for i_v2 in T.Serial(dim_v):
                    for i_k in T.Parallel(dim_k):
                        dg_inter[i_k] = dg_inter[i_k] + (
                            h[i_b, i_c, i_h, i_k, i_v2]
                            * T.cast(dh[i_b, i_c, i_h, i_k, i_v2], accum_dtype))
                for i_k in T.Parallel(dim_k):
                    dg_inter[i_k] = dg_inter[i_k] * T.exp2(g_last[i_k] * LOG2_E)

                # corr += k * dk_inter_raw * exp(g_last - g)
                for s in T.Serial(NS):
                    t0 = chunk_start + s * BC
                    T.copy(k[i_b, t0:t0 + BC, i_h, :], k_s, disable_tma=True)
                    T.copy(v[i_b, t0:t0 + BC, i_h, :], v_s, disable_tma=True)
                    T.copy(
                        g_cumsum[i_b, t0:t0 + BC, i_h, :],
                        g_s,
                        disable_tma=True,
                    )
                    dk_inter = T.alloc_fragment([BC, dim_k], accum_dtype)
                    T.fill(dk_inter, 0.0)
                    T.gemm(
                        v_s,
                        dh_s,
                        dk_inter,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i_t, i_k in T.Parallel(BC, dim_k):
                        dg_s[s * BC + i_t, i_k] = (
                            T.cast(k_s[i_t, i_k], accum_dtype)
                            * dk_inter[i_t, i_k]
                            * T.exp2(
                                (g_last[i_k] - g_s[i_t, i_k]) * LOG2_E))
                for i_t in T.Serial(BT):
                    for i_k in T.Parallel(dim_k):
                        dg_inter[i_k] = dg_inter[i_k] + dg_s[i_t, i_k]

                for s in T.Serial(NS):
                    t0 = chunk_start + s * BC
                    T.copy(q[i_b, t0:t0 + BC, i_h, :], q_s, disable_tma=True)
                    T.copy(k[i_b, t0:t0 + BC, i_h, :], k_s, disable_tma=True)
                    T.copy(dq[i_b, t0:t0 + BC, i_h, :], dq_s, disable_tma=True)
                    T.copy(dk[i_b, t0:t0 + BC, i_h, :], dk_s, disable_tma=True)
                    for i_t, i_k in T.Parallel(BC, dim_k):
                        dg_s[s * BC + i_t, i_k] = (
                            T.cast(q_s[i_t, i_k], accum_dtype) * dq_s[i_t, i_k]
                            - T.cast(k_s[i_t, i_k], accum_dtype)
                            * dk_s[i_t, i_k])

                for s in T.Serial(BT - 1):
                    i_t_rev = BT - 2 - s
                    for i_k in T.Parallel(dim_k):
                        dg_s[i_t_rev, i_k] = (
                            dg_s[i_t_rev, i_k] + dg_s[i_t_rev + 1, i_k])

                for i_t, i_k in T.Parallel(BT, dim_k):
                    dg_out[i_b, chunk_start + i_t, i_h, i_k] = (
                        dg_s[i_t, i_k] + dg_inter[i_k])

        return _main

    return _dg_func


class GLABwdMACAKernel(Kernel):
    """GLA backward kernel for MACA (smem-safe tiled stage-2).

    Pass 1: same reverse dh accumulation as CUDA.
    Pass 2: separate dq / dkdv / dg kernels with sub-chunk output tiling.
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale if scale > 0 else dim_k**-0.5
        self.dtype = dtype
        self.dtype_name = str(dtype).split(".")[-1]
        self.init_config(config, tune)
        if not tune:
            self._build_kernels(self.config)

    @property
    def default_config(self) -> dict:
        # Stage-2 must use 1 MACA warp (64 threads): BC=16 GEMMs are 16x16,
        # and FullRow with >=2 warps yields warp_cols=0 (divide-by-zero).
        return {
            "num_stages": 1,
            "threads_par": 64,
            "threads_seq": 256,
            "num_v_partitions": 4,
            "sub_chunk_size": 16,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for ns in [1, 2]:
            for t_seq in [128, 256]:
                for nvp in [2, 4]:
                    configs.append({
                        "num_stages": ns,
                        "threads_par": 64,
                        "threads_seq": t_seq,
                        "num_v_partitions": nvp,
                        "sub_chunk_size": 16,
                    })
        return configs

    def _build_kernels(self, config: dict) -> None:
        ns = config.get("num_stages", 1)
        thr_seq = config.get("threads_seq", 256)
        thr_par = config.get("threads_par", 64)
        num_vp = config.get("num_v_partitions", 4)
        bc = config.get("sub_chunk_size", 16)
        self._g_fn = _gla_precompute_g_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.chunk_size,
            self.dtype_name,
        )(ns, thr_par)
        self._dh_fn = _gla_bwd_dh_kernel_maca(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.scale,
            False,
            self.dtype_name,
            num_v_partitions=num_vp,
        )(1, thr_seq)
        self._dh_fn_with_init = _gla_bwd_dh_kernel_maca(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.scale,
            True,
            self.dtype_name,
            num_v_partitions=num_vp,
        )(1, thr_seq)
        self._dq_fn = _gla_bwd_dq_kernel_maca(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.scale,
            self.dtype_name,
            sub_chunk_size=bc,
        )(ns, thr_par)
        self._dkdv_fn = _gla_bwd_dkdv_kernel_maca(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.scale,
            self.dtype_name,
            sub_chunk_size=bc,
        )(ns, thr_par)
        self._dg_fn = _gla_bwd_dg_kernel_maca(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.dtype_name,
            sub_chunk_size=bc,
        )(ns, thr_par)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        if self.autotune_configs is None:
            return
        print(f"Start autotuning {self.__class__.__name__} "
              f"({len(self.autotune_configs)} configs)...")

        B, T, H, K, V = (
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
        )
        BT = self.chunk_size
        NT = T // BT
        dtype_torch = self.dtype

        q = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype_torch).abs()
        h = torch.randn(
            B, NT + 1, H, K, V, device="cuda", dtype=torch.float32) * 0.01
        do = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        dht = torch.zeros(B, H, K, V, dtype=torch.float32, device="cuda")

        best_lat = float("inf")
        best_cfg = None
        for cfg in self.autotune_configs:
            try:
                self._build_kernels(cfg)
                self.forward(q, k, v, g, h, do, dht)
                torch.cuda.synchronize()
                lat = do_bench(
                    lambda: self.forward(q, k, v, g, h, do, dht),
                    warmup=warmup,
                    rep=rep,
                )
                print(f"  config={cfg} -> {lat:.3f}ms")
                if lat < best_lat:
                    best_lat = lat
                    best_cfg = cfg
            except Exception as e:
                print(f"  config={cfg} -> FAILED: {e}")
                continue

        if best_cfg is not None:
            self.config = best_cfg
            self._build_kernels(best_cfg)
            print(f"Best config: {best_cfg} ({best_lat:.3f}ms)")
        else:
            print("Autotuning failed, using default config")
            self.config = self.default_config
            self._build_kernels(self.config)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        has_initial_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype_torch = self.dtype

        g_cumsum = self._g_fn(g.to(dtype_torch))
        dh_fn = self._dh_fn_with_init if has_initial_state else self._dh_fn
        dh_out, _dh0 = dh_fn(
            q.to(dtype_torch),
            g_cumsum,
            do.to(dtype_torch),
            dht,
        )

        dq = self._dq_fn(
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            do.to(dtype_torch),
            h.to(torch.float32),
        )
        dk, dv = self._dkdv_fn(
            q.to(dtype_torch),
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            do.to(dtype_torch),
            dh_out,
        )
        dg = self._dg_fn(
            q.to(dtype_torch),
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            dq,
            dk,
            h.to(torch.float32),
            dh_out,
        )

        return dq, dk, dv, dg
