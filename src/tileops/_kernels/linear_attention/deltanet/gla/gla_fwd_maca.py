"""GLA forward MACA path: smem-safe o-kernel via sub-chunk / V-tile streaming.

Keeps CUDA ``GLAFwdKernel`` unchanged. Owns a self-contained copy of the three
forward passes; only Pass-2 ``o`` shrinks shared buffers when the full-chunk
layout would exceed MACA's 64 KiB/block budget.
"""

import functools
from typing import Callable, Optional, Tuple

import tilelang
import torch
from tilelang import language as T
from tilelang.profiler import do_bench

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention.deltanet.v_tile import GEMM_MIN_N, resolve_block_v

__all__ = ["GLAFwdMACAKernel"]

LOG2_E = 1.44269504
_MACA_SMEM_CAP = 65536
_MACA_SMEM_SLACK = 2048


def _dtype_nbytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


def _o_smem_bytes(
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    sub_chunk_size: int,
    block_v: int,
) -> int:
    """Peak shared-memory estimate for the tiled MACA o-kernel (7 buffers)."""
    elem = _dtype_nbytes(dtype)
    bc = sub_chunk_size
    bt = chunk_size
    bv = dim_v if block_v <= 0 else block_v
    # h_cast, k_s, v_s, g_cumsum_s, q_s, q_gated_s, A_s
    return (
        dim_k * bv * elem
        + bt * dim_k * elem
        + bt * bv * elem
        + bt * dim_k * 4
        + bc * dim_k * elem
        + bc * dim_k * elem
        + bc * bt * elem
    )


def _pick_o_tiles(
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
) -> Tuple[int, int]:
    """Pick the largest (sub_chunk_size, block_v) that fits MACA smem."""
    full_used = _o_smem_bytes(chunk_size, dim_k, dim_v, dtype, chunk_size, dim_v)
    if full_used <= _MACA_SMEM_CAP:
        return chunk_size, dim_v

    best: Tuple[int, int, int] | None = None
    bc = chunk_size
    while bc >= GEMM_MIN_N:
        if chunk_size % bc != 0:
            bc //= 2
            continue
        bv = dim_v
        while bv >= GEMM_MIN_N:
            if dim_v % bv != 0:
                bv //= 2
                continue
            used = _o_smem_bytes(chunk_size, dim_k, dim_v, dtype, bc, bv)
            if used + _MACA_SMEM_SLACK <= _MACA_SMEM_CAP:
                score = (bc, bv)
                if best is None or score > (best[0], best[1]):
                    best = (bc, bv, used)
            bv //= 2
        bc //= 2

    if best is None:
        raise ValueError(
            f"MACA GLA o-kernel: no sub-chunk/V-tile fits under "
            f"{_MACA_SMEM_CAP} bytes for chunk={chunk_size} "
            f"dim_k={dim_k} dim_v={dim_v} dtype={dtype}"
        )
    return best[0], best[1]


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _gla_precompute_g_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    chunk_size: int,
    dtype: str,
) -> Callable:
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _fn(num_stages, threads=128):
        g_shape = [batch, seq_len, heads, dim_k]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]

        @T.prim_func
        def _main(
            g: T.Tensor(g_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                cs = i_c * chunk_size

                g_s = T.alloc_shared([chunk_size, dim_k], dtype)
                g_out_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)

                T.copy(g[i_b, cs : cs + chunk_size, i_h, :], g_s, disable_tma=True)

                for i_k in T.Parallel(dim_k):
                    g_out_s[0, i_k] = T.cast(g_s[0, i_k], accum_dtype)
                for i_t in T.Serial(1, chunk_size):
                    for i_k in T.Parallel(dim_k):
                        g_out_s[i_t, i_k] = g_out_s[i_t - 1, i_k] + T.cast(
                            g_s[i_t, i_k], accum_dtype
                        )

                T.copy(g_out_s, g_cumsum[i_b, cs : cs + chunk_size, i_h, :])

        return _main

    return _fn


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _gla_fwd_h_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    num_v_partitions: int = 1,
    num_k_partitions: int = 1,
) -> Callable:
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    dim_v_part = dim_v // num_v_partitions
    if dim_v_part < GEMM_MIN_N:
        raise ValueError(
            f"dim_v ({dim_v}) split across num_v_partitions "
            f"({num_v_partitions}) gives a {dim_v_part}-column T.gemm B "
            f"operand, below the minimum N extent ({GEMM_MIN_N})"
        )
    dim_k_part = dim_k // num_k_partitions
    num_kv = num_k_partitions * num_v_partitions

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _h_func(num_stages, threads=128):
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        init_state_shape = [batch, heads, dim_k, dim_v]
        h_out_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            initial_state: T.Tensor(init_state_shape, accum_dtype),
            h_out: T.Tensor(h_out_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_kv, threads=threads) as bx:
                i_b = bx // (heads * num_kv)
                i_h = (bx // num_kv) % heads
                i_kv = bx % num_kv
                i_kp = i_kv // num_v_partitions
                i_vp = i_kv % num_v_partitions
                k_offset = i_kp * dim_k_part
                v_offset = i_vp * dim_v_part

                h_s = T.alloc_shared([dim_k_part, dim_v_part], accum_dtype)
                k_s = T.alloc_shared([chunk_size, dim_k_part], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k_part], accum_dtype)

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_s[i_k, i_v] = initial_state[i_b, i_h, k_offset + i_k, v_offset + i_v]

                for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(
                        k[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        k_s,
                        disable_tma=True,
                    )
                    T.copy(
                        v[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            v_offset : v_offset + dim_v_part,
                        ],
                        v_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        g_cumsum_s,
                        disable_tma=True,
                    )

                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_out[i_b, i_c, i_h, k_offset + i_k, v_offset + i_v] = h_s[i_k, i_v]

                    g_last = T.alloc_fragment([dim_k_part], accum_dtype)
                    for i_k in T.Parallel(dim_k_part):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] * T.exp2(g_last[i_k] * LOG2_E)

                    k_adj_f = T.alloc_fragment([chunk_size, dim_k_part], dtype)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k_part):
                        k_adj_f[i_t, i_k] = T.cast(
                            T.cast(k_s[i_t, i_k], accum_dtype)
                            * T.exp2((g_last[i_k] - g_cumsum_s[i_t, i_k]) * LOG2_E),
                            dtype,
                        )

                    delta_h = T.alloc_fragment([dim_k_part, dim_v_part], accum_dtype)
                    T.fill(delta_h, 0.0)
                    T.gemm(k_adj_f, v_s, delta_h, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] + delta_h[i_k, i_v]

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_out[i_b, num_chunks, i_h, k_offset + i_k, v_offset + i_v] = h_s[i_k, i_v]

        return _main

    return _h_func


def _o_jit_pass_configs() -> dict:
    return {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _gla_fwd_o_kernel_maca_full(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
) -> Callable:
    """Full-chunk o kernel — identical layout to CUDA GLAFwdKernel."""
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size

    @tilelang.jit(out_idx=[-1], pass_configs=_o_jit_pass_configs())
    def _o_func(num_stages, threads=128):
        q_shape = [batch, seq_len, heads, dim_k]
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        h_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]
        o_shape = [batch, seq_len, heads, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            h: T.Tensor(h_shape, accum_dtype),
            o: T.Tensor(o_shape, dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                chunk_start = i_c * chunk_size

                h_cast_s = T.alloc_shared([dim_k, dim_v], dtype)
                q_s = T.alloc_shared([chunk_size, dim_k], dtype)
                k_s = T.alloc_shared([chunk_size, dim_k], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                q_gated_s = T.alloc_shared([chunk_size, dim_k], dtype)
                A_s = T.alloc_shared([chunk_size, chunk_size], dtype)

                T.copy(
                    q[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                    q_s,
                    disable_tma=True,
                )
                T.copy(
                    k[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                    k_s,
                    disable_tma=True,
                )
                T.copy(
                    v[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                    v_s,
                    disable_tma=True,
                )
                T.copy(
                    g_cumsum[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                    g_cumsum_s,
                    disable_tma=True,
                )

                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    h_cast_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, i_v], dtype)

                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    q_gated_s[i_t, i_k] = T.cast(
                        T.cast(q_s[i_t, i_k], accum_dtype) * T.exp2(g_cumsum_s[i_t, i_k] * LOG2_E),
                        dtype,
                    )

                A_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                T.fill(A_frag, 0.0)
                for i_k in T.Serial(dim_k):
                    for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                        A_frag[i_t, i_j] = A_frag[i_t, i_j] + (
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.cast(k_s[i_j, i_k], accum_dtype)
                            * T.exp2((g_cumsum_s[i_t, i_k] - g_cumsum_s[i_j, i_k]) * LOG2_E)
                        )
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    A_s[i_t, i_j] = T.cast(
                        T.if_then_else(i_j <= i_t, A_frag[i_t, i_j] * scale, 0.0), dtype
                    )

                acc = T.alloc_fragment([chunk_size, dim_v], accum_dtype)
                T.fill(acc, 0.0)
                T.gemm(q_gated_s, h_cast_s, acc, policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    acc[i_t, i_v] = acc[i_t, i_v] * scale
                T.gemm(A_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)

                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    o[i_b, chunk_start + i_t, i_h, i_v] = T.cast(acc[i_t, i_v], dtype)

        return _main

    return _o_func


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _gla_fwd_o_kernel_maca_tiled(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    sub_chunk_size: int,
    block_v: int,
) -> Callable:
    """Stream output rows and V bands to stay under MACA smem."""
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    bc = sub_chunk_size
    bt = chunk_size
    bv = resolve_block_v(dim_v, block_v)
    ns = bt // bc
    nv = dim_v // bv

    @tilelang.jit(out_idx=[-1], pass_configs=_o_jit_pass_configs())
    def _o_func(num_stages, threads=128):
        q_shape = [batch, seq_len, heads, dim_k]
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        h_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]
        o_shape = [batch, seq_len, heads, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            h: T.Tensor(h_shape, accum_dtype),
            o: T.Tensor(o_shape, dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                chunk_start = i_c * chunk_size

                k_s = T.alloc_shared([bt, dim_k], dtype)
                g_cumsum_s = T.alloc_shared([bt, dim_k], accum_dtype)
                q_s = T.alloc_shared([bc, dim_k], dtype)
                q_gated_s = T.alloc_shared([bc, dim_k], dtype)
                A_s = T.alloc_shared([bc, bt], dtype)
                h_cast_s = T.alloc_shared([dim_k, bv], dtype)
                v_s = T.alloc_shared([bt, bv], dtype)
                T.copy(
                    k[i_b, chunk_start : chunk_start + bt, i_h, :],
                    k_s,
                    disable_tma=True,
                )
                T.copy(
                    g_cumsum[i_b, chunk_start : chunk_start + bt, i_h, :],
                    g_cumsum_s,
                    disable_tma=True,
                )

                for s_i in T.Serial(ns):
                    q_start = chunk_start + s_i * bc

                    T.copy(
                        q[i_b, q_start : q_start + bc, i_h, :],
                        q_s,
                        disable_tma=True,
                    )

                    for i_t, i_k in T.Parallel(bc, dim_k):
                        row = s_i * bc + i_t
                        q_gated_s[i_t, i_k] = T.cast(
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.exp2(g_cumsum_s[row, i_k] * LOG2_E),
                            dtype,
                        )

                    A_frag = T.alloc_fragment([bc, bt], accum_dtype)
                    T.fill(A_frag, 0.0)
                    for i_k in T.Serial(dim_k):
                        for i_t, i_j in T.Parallel(bc, bt):
                            row = s_i * bc + i_t
                            A_frag[i_t, i_j] = A_frag[i_t, i_j] + (
                                T.cast(q_s[i_t, i_k], accum_dtype)
                                * T.cast(k_s[i_j, i_k], accum_dtype)
                                * T.exp2((g_cumsum_s[row, i_k] - g_cumsum_s[i_j, i_k]) * LOG2_E)
                            )
                    for i_t, i_j in T.Parallel(bc, bt):
                        global_i = q_start + i_t - chunk_start
                        A_s[i_t, i_j] = T.cast(
                            T.if_then_else(
                                i_j <= global_i,
                                A_frag[i_t, i_j] * scale,
                                T.float32(0.0),
                            ),
                            dtype,
                        )

                    for v_i in T.Serial(nv):
                        v_off = v_i * bv

                        for i_k, i_v in T.Parallel(dim_k, bv):
                            h_cast_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, v_off + i_v], dtype)
                        T.copy(
                            v[
                                i_b,
                                chunk_start : chunk_start + bt,
                                i_h,
                                v_off : v_off + bv,
                            ],
                            v_s,
                            disable_tma=True,
                        )

                        acc = T.alloc_fragment([bc, bv], accum_dtype)
                        T.fill(acc, 0.0)
                        T.gemm(q_gated_s, h_cast_s, acc, policy=T.GemmWarpPolicy.FullRow)
                        for i_t, i_v in T.Parallel(bc, bv):
                            acc[i_t, i_v] = acc[i_t, i_v] * scale
                        T.gemm(A_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)

                        for i_t, i_v in T.Parallel(bc, bv):
                            o[i_b, q_start + i_t, i_h, v_off + i_v] = T.cast(acc[i_t, i_v], dtype)

        return _main

    return _o_func


def _gla_fwd_o_kernel_maca(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    sub_chunk_size: int,
    block_v: int,
) -> Callable:
    if sub_chunk_size == chunk_size and block_v == dim_v:
        return _gla_fwd_o_kernel_maca_full(
            batch, seq_len, heads, dim_k, dim_v, chunk_size, scale, dtype
        )
    return _gla_fwd_o_kernel_maca_tiled(
        batch,
        seq_len,
        heads,
        dim_k,
        dim_v,
        chunk_size,
        scale,
        dtype,
        sub_chunk_size,
        block_v,
    )


class GLAFwdMACAKernel(Kernel):
    """GLA forward for MACA — CUDA-equivalent g/h, smem-safe o."""

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
        output_final_state: bool = False,
        dtype: torch.dtype = torch.float16,
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
        self.output_final_state = output_final_state
        self.dtype_name = str(dtype).split(".")[-1]
        self._o_sub_chunk, self._o_block_v = _pick_o_tiles(
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_name,
        )
        self.init_config(config)
        if tune:
            self.request_tune()
        if not tune:
            self._build_kernels(self.config)

    @property
    def default_config(self) -> dict:
        return {
            "num_stages": 3,
            "threads": 64,
            "num_v_partitions": 4,
            "num_k_partitions": 2,
            "sub_chunk_size": self._o_sub_chunk,
            "block_v": self._o_block_v,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        sub_chunk = self._o_sub_chunk
        block_v = self._o_block_v
        configs = []
        for ns in [1, 2, 3]:
            for t_par in [64, 128, 256]:
                for t_seq in [64, 128, 256]:
                    for nvp in [2, 4]:
                        for nkp in [1, 2]:
                            configs.append(
                                {
                                    "num_stages": ns,
                                    "threads_par": t_par,
                                    "threads_seq": t_seq,
                                    "num_v_partitions": nvp,
                                    "num_k_partitions": nkp,
                                    "sub_chunk_size": sub_chunk,
                                    "block_v": block_v,
                                }
                            )
        return configs

    def _build_kernels(self, config: dict) -> None:
        ns = config.get("num_stages", 2)
        thr_seq = config.get("threads_seq", config.get("threads", 256))
        thr_par = config.get("threads_par", config.get("threads", 256))
        num_vp = config.get("num_v_partitions", 4)
        num_kp = config.get("num_k_partitions", 1)
        sub_chunk = config.get("sub_chunk_size", self._o_sub_chunk)
        block_v = config.get("block_v", self._o_block_v)
        self._g_fn = _gla_precompute_g_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.chunk_size,
            self.dtype_name,
        )(ns, thr_par)
        self._h_fn = _gla_fwd_h_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.dtype_name,
            num_v_partitions=num_vp,
            num_k_partitions=num_kp,
        )(ns, thr_seq)
        self._o_fn = _gla_fwd_o_kernel_maca(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.scale,
            self.dtype_name,
            sub_chunk,
            block_v,
        )(ns, thr_par)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        if self.autotune_configs is None:
            return
        print(
            f"Start autotuning {self.__class__.__name__} ({len(self.autotune_configs)} configs)..."
        )

        B, T, H, K, V = (self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v)
        dtype_torch = getattr(torch, self.dtype_name)

        q = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype_torch).abs()

        best_lat = float("inf")
        best_cfg = None

        for cfg in self.autotune_configs:
            try:
                self._build_kernels(cfg)
                self.forward(q, k, v, g)
                torch.cuda.synchronize()

                lat = do_bench(
                    lambda: self.forward(q, k, v, g),
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
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, K, V = self.batch, self.heads, self.dim_k, self.dim_v
        dtype_torch = getattr(torch, self.dtype_name)

        if initial_state is None:
            init_state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        else:
            init_state = initial_state.to(torch.float32)

        g_cumsum = self._g_fn(g.to(dtype_torch))
        h_out = self._h_fn(
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            init_state,
        )
        o = self._o_fn(
            q.to(dtype_torch),
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            h_out,
        )

        self._h_out = h_out
        final_state = h_out[:, -1] if self.output_final_state else None
        return o, final_state
