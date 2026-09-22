"""Dense GEMM kernel — MACA / non-Hopper path.

Uses Pipelined + ``T.gemm`` (no Hopper TMA / WGMMA / mbarrier). Avoids
``T.alloc_barrier``, which lowers to ``tirx.ptx_init_barrier_thread_count``
and is unresolved on MACA codegen.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.gemm.dense import swap_ab_grid_underfills
from tileops.kernels.kernel_base import Kernel

__all__ = ["GemmMACAKernel", "GemvMACAKernel", "SmallBatchGemmMACAKernel"]

_DEFAULT_CONFIG = {
    "block_m": 64,
    "block_n": 128,
    "block_k": 64,
    "num_stages": 1,
    "threads": 256,
}


@functools.lru_cache(maxsize=32)
def _gemm_kernel_maca(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
) -> Callable:
    """Dense GEMM ``C = op(A) @ op(B)`` via Pipelined + T.gemm.

    Supports all four ``(trans_a, trans_b)`` layouts. No mbarrier / TMA / WGMMA.

    Freevars must stay scalar (int/bool/str) so TileLang autotune can serialize
    the JIT factory closure — do not capture shape tuples here.
    """
    accum_dtype = "float32"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_func(
        block_m: int = 64,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 1,
        threads: int = 256,
    ) -> Callable:
        # Tile shapes are locals of this factory (not freevars of the JIT fn).
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)

        @T.prim_func
        def _gemm_main(
            a: T.Tensor((k, m) if trans_a else (m, k), dtype),  # type: ignore
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                a_shared = T.alloc_shared(a_tile, dtype)
                b_shared = T.alloc_shared(b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    }
                )

                m_start = by * block_m
                n_start = bx * block_n
                T.clear(c_local)

                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k
                    if trans_a:
                        for i, j in T.Parallel(block_k, block_m):
                            a_shared[i, j] = T.if_then_else(
                                (k_start + i < k) & (m_start + j < m),
                                a[k_start + i, m_start + j],
                                T.cast(0, dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_m, block_k):
                            a_shared[i, j] = T.if_then_else(
                                (m_start + i < m) & (k_start + j < k),
                                a[m_start + i, k_start + j],
                                T.cast(0, dtype),
                            )
                    if trans_b:
                        for i, j in T.Parallel(block_n, block_k):
                            b_shared[i, j] = T.if_then_else(
                                (n_start + i < n) & (k_start + j < k),
                                b[n_start + i, k_start + j],
                                T.cast(0, dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_k, block_n):
                            b_shared[i, j] = T.if_then_else(
                                (k_start + i < k) & (n_start + j < n),
                                b[k_start + i, n_start + j],
                                T.cast(0, dtype),
                            )
                    T.gemm(
                        a_shared,
                        b_shared,
                        c_local,
                        transpose_A=trans_a,
                        transpose_B=trans_b,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_m, block_n):
                    if (m_start + i < m) & (n_start + j < n):
                        c[m_start + i, n_start + j] = c_local[i, j]

        return _gemm_main

    return _gemm_func


class GemmMACAKernel(Kernel):
    """Dense GEMM for MACA / Ampere-class devices (Pipelined + T.gemm).

    Same forward signature as ``GemmKernel``, but avoids Hopper TMA / WGMMA /
    mbarrier so MACA codegen can resolve the kernel.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    # Mirror GemmKernel: general fallback behind specialised roles (e.g. Gemv).
    general = True

    @classmethod
    def entry_for(cls, call):
        """Construct the MACA kernel from the dispatch call record."""
        identity = (
            call.m,
            call.n,
            call.k,
            call.dtype,
            call.tune,
            call.trans_a,
            call.trans_b,
            call.device.index if call.device is not None else None,
        )
        return identity, lambda: cls(
            call.m,
            call.n,
            call.k,
            call.dtype,
            tune=call.tune,
            trans_a=call.trans_a,
            trans_b=call.trans_b,
        )

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        trans_a: bool = False,
        trans_b: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.kernel = _gemm_kernel_maca(m, n, k, trans_a, trans_b, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        config = dict(_DEFAULT_CONFIG)

        if self.m <= 32:
            config["block_n"] = 32
            config["num_stages"] = 2
        elif self.m <= 64:
            config["block_n"] = 64
        elif self.m <= 128:
            config["block_n"] = 32
            if not self.trans_a and self.trans_b and self.k > self.n:
                config["block_k"] = 128

        return config

    @property
    def autotune_configs(self) -> list[dict]:
        # Keep a single safe config: sweeping block/stages on large MNK (and
        # compiling each candidate) OOMs CI hosts / MACA devices.
        return [self.default_config]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        return self.kernel(
            cfg["block_m"],
            cfg["block_n"],
            cfg["block_k"],
            cfg["num_stages"],
            cfg["threads"],
        )(a, b)


@functools.lru_cache(maxsize=32)
def _small_batch_gemm_kernel_maca(
    m: int,
    n: int,
    k: int,
    dtype: str = "float16",
) -> Callable:
    """Bandwidth-oriented MACA kernel for one- and two-row NT GEMMs.

    A 64-thread reduction group computes one output column. Reduction is
    performed explicitly in shared memory because the SM90 thread-allreduce
    path is not valid on MACA.
    """
    accum_dtype = "float32"
    vector_width = 8

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _small_batch_func(
        block_n: int = 4,
        reduce_threads: int = 64,
        num_stages: int = 2,
    ) -> Callable:
        block_k = reduce_threads * vector_width
        is_aligned = n % block_n == 0 and k % block_k == 0

        @T.prim_func
        def _small_batch_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n),
                threads=(reduce_threads, block_n),
            ) as bn:
                tk = T.get_thread_binding(0)
                tn = T.get_thread_binding(1)

                b_shared = T.alloc_shared((block_n, block_k), dtype)
                reduce_shared = T.alloc_shared((block_n, reduce_threads), accum_dtype)
                a_local = T.alloc_local((m, vector_width), dtype)
                c_accum = T.alloc_local((m,), accum_dtype)
                c_reduced = T.alloc_local((1,), accum_dtype)

                T.clear(c_accum)

                for bk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    if is_aligned:
                        T.copy(
                            b[bn * block_n, bk * block_k],
                            b_shared,
                            disable_tma=True,
                        )
                    else:
                        for ni, ki in T.Parallel(block_n, block_k):
                            b_shared[ni, ki] = T.if_then_else(
                                (bn * block_n + ni < n) & (bk * block_k + ki < k),
                                b[bn * block_n + ni, bk * block_k + ki],
                                T.cast(0, dtype),
                            )

                    for mi in T.serial(m):
                        for ki in T.vectorized(vector_width):
                            offset = bk * block_k + tk * vector_width + ki
                            a_local[mi, ki] = T.if_then_else(
                                offset < k,
                                a[mi, offset],
                                T.cast(0, dtype),
                            )

                    for mi in T.serial(m):
                        for ki in T.serial(vector_width):
                            c_accum[mi] += a_local[mi, ki].astype(accum_dtype) * b_shared[
                                tn, tk * vector_width + ki
                            ].astype(accum_dtype)

                for mi in T.serial(m):
                    reduce_shared[tn, tk] = c_accum[mi]
                    T.sync_threads()

                    if tk == 0:
                        c_reduced[0] = T.cast(0, accum_dtype)
                        for rk in T.serial(reduce_threads):
                            c_reduced[0] += reduce_shared[tn, rk]

                        if bn * block_n + tn < n:
                            c[mi, bn * block_n + tn] = c_reduced[0]

                    T.sync_threads()

        return _small_batch_main

    return _small_batch_func


class GemvMACAKernel(Kernel):
    """MACA specialization for one-row NT GEMMs."""

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call) -> bool:
        return call.gemv_mode is not None

    def __init__(
        self,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self.dtype = dtype
        self.kernel = _small_batch_gemm_kernel_maca(1, n, k, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_n": 4, "reduce_threads": 64, "num_stages": 2}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        return self.kernel(
            cfg["block_n"],
            cfg["reduce_threads"],
            cfg["num_stages"],
        )(a.reshape(1, -1).contiguous(), b).reshape(self.n)


class SmallBatchGemmMACAKernel(Kernel):
    """MACA specialization for two-row NT GEMMs."""

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call) -> bool:
        if call.trans_a or not call.trans_b or call.m != 2:
            return False
        return swap_ab_grid_underfills(call.n, call.sm_count)

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.kernel = _small_batch_gemm_kernel_maca(m, n, k, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_n": 4, "reduce_threads": 64, "num_stages": 2}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        return self.kernel(
            cfg["block_n"],
            cfg["reduce_threads"],
            cfg["num_stages"],
        )(a, b)
