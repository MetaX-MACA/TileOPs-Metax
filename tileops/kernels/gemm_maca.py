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

from tileops.kernels.kernel_base import Kernel

__all__ = ["GemmMACAKernel"]

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
            with T.Kernel(
                    T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
                a_shared = T.alloc_shared(a_tile, dtype)
                b_shared = T.alloc_shared(b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout({
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                })

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
        return dict(_DEFAULT_CONFIG)

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
