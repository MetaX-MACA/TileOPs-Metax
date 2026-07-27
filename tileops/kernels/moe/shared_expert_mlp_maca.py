"""Shared Expert MLP Kernel — MACA smem-safe path.

Uses Pipelined + T.gemm (no Hopper TMA / WGMMA / mbarrier). Avoids
``T.alloc_barrier``, which lowers to ``tirx.ptx_init_barrier_thread_count``
and is unresolved on MACA codegen.

SiLU-mul reuses the CUDA-safe factory from ``shared_expert_mlp``.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from .shared_expert_mlp import _silu_mul_fused_kernel

__all__ = ["SharedExpertMLPMACAKernel"]

_DEFAULT_CONFIG = {
    "block_m": 64,
    "block_n": 128,
    "block_k": 64,
    "num_stages": 1,
    "threads": 256,
}


@functools.lru_cache(maxsize=32)
def _dense_gemm_nt_kernel_maca(m: int, n: int, k: int, dtype: str) -> Callable:
    """Dense GEMM C = A @ B^T via Pipelined + T.gemm (no mbarrier / TMA / WGMMA)."""
    accum_dtype = "float"

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
        @T.prim_func
        def _gemm_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
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
                    for i, j in T.Parallel(block_m, block_k):
                        a_shared[i, j] = T.if_then_else(
                            (m_start + i < m) & (k_start + j < k),
                            a[m_start + i, k_start + j],
                            T.cast(0, dtype),
                        )
                    for i, j in T.Parallel(block_n, block_k):
                        b_shared[i, j] = T.if_then_else(
                            (n_start + i < n) & (k_start + j < k),
                            b[n_start + i, k_start + j],
                            T.cast(0, dtype),
                        )
                    T.gemm(
                        a_shared,
                        b_shared,
                        c_local,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        c[m_start + i, n_start + j] = c_local[i, j]

        return _gemm_main

    return _gemm_func


class SharedExpertMLPMACAKernel(Kernel):
    """Shared expert MLP on MACA: gate_up GEMM + SiLU + down GEMM.

    Same forward signature as ``SharedExpertMLPKernel``, but GEMMs use a
    Pipelined + ``T.gemm`` path instead of Hopper ``GemmKernel``.

    Forward signature:
        hidden:    [T, H]
        w_gate_up: [2F, H]  — gate and up weights concatenated along dim 0
        w_down:    [H, F]
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        ffn_size: int,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype
        self.init_config(config, tune)

        self._gemm_gate_up = _dense_gemm_nt_kernel_maca(
            num_tokens, ffn_size * 2, hidden_size, self.dtype_str
        )
        self._gemm_down = _dense_gemm_nt_kernel_maca(
            num_tokens, hidden_size, ffn_size, self.dtype_str
        )

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(
        self,
        hidden: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        T_dim = self.num_tokens
        F = self.ffn_size
        cfg = self.config

        # [T, H] @ [2F, H]^T -> [T, 2F]
        gate_up_out = self._gemm_gate_up(
            cfg["block_m"], cfg["block_n"], cfg["block_k"],
            cfg["num_stages"], cfg["threads"],
        )(hidden, w_gate_up)

        silu_mul_fn = _silu_mul_fused_kernel(T_dim, F, self.dtype_str)(
            cfg["block_m"], cfg["block_n"], cfg["threads"])
        gate_up = silu_mul_fn(gate_up_out)

        # [T, F] @ [H, F]^T -> [T, H]
        return self._gemm_down(
            cfg["block_m"], cfg["block_n"], cfg["block_k"],
            cfg["num_stages"], cfg["threads"],
        )(gate_up, w_down)
