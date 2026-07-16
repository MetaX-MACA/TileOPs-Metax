# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import functools
import itertools
import os
import weakref
from contextlib import contextmanager
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.trace import trace
from tileops.utils import get_sm_version, is_metax_c500, str2dtype

__all__ = [
    "GemmFp8BlockScaledKernel",
    "GemmFp8EpilogueKernel",
    "GemmKernel",
    "GemvKernel",
]


class GemmFp8EpilogueKernel(Kernel):
    """Simple TileLang FP8 GEMM for per-tensor scales."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.kernel = _gemm_fp8_kernel(
            m, n, k, self.dtype_str, self.out_dtype_str, block_scaled=False
        )
        self.init_config(config, tune)

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 3,
            "threads": 256,
        }

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                f"GemmFp8EpilogueKernel only supports torch.float8_e4m3fn, got {self.dtype}"
            )
        compiled = _gemm_fp8_kernel(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.out_dtype_str,
            block_scaled=False,
            has_bias=bias is not None,
        )(**self.config)
        if bias is not None:
            return compiled(a, b, scale_a, scale_b, bias)
        return compiled(a, b, scale_a, scale_b)


class GemmFp8BlockScaledKernel(Kernel):
    """Simple TileLang FP8 GEMM for K-block scales."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.kernel = _gemm_fp8_kernel(
            m, n, k, self.dtype_str, self.out_dtype_str, block_scaled=True
        )
        self.init_config(config, tune)

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 3,
            "threads": 256,
        }

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                f"GemmFp8BlockScaledKernel only supports torch.float8_e4m3fn, got {self.dtype}"
            )
        compiled = _gemm_fp8_kernel(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.out_dtype_str,
            block_scaled=True,
            has_bias=bias is not None,
        )(**self.config)
        if bias is not None:
            return compiled(a, b, scale_a, scale_b, bias)
        return compiled(a, b, scale_a, scale_b)


@functools.lru_cache(maxsize=32)
def _gemm_fp8_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    block_scaled: bool,
    has_bias: bool = False,
) -> Callable:
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_fp8_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 128,
        num_stages: int = 3,
        threads: int = 256,
    ) -> Callable:
        if block_scaled:
            if block_k > 128:
                raise ValueError(f"block_k must be <= 128 for block128 scaling, got {block_k}")
            if 128 % block_k != 0:
                raise ValueError(f"128 must be divisible by block_k, got {block_k}")
        scale_k = (k + 127) // 128 if block_scaled else 1
        scale_a_shape = (m, scale_k) if block_scaled else (1, 1)
        scale_b_shape = (n, scale_k) if block_scaled else (1, 1)

        @T.prim_func
        def _gemm_fp8_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                if block_scaled:
                    partial = T.alloc_fragment((block_m, block_n), accum_dtype)

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
                    if block_scaled:
                        scale_idx = kk * block_k // 128
                        T.clear(partial)
                        T.gemm(
                            a_shared,
                            b_shared,
                            partial,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c_local[i, j] += (
                                    partial[i, j]
                                    * scale_a[m_start + i, scale_idx]
                                    * scale_b[n_start + j, scale_idx]
                                )
                    else:
                        T.gemm(
                            a_shared,
                            b_shared,
                            c_local,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        if block_scaled:
                            c[m_start + i, n_start + j] = c_local[i, j]
                        else:
                            c[m_start + i, n_start + j] = (
                                c_local[i, j] * scale_a[0, 0] * scale_b[0, 0]
                            )

        @T.prim_func
        def _gemm_fp8_bias_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            bias: T.Tensor((n,), out_dtype),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                if block_scaled:
                    partial = T.alloc_fragment((block_m, block_n), accum_dtype)

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
                    if block_scaled:
                        scale_idx = kk * block_k // 128
                        T.clear(partial)
                        T.gemm(
                            a_shared,
                            b_shared,
                            partial,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c_local[i, j] += (
                                    partial[i, j]
                                    * scale_a[m_start + i, scale_idx]
                                    * scale_b[n_start + j, scale_idx]
                                )
                    else:
                        T.gemm(
                            a_shared,
                            b_shared,
                            c_local,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        if block_scaled:
                            c[m_start + i, n_start + j] = c_local[i, j] + bias[n_start + j]
                        else:
                            c[m_start + i, n_start + j] = (
                                c_local[i, j] * scale_a[0, 0] * scale_b[0, 0] + bias[n_start + j]
                            )

        return _gemm_fp8_bias_main if has_bias else _gemm_fp8_main

    return _gemm_fp8_func


def _gemm_compile_flags(use_maca: Optional[bool] = None) -> list[str]:
    if use_maca is None:
        use_maca = is_metax_c500()
    flags = ["-O3", "-DENABLE_BF16"]
    if use_maca:
        flags.extend(
            [
                "-mllvm -metaxgpu-direct-address=disshared",
                "-mllvm -metaxgpu-force-global-saddr=1",
                "-gcc-version 11",
            ]
        )
    return flags


def _make_maca_gemm_ab_layout(buf, kfactor: int):
    layout_fn = getattr(tilelang.layout, "make_maca_gemm_ab_layout", None)
    if layout_fn is None:
        return tilelang.layout.make_swizzled_layout(buf)
    return layout_fn(buf, kfactor)


def _make_contiguous_output_layout(buf):
    layout_fn = getattr(tilelang.layout, "make_linear_layout", None)
    if layout_fn is None:
        return tilelang.layout.make_swizzled_layout(buf)
    return layout_fn(buf)


def _get_maca_bsm_split_k() -> int:
    value = os.environ.get("TILEOPS_GEMM_SPLIT_K", "1").strip()
    if not value:
        return 1
    return max(1, int(value))


def _get_maca_bsm_split_k_override() -> Optional[int]:
    value = os.environ.get("TILEOPS_GEMM_SPLIT_K")
    if value is None or not value.strip():
        return None
    return max(1, int(value))


def _get_bool_env_override(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip().lower()
    return value in {"1", "true", "yes", "on"}


def _get_maca_bsm_packed_b_tile() -> bool:
    value = _get_bool_env_override("TILEOPS_GEMM_PACKED_B_TILE")
    return bool(value)


def _get_maca_bsm_packed_b_async_pipeline() -> bool:
    value = _get_bool_env_override("TILEOPS_GEMM_PACKED_B_ASYNC_PIPELINE")
    return bool(value)


def _should_auto_use_maca_compiler_splitk_packed(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    trans_a: bool,
    trans_b: bool,
    tune: bool,
) -> bool:
    return (
        is_metax_c500()
        and dtype == torch.float16
        and not trans_a
        and not trans_b
        and not tune
        and m % 128 == 0
        and n % 128 == 0
        and k % 128 == 0
        and k >= 131072
    )


@contextmanager
def _temporary_env(updates: dict[str, str]):
    old_values = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@functools.lru_cache(maxsize=32)
def _gemm_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    traced: bool = False,
) -> Callable:
    """Hand-written warp-specialized GEMM ``C = op(A) @ op(B)`` for Hopper (SM90).

    One producer warpgroup (128 threads) issues TMA loads into a double-buffered
    SMEM ring; one consumer warpgroup (128 threads) runs the WGMMA and accumulates
    over K. All four layouts are covered by ``trans_a`` / ``trans_b`` (forwarded to
    the WGMMA transpose flags): ``A`` is ``[M,K]`` (or ``[K,M]`` transposed), ``B``
    is ``[K,N]`` (or ``[N,K]`` transposed), ``C`` is ``[M,N]``. fp16 / bf16 inputs,
    fp32 accumulation. The auto warp-specialization pass is disabled so it does not
    fire on top of this manual layout.

    TMA constraint: each input's innermost (contiguous) dimension must be a
    multiple of 8 elements (16-byte alignment). That dim is ``K`` for a
    non-transposed ``A`` and a transposed ``B``, ``M`` for a transposed ``A``, and
    ``N`` for a non-transposed ``B``. So NT (``A@Bᵀ``) only requires ``K % 8 == 0``;
    other layouts additionally require the relevant ``M`` / ``N`` to be aligned.

    Args:
        m: Rows of ``op(A)`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Whether ``A`` is stored transposed (``[K, M]``).
        trans_b: Whether ``B`` is stored transposed (``[N, K]``).
        dtype: Activation / weight dtype string (``"float16"`` or ``"bfloat16"``).
        traced: Build with in-kernel timeline markers materialized (``True``) or
            stripped to zero cost (``False``). **Part of the cache key**: traced
            and untraced builds are distinct cached kernels, so flipping the
            process trace switch never returns a stale variant. Callers pass
            ``trace.enabled`` explicitly rather than letting the build read the
            global switch.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n,
        block_k, num_stages)`` returns the compiled ``prim_func``. When ``traced``
        it materializes the markers and appends a trailing ``slots`` output (so
        ``out_idx`` returns ``(C, slots)``); otherwise ``C`` is the lone output.
    """
    accum_dtype = "float"
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    @tilelang.jit(
        out_idx=trace.out_idx(1, traced),
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_func(
        block_m: int = 128, block_n: int = 128, block_k: int = 64, num_stages: int = 3
    ) -> Callable:
        # Manual 2-warpgroup WS: 1 producer WG (128 threads) issues TMA, 1
        # consumer WG (128 threads) runs WGMMA. Barrier arrive_counts (128) are
        # bound to this layout, so threads is fixed at 256.
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        # SMEM tile shapes follow the storage layout; the WGMMA transpose flags
        # reconcile them with the logical (M,K) x (K,N) contraction.
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)

        @T.prim_func
        def _gemm_main(
            a: T.Tensor(a_shape, dtype),  # type: ignore
            b: T.Tensor(b_shape, dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                # Multi-stage ring of A/B SMEM buffers. Indexed by stage = gi %
                # num_stages; the phase bit flips every num_stages iterations.
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout(
                    {
                        a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                    }
                )

                # Producer→consumer (buffer full) and consumer→producer (buffer
                # empty) barriers, one per ring slot. Each is arrived by exactly
                # one warpgroup (128 threads). Allocated as length-num_stages
                # barrier arrays and indexed by the static slot id.
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)

                # Monotonic per-warpgroup iteration counters; stage = gi %
                # num_stages, phase = (gi // num_stages) % 2.
                gi_prod = T.alloc_var("int32", init=0)
                gi_cons = T.alloc_var("int32", init=0)

                m_start = by * block_m
                n_start = bx * block_n

                tx = T.get_thread_binding()

                if tx < 128:
                    # ── Producer warpgroup: issue TMA loads of A and B tiles. ──
                    # Intern the "producer" group first so it gets gid 0.
                    T.dec_max_nreg(24)
                    with trace.group("producer", lead=0):
                        for ki in T.serial(k_iters):
                            stage = gi_prod % num_stages
                            phase = (gi_prod // num_stages) % 2
                            k_start = ki * block_k
                            # Unroll the ring-slot dispatch at trace time: each
                            # slot gets a static SMEM/barrier index under a
                            # dynamic `stage == s` guard.
                            for s in range(num_stages):
                                if stage == s:
                                    # Wait for this slot to be drained before
                                    # reuse. The consumer leaves the slot in
                                    # empty-phase (phase ^ 1) for the round the
                                    # producer is about to refill; rounds
                                    # 0..num_stages-1 see the init-0 state (phase
                                    # ^ 1 == 1) which is already satisfied by the
                                    # barrier's initial parity.
                                    T.barrier_wait(ab_empty[s], phase ^ 1)
                                    with trace.range("tma", lane="tma"):
                                        if trans_a:
                                            T.tma_copy(
                                                a[
                                                    k_start : k_start + block_k,
                                                    m_start : m_start + block_m,
                                                ],
                                                a_smem[s, :, :],
                                                barrier=ab_full[s],
                                            )
                                        else:
                                            T.tma_copy(
                                                a[
                                                    m_start : m_start + block_m,
                                                    k_start : k_start + block_k,
                                                ],
                                                a_smem[s, :, :],
                                                barrier=ab_full[s],
                                            )
                                        if trans_b:
                                            T.tma_copy(
                                                b[
                                                    n_start : n_start + block_n,
                                                    k_start : k_start + block_k,
                                                ],
                                                b_smem[s, :, :],
                                                barrier=ab_full[s],
                                            )
                                        else:
                                            T.tma_copy(
                                                b[
                                                    k_start : k_start + block_k,
                                                    n_start : n_start + block_n,
                                                ],
                                                b_smem[s, :, :],
                                                barrier=ab_full[s],
                                            )
                                    with trace.range("arrive", lane="barrier"):
                                        T.barrier_arrive(ab_full[s])
                            gi_prod = gi_prod + 1
                else:
                    # ── Consumer warpgroup: run WGMMA, accumulate over K. ──
                    T.inc_max_nreg(240)
                    T.clear(c_local)
                    with trace.group("consumer", lead=128):
                        for ki in T.serial(k_iters):
                            stage = gi_cons % num_stages
                            phase = (gi_cons // num_stages) % 2
                            for s in range(num_stages):
                                if stage == s:
                                    with trace.range("wait", lane="barrier"):
                                        T.barrier_wait(ab_full[s], phase)
                                    with trace.range("mma", lane="wgmma"):
                                        T.wgmma_gemm(
                                            a_smem[s, :, :],
                                            b_smem[s, :, :],
                                            c_local,
                                            transpose_A=trans_a,
                                            transpose_B=trans_b,
                                            policy=T.GemmWarpPolicy.FullRow,
                                            clear_accum=(ki == 0),
                                        )
                                        T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(c_local, num_regs=64)
                                    T.barrier_arrive(ab_empty[s])
                            gi_cons = gi_cons + 1

                        # Epilogue: guard the M/N tail so partial tiles don't
                        # write out of bounds (m / n need not be multiples of the
                        # block sizes; K tails are zero-filled by TMA).
                        with trace.range("epilogue"):
                            for i, j in T.Parallel(block_m, block_n):
                                if m_start + i < m and n_start + j < n:
                                    c[m_start + i, n_start + j] = c_local[i, j]

                # Build-time flow declaration: producer "arrive" → consumer
                # "wait" (fixed per-iter pairing).
                trace.dag("arrive", "wait")

        # Materialize markers + append ``slots`` when traced; no-op them (identical
        # CUDA to an un-instrumented build) otherwise. Pairs with ``out_idx`` above.
        return trace.finalize(_gemm_main, traced=traced, max_events=1024)

    return _gemm_func


@functools.lru_cache(maxsize=32)
def _gemm_kernel_bsm(m: int, n: int, k: int, dtype: str = "float16") -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=_gemm_compile_flags(use_maca=True))
    def _gemm_func(
        block_m: int,
        block_n: int,
        block_k: int,
        threads: int,
        num_stages: int,
        enable_rasterization: bool,
    ) -> Callable:

        a_shape = (m, k)
        b_shape = (n, k)
        a_shared_shape = (block_m, block_k)
        b_shared_shape = (block_n, block_k)

        @T.prim_func
        def _gemm_main(
            a: T.Tensor(a_shape, dtype),  # type: ignore
            b: T.Tensor(b_shape, dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                a_operand = T.alloc_shared(a_shared_shape, dtype)
                b_shared = T.alloc_shared(b_shared_shape, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                c_shared = T.alloc_shared((block_m, block_n), dtype)

                T.annotate_layout(
                    {
                        a_operand: _make_maca_gemm_ab_layout(a_operand, 2),
                        c_shared: _make_contiguous_output_layout(c_shared),
                    }
                )
                T.use_swizzle(10, enable=enable_rasterization)

                T.clear(c_local)
                for _k in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(a[by * block_m, _k * block_k], a_operand)
                    T.copy(b[bx * block_n, _k * block_k], b_shared)
                    T.gemm(a_operand, b_shared, c_local, False, True)

                T.copy(c_local, c_shared)
                T.copy(c_shared, c[by * block_m, bx * block_n])

        return _gemm_main

    return _gemm_func


@functools.lru_cache(maxsize=32)
def _gemm_kernel_bsm_packed_b_tile(m: int, n: int, k: int, dtype: str = "float16") -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=_gemm_compile_flags(use_maca=True))
    def _gemm_func(
        block_m: int,
        block_n: int,
        block_k: int,
        threads: int,
        num_stages: int,
        enable_rasterization: bool,
    ) -> Callable:

        a_shape = (m, k)
        b_shape = (n // block_n, k // block_k, block_n, block_k)
        a_shared_shape = (block_m, block_k)
        b_shared_shape = (block_n, block_k)

        @T.prim_func
        def _gemm_main(
            a: T.Tensor(a_shape, dtype),  # type: ignore
            b: T.Tensor(b_shape, dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                a_operand = T.alloc_shared(a_shared_shape, dtype)
                b_shared = T.alloc_shared(b_shared_shape, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                c_shared = T.alloc_shared((block_m, block_n), dtype)

                T.annotate_layout(
                    {
                        a_operand: _make_maca_gemm_ab_layout(a_operand, 2),
                        c_shared: _make_contiguous_output_layout(c_shared),
                    }
                )
                T.use_swizzle(10, enable=enable_rasterization)

                T.clear(c_local)
                for _k in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(a[by * block_m, _k * block_k], a_operand)
                    T.copy(b[bx, _k, 0, 0], b_shared)
                    T.gemm(a_operand, b_shared, c_local, False, True)

                T.copy(c_local, c_shared)
                T.copy(c_shared, c[by * block_m, bx * block_n])

        return _gemm_main

    return _gemm_func


@functools.lru_cache(maxsize=32)
def _gemm_kernel_bsm_splitk(
    m: int, n: int, k: int, split_k: int, dtype: str = "float16", partial_dtype: str = "float32"
) -> Callable:
    accum_dtype = "float"
    k_chunk = k // split_k

    @tilelang.jit(out_idx=[2], compile_flags=_gemm_compile_flags(use_maca=True))
    def _gemm_func(
        block_m: int,
        block_n: int,
        block_k: int,
        threads: int,
        num_stages: int,
        enable_rasterization: bool,
    ) -> Callable:

        split_block_k = block_k // split_k
        a_shape = (m, k)
        b_shape = (n, k)
        partial_c_shape = (split_k, m, n)
        a_shared_shape = (block_m, split_block_k)
        b_shared_shape = (block_n, split_block_k)

        @T.prim_func
        def _gemm_main(
            a: T.Tensor(a_shape, dtype),  # type: ignore
            b: T.Tensor(b_shape, dtype),  # type: ignore
            partial_c: T.Tensor(partial_c_shape, partial_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=threads
            ) as (bx, by, bz):
                a_operand = T.alloc_shared(a_shared_shape, dtype)
                b_shared = T.alloc_shared(b_shared_shape, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout(
                    {
                        a_operand: _make_maca_gemm_ab_layout(a_operand, 2),
                    }
                )
                T.use_swizzle(10, enable=enable_rasterization)

                T.clear(c_local)
                for _k in T.Pipelined(T.ceildiv(k_chunk, split_block_k), num_stages=num_stages):
                    k_offset = bz * k_chunk + _k * split_block_k
                    T.copy(a[by * block_m, k_offset], a_operand)
                    T.copy(b[bx * block_n, k_offset], b_shared)
                    T.gemm(a_operand, b_shared, c_local, False, True)

                T.copy(c_local, partial_c[bz, by * block_m, bx * block_n])

        return _gemm_main

    return _gemm_func


@functools.lru_cache(maxsize=32)
def _gemm_kernel_bsm_splitk_packed_b_tile(
    m: int, n: int, k: int, split_k: int, dtype: str = "float16", partial_dtype: str = "float32"
) -> Callable:
    accum_dtype = "float"
    k_chunk = k // split_k

    @tilelang.jit(out_idx=[2], compile_flags=_gemm_compile_flags(use_maca=True))
    def _gemm_func(
        block_m: int,
        block_n: int,
        block_k: int,
        threads: int,
        num_stages: int,
        enable_rasterization: bool,
    ) -> Callable:

        split_block_k = block_k // split_k
        k_tiles_per_split = k_chunk // split_block_k
        a_shape = (m, k)
        b_shape = (n // block_n, k // split_block_k, block_n, split_block_k)
        partial_c_shape = (split_k, m, n)
        a_shared_shape = (block_m, split_block_k)
        b_shared_shape = (block_n, split_block_k)

        @T.prim_func
        def _gemm_main(
            a: T.Tensor(a_shape, dtype),  # type: ignore
            b: T.Tensor(b_shape, dtype),  # type: ignore
            partial_c: T.Tensor(partial_c_shape, partial_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=threads
            ) as (bx, by, bz):
                a_operand = T.alloc_shared(a_shared_shape, dtype)
                b_shared = T.alloc_shared(b_shared_shape, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout(
                    {
                        a_operand: _make_maca_gemm_ab_layout(a_operand, 2),
                    }
                )
                T.use_swizzle(10, enable=enable_rasterization)

                T.clear(c_local)
                for _k in T.Pipelined(T.ceildiv(k_chunk, split_block_k), num_stages=num_stages):
                    k_offset = bz * k_chunk + _k * split_block_k
                    k_tile = bz * k_tiles_per_split + _k
                    T.copy(a[by * block_m, k_offset], a_operand)
                    T.copy(b[bx, k_tile, 0, 0], b_shared)
                    gemm_annotations = {
                        "maca_wsm_a_source_ptr": T.address_of(a[by * block_m, k_offset]),
                        "maca_wsm_b_source_ptr": T.address_of(b[bx, k_tile, 0, 0]),
                        "maca_wsm_a_stride": k,
                    }
                    T.gemm(
                        a_operand,
                        b_shared,
                        c_local,
                        False,
                        True,
                        annotations=gemm_annotations,
                    )

                T.copy(c_local, partial_c[bz, by * block_m, bx * block_n])

        return _gemm_main

    return _gemm_func


@functools.lru_cache(maxsize=32)
def _gemm_kernel_bsm_splitk_packed_b_tile_async(
    m: int, n: int, k: int, split_k: int, dtype: str = "float16", partial_dtype: str = "float32"
) -> Callable:
    accum_dtype = "float"
    k_chunk = k // split_k
    stage_count = 2

    @tilelang.jit(out_idx=[2], compile_flags=_gemm_compile_flags(use_maca=True))
    def _gemm_func(
        block_m: int,
        block_n: int,
        block_k: int,
        threads: int,
        num_stages: int,
        enable_rasterization: bool,
    ) -> Callable:

        split_block_k = block_k // split_k
        a_shape = (m, k)
        b_shape = (n // block_n, k // split_block_k, block_n, split_block_k)
        partial_c_shape = (split_k, m, n)
        a_shared_shape = (stage_count, block_m, split_block_k)
        b_shared_shape = (stage_count, block_n, split_block_k)
        num_k_tiles = (k_chunk + split_block_k - 1) // split_block_k

        @T.prim_func
        def _gemm_main(
            a: T.Tensor(a_shape, dtype),  # type: ignore
            b: T.Tensor(b_shape, dtype),  # type: ignore
            partial_c: T.Tensor(partial_c_shape, partial_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=threads
            ) as (bx, by, bz):
                a_operand = T.alloc_shared(a_shared_shape, dtype)
                b_shared = T.alloc_shared(b_shared_shape, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                bar = T.alloc_maca_barrier(stage_count)

                T.annotate_layout(
                    {
                        a_operand: _make_maca_gemm_ab_layout(a_operand, 2),
                    }
                )
                T.use_swizzle(10, enable=enable_rasterization)

                first_k_offset = bz * k_chunk
                first_k_tile = first_k_offset // split_block_k
                T.maca_async_copy(
                    a[
                        by * block_m : (by + 1) * block_m,
                        first_k_offset : first_k_offset + split_block_k,
                    ],
                    a_operand[0, :, :],
                    barrier=bar[0],
                )
                T.maca_async_copy(
                    b[bx, first_k_tile, :, :],
                    b_shared[0, :, :],
                    barrier=bar[0],
                )
                T.clear(c_local)

                for _k in T.serial(num_k_tiles):
                    stage = _k % stage_count
                    next_k = _k + 1
                    if next_k < num_k_tiles:
                        next_stage = next_k % stage_count
                        next_k_offset = bz * k_chunk + next_k * split_block_k
                        next_k_tile = next_k_offset // split_block_k
                        T.maca_async_copy(
                            a[
                                by * block_m : (by + 1) * block_m,
                                next_k_offset : next_k_offset + split_block_k,
                            ],
                            a_operand[next_stage, :, :],
                            barrier=bar[next_stage],
                        )
                        T.maca_async_copy(
                            b[bx, next_k_tile, :, :],
                            b_shared[next_stage, :, :],
                            barrier=bar[next_stage],
                        )
                    T.gemm(
                        a_operand[stage, :, :],
                        b_shared[stage, :, :],
                        c_local,
                        False,
                        True,
                        mbar=bar[stage],
                    )

                T.copy(c_local, partial_c[bz, by * block_m, bx * block_n])

        return _gemm_main

    return _gemm_func


@functools.lru_cache(maxsize=32)
def _gemm_splitk_reduce_kernel(
    m: int, n: int, split_k: int, dtype: str = "float16", partial_dtype: str = "float32"
) -> Callable:
    threads = 256
    items_per_thread = 16
    block_size = threads * items_per_thread
    total = m * n

    @tilelang.jit(out_idx=[1], compile_flags=_gemm_compile_flags(use_maca=True))
    def _reduce_func() -> Callable:

        if split_k == 2:

            @T.prim_func
            def _reduce_main(
                partial_c: T.Tensor((split_k, m, n), partial_dtype),  # type: ignore
                c: T.Tensor((m, n), dtype),  # type: ignore
            ) -> None:
                with T.Kernel(T.ceildiv(total, block_size), threads=threads) as bx:
                    for tx, item in T.Parallel(threads, items_per_thread):
                        flat_idx = (bx * threads + tx) * items_per_thread + item
                        if flat_idx < total:
                            row = flat_idx // n
                            col = flat_idx - row * n
                            c[row, col] = T.cast(
                                T.cast(partial_c[0, row, col], "float32")
                                + T.cast(partial_c[1, row, col], "float32"),
                                dtype,
                            )

            return _reduce_main

        @T.prim_func
        def _reduce_main(
            partial_c: T.Tensor((split_k, m, n), partial_dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(total, block_size), threads=threads) as bx:
                for tx, item in T.Parallel(threads, items_per_thread):
                    flat_idx = (bx * threads + tx) * items_per_thread + item
                    if flat_idx < total:
                        row = flat_idx // n
                        col = flat_idx - row * n
                        acc = T.alloc_local((1,), "float32")
                        acc[0] = 0.0
                        for sk in T.serial(split_k):
                            acc[0] += T.cast(partial_c[sk, row, col], "float32")
                        c[row, col] = T.cast(acc[0], dtype)

        return _reduce_main

    return _reduce_func


@functools.lru_cache(maxsize=32)
def _gemm_kernel_col_major(m: int, n: int, k: int, dtype: str = "float16") -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=_gemm_compile_flags())
    def _gemm_func(
        block_m: int,
        block_n: int,
        block_k: int,
        threads: int,
        num_stages: int,
        enable_rasterization: bool,
    ) -> Callable:

        a_shape = (m, k)
        b_shape = (n, k)
        a_shared_shape = (block_m, block_k)
        b_shared_shape = (block_n, block_k)

        @T.prim_func
        def _gemm_main(
            a: T.Tensor(a_shape, dtype),  # type: ignore
            b: T.Tensor(b_shape, dtype),  # type: ignore
            c_col_major: T.Tensor((n, m), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), threads=threads) as (
                bx,
                by,
            ):
                a_shared = T.alloc_shared(a_shared_shape, dtype)
                b_shared = T.alloc_shared(b_shared_shape, dtype)
                c_local = T.alloc_fragment((block_n, block_m), accum_dtype)
                c_shared = T.alloc_shared((block_n, block_m), dtype)

                T.annotate_layout(
                    {
                        c_shared: tilelang.layout.make_swizzled_layout(c_shared),
                    }
                )
                T.use_swizzle(10, enable=enable_rasterization)

                T.clear(c_local)

                for _k in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(a[bx * block_m, _k * block_k], a_shared)
                    T.copy(b[by * block_n, _k * block_k], b_shared)
                    T.gemm(b_shared, a_shared, c_local, False, True)

                T.copy(c_local, c_shared)
                T.copy(c_shared, c_col_major[by * block_n, bx * block_m])

        return _gemm_main

    return _gemm_func


@torch.library.custom_op("top::gemm_wrapped_kernel", mutates_args=())
def _gemm_wrapped_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Run the warp-specialized GEMM ``C = op(A) @ op(B)`` (torch custom op).

    Kept for ``torch.compile`` compatibility (registered op + ``register_fake``).
    ``GemmKernel.forward`` calls the compiled JIT directly (cf. ``GemvKernel``),
    so this wrapper is not on the eager forward path.
    """
    return _gemm_kernel(m, n, k, trans_a, trans_b, dtype)(block_m, block_n, block_k, num_stages)(
        a, b
    )


@_gemm_wrapped_kernel.register_fake
def _(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return torch.empty((m, n), dtype=inputs[0].dtype, device=inputs[0].device)


class GemmKernel(Kernel):
    """Dense GEMM kernel with C500 compiler and SM90 WGMMA implementations.

    Computes ``C = op(A) @ op(B)`` for any ``(trans_a, trans_b)`` layout. C500
    reports capability 8.0 through PyTorch and selects the MACA compiler path;
    other supported hardware uses the SM90 WGMMA path.
    """

    supported_archs: list[int] = [80, 90]

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
        self._is_maca_c500 = is_metax_c500()
        self._use_wgmma_path = not self._is_maca_c500 and get_sm_version() == 90
        split_k_override = _get_maca_bsm_split_k_override()
        packed_b_override = _get_bool_env_override("TILEOPS_GEMM_PACKED_B_TILE")
        # Keep the long-K C500 compiler split-K packed-B path selected by default.
        auto_splitk_packed = (
            split_k_override is None
            and packed_b_override is None
            and _should_auto_use_maca_compiler_splitk_packed(m, n, k, dtype, trans_a, trans_b, tune)
        )
        self.split_k = (
            split_k_override if split_k_override is not None else (2 if auto_splitk_packed else 1)
        )
        self._use_maca_bsm_path = (
            self._is_maca_c500
            and dtype == torch.float16
            and not trans_a
            and m % 128 == 0
            and n % 128 == 0
            and k % 128 == 0
            and not tune
        )
        if self.split_k > 1 and not self._use_maca_bsm_path:
            raise RuntimeError(
                "TILEOPS_GEMM_SPLIT_K currently only applies to the MetaX C500 BSM path"
            )
        if self.split_k > 1 and k % self.split_k != 0:
            raise RuntimeError("TILEOPS_GEMM_SPLIT_K requires K to be divisible by split_k")
        self._use_packed_b_tile_path = (
            self._use_maca_bsm_path
            and not trans_b
            and (packed_b_override if packed_b_override is not None else auto_splitk_packed)
        )
        self._use_col_major_output = (
            not trans_a and dtype == torch.float16 and not self._use_maca_bsm_path
        )
        self._use_split_k_path = self._use_maca_bsm_path and self.split_k > 1
        self._use_packed_b_async_pipeline_path = (
            self._use_split_k_path
            and self._use_packed_b_tile_path
            and _get_maca_bsm_packed_b_async_pipeline()
        )
        self._compiler_env: dict[str, str] = {}
        if self._use_maca_bsm_path:
            self._compiler_env["TILELANG_DEFAULT_TARGET"] = "maca"
        if (
            self._use_split_k_path
            and self._use_packed_b_tile_path
            and not self._use_packed_b_async_pipeline_path
        ):
            self._compiler_env.update(
                {
                    "TILELANG_MACA_GEMM_USE_TEMPLATE": "1",
                    "TILELANG_MACA_GEMM_K_PACK": "1",
                }
            )

        if self._use_wgmma_path:
            self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str)
        elif self._use_split_k_path:
            if self._use_packed_b_async_pipeline_path:
                self.kernel = _gemm_kernel_bsm_splitk_packed_b_tile_async(
                    m,
                    n,
                    k,
                    self.split_k,
                    self.dtype_str,
                )
            elif self._use_packed_b_tile_path:
                self.kernel = _gemm_kernel_bsm_splitk_packed_b_tile(
                    m,
                    n,
                    k,
                    self.split_k,
                    self.dtype_str,
                )
            else:
                self.kernel = _gemm_kernel_bsm_splitk(m, n, k, self.split_k, self.dtype_str)
            self.reduce_kernel = _gemm_splitk_reduce_kernel(m, n, self.split_k, self.dtype_str)
        elif self._use_maca_bsm_path:
            if self._use_packed_b_tile_path:
                self.kernel = _gemm_kernel_bsm_packed_b_tile(m, n, k, self.dtype_str)
            else:
                self.kernel = _gemm_kernel_bsm(m, n, k, self.dtype_str)
        elif self._use_col_major_output:
            self.kernel = _gemm_kernel_col_major(m, n, k, self.dtype_str)
        else:
            self.kernel = _gemm_kernel(m, n, k, trans_a, True, self.dtype_str)
        self.init_config(config, tune)
        if self._use_split_k_path:
            if self.config["block_k"] % self.split_k != 0:
                raise RuntimeError("TILEOPS_GEMM_SPLIT_K requires block_k divisible by split_k")
            if self.config["block_k"] < self.split_k:
                raise RuntimeError("TILEOPS_GEMM_SPLIT_K requires block_k >= split_k")
        if not self._use_split_k_path:
            self.reduce_kernel = None
        self._b_native_cache_source = None
        self._b_native_cache_version = -1
        self._b_native_cache_tensor: Optional[torch.Tensor] = None
        self._compiled_kernel_config: Optional[tuple] = None
        self._compiled_kernel = None
        self._compiled_reduce_kernel = None

    @property
    def default_config(self) -> dict:
        # From tilelang/examples/gemm/example_gemm_autotune.py
        if self._use_wgmma_path:
            return {
                "block_m": 128,
                "block_n": 128,
                "block_k": 64,
                "num_stages": 3,
            }
        if self._use_maca_bsm_path:
            if self._use_split_k_path and self._use_packed_b_tile_path and self.k >= 131072:
                return {
                    "block_m": 128,
                    "block_n": 128,
                    "block_k": 128,
                    "num_stages": 0,
                    "threads": 256,
                    "enable_rasterization": True,
                }
            return {
                "block_m": 128,
                "block_n": 128,
                "block_k": 128,
                "num_stages": 0,
                "threads": 256,
                "enable_rasterization": True,
            }

        sm_version = get_sm_version()

        if sm_version in {80}:
            return {
                "block_m": 128,
                "block_n": 128,
                "block_k": 128,
                "num_stages": 0,
                "threads": 256,
                "enable_rasterization": True,
            }
        if sm_version in {90}:
            return {
                "block_m": 128,
                "block_n": 256,
                "block_k": 64,
                "num_stages": 3,
                "threads": 256,
                "enable_rasterization": True,
            }
        return {
            "block_m": 128,
            "block_n": 256,
            "block_k": 32,
            "num_stages": 0,
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_ms = [64, 128, 256]
        block_ns = [64, 128, 256]
        block_ks = [32, 64]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        enable_rasterization = [True, False]
        configs = itertools.product(
            block_ms, block_ns, block_ks, num_stages, threads, enable_rasterization
        )
        return [
            {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_stages": stages,
                "threads": num_threads,
                "enable_rasterization": rasterization,
            }
            for block_m, block_n, block_k, stages, num_threads, rasterization in configs
        ]

    @property
    def execution_info(self) -> dict[str, object]:
        if self._use_wgmma_path:
            backend = "wgmma"
        elif self._use_split_k_path and self._use_packed_b_tile_path:
            backend = "compiler-splitk-packed"
        elif self._use_split_k_path:
            backend = "compiler-splitk"
        elif self._use_maca_bsm_path and self._use_packed_b_tile_path:
            backend = "compiler-packed-b"
        elif self._use_maca_bsm_path:
            backend = "compiler-bsm"
        elif self._use_col_major_output:
            backend = "compiler-col-major"
        else:
            backend = "compiler-generic"
        return {
            "backend": backend,
            "split_k": self.split_k,
            "packed_b_tile": self._use_packed_b_tile_path,
            "template": self._compiler_env.get("TILELANG_MACA_GEMM_USE_TEMPLATE") == "1",
            "specialized_reduce": self._use_split_k_path,
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.forward_with_prepared_b(a, self.prepare_b(b))

    def prepare_a(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def prepare_b(self, b: torch.Tensor) -> torch.Tensor:
        if self._use_wgmma_path or self.trans_b:
            return b

        version = int(getattr(b, "_version", 0))
        if (
            self._b_native_cache_tensor is not None
            and self._b_native_cache_version == version
            and self._b_native_cache_source is not None
            and self._b_native_cache_source() is b
        ):
            return self._b_native_cache_tensor

        native_b = b.transpose(0, 1).contiguous()
        if self._use_packed_b_tile_path:
            block_n = self.config["block_n"]
            block_k = self.config["block_k"]
            tile_k = block_k // self.split_k if self._use_split_k_path else block_k
            if self.n % block_n != 0:
                raise RuntimeError("TILEOPS_GEMM_PACKED_B_TILE requires N divisible by block_n")
            if self.k % tile_k != 0:
                raise RuntimeError("TILEOPS_GEMM_PACKED_B_TILE requires K divisible by tile_k")
            native_b = (
                native_b.view(
                    self.n // block_n,
                    block_n,
                    self.k // tile_k,
                    tile_k,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        try:
            self._b_native_cache_source = weakref.ref(b)
            self._b_native_cache_version = version
            self._b_native_cache_tensor = native_b
        except TypeError:
            self._b_native_cache_source = None
            self._b_native_cache_version = -1
            self._b_native_cache_tensor = None
        return native_b

    def _compiled_kernel_config_key(self) -> tuple:
        if self._use_wgmma_path:
            return (
                self.config["block_m"],
                self.config["block_n"],
                self.config["block_k"],
                self.config["num_stages"],
            )
        return (
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["threads"],
            self.config["num_stages"],
            self.config["enable_rasterization"],
        )

    def _get_compiled_kernel(self):
        config_key = self._compiled_kernel_config_key()
        if self._compiled_kernel is None or self._compiled_kernel_config != config_key:
            with _temporary_env(self._compiler_env):
                self._compiled_kernel = self.kernel(*config_key)
            self._compiled_kernel_config = config_key
        return self._compiled_kernel

    def _get_compiled_reduce_kernel(self):
        if self._compiled_reduce_kernel is None:
            assert self.reduce_kernel is not None
            with _temporary_env(self._compiler_env):
                self._compiled_reduce_kernel = self.reduce_kernel()
        return self._compiled_reduce_kernel

    def forward_with_prepared_b(self, a: torch.Tensor, b_prepared: torch.Tensor) -> torch.Tensor:
        kernel = self._get_compiled_kernel()
        if self._use_wgmma_path:
            layout = f"{'T' if self.trans_a else 'N'}{'T' if self.trans_b else 'N'}"
            return trace.run(
                kernel,
                (a, b_prepared),
                stem=f"gemm_{self.m}x{self.n}x{self.k}_{layout}_{self.dtype_str}",
            )
        if self._use_split_k_path:
            partial_c = kernel(a, b_prepared)
            return self._get_compiled_reduce_kernel()(partial_c)
        if self._use_col_major_output:
            return kernel(a, b_prepared).transpose(0, 1)
        return kernel(a, b_prepared)

    def forward_with_prepared_a_and_b(
        self, a_prepared: torch.Tensor, b_prepared: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_with_prepared_b(a_prepared, b_prepared)


# TODO: add persistent, split-k, steam-k...


@functools.lru_cache(maxsize=32)
def _gemv_kernel(n: int, k: int, dtype: str = "float16") -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemv_func(
        block_n: int = 8,
        reduce_threads: int = 32,
        num_stages: int = 2,
    ) -> Callable:

        max_transaction_size_in_bits = 128
        tile_k = max_transaction_size_in_bits // (str2dtype[dtype].itemsize * 8)
        block_k = reduce_threads * tile_k

        @T.prim_func
        def _gemv_main(
            a: T.Tensor((k,), dtype),
            b: T.Tensor((n, k), dtype),
            c: T.Tensor((n,), dtype),
        ):
            # threads=(reduce_threads, block_n): tk=threadIdx.x is the fast-varying
            # dimension so consecutive warp threads access consecutive columns of B
            # (same row, stride-1) → coalesced 128-bit loads.
            with T.Kernel(T.ceildiv(n, block_n), threads=(reduce_threads, block_n)) as bn:
                tk = T.get_thread_binding(0)  # threadIdx.x — varies within a warp
                tn = T.get_thread_binding(
                    1
                )  # threadIdx.y — one row per warp (when reduce_threads=32)
                c_accum = T.alloc_local((1,), accum_dtype)

                T.clear(c_accum)

                # O3: pipeline B loads through shared memory using T.Pipelined.
                # T.copy issues cp.async for the next tile while the current tile is
                # being consumed, hiding HBM3e latency.
                # num_stages=1 → sequential (no overlap), num_stages>=2 → actual pipeline.
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                a_local = T.alloc_local((tile_k,), dtype)

                for bk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    # disable_tma=True: use cp.async instead of TMA to avoid
                    # mbarrier requirements that TileLang cannot infer for
                    # manually-indexed b_shared in a non-wgmma kernel.
                    T.copy(b[bn * block_n, bk * block_k], b_shared, disable_tma=True)
                    # a is tiny (fits in L1), load directly to registers
                    for _k in T.vectorized(tile_k):
                        a_local[_k] = a[bk * block_k + tk * tile_k + _k]
                    # FMA
                    for _k in T.serial(tile_k):
                        c_accum[0] += a_local[_k].astype(accum_dtype) * b_shared[
                            tn, tk * tile_k + _k
                        ].astype(accum_dtype)

                c_reduced = T.alloc_local((1,), accum_dtype)
                with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            c_accum[0],
                            True,
                            c_reduced[0],
                            tk,
                            dtype="handle",
                        )
                    )

                c[bn * block_n + tn] = c_reduced[0]

        return _gemv_main

    return _gemv_func


@torch.library.custom_op("top::gemv_wrapped_kernel", mutates_args=())
def _gemv_wrapped_kernel(
    n: int,
    k: int,
    dtype: str,
    block_n: int,
    reduce_threads: int,
    num_stages: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return _gemv_kernel(n, k, dtype)(block_n, reduce_threads, num_stages)(a, b)


@_gemv_wrapped_kernel.register_fake
def _(
    n: int,
    k: int,
    dtype: str,
    block_n: int,
    reduce_threads: int,
    num_stages: int,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return torch.empty((n,), dtype=inputs[0].dtype, device=inputs[0].device)


class GemvKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(
        self, n: int, k: int, dtype: torch.dtype, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self.dtype = dtype

        self.kernel = _gemv_kernel(n, k, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()

        if sm_version in {90}:
            # reduce_threads=32: full warp per row → coalesced B access + warp shuffle reduce
            # block_n=8: 256 threads/block, 448 blocks for n=7168 → ~3.4 blocks/SM on H200
            # num_stages=2: double-buffer B tile to hide HBM3e latency
            return {
                "block_n": 8,
                "reduce_threads": 32,
                "num_stages": 2,
            }

        return {
            "block_n": 32,
            "reduce_threads": 32,
            "num_stages": 1,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # num_stages=1: sequential shared-memory path (no overlap, baseline for comparison)
        # num_stages>=2: actual pipeline with cp.async prefetch to hide HBM latency
        return [
            {"block_n": bn, "reduce_threads": rt, "num_stages": ns}
            for bn in [1, 2, 4, 8, 16]
            for rt in [32]
            for ns in [1, 2, 3]
        ]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.flatten().contiguous()
        # Call the JIT-compiled kernel directly to avoid Python overhead from
        # closure recreation + JIT cache lookup in _gemv_wrapped_kernel on every
        # forward pass. _gemv_wrapped_kernel is kept for torch.compile compatibility.
        return self.kernel(
            self.config["block_n"],
            self.config["reduce_threads"],
            self.config["num_stages"],
        )(a, b)
