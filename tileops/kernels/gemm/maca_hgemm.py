# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import csv
import functools
import os
import sys
import weakref
from pathlib import Path
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.gemm.gemm import GemmKernel
from tileops.kernels.kernel import Kernel
from tileops.utils import is_metax_c500

__all__ = ["MacaHGemmKernel"]


_TILE_M = 128
_TILE_N = 128
_TILE_K = 128
_THREADS = 256
_C500_AP_COUNT = 104
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_DISABLED_BODY_VARIANTS = (
    (
        "TILEOPS_MACA_HGEMM_USE_128X64_BODY",
        "TILEOPS_MACA_HGEMM_USE_128X64_BODY is disabled: the 128x64 packed-BSM "
        "probe failed correctness validation on MetaX C500.",
    ),
    (
        "TILEOPS_MACA_HGEMM_USE_64X64_BODY",
        "TILEOPS_MACA_HGEMM_USE_64X64_BODY is disabled: the 64x64 packed-BSM "
        "prototype failed correctness validation on MetaX C500.",
    ),
    (
        "TILEOPS_MACA_HGEMM_USE_64X128_BODY",
        "TILEOPS_MACA_HGEMM_USE_64X128_BODY is disabled: the corrected 256-thread "
        "64x128 packed-BSM prototype passes small-shape correctness on MetaX C500 "
        "but is far slower than the 128x128 packed-BSM baseline.",
    ),
)
_WRAPPER_HEADER = Path(
    os.environ.get("TILEOPS_MACA_HGEMM_WRAPPER_HEADER",
                   str(Path(__file__).with_name("maca_hgemm_wrapper.hpp"))))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def _disabled_env_flag(name: str, message: str) -> bool:
    if _env_flag(name):
        raise RuntimeError(message)
    return False


def _guard_disabled_body_variants() -> None:
    for name, message in _DISABLED_BODY_VARIANTS:
        _disabled_env_flag(name, message)


def _reference_layout_repo() -> Path:
    override = os.environ.get("TILEOPS_MACA_HGEMM_REFERENCE_REPO", "").strip()
    if override:
        return Path(override).expanduser()

    workspace_sibling = Path(__file__).resolve().parents[3].parent / "muxi_native_layout_kernels"
    return workspace_sibling


def _use_layout_ab_body() -> bool:
    return _env_flag("TILEOPS_MACA_HGEMM_USE_LAYOUT_AB")


def _use_reference_layout_ab_continuous_c_body() -> bool:
    return _env_flag("TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_AB_CONTINUOUS_C")


def _use_reference_layout_a_body() -> bool:
    return _env_flag("TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_A_BODY")


def _use_async_a_staging() -> bool:
    return _disabled_env_flag(
        "TILEOPS_MACA_HGEMM_USE_ASYNC_A_STAGING",
        "TILEOPS_MACA_HGEMM_USE_ASYNC_A_STAGING is disabled: the macro-level "
        "async-A probe stores the async load back through shared memory and "
        "measured flat versus the packed-BSM baseline. A future async-A path "
        "must be a real register-resident body rewrite.",
    )


def _use_example_async_a_staging() -> bool:
    return _disabled_env_flag(
        "TILEOPS_MACA_HGEMM_USE_EXAMPLE_ASYNC_A_STAGING",
        "TILEOPS_MACA_HGEMM_USE_EXAMPLE_ASYNC_A_STAGING is disabled: the "
        "default hgemm-example body cannot replace WSM A-side BSM loads with "
        "global async loads without a real register-resident body rewrite.",
    )


def _use_m_group_swizzle() -> bool:
    return _env_flag("TILEOPS_MACA_HGEMM_USE_M_GROUP_SWIZZLE")


def _use_n_group_swizzle() -> bool:
    return _env_flag("TILEOPS_MACA_HGEMM_USE_N_GROUP_SWIZZLE")


def _auto_launch_order(m: int, n: int, body_variant: str, split_k: int) -> str:
    if body_variant != "m128n128" or split_k != 1:
        return "grid_xy"

    num_m_tiles = (m + _TILE_M - 1) // _TILE_M
    num_n_tiles = (n + _TILE_N - 1) // _TILE_N
    if num_m_tiles * num_n_tiles <= _C500_AP_COUNT:
        return "grid_xy"
    if num_m_tiles >= num_n_tiles:
        return "m_group_swizzle"
    return "grid_xy"


def _use_rowa_layout_b_body() -> bool:
    _disabled_env_flag(
        "TILEOPS_MACA_HGEMM_USE_ROWA_LAYOUT_B_BODY",
        "TILEOPS_MACA_HGEMM_USE_ROWA_LAYOUT_B_BODY is disabled: the row-major-A "
        "and packed-layoutB hybrid failed correctness and long-K validation on MetaX C500.",
    )
    _disabled_env_flag(
        "TILEOPS_MACA_HGEMM_EXPERIMENTAL_ROWA_LAYOUT_B_BODY",
        "TILEOPS_MACA_HGEMM_EXPERIMENTAL_ROWA_LAYOUT_B_BODY is disabled: the "
        "row-major-A and packed-layoutB hybrid failed correctness and long-K "
        "validation on MetaX C500.",
    )
    return False


@functools.lru_cache(maxsize=1)
def _reference_muxi_layout_kernels() -> object:
    repo_root = str(_reference_layout_repo().resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import muxi_layout_kernels  # type: ignore

    return muxi_layout_kernels


@functools.lru_cache(maxsize=1)
def _reference_continuous_c_supported_shapes() -> frozenset[tuple[int, int, int]]:
    csv_path = _reference_layout_repo() / "muxi_layout_kernels" / "continuous_gemm_dispatch_arg.csv"
    if not csv_path.exists():
        return frozenset()

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return frozenset(
            (int(row["m"]), int(row["n"]), int(row["k"]))
            for row in reader
            if row.get("m") and row.get("n") and row.get("k")
        )


def _compile_profile() -> str:
    value = os.environ.get("TILEOPS_MACA_HGEMM_COMPILE_PROFILE", "aggressive").strip().lower()
    return value or "aggressive"


def _compile_flags() -> list[str]:
    profile = _compile_profile()
    flags = [
        "-O3",
        "-DENABLE_BF16",
        "-mllvm -metaxgpu-direct-address=disshared",
        "-mllvm -metaxgpu-force-global-saddr=1",
    ]
    if _use_async_a_staging():
        flags.append("-DTILEOPS_MACA_HGEMM_USE_ASYNC_A_STAGING=1")
    if _use_example_async_a_staging():
        flags.append("-DTILEOPS_MACA_HGEMM_USE_EXAMPLE_ASYNC_A_STAGING=1")
    if profile == "aggressive":
        flags.extend([
            "-mllvm -metaxgpu-sched-regpressure=true",
            "-mllvm -metaxgpu-sinkload=false",
            "-mllvm -metaxgpu-vectorize-slp=true",
            "-mllvm -metaxgpu-igroup",
            "-mllvm -metaxgpu-aggressive-4g-addr-opt=true",
            "-mllvm -metaxgpu-shl-add-combine=false",
            "-mllvm -misched-postra=true",
            "-mllvm -enable-post-misched=true",
        ])
    elif profile == "mma_sched":
        flags.extend([
            "-mllvm -metaxgpu-mma-sched=true",
            "-mllvm -metaxgpu-sched-select=metaxgpu-minreg",
            "-mllvm -map-use-pk-fma=1",
        ])
    elif profile not in {"example", "minimal"}:
        raise ValueError(
            "TILEOPS_MACA_HGEMM_COMPILE_PROFILE must be 'aggressive', 'mma_sched', "
            "'example', or 'minimal'"
        )
    flags.extend([
        "-gcc-version 11",
        f"-include {_WRAPPER_HEADER}",
    ])
    extra = os.environ.get("TILEOPS_MACA_HGEMM_EXTRA_COMPILE_FLAGS", "").strip()
    if extra:
        flags.extend(flag for flag in extra.split() if flag)
    return flags


def _is_supported_hgemm(m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool,
                        trans_b: bool) -> bool:
    return (
        is_metax_c500()
        and dtype == torch.float16
        and not trans_a
        and trans_b in {False, True}
        and m % 4 == 0
        and k % 8 == 0
        and m > 0
        and n > 0
        and k >= _TILE_K
    )


def _get_split_k() -> int:
    value = os.environ.get("TILEOPS_MACA_HGEMM_SPLIT_K", "1").strip()
    if not value:
        return 1
    return max(1, int(value))


def _get_split_k_mode() -> str:
    value = os.environ.get("TILEOPS_MACA_HGEMM_SPLIT_K_MODE", "reduce").strip().lower()
    if value not in {"reduce", "reduce_f32", "atomic", "atomic_f32"}:
        raise ValueError(
            "TILEOPS_MACA_HGEMM_SPLIT_K_MODE must be 'reduce', 'reduce_f32', "
            "'atomic', or 'atomic_f32'")
    return value


def _is_supported_split_k(m: int, n: int, k: int, split_k: int) -> bool:
    if split_k <= 1:
        return False
    if k % split_k != 0:
        return False
    k_chunk = k // split_k
    return k_chunk >= _TILE_K and k_chunk % _TILE_K == 0 and m % 4 == 0


def _is_supported_layout_ab_hgemm(m: int, n: int, k: int, dtype: torch.dtype,
                                  trans_a: bool, trans_b: bool) -> bool:
    return (
        is_metax_c500()
        and dtype == torch.float16
        and not trans_a
        and trans_b in {False, True}
        and m % 128 == 0
        and n % 16 == 0
        and k % 128 == 0
        and m > 0
        and n > 0
        and k > 0
    )


@functools.lru_cache(maxsize=32)
def _maca_hgemm_tn_kernel_impl(m: int, n: int, k: int, dtype: str, tile_m: int, tile_n: int,
                               threads: int, call_name: str) -> Callable:
    @tilelang.jit(out_idx=[-1], compile_flags=_compile_flags())
    def _hgemm_func() -> Callable:

        @T.prim_func
        def _hgemm_main(
                a: T.Tensor((m, k), dtype),  # type: ignore
                b: T.Tensor((n, k), dtype),  # type: ignore
                c_col_major: T.Tensor((n, m), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(m, tile_m), T.ceildiv(n, tile_n), threads=threads) as (
                    bidx, bidy):
                with T.sblock("maca_hgemm_tn_call"):
                    T.reads(a[0:m, 0:k], b[0:n, 0:k])
                    T.writes(c_col_major[0:n, 0:m])
                    T.evaluate(
                        T.call_extern(
                            "handle",
                            call_name,
                            T.tvm_access_ptr(T.type_annotation(dtype), a.data, 0, m * k, 1),
                            T.tvm_access_ptr(T.type_annotation(dtype), b.data, 0, n * k, 1),
                            T.tvm_access_ptr(
                                T.type_annotation(dtype), c_col_major.data, 0, n * m, 2),
                            m,
                            n,
                            k,
                            k,
                            k,
                            m,
                            bidx,
                            bidy,
                        ))

        return _hgemm_main

    return _hgemm_func


@functools.lru_cache(maxsize=32)
def _maca_hgemm_tn_m_group_swizzle_kernel_impl(m: int, n: int, k: int, dtype: str,
                                               tile_m: int, tile_n: int, threads: int,
                                               call_name: str) -> Callable:
    num_m_tiles = (m + tile_m - 1) // tile_m
    num_n_tiles = (n + tile_n - 1) // tile_n
    n_tiles_for_group = min(num_n_tiles, 8)
    m_group_tiles = min(num_m_tiles, max(1, _C500_AP_COUNT // max(1, n_tiles_for_group)))
    full_m_groups = num_m_tiles // m_group_tiles
    tail_m_tiles = num_m_tiles - full_m_groups * m_group_tiles
    full_tile_count = full_m_groups * m_group_tiles * num_n_tiles
    total_tiles = num_m_tiles * num_n_tiles
    group_tile_count = m_group_tiles * num_n_tiles

    @tilelang.jit(out_idx=[-1], compile_flags=_compile_flags())
    def _hgemm_func() -> Callable:

        @T.prim_func
        def _hgemm_main(
                a: T.Tensor((m, k), dtype),  # type: ignore
                b: T.Tensor((n, k), dtype),  # type: ignore
                c_col_major: T.Tensor((n, m), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(total_tiles, threads=threads) as tile_id:
                with T.sblock("maca_hgemm_tn_m_group_swizzle_call"):
                    T.reads(a[0:m, 0:k], b[0:n, 0:k])
                    T.writes(c_col_major[0:n, 0:m])
                    if tail_m_tiles == 0:
                        group_id = tile_id // group_tile_count
                        local_id = tile_id - group_id * group_tile_count
                        bidx = group_id * m_group_tiles + local_id % m_group_tiles
                        bidy = local_id // m_group_tiles
                        T.evaluate(
                            T.call_extern(
                                "handle",
                                call_name,
                                T.tvm_access_ptr(T.type_annotation(dtype), a.data, 0, m * k, 1),
                                T.tvm_access_ptr(T.type_annotation(dtype), b.data, 0, n * k, 1),
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype), c_col_major.data, 0, n * m, 2),
                                m,
                                n,
                                k,
                                k,
                                k,
                                m,
                                bidx,
                                bidy,
                            ))
                    elif full_m_groups == 0:
                        bidx = tile_id % tail_m_tiles
                        bidy = tile_id // tail_m_tiles
                        T.evaluate(
                            T.call_extern(
                                "handle",
                                call_name,
                                T.tvm_access_ptr(T.type_annotation(dtype), a.data, 0, m * k, 1),
                                T.tvm_access_ptr(T.type_annotation(dtype), b.data, 0, n * k, 1),
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype), c_col_major.data, 0, n * m, 2),
                                m,
                                n,
                                k,
                                k,
                                k,
                                m,
                                bidx,
                                bidy,
                            ))
                    else:
                        if tile_id < full_tile_count:
                            group_id = tile_id // group_tile_count
                            local_id = tile_id - group_id * group_tile_count
                            bidx = group_id * m_group_tiles + local_id % m_group_tiles
                            bidy = local_id // m_group_tiles
                            T.evaluate(
                                T.call_extern(
                                    "handle",
                                    call_name,
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), a.data, 0, m * k, 1),
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), b.data, 0, n * k, 1),
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), c_col_major.data, 0, n * m, 2),
                                    m,
                                    n,
                                    k,
                                    k,
                                    k,
                                    m,
                                    bidx,
                                    bidy,
                                ))
                        else:
                            tail_id = tile_id - full_tile_count
                            bidx = full_m_groups * m_group_tiles + tail_id % tail_m_tiles
                            bidy = tail_id // tail_m_tiles
                            T.evaluate(
                                T.call_extern(
                                    "handle",
                                    call_name,
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), a.data, 0, m * k, 1),
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), b.data, 0, n * k, 1),
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), c_col_major.data, 0, n * m, 2),
                                    m,
                                    n,
                                    k,
                                    k,
                                    k,
                                    m,
                                    bidx,
                                    bidy,
                                ))

        return _hgemm_main

    return _hgemm_func


@functools.lru_cache(maxsize=32)
def _maca_hgemm_tn_m_group_swizzle_kernel(m: int, n: int, k: int, dtype: str) -> Callable:
    return _maca_hgemm_tn_m_group_swizzle_kernel_impl(
        m, n, k, dtype, _TILE_M, _TILE_N, _THREADS,
        "tileops_maca_hgemm_tn_f16_beta0")


@functools.lru_cache(maxsize=32)
def _maca_hgemm_tn_n_group_swizzle_kernel_impl(m: int, n: int, k: int, dtype: str,
                                               tile_m: int, tile_n: int, threads: int,
                                               call_name: str) -> Callable:
    num_m_tiles = (m + tile_m - 1) // tile_m
    num_n_tiles = (n + tile_n - 1) // tile_n
    m_tiles_for_group = min(num_m_tiles, 8)
    n_group_tiles = min(num_n_tiles, max(1, _C500_AP_COUNT // max(1, m_tiles_for_group)))
    full_n_groups = num_n_tiles // n_group_tiles
    tail_n_tiles = num_n_tiles - full_n_groups * n_group_tiles
    full_tile_count = full_n_groups * n_group_tiles * num_m_tiles
    total_tiles = num_m_tiles * num_n_tiles
    group_tile_count = n_group_tiles * num_m_tiles

    @tilelang.jit(out_idx=[-1], compile_flags=_compile_flags())
    def _hgemm_func() -> Callable:

        @T.prim_func
        def _hgemm_main(
                a: T.Tensor((m, k), dtype),  # type: ignore
                b: T.Tensor((n, k), dtype),  # type: ignore
                c_col_major: T.Tensor((n, m), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(total_tiles, threads=threads) as tile_id:
                with T.sblock("maca_hgemm_tn_n_group_swizzle_call"):
                    T.reads(a[0:m, 0:k], b[0:n, 0:k])
                    T.writes(c_col_major[0:n, 0:m])
                    if tail_n_tiles == 0:
                        group_id = tile_id // group_tile_count
                        local_id = tile_id - group_id * group_tile_count
                        bidx = local_id % num_m_tiles
                        bidy = group_id * n_group_tiles + local_id // num_m_tiles
                        T.evaluate(
                            T.call_extern(
                                "handle",
                                call_name,
                                T.tvm_access_ptr(T.type_annotation(dtype), a.data, 0, m * k, 1),
                                T.tvm_access_ptr(T.type_annotation(dtype), b.data, 0, n * k, 1),
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype), c_col_major.data, 0, n * m, 2),
                                m,
                                n,
                                k,
                                k,
                                k,
                                m,
                                bidx,
                                bidy,
                            ))
                    elif full_n_groups == 0:
                        bidx = tile_id % num_m_tiles
                        bidy = tile_id // num_m_tiles
                        T.evaluate(
                            T.call_extern(
                                "handle",
                                call_name,
                                T.tvm_access_ptr(T.type_annotation(dtype), a.data, 0, m * k, 1),
                                T.tvm_access_ptr(T.type_annotation(dtype), b.data, 0, n * k, 1),
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype), c_col_major.data, 0, n * m, 2),
                                m,
                                n,
                                k,
                                k,
                                k,
                                m,
                                bidx,
                                bidy,
                            ))
                    else:
                        if tile_id < full_tile_count:
                            group_id = tile_id // group_tile_count
                            local_id = tile_id - group_id * group_tile_count
                            bidx = local_id % num_m_tiles
                            bidy = group_id * n_group_tiles + local_id // num_m_tiles
                            T.evaluate(
                                T.call_extern(
                                    "handle",
                                    call_name,
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), a.data, 0, m * k, 1),
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), b.data, 0, n * k, 1),
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), c_col_major.data, 0, n * m, 2),
                                    m,
                                    n,
                                    k,
                                    k,
                                    k,
                                    m,
                                    bidx,
                                    bidy,
                                ))
                        else:
                            tail_id = tile_id - full_tile_count
                            bidx = tail_id % num_m_tiles
                            bidy = full_n_groups * n_group_tiles + tail_id // num_m_tiles
                            T.evaluate(
                                T.call_extern(
                                    "handle",
                                    call_name,
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), a.data, 0, m * k, 1),
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), b.data, 0, n * k, 1),
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype), c_col_major.data, 0, n * m, 2),
                                    m,
                                    n,
                                    k,
                                    k,
                                    k,
                                    m,
                                    bidx,
                                    bidy,
                                ))

        return _hgemm_main

    return _hgemm_func


@functools.lru_cache(maxsize=32)
def _maca_hgemm_tn_n_group_swizzle_kernel(m: int, n: int, k: int, dtype: str) -> Callable:
    return _maca_hgemm_tn_n_group_swizzle_kernel_impl(
        m, n, k, dtype, _TILE_M, _TILE_N, _THREADS,
        "tileops_maca_hgemm_tn_f16_beta0")


@functools.lru_cache(maxsize=32)
def _maca_hgemm_tn_kernel(m: int, n: int, k: int, dtype: str) -> Callable:
    return _maca_hgemm_tn_kernel_impl(m, n, k, dtype, _TILE_M, _TILE_N, _THREADS,
                                      "tileops_maca_hgemm_tn_f16_beta0")


@functools.lru_cache(maxsize=32)
def _maca_hgemm_layout_ab_kernel(m: int, n: int, k: int, dtype: str) -> Callable:
    @tilelang.jit(out_idx=[-1], compile_flags=_compile_flags())
    def _hgemm_layout_ab_func() -> Callable:

        @T.prim_func
        def _hgemm_layout_ab_main(
                a_layout: T.Tensor((m // 16, k // 8, 16, 8), dtype),  # type: ignore
                b_layout: T.Tensor((k // 32, n // 16, 4, 16, 8), dtype),  # type: ignore
                c_col_major: T.Tensor((n, m), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(m, _TILE_M), T.ceildiv(n, _TILE_N), threads=_THREADS) as (
                    bidx, bidy):
                with T.sblock("maca_hgemm_layout_ab_call"):
                    T.reads(a_layout[0:m // 16, 0:k // 8, 0:16, 0:8],
                            b_layout[0:k // 32, 0:n // 16, 0:4, 0:16, 0:8])
                    T.writes(c_col_major[0:n, 0:m])
                    T.evaluate(
                        T.call_extern(
                            "handle",
                            "tileops_maca_hgemm_layout_ab_tn_f16_beta0",
                            T.tvm_access_ptr(
                                T.type_annotation(dtype), a_layout.data, 0, m * k, 1),
                            T.tvm_access_ptr(
                                T.type_annotation(dtype), b_layout.data, 0, n * k, 1),
                            T.tvm_access_ptr(
                                T.type_annotation(dtype), c_col_major.data, 0, n * m, 2),
                            m,
                            n,
                            k,
                            k,
                            k,
                            m,
                            bidx,
                            bidy,
                        ))

        return _hgemm_layout_ab_main

    return _hgemm_layout_ab_func


@functools.lru_cache(maxsize=32)
def _maca_hgemm_tn_splitk_partial_kernel(m: int, n: int, k: int, split_k: int,
                                         dtype: str, partial_dtype: str) -> Callable:
    k_chunk = k // split_k
    call_name = ("tileops_maca_hgemm_tn_f16_to_f32_beta0"
                 if partial_dtype == "float32" else "tileops_maca_hgemm_tn_f16_beta0")

    @tilelang.jit(compile_flags=_compile_flags())
    def _hgemm_splitk_partial_func() -> Callable:

        @T.prim_func
        def _hgemm_splitk_partial_main(
                a: T.Tensor((m, k), dtype),  # type: ignore
                b: T.Tensor((n, k), dtype),  # type: ignore
                partial_c_col_major: T.Tensor((split_k, n, m), partial_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(m, _TILE_M),
                    T.ceildiv(n, _TILE_N),
                    split_k,
                    threads=_THREADS,
            ) as (bidx, bidy, bidz):
                with T.sblock("maca_hgemm_tn_splitk_partial_call"):
                    k_start = bidz * k_chunk
                    T.reads(a[0:m, 0:k], b[0:n, 0:k])
                    T.writes(partial_c_col_major[0:split_k, 0:n, 0:m])
                    T.evaluate(
                        T.call_extern(
                            "handle",
                            call_name,
                            T.tvm_access_ptr(
                                T.type_annotation(dtype), a.data, k_start, m * k - k_start, 1),
                            T.tvm_access_ptr(
                                T.type_annotation(dtype), b.data, k_start, n * k - k_start, 1),
                            T.tvm_access_ptr(
                                T.type_annotation(partial_dtype),
                                partial_c_col_major.data,
                                bidz * n * m,
                                split_k * n * m - bidz * n * m,
                                2,
                            ),
                            m,
                            n,
                            k_chunk,
                            k,
                            k,
                            m,
                            bidx,
                            bidy,
                        ))

        return _hgemm_splitk_partial_main

    return _hgemm_splitk_partial_func


@functools.lru_cache(maxsize=32)
def _maca_hgemm_splitk_reduce_kernel(m: int, n: int, split_k: int, dtype: str,
                                     partial_dtype: str) -> Callable:
    threads = 256
    items_per_thread = 4
    block_size = threads * items_per_thread
    total = m * n
    call_name = ("tileops_maca_hgemm_splitk_reduce_f32_to_f16"
                 if partial_dtype == "float32" else "tileops_maca_hgemm_splitk_reduce_f16_to_f16")

    @tilelang.jit(out_idx=[1], compile_flags=_compile_flags())
    def _splitk_reduce_func() -> Callable:

        @T.prim_func
        def _splitk_reduce_main(
                partial_c_col_major: T.Tensor((split_k, n, m), partial_dtype),  # type: ignore
                c_col_major: T.Tensor((n, m), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(total, block_size), threads=threads) as bx:
                for tx, item in T.Parallel(threads, items_per_thread):
                    flat_idx = (bx * threads + tx) * items_per_thread + item
                    if flat_idx < total:
                        T.evaluate(
                            T.call_extern(
                                "handle",
                                call_name,
                                T.tvm_access_ptr(
                                    T.type_annotation(partial_dtype),
                                    partial_c_col_major.data,
                                    0,
                                    split_k * n * m,
                                    1,
                                ),
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype),
                                    c_col_major.data,
                                    0,
                                    n * m,
                                    2,
                                ),
                                m,
                                n,
                                split_k,
                                total,
                                flat_idx,
                            ))

        return _splitk_reduce_main

    return _splitk_reduce_func


@functools.lru_cache(maxsize=32)
def _maca_hgemm_cast_f32_kernel(m: int, n: int, dtype: str) -> Callable:
    threads = 256
    items_per_thread = 4
    block_size = threads * items_per_thread
    total = m * n

    @tilelang.jit(out_idx=[1], compile_flags=_compile_flags())
    def _cast_func() -> Callable:

        @T.prim_func
        def _cast_main(
                accum_c_col_major: T.Tensor((n, m), "float32"),  # type: ignore
                c_col_major: T.Tensor((n, m), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(total, block_size), threads=threads) as bx:
                for tx, item in T.Parallel(threads, items_per_thread):
                    flat_idx = (bx * threads + tx) * items_per_thread + item
                    if flat_idx < total:
                        T.evaluate(
                            T.call_extern(
                                "handle",
                                "tileops_maca_hgemm_cast_f32_to_f16",
                                T.tvm_access_ptr(
                                    T.type_annotation("float32"),
                                    accum_c_col_major.data,
                                    0,
                                    total,
                                    1,
                                ),
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype),
                                    c_col_major.data,
                                    0,
                                    total,
                                    2,
                                ),
                                total,
                                flat_idx,
                            ))

        return _cast_main

    return _cast_func


@functools.lru_cache(maxsize=32)
def _maca_hgemm_tn_splitk_atomic_kernel(m: int, n: int, k: int, split_k: int,
                                        dtype: str, accum_dtype: str) -> Callable:
    k_chunk = k // split_k
    call_name = ("tileops_maca_hgemm_tn_f32_splitk_atomic_beta0"
                 if accum_dtype == "float32" else
                 "tileops_maca_hgemm_tn_f16_splitk_atomic_beta0")

    @tilelang.jit(compile_flags=_compile_flags())
    def _hgemm_splitk_atomic_func() -> Callable:

        @T.prim_func
        def _hgemm_splitk_atomic_main(
                a: T.Tensor((m, k), dtype),  # type: ignore
                b: T.Tensor((n, k), dtype),  # type: ignore
                c_col_major: T.Tensor((n, m), accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(m, _TILE_M),
                    T.ceildiv(n, _TILE_N),
                    split_k,
                    threads=_THREADS,
            ) as (bidx, bidy, bidz):
                with T.sblock("maca_hgemm_tn_splitk_atomic_call"):
                    k_start = bidz * k_chunk
                    T.reads(a[0:m, 0:k], b[0:n, 0:k], c_col_major[0:n, 0:m])
                    T.writes(c_col_major[0:n, 0:m])
                    T.evaluate(
                        T.call_extern(
                            "handle",
                            call_name,
                            T.tvm_access_ptr(
                                T.type_annotation(dtype), a.data, k_start, m * k - k_start, 1),
                            T.tvm_access_ptr(
                                T.type_annotation(dtype), b.data, k_start, n * k - k_start, 1),
                            T.tvm_access_ptr(
                                T.type_annotation(accum_dtype), c_col_major.data, 0, n * m, 2),
                            m,
                            n,
                            k_chunk,
                            k,
                            k,
                            m,
                            bidx,
                            bidy,
                        ))

        return _hgemm_splitk_atomic_main

    return _hgemm_splitk_atomic_func


class MacaHGemmKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 m: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False,
                 trans_a: bool = False,
                 trans_b: bool = False) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.fallback: Optional[GemmKernel] = None
        self.use_reference_layout_ab_continuous_c = _use_reference_layout_ab_continuous_c_body()
        self.use_reference_layout_a = _use_reference_layout_a_body()
        self.use_layout_ab = _use_layout_ab_body()
        _use_rowa_layout_b_body()
        self.use_m_group_swizzle = _use_m_group_swizzle()
        self.use_n_group_swizzle = _use_n_group_swizzle()
        if sum([
                self.use_reference_layout_ab_continuous_c,
                self.use_reference_layout_a,
                self.use_layout_ab,
        ]) > 1:
            raise RuntimeError(
                "TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_AB_CONTINUOUS_C, "
                "TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_A_BODY, "
                "TILEOPS_MACA_HGEMM_USE_LAYOUT_AB are mutually exclusive.")
        if self.use_reference_layout_ab_continuous_c:
            self.body_variant = "reference_layout_ab_continuous_c"
        elif self.use_reference_layout_a:
            self.body_variant = "reference_layout_a"
        elif self.use_layout_ab:
            self.body_variant = "layout_ab"
        else:
            _guard_disabled_body_variants()
            self.body_variant = "m128n128"
        self.split_k = _get_split_k()
        self.split_k_mode = _get_split_k_mode()
        launch_order_env_set = (
            "TILEOPS_MACA_HGEMM_USE_M_GROUP_SWIZZLE" in os.environ
            or "TILEOPS_MACA_HGEMM_USE_N_GROUP_SWIZZLE" in os.environ
        )
        if (not launch_order_env_set
                and not self.use_m_group_swizzle
                and not self.use_n_group_swizzle):
            auto_launch_order = _auto_launch_order(m, n, self.body_variant, self.split_k)
            self.use_m_group_swizzle = auto_launch_order == "m_group_swizzle"
            self.use_n_group_swizzle = auto_launch_order == "n_group_swizzle"
        if self.use_m_group_swizzle and (
                self.use_reference_layout_ab_continuous_c
                or self.use_reference_layout_a
                or self.use_layout_ab):
            raise RuntimeError(
                "TILEOPS_MACA_HGEMM_USE_M_GROUP_SWIZZLE only applies to the default "
                "packed-BSM body.")
        if self.use_n_group_swizzle and (
                self.use_reference_layout_ab_continuous_c
                or self.use_reference_layout_a
                or self.use_layout_ab):
            raise RuntimeError(
                "TILEOPS_MACA_HGEMM_USE_N_GROUP_SWIZZLE only applies to the default "
                "packed-BSM body.")
        if self.use_m_group_swizzle and self.use_n_group_swizzle:
            raise RuntimeError(
                "TILEOPS_MACA_HGEMM_USE_M_GROUP_SWIZZLE and "
                "TILEOPS_MACA_HGEMM_USE_N_GROUP_SWIZZLE are mutually exclusive.")
        if self.use_m_group_swizzle and self.split_k != 1:
            raise RuntimeError(
                "TILEOPS_MACA_HGEMM_USE_M_GROUP_SWIZZLE does not support split-K.")
        if self.use_n_group_swizzle and self.split_k != 1:
            raise RuntimeError(
                "TILEOPS_MACA_HGEMM_USE_N_GROUP_SWIZZLE does not support split-K.")
        self.layout_adapter = (
            "reference_layout_ab_continuous_c"
            if self.use_reference_layout_ab_continuous_c else
            ("reference_layout_a" if self.use_reference_layout_a else
             ("layout_ab" if self.use_layout_ab else
              ("native_tn" if trans_b else "pretranspose_b"))))
        self.split_k_partial_dtype = (
            "float32" if self.split_k_mode in {"reduce_f32", "atomic_f32"} else self.dtype_str)
        self.reduce_kernel: Optional[Callable] = None
        self.cast_kernel: Optional[Callable] = None
        self._b_native_cache_source = None
        self._b_native_cache_version = -1
        self._b_native_cache_tensor: Optional[torch.Tensor] = None

        if self.use_reference_layout_ab_continuous_c:
            if not _is_supported_layout_ab_hgemm(m, n, k, dtype, trans_a, trans_b):
                raise RuntimeError(
                    "TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_AB_CONTINUOUS_C requires "
                    "C500 float16 GEMM with trans_a=False, M multiple of 128, N multiple "
                    "of 16, and K multiple of 128.")
            if self.split_k != 1:
                raise RuntimeError(
                    "TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_AB_CONTINUOUS_C does not "
                    "support split-K.")
            if tune:
                raise RuntimeError(
                    "TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_AB_CONTINUOUS_C does not "
                    "support autotuning.")
            if (m, n, k) not in _reference_continuous_c_supported_shapes():
                raise RuntimeError(
                    "TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_AB_CONTINUOUS_C only supports "
                    "shapes listed in muxi_native_layout_kernels "
                    f"continuous_gemm_dispatch_arg.csv; got {(m, n, k)}.")
            self.kernel = None
            self.init_config(config, tune=False)
        elif self.use_layout_ab:
            if not _is_supported_layout_ab_hgemm(m, n, k, dtype, trans_a, trans_b):
                raise RuntimeError(
                    "TILEOPS_MACA_HGEMM_USE_LAYOUT_AB requires C500 float16 GEMM with "
                    "trans_a=False, M multiple of 128, N multiple of 16, and K multiple of 128.")
            if self.split_k != 1:
                raise RuntimeError("TILEOPS_MACA_HGEMM_USE_LAYOUT_AB does not support split-K.")
            self.kernel = _maca_hgemm_layout_ab_kernel(m, n, k, self.dtype_str)
            self.init_config(config, tune)
        elif self.use_reference_layout_a:
            if not _is_supported_layout_ab_hgemm(m, n, k, dtype, trans_a, trans_b):
                raise RuntimeError(
                    "TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_A_BODY requires C500 "
                    "float16 GEMM with trans_a=False, M multiple of 128, N multiple "
                    "of 16, and K multiple of 128.")
            if self.split_k != 1:
                raise RuntimeError(
                    "TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_A_BODY does not support split-K.")
            if tune:
                raise RuntimeError(
                    "TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_A_BODY does not support autotuning.")
            self.kernel = None
            self.init_config(config, tune=False)
        elif _is_supported_hgemm(m, n, k, dtype, trans_a, trans_b):
            if _is_supported_split_k(m, n, k, self.split_k):
                if self.split_k_mode in {"atomic", "atomic_f32"}:
                    self.kernel = _maca_hgemm_tn_splitk_atomic_kernel(
                        m, n, k, self.split_k, self.dtype_str, self.split_k_partial_dtype)
                    if self.split_k_mode == "atomic_f32":
                        self.cast_kernel = _maca_hgemm_cast_f32_kernel(
                            m, n, self.dtype_str)
                else:
                    self.kernel = _maca_hgemm_tn_splitk_partial_kernel(
                        m, n, k, self.split_k, self.dtype_str, self.split_k_partial_dtype)
                    self.reduce_kernel = _maca_hgemm_splitk_reduce_kernel(
                        m, n, self.split_k, self.dtype_str, self.split_k_partial_dtype)
            else:
                self.split_k = 1
                if self.use_m_group_swizzle:
                    self.kernel = _maca_hgemm_tn_m_group_swizzle_kernel(m, n, k, self.dtype_str)
                elif self.use_n_group_swizzle:
                    self.kernel = _maca_hgemm_tn_n_group_swizzle_kernel(m, n, k, self.dtype_str)
                else:
                    self.kernel = _maca_hgemm_tn_kernel(m, n, k, self.dtype_str)
            self.init_config(config, tune)
        else:
            self.fallback = GemmKernel(
                m,
                n,
                k,
                dtype,
                config=config,
                tune=tune,
                trans_a=trans_a,
                trans_b=trans_b,
            )
            self.config = {
                "backend": "tilelang_fallback",
                "reason": "maca_hgemm supports only MetaX C500 float16 trans_a=False GEMM",
                "fallback_config": self.fallback.config,
            }
            print(f"{self.__class__.__name__} initialized with config: {self.config}")

    @property
    def default_config(self) -> dict:
        if self.use_reference_layout_a:
            return {
                "backend": "maca_hgemm_reference_layout_a",
                "tile_m": None,
                "tile_n": None,
                "tile_k": None,
                "threads": None,
                "header": str(_WRAPPER_HEADER),
                "reference_repo": str(_reference_layout_repo()),
                "compile_profile": _compile_profile(),
                "layout_adapter": self.layout_adapter,
                "body_variant": self.body_variant,
                "split_k": self.split_k,
                "split_k_mode": self.split_k_mode,
                "partial_dtype": None,
                "cast_dtype": None,
            }
        if self.use_reference_layout_ab_continuous_c:
            return {
                "backend": "maca_hgemm_reference_layout_ab_continuous_c",
                "tile_m": None,
                "tile_n": None,
                "tile_k": None,
                "threads": None,
                "header": str(_WRAPPER_HEADER),
                "reference_repo": str(_reference_layout_repo()),
                "compile_profile": _compile_profile(),
                "layout_adapter": self.layout_adapter,
                "body_variant": self.body_variant,
                "split_k": self.split_k,
                "split_k_mode": self.split_k_mode,
                "partial_dtype": None,
                "cast_dtype": None,
            }
        return {
            "backend": "maca_hgemm_tn_native",
            "tile_m": _TILE_M,
            "tile_n": _TILE_N,
            "tile_k": _TILE_K,
            "threads": _THREADS,
            "header": str(_WRAPPER_HEADER),
            "compile_profile": _compile_profile(),
            "layout_adapter": self.layout_adapter,
            "body_variant": self.body_variant,
            "launch_order": (
                "m_group_swizzle" if self.use_m_group_swizzle else
                ("n_group_swizzle" if self.use_n_group_swizzle else "grid_xy")),
            "split_k": self.split_k,
            "split_k_mode": self.split_k_mode,
            "partial_dtype": self.split_k_partial_dtype if self.split_k > 1 else None,
            "cast_dtype": self.dtype_str if self.split_k_mode == "atomic_f32" else None,
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.fallback is not None:
            return self.fallback(a, b)

        b_native = self.prepare_b(b)
        return self.forward_with_prepared_b(a, b_native)

    def prepare_a(self, a: torch.Tensor) -> torch.Tensor:
        if self.fallback is not None:
            raise NotImplementedError("packed A preparation is only available on the MACA HGEMM path")
        if (self.use_layout_ab
                or self.use_reference_layout_ab_continuous_c
                or self.use_reference_layout_a):
            return self._layout_a_tensor(a)
        return a

    def prepare_b(self, b: torch.Tensor) -> torch.Tensor:
        if self.fallback is not None:
            raise NotImplementedError("packed B preparation is only available on the MACA HGEMM path")
        if self.layout_adapter == "native_tn":
            return b
        return self._native_b_tensor(b)

    def forward_with_prepared_b(self, a: torch.Tensor, b_native: torch.Tensor) -> torch.Tensor:
        if self.fallback is not None:
            raise NotImplementedError(
                "prepacked-B execution is only available on the MACA HGEMM path")

        if (self.use_layout_ab
                or self.use_reference_layout_ab_continuous_c
                or self.use_reference_layout_a):
            return self.forward_with_prepared_a_and_b(self.prepare_a(a), b_native)

        if self.split_k > 1:
            if self.split_k_mode == "atomic":
                c_col_major = torch.zeros((self.n, self.m), device=a.device, dtype=self.dtype)
                self.kernel()(a, b_native, c_col_major)
            elif self.split_k_mode == "atomic_f32":
                assert self.cast_kernel is not None
                accum_c_col_major = torch.zeros(
                    (self.n, self.m), device=a.device, dtype=torch.float32)
                self.kernel()(a, b_native, accum_c_col_major)
                c_col_major = self.cast_kernel()(accum_c_col_major)
            else:
                assert self.reduce_kernel is not None
                partial_c_col_major = torch.empty(
                    (self.split_k, self.n, self.m),
                    device=a.device,
                    dtype=torch.float32 if self.split_k_partial_dtype == "float32" else self.dtype,
                )
                self.kernel()(a, b_native, partial_c_col_major)
                c_col_major = self.reduce_kernel()(partial_c_col_major)
            return c_col_major.transpose(0, 1)

        c_col_major = self.kernel()(a, b_native)
        return c_col_major.transpose(0, 1)

    def forward_with_prepared_a_and_b(self, a_prepared: torch.Tensor,
                                      b_prepared: torch.Tensor) -> torch.Tensor:
        if self.fallback is not None:
            raise NotImplementedError(
                "prepacked-A/B execution is only available on the MACA HGEMM path")
        if self.use_reference_layout_ab_continuous_c:
            muxi = _reference_muxi_layout_kernels()
            c_native = muxi.gemm_layoutAB_ContinuousC(a_prepared, b_prepared, 1.0, 0.0)
            return c_native.transpose(0, 1).contiguous()
        if self.use_reference_layout_a:
            muxi = _reference_muxi_layout_kernels()
            c_native = muxi.muxi_hgemm_layoutA(a_prepared, b_prepared, 1.0, 0.0)
            return c_native.transpose(0, 1).contiguous()
        if self.use_layout_ab:
            c_col_major = self.kernel()(a_prepared, b_prepared)
            return c_col_major.transpose(0, 1)
        return self.forward_with_prepared_b(a_prepared, b_prepared)

    def _native_b_tensor(self, b: torch.Tensor) -> torch.Tensor:
        if self.use_layout_ab or self.use_reference_layout_ab_continuous_c:
            return self._layout_b_tensor(b)

        cache_enabled = os.environ.get("TILEOPS_MACA_HGEMM_CACHE_B_NATIVE", "1").strip().lower()
        if cache_enabled in {"0", "false", "no"}:
            return b.transpose(0, 1).contiguous()

        version = int(getattr(b, "_version", 0))
        if (self._b_native_cache_tensor is not None and self._b_native_cache_version == version
                and self._b_native_cache_source is not None
                and self._b_native_cache_source() is b):
            return self._b_native_cache_tensor

        native_b = b.transpose(0, 1).contiguous()
        try:
            self._b_native_cache_source = weakref.ref(b)
            self._b_native_cache_version = version
            self._b_native_cache_tensor = native_b
        except TypeError:
            self._b_native_cache_source = None
            self._b_native_cache_version = -1
            self._b_native_cache_tensor = None
        return native_b

    def _layout_a_tensor(self, a: torch.Tensor) -> torch.Tensor:
        if a.shape != (self.m, self.k):
            raise ValueError(f"expected A shape {(self.m, self.k)}, got {tuple(a.shape)}")
        return a.view(self.m // 16, 16, self.k // 8, 8).permute(0, 2, 1, 3).contiguous()

    def _layout_b_tensor(self, b: torch.Tensor) -> torch.Tensor:
        if self.trans_b:
            if b.shape != (self.n, self.k):
                raise ValueError(f"expected transposed B shape {(self.n, self.k)}, got {tuple(b.shape)}")
            b_nk = b
        else:
            if b.shape != (self.k, self.n):
                raise ValueError(f"expected B shape {(self.k, self.n)}, got {tuple(b.shape)}")
            b_nk = b.transpose(0, 1)

        return b_nk.contiguous().view(
            self.n // 16, 16, self.k // 32, 4, 8).permute(2, 0, 3, 1, 4).contiguous()
