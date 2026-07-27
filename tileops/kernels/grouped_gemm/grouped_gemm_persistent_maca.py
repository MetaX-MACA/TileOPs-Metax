"""Persistent grouped GEMM kernel — MACA smem-safe path.

Single-kernel design (no separate tile scheduler):
  - sm_count CTAs form a fixed persistent grid
  - Each CTA atomically claims tiles from a global counter
  - Tile metadata computed inline via prefix sum + binary search
  - GEMM uses predicated T.Pipelined + T.gemm (no Hopper TMA/WGMMA)

Used on MetaX MACA where ``GroupedGemmPersistentKernel`` (Hopper 2-WG) and
``GroupedGemmPersistent3WGKernel`` are unavailable.
"""

import functools
import math

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import device_smem_budget

__all__ = [
    "GroupedGemmPersistentMACAKernel",
]


_DEFAULT_CONFIG = {
    "block_m": 64,
    "block_n": 128,
    "block_k": 64,
    "num_stages": 1,
    "threads": 256,
    "group_size_m": 1,
}


def _estimate_grouped_gemm_maca_smem_bytes(
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
) -> int:
    """Upper-bound dynamic shared memory for the MACA persistent grouped GEMM."""
    stages = max(num_stages, 1)
    bytes_per_elem = 2
    return 2 * stages * (block_m * block_k + block_n * block_k) * bytes_per_elem


def _config_fits_smem(cfg: dict, smem_budget: int) -> bool:
    return _estimate_grouped_gemm_maca_smem_bytes(
        block_m=cfg["block_m"],
        block_n=cfg["block_n"],
        block_k=cfg["block_k"],
        num_stages=cfg["num_stages"],
    ) <= smem_budget


def _build_persistent_gemm_maca(
    *,
    numel: int,
    num_experts: int,
    N: int,
    K: int,
    dtype: str,
    sm_count: int,
    block_m: int,
    block_n: int,
    block_k: int,
    threads: int,
    group_size_m: int,
    log2_up: int,
    num_pid_n: int,
    max_iters: int,
    num_stages: int,
    k_aligned: bool,
):
    accum_dtype = "float"
    A_shape = (numel + block_m, K) if k_aligned else (numel, K)

    if k_aligned:
        k_loop_extent = K // block_k

        @T.prim_func
        def _gemm_main(
            A: T.Tensor(A_shape, dtype),                         # type: ignore  # noqa: F821
            B: T.Tensor((num_experts, N, K), dtype),             # type: ignore  # noqa: F821
            true_sizes: T.Tensor((num_experts,), "int32"),       # noqa: F821
            true_offsets: T.Tensor((num_experts,), "int32"),     # noqa: F821
            C: T.Tensor((numel, N), dtype),                      # type: ignore  # noqa: F821
            tile_counter: T.Tensor((1,), "int32"),               # noqa: F821
        ):
            with T.Kernel(sm_count, threads=threads) as (pid,):
                A_shared = T.alloc_shared((block_m, block_k), dtype)
                B_shared = T.alloc_shared((block_n, block_k), dtype)
                C_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                s_cum = T.alloc_shared((num_experts + 1,), "int32")
                s_total = T.alloc_shared((1,), "int32")
                s_tile = T.alloc_shared((1,), "int32")
                s_expert = T.alloc_shared((1,), "int32")
                lo = T.alloc_local((1,), "int32")
                hi = T.alloc_local((1,), "int32")

                T.annotate_layout({
                    A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                    B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                })

                tx = T.get_thread_binding()

                if tx == 0:
                    s_cum[0] = T.int32(0)
                    for e in T.serial(num_experts):
                        s_cum[e + 1] = s_cum[e] + (true_sizes[e] + (block_m - 1)) // block_m
                    s_total[0] = s_cum[num_experts] * T.int32(num_pid_n)
                T.sync_threads()

                for _iter in T.serial(max_iters):
                    if tx == 0:
                        s_tile[0] = T.atomic_add(tile_counter[0], 1, return_prev=True)
                    T.sync_threads()

                    flat_id = s_tile[0]
                    total = s_total[0]

                    if flat_id < total:
                        if group_size_m == 1:
                            m_tile = flat_id // T.int32(num_pid_n)
                            n_tile = flat_id % T.int32(num_pid_n)
                        else:
                            num_pid_in_group = group_size_m * num_pid_n
                            m_tiles_total = s_cum[num_experts]
                            pid_in_group = flat_id % T.int32(num_pid_in_group)
                            group_id = flat_id // T.int32(num_pid_in_group)
                            first_pid_m = group_id * T.int32(group_size_m)
                            actual_gsm = T.min(m_tiles_total - first_pid_m,
                                               T.int32(group_size_m))
                            m_tile = first_pid_m + pid_in_group % actual_gsm
                            n_tile = pid_in_group // actual_gsm

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        if tx == 0:
                            s_expert[0] = lo[0]
                        T.sync_threads()
                        expert_id = s_expert[0]
                        row_in_expert = (m_tile - s_cum[expert_id]) * T.int32(block_m)
                        m_start = true_offsets[expert_id] + row_in_expert
                        n_start = n_tile * T.int32(block_n)
                        actual_rows = T.min(T.int32(block_m),
                                            true_sizes[expert_id] - row_in_expert)
                        actual_cols = T.min(T.int32(block_n),
                                            T.int32(N) - n_start)

                        T.clear(C_local)

                        for k in T.Pipelined(k_loop_extent, num_stages=num_stages):
                            k_start = k * block_k
                            if actual_rows == T.int32(block_m) and actual_cols == T.int32(block_n):
                                T.copy(
                                    A[m_start:m_start + block_m, k_start:k_start + block_k],
                                    A_shared,
                                    disable_tma=True,
                                )
                                T.copy(
                                    B[expert_id, n_start:n_start + block_n,
                                      k_start:k_start + block_k],
                                    B_shared,
                                    disable_tma=True,
                                )
                            else:
                                for i, j in T.Parallel(block_m, block_k):
                                    A_shared[i, j] = T.if_then_else(
                                        i < actual_rows and j < K - k * block_k,
                                        A[m_start + i, k * block_k + j], 0)
                                for i, j in T.Parallel(block_n, block_k):
                                    B_shared[i, j] = T.if_then_else(
                                        j < K - k * block_k and i < actual_cols,
                                        B[expert_id, n_start + i, k * block_k + j], 0)
                            T.gemm(
                                A_shared,
                                B_shared,
                                C_local,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                            )

                        for i, j in T.Parallel(block_m, block_n):
                            if i < actual_rows and j < actual_cols:
                                C[m_start + i, n_start + j] = C_local[i, j]
    else:
        @T.prim_func
        def _gemm_main(
            A: T.Tensor(A_shape, dtype),                         # type: ignore  # noqa: F821
            B: T.Tensor((num_experts, N, K), dtype),             # type: ignore  # noqa: F821
            true_sizes: T.Tensor((num_experts,), "int32"),       # noqa: F821
            true_offsets: T.Tensor((num_experts,), "int32"),     # noqa: F821
            C: T.Tensor((numel, N), dtype),                      # type: ignore  # noqa: F821
            tile_counter: T.Tensor((1,), "int32"),               # noqa: F821
        ):
            with T.Kernel(sm_count, threads=threads) as (pid,):
                A_shared = T.alloc_shared((block_m, block_k), dtype)
                B_shared = T.alloc_shared((block_n, block_k), dtype)
                C_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                s_cum = T.alloc_shared((num_experts + 1,), "int32")
                s_total = T.alloc_shared((1,), "int32")
                s_tile = T.alloc_shared((1,), "int32")
                s_expert = T.alloc_shared((1,), "int32")
                lo = T.alloc_local((1,), "int32")
                hi = T.alloc_local((1,), "int32")

                T.annotate_layout({
                    A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                    B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                })

                tx = T.get_thread_binding()

                if tx == 0:
                    s_cum[0] = T.int32(0)
                    for e in T.serial(num_experts):
                        s_cum[e + 1] = s_cum[e] + (true_sizes[e] + (block_m - 1)) // block_m
                    s_total[0] = s_cum[num_experts] * T.int32(num_pid_n)
                T.sync_threads()

                for _iter in T.serial(max_iters):
                    if tx == 0:
                        s_tile[0] = T.atomic_add(tile_counter[0], 1, return_prev=True)
                    T.sync_threads()

                    flat_id = s_tile[0]
                    total = s_total[0]

                    if flat_id < total:
                        if group_size_m == 1:
                            m_tile = flat_id // T.int32(num_pid_n)
                            n_tile = flat_id % T.int32(num_pid_n)
                        else:
                            num_pid_in_group = group_size_m * num_pid_n
                            m_tiles_total = s_cum[num_experts]
                            pid_in_group = flat_id % T.int32(num_pid_in_group)
                            group_id = flat_id // T.int32(num_pid_in_group)
                            first_pid_m = group_id * T.int32(group_size_m)
                            actual_gsm = T.min(m_tiles_total - first_pid_m,
                                               T.int32(group_size_m))
                            m_tile = first_pid_m + pid_in_group % actual_gsm
                            n_tile = pid_in_group // actual_gsm

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        if tx == 0:
                            s_expert[0] = lo[0]
                        T.sync_threads()
                        expert_id = s_expert[0]
                        row_in_expert = (m_tile - s_cum[expert_id]) * T.int32(block_m)
                        m_start = true_offsets[expert_id] + row_in_expert
                        n_start = n_tile * T.int32(block_n)
                        actual_rows = T.min(T.int32(block_m),
                                            true_sizes[expert_id] - row_in_expert)
                        actual_cols = T.min(T.int32(block_n),
                                            T.int32(N) - n_start)

                        T.clear(C_local)

                        for k in T.Pipelined(T.ceildiv(K, block_k),
                                             num_stages=num_stages):
                            for i, j in T.Parallel(block_m, block_k):
                                A_shared[i, j] = T.if_then_else(
                                    i < actual_rows and j < K - k * block_k,
                                    A[m_start + i, k * block_k + j], 0)
                            for i, j in T.Parallel(block_n, block_k):
                                B_shared[i, j] = T.if_then_else(
                                    j < K - k * block_k and i < actual_cols,
                                    B[expert_id, n_start + i, k * block_k + j], 0)
                            T.gemm(
                                A_shared,
                                B_shared,
                                C_local,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                            )

                        for i, j in T.Parallel(block_m, block_n):
                            if i < actual_rows and j < actual_cols:
                                C[m_start + i, n_start + j] = C_local[i, j]

    return _gemm_main


@functools.lru_cache(maxsize=64)
def _persistent_grouped_gemm_kernel_maca(
    numel: int,
    num_experts: int,
    N: int,
    K: int,
    dtype: str,
    sm_count: int,
    block_k: int,
):
    """Build a MACA persistent grouped-GEMM JIT factory."""
    log2_up = max(1, math.ceil(math.log2(num_experts + 1)))

    @tilelang.jit(
        out_idx=[],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_m, block_n, block_k, num_stages, threads, group_size_m):
        num_pid_n = math.ceil(N / block_n)
        max_tiles = numel // block_m + num_experts
        total_ctas_ub = max_tiles * num_pid_n
        max_iters = (total_ctas_ub + sm_count - 1) // sm_count + 2
        k_aligned = (K % block_k == 0)
        return _build_persistent_gemm_maca(
            numel=numel,
            num_experts=num_experts,
            N=N,
            K=K,
            dtype=dtype,
            sm_count=sm_count,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            group_size_m=group_size_m,
            log2_up=log2_up,
            num_pid_n=num_pid_n,
            max_iters=max_iters,
            num_stages=num_stages,
            k_aligned=k_aligned,
        )

    return _func


class GroupedGemmPersistentMACAKernel(Kernel):
    """MACA persistent grouped GEMM with global atomic work-stealing."""

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        numel: int,
        num_experts: int,
        N: int,
        K: int,
        dtype: torch.dtype = torch.bfloat16,
        sm_count: int | None = None,
        config=None,
        tune: bool = False,
    ):
        super().__init__()
        self.numel = numel
        self.num_experts = num_experts
        self.N = N
        self.K = K
        self.dtype = dtype
        if sm_count is None:
            sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        self.sm_count = sm_count
        self._smem_budget = device_smem_budget()
        self.init_config(config, tune)
        self._tile_counter: torch.Tensor | None = None

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for block_m in (64, 128):
            for block_n in (128, 256):
                for block_k in (64, 128):
                    for num_stages in (2, 3, 4):
                        cfg = {
                            "block_m": block_m,
                            "block_n": block_n,
                            "block_k": block_k,
                            "num_stages": num_stages,
                            "threads": 256,
                            "group_size_m": 1,
                        }
                        if _config_fits_smem(cfg, self._smem_budget):
                            configs.append(cfg)
        return configs or [self.default_config]

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        true_sizes: torch.Tensor,
        true_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if self._tile_counter is None or self._tile_counter.device != A.device:
            self._tile_counter = torch.zeros(1, dtype=torch.int32, device=A.device)
        else:
            self._tile_counter.zero_()
        C = torch.zeros(self.numel, self.N, dtype=self.dtype, device=A.device)
        block_m = self.config["block_m"]
        block_k = self.config["block_k"]
        if self.K % block_k == 0:
            A = F.pad(A, (0, 0, 0, block_m))
        gemm_fn = _persistent_grouped_gemm_kernel_maca(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.sm_count, block_k,
        )(
            block_m,
            self.config["block_n"],
            block_k,
            self.config["num_stages"],
            self.config["threads"],
            self.config.get("group_size_m", 1),
        )
        gemm_fn(A, B, true_sizes, true_offsets, C, self._tile_counter)
        return C
