"""Argreduce MACA path: N-tiled argmax/argmin under 64 KiB shared memory.

Implements a two-step kernel: first finds the extreme value via parallel reduce,
then scans for the first index matching that value.
Operates on 2D (M, N_padded) tensors; the Op layer handles reshape.

For large N that does not fit in shared memory, tiles over N in chunks of
``tile_n`` columns and merges per-tile extrema while preserving leftmost-index
semantics.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    MAX_SINGLE_TILE_COLS,
    align_up,
    compute_tile_n,
    device_smem_budget,
)

__all__ = ["ArgreduceMACAKernel"]

_ARGREDUCE_KINDS = {"argmax", "argmin"}


@functools.lru_cache(maxsize=32)
def _argreduce_kernel_single(M: int, N: int, op_kind: str, dtype: str):
    """Build a single-tile TileLang argmax/argmin kernel."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_f32 = T.alloc_shared((block_m, N_padded), "float32")
                row_extreme = T.alloc_fragment((block_m,), "float32")
                out_idx = T.alloc_fragment((block_m,), "int64")

                T.copy(x[pid_m * block_m, 0], shared_buf)

                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                if op_kind == "argmax":
                    T.fill(row_extreme, -T.infinity("float32"))
                    T.reduce_max(x_f32, row_extreme, dim=1, clear=False)
                else:
                    neg_x = T.alloc_fragment((block_m, N_padded), "float32")
                    for i, j in T.Parallel(block_m, N_padded):
                        neg_x[i, j] = -x_f32[i, j]
                    T.fill(row_extreme, -T.infinity("float32"))
                    T.reduce_max(neg_x, row_extreme, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        row_extreme[i] = -row_extreme[i]

                T.fill(out_idx, T.cast(0, "int64"))
                for i in T.Parallel(block_m):
                    for j in T.Serial(N):
                        if x_f32[i, j] == row_extreme[i]:
                            out_idx[i] = T.cast(j, "int64")
                            T.loop_break()

                T.copy(out_idx, out[pid_m * block_m])

        return main

    return _func


@functools.lru_cache(maxsize=64)
def _argreduce_kernel_tiled(M: int, N: int, op_kind: str, dtype: str, tile_n: int):
    """Build a multi-tile argmax/argmin kernel."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        if op_kind == "argmax":

            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                out: T.Tensor[(M,), "int64"],  # noqa: F821
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    x_f32 = T.alloc_shared((block_m, tile_n), "float32")
                    tile_extreme = T.alloc_fragment((block_m,), "float32")
                    tile_idx = T.alloc_fragment((block_m,), "int64")
                    row_extreme = T.alloc_fragment((block_m,), "float32")
                    out_idx = T.alloc_fragment((block_m,), "int64")

                    T.fill(row_extreme, -T.infinity("float32"))
                    T.fill(out_idx, T.cast(0, "int64"))

                    for t in T.Serial(num_tiles):
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        for i, j in T.Parallel(block_m, tile_n):
                            x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.fill(tile_extreme, -T.infinity("float32"))
                        T.reduce_max(x_f32, tile_extreme, dim=1, clear=False)

                        T.fill(tile_idx, T.cast(0, "int64"))
                        for i in T.Parallel(block_m):
                            for j in T.Serial(tile_n):
                                if t * tile_n + j < N and x_f32[i, j] == tile_extreme[i]:
                                    tile_idx[i] = T.cast(t * tile_n + j, "int64")
                                    T.loop_break()

                        for i in T.Parallel(block_m):
                            row_extreme[i] = T.if_then_else(
                                tile_extreme[i] > row_extreme[i],
                                tile_extreme[i],
                                row_extreme[i],
                            )
                            out_idx[i] = T.if_then_else(
                                tile_extreme[i] > row_extreme[i],
                                tile_idx[i],
                                out_idx[i],
                            )

                    T.copy(out_idx, out[pid_m * block_m])

        else:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                out: T.Tensor[(M,), "int64"],  # noqa: F821
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    x_f32 = T.alloc_shared((block_m, tile_n), "float32")
                    neg_x = T.alloc_fragment((block_m, tile_n), "float32")
                    tile_extreme = T.alloc_fragment((block_m,), "float32")
                    tile_idx = T.alloc_fragment((block_m,), "int64")
                    row_extreme = T.alloc_fragment((block_m,), "float32")
                    out_idx = T.alloc_fragment((block_m,), "int64")

                    T.fill(row_extreme, T.infinity("float32"))
                    T.fill(out_idx, T.cast(0, "int64"))

                    for t in T.Serial(num_tiles):
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        for i, j in T.Parallel(block_m, tile_n):
                            x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        for i, j in T.Parallel(block_m, tile_n):
                            neg_x[i, j] = -x_f32[i, j]
                        T.fill(tile_extreme, -T.infinity("float32"))
                        T.reduce_max(neg_x, tile_extreme, dim=1, clear=False)
                        for i in T.Parallel(block_m):
                            tile_extreme[i] = -tile_extreme[i]

                        T.fill(tile_idx, T.cast(0, "int64"))
                        for i in T.Parallel(block_m):
                            for j in T.Serial(tile_n):
                                if t * tile_n + j < N and x_f32[i, j] == tile_extreme[i]:
                                    tile_idx[i] = T.cast(t * tile_n + j, "int64")
                                    T.loop_break()

                        for i in T.Parallel(block_m):
                            row_extreme[i] = T.if_then_else(
                                tile_extreme[i] < row_extreme[i],
                                tile_extreme[i],
                                row_extreme[i],
                            )
                            out_idx[i] = T.if_then_else(
                                tile_extreme[i] < row_extreme[i],
                                tile_idx[i],
                                out_idx[i],
                            )

                    T.copy(out_idx, out[pid_m * block_m])

        return main

    return _func


def _argreduce_kernel_maca(M: int, N: int, op_kind: str, dtype: str, tile_n: int = 0):
    """Build the appropriate MACA argmax/argmin kernel."""
    if tile_n == 0:
        return _argreduce_kernel_single(M, N, op_kind, dtype)
    return _argreduce_kernel_tiled(M, N, op_kind, dtype, tile_n)


@torch.library.custom_op("top::argreduce_fwd_maca", mutates_args=())
def _argreduce_fwd_wrapped_maca(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    threads: int,
    tile_n: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _argreduce_kernel_maca(M, N, op_kind, dtype_str, tile_n)(block_m, threads)(x)


@_argreduce_fwd_wrapped_maca.register_fake
def _argreduce_fwd_wrapped_maca_fake(M, N, op_kind, dtype_str, block_m, threads, tile_n, x):
    return torch.empty((M,), dtype=torch.int64, device=x.device)


class ArgreduceMACAKernel(Kernel):
    """Argmax / argmin forward kernel for MACA (64 KiB smem, N-tiled path)."""

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        if op_kind not in _ARGREDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_ARGREDUCE_KINDS)}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = torch.tensor([], dtype=dtype).element_size()
        self._combined_bytes = self._elem_bytes + 4
        self._smem_budget = device_smem_budget()

        self._tile_n = self.default_config["tile_n"]
        self.kernel = _argreduce_kernel_maca(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self._tile_n,
        )
        self.init_config(config, tune)

        if not tune:
            caller_tile_n = config.get("tile_n") if config is not None else None
            if caller_tile_n is not None:
                target_tile_n = caller_tile_n
            else:
                target_tile_n = self._tile_n_for_block_m(self.config["block_m"])
            if target_tile_n != self._tile_n:
                self._tile_n = target_tile_n
                self.kernel = _argreduce_kernel_maca(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.dtype_str,
                    self._tile_n,
                )
            self.config["tile_n"] = self._tile_n

    def _tile_n_for_block_m(self, block_m: int) -> int:
        """Return tile_n for a given block_m (0 means no tiling needed)."""
        budget = self._smem_budget
        if self.N_padded <= MAX_SINGLE_TILE_COLS:
            single = compute_tile_n(
                block_m,
                self._combined_bytes,
                self.N_padded,
                budget=budget,
            )
            if single == self.N_padded:
                return 0
        col_budget = MAX_SINGLE_TILE_COLS * block_m * self._combined_bytes
        effective_budget = min(budget, col_budget)
        return compute_tile_n(
            block_m,
            self._combined_bytes,
            self.N_padded,
            budget=effective_budget,
        )

    def _heuristic_tile_n(self) -> int:
        """Return the default tile_n for the tiled path (always > 0)."""
        best_tile_n = self._tile_n_for_block_m(1)
        for bm in [2, 4, 8]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            best_num = (self.N_padded + best_tile_n - 1) // best_tile_n
            curr_num = (self.N_padded + tn - 1) // tn
            if curr_num < best_num:
                best_tile_n = tn
        return best_tile_n

    def _single_tile_default_config(self) -> dict:
        """Default config when the full row fits in shared memory."""
        smem_per_row = self.N_padded * self._combined_bytes
        budget = self._smem_budget
        max_block_m_smem = budget // smem_per_row
        threads = 128
        max_block_m = max_block_m_smem
        if self.N < DEFAULT_ALIGNMENT:
            max_block_m_layout = (2 * threads) // self.N_padded
            max_block_m = min(max_block_m_smem, max(max_block_m_layout, 1))
        block_m = 1
        for bm in [1, 2, 4, 8]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": threads, "tile_n": 0}

    @property
    def default_config(self) -> dict:
        """Select default block_m and tile_n based on shared memory budget."""
        if self.N_padded == 0:
            raise ValueError(
                "Reduction dimension is empty (N=0). "
                "argmax/argmin over an empty dimension is undefined."
            )
        if self._tile_n_for_block_m(1) == 0:
            return self._single_tile_default_config()

        best_bm = 1
        best_tile_n = self._tile_n_for_block_m(1)
        for bm in [2, 4, 8]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            best_num = (self.N_padded + best_tile_n - 1) // best_tile_n
            curr_num = (self.N_padded + tn - 1) // tn
            if curr_num < best_num:
                best_bm = bm
                best_tile_n = tn
        return {"block_m": best_bm, "threads": 128, "tile_n": best_tile_n}

    @property
    def autotune_configs(self) -> list[dict]:
        if self.N_padded == 0:
            raise ValueError(
                "Reduction dimension is empty (N=0). "
                "argmax/argmin over an empty dimension is undefined."
            )
        if self._tile_n_for_block_m(1) == 0:
            smem_per_row = self.N_padded * self._combined_bytes
            budget = self._smem_budget
            max_block_m_smem = budget // smem_per_row
            threads_list = [128, 256]
            configs = []
            for threads in threads_list:
                max_block_m = max_block_m_smem
                if self.N < DEFAULT_ALIGNMENT:
                    max_block_m_layout = (2 * threads) // self.N_padded
                    max_block_m = min(max_block_m_smem, max(max_block_m_layout, 1))
                for bm in [1, 2, 4, 8]:
                    if bm <= max_block_m:
                        configs.append({"block_m": bm, "threads": threads, "tile_n": 0})
            return configs

        fixed_tile_n = self._heuristic_tile_n()
        threads_list = [128, 256]
        configs = []
        for threads in threads_list:
            for bm in [1, 2, 4, 8]:
                try:
                    tn = self._tile_n_for_block_m(bm)
                except ValueError:
                    continue
                if tn == fixed_tile_n:
                    configs.append({"block_m": bm, "threads": threads, "tile_n": tn})
        return configs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the argmax/argmin kernel."""
        return _argreduce_fwd_wrapped_maca(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            self.config["tile_n"],
            x,
        )
