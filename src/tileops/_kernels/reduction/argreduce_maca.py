"""Argreduce MACA path: N-tiled argmax/argmin under 64 KiB shared memory.

Implements a two-step kernel: first finds the extreme value via parallel reduce,
then scans for the first index matching that value.
Operates on raw ``(M, N)`` inputs; alignment padding is handled inside the
kernel via masked loads (Op layer never pads).

For large N that does not fit in shared memory, tiles over N in chunks of
``tile_n`` columns and merges per-tile extrema while preserving leftmost-index
and PyTorch NaN semantics (any NaN wins; first NaN index).

Shared-memory accounting matches the original dual-buffer layout
(``elem_bytes + 4`` per column). Full-width float32 work stays in shared
``x_f32`` (NaNs scrubbed in place); only argmin allocates a row-wide
``neg_x`` fragment — never a second full-width ``numeric`` fragment.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    MAX_SINGLE_TILE_COLS,
    align_up,
    compute_tile_n,
    device_smem_budget,
    restore_reduced,
)
from tileops._kernels.reduction.call_spec import ArgreduceCall, ArgreduceFwdInterface
from tileops._utils import is_maca

__all__ = ["ArgreduceMACAKernel"]

_ARGREDUCE_KINDS = {"argmax", "argmin"}


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _argreduce_kernel_single(M: int, N: int, op_kind: str, dtype: str):
    """Build a single-tile TileLang argmax/argmin kernel."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    _pad_fill = float("-inf") if op_kind == "argmax" else float("inf")

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_f32 = T.alloc_shared((block_m, N_padded), "float32")
                row_extreme = T.alloc_fragment((block_m,), "float32")
                out_idx = T.alloc_fragment((block_m,), "int64")
                nan_idx = T.alloc_fragment((block_m,), "int64")

                if _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_pad_fill, "float32"),
                            )
                else:
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                # First NaN wins (PyTorch); sentinel N means no NaN in the row.
                T.fill(nan_idx, T.cast(N, "int64"))
                for i in T.Parallel(block_m):
                    for j in T.Serial(N):
                        if T.isnan(x_f32[i, j]):
                            nan_idx[i] = T.cast(j, "int64")
                            T.loop_break()

                # Scrub NaNs in shared so reduce_max stays stable (no extra fragment).
                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = T.if_then_else(
                        T.isnan(x_f32[i, j]),
                        T.cast(_pad_fill, "float32"),
                        x_f32[i, j],
                    )

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

                for i in T.Parallel(block_m):
                    out_idx[i] = T.if_then_else(
                        nan_idx[i] < T.cast(N, "int64"),
                        nan_idx[i],
                        out_idx[i],
                    )

                T.copy(out_idx, out[pid_m * block_m])

        return main

    return _func


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=64
)
def _argreduce_kernel_tiled(M: int, N: int, op_kind: str, dtype: str, tile_n: int):
    """Build a multi-tile argmax/argmin kernel."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    total_cols = num_tiles * tile_n
    _needs_mask = total_cols > N
    _pad_fill = float("-inf") if op_kind == "argmax" else float("inf")
    _is_argmax = op_kind == "argmax"

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        if _is_argmax:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out: T.Tensor[(M,), "int64"],  # noqa: F821
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    x_f32 = T.alloc_shared((block_m, tile_n), "float32")
                    tile_extreme = T.alloc_fragment((block_m,), "float32")
                    tile_idx = T.alloc_fragment((block_m,), "int64")
                    tile_nan_idx = T.alloc_fragment((block_m,), "int64")
                    row_extreme = T.alloc_fragment((block_m,), "float32")
                    out_idx = T.alloc_fragment((block_m,), "int64")
                    initialized = T.alloc_fragment((block_m,), "int32")

                    T.fill(row_extreme, -T.infinity("float32"))
                    T.fill(out_idx, T.cast(0, "int64"))
                    T.fill(initialized, 0)

                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                with T.Else():
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            x_f32[i, j] = T.if_then_else(
                                                T.And(
                                                    pid_m * block_m + i < M,
                                                    t * tile_n + j < N,
                                                ),
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                ),
                                                T.cast(_pad_fill, "float32"),
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                            for i, j in T.Parallel(block_m, tile_n):
                                x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.fill(tile_nan_idx, T.cast(N, "int64"))
                        for i in T.Parallel(block_m):
                            for j in T.Serial(tile_n):
                                if t * tile_n + j < N and T.isnan(x_f32[i, j]):
                                    tile_nan_idx[i] = T.cast(t * tile_n + j, "int64")
                                    T.loop_break()

                        for i, j in T.Parallel(block_m, tile_n):
                            x_f32[i, j] = T.if_then_else(
                                T.isnan(x_f32[i, j]),
                                T.cast(_pad_fill, "float32"),
                                x_f32[i, j],
                            )

                        T.fill(tile_extreme, -T.infinity("float32"))
                        T.reduce_max(x_f32, tile_extreme, dim=1, clear=False)

                        T.fill(tile_idx, T.cast(0, "int64"))
                        for i in T.Parallel(block_m):
                            for j in T.Serial(tile_n):
                                if t * tile_n + j < N and x_f32[i, j] == tile_extreme[i]:
                                    tile_idx[i] = T.cast(t * tile_n + j, "int64")
                                    T.loop_break()

                        for i in T.Parallel(block_m):
                            tile_idx[i] = T.if_then_else(
                                tile_nan_idx[i] < T.cast(N, "int64"),
                                tile_nan_idx[i],
                                tile_idx[i],
                            )
                            tile_extreme[i] = T.if_then_else(
                                tile_nan_idx[i] < T.cast(N, "int64"),
                                T.cast(float("nan"), "float32"),
                                tile_extreme[i],
                            )

                        for i in T.Parallel(block_m):
                            tile_is_nan = T.isnan(tile_extreme[i])
                            row_is_nan = T.isnan(row_extreme[i])
                            numeric_better = (
                                (not tile_is_nan)
                                and (not row_is_nan)
                                and tile_extreme[i] > row_extreme[i]
                            )
                            take_tile = (
                                (initialized[i] == 0)
                                or (tile_is_nan and not row_is_nan)
                                or (tile_is_nan and row_is_nan and tile_idx[i] < out_idx[i])
                                or numeric_better
                            )
                            out_idx[i] = T.if_then_else(take_tile, tile_idx[i], out_idx[i])
                            row_extreme[i] = T.if_then_else(
                                take_tile, tile_extreme[i], row_extreme[i]
                            )
                            initialized[i] = 1

                    T.copy(out_idx, out[pid_m * block_m])

        else:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out: T.Tensor[(M,), "int64"],  # noqa: F821
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    x_f32 = T.alloc_shared((block_m, tile_n), "float32")
                    neg_x = T.alloc_fragment((block_m, tile_n), "float32")
                    tile_extreme = T.alloc_fragment((block_m,), "float32")
                    tile_idx = T.alloc_fragment((block_m,), "int64")
                    tile_nan_idx = T.alloc_fragment((block_m,), "int64")
                    row_extreme = T.alloc_fragment((block_m,), "float32")
                    out_idx = T.alloc_fragment((block_m,), "int64")
                    initialized = T.alloc_fragment((block_m,), "int32")

                    T.fill(row_extreme, T.infinity("float32"))
                    T.fill(out_idx, T.cast(0, "int64"))
                    T.fill(initialized, 0)

                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                with T.Else():
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            x_f32[i, j] = T.if_then_else(
                                                T.And(
                                                    pid_m * block_m + i < M,
                                                    t * tile_n + j < N,
                                                ),
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                ),
                                                T.cast(_pad_fill, "float32"),
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                            for i, j in T.Parallel(block_m, tile_n):
                                x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.fill(tile_nan_idx, T.cast(N, "int64"))
                        for i in T.Parallel(block_m):
                            for j in T.Serial(tile_n):
                                if t * tile_n + j < N and T.isnan(x_f32[i, j]):
                                    tile_nan_idx[i] = T.cast(t * tile_n + j, "int64")
                                    T.loop_break()

                        for i, j in T.Parallel(block_m, tile_n):
                            x_f32[i, j] = T.if_then_else(
                                T.isnan(x_f32[i, j]),
                                T.cast(_pad_fill, "float32"),
                                x_f32[i, j],
                            )

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
                            tile_idx[i] = T.if_then_else(
                                tile_nan_idx[i] < T.cast(N, "int64"),
                                tile_nan_idx[i],
                                tile_idx[i],
                            )
                            tile_extreme[i] = T.if_then_else(
                                tile_nan_idx[i] < T.cast(N, "int64"),
                                T.cast(float("nan"), "float32"),
                                tile_extreme[i],
                            )

                        for i in T.Parallel(block_m):
                            tile_is_nan = T.isnan(tile_extreme[i])
                            row_is_nan = T.isnan(row_extreme[i])
                            numeric_better = (
                                (not tile_is_nan)
                                and (not row_is_nan)
                                and tile_extreme[i] < row_extreme[i]
                            )
                            take_tile = (
                                (initialized[i] == 0)
                                or (tile_is_nan and not row_is_nan)
                                or (tile_is_nan and row_is_nan and tile_idx[i] < out_idx[i])
                                or numeric_better
                            )
                            out_idx[i] = T.if_then_else(take_tile, tile_idx[i], out_idx[i])
                            row_extreme[i] = T.if_then_else(
                                take_tile, tile_extreme[i], row_extreme[i]
                            )
                            initialized[i] = 1

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


class ArgreduceMACAKernel(Kernel, ArgreduceFwdInterface):
    """Argmax / argmin forward kernel for MACA (64 KiB smem, N-tiled path)."""

    devices = frozenset({"cuda"})
    supported_archs: list[int] = [80, 86, 89, 90]
    preferred_over = frozenset({"argreduce", "argreduce_split", "argreduce_strided"})
    # Contiguous last-axis reduction; no CUDA-style output/warp/cta strategy.
    strategy: str = "contiguous"

    @classmethod
    def refusal(cls, call: ArgreduceCall) -> str | None:
        if not is_maca():
            return "serves MACA devices only"
        if len(call.axes) != 1 or call.inner_stride != 1:
            return "serves one contiguous last-axis reduction"
        return super().refusal(call)

    def __init__(self, call: ArgreduceCall, config: Optional[dict] = None):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        if call.op_kind not in _ARGREDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{call.op_kind}'. Expected one of {sorted(_ARGREDUCE_KINDS)}."
            )
        self.M = call.m
        self.N = call.n
        self.op_kind = call.op_kind
        self.dtype = call.dtype
        self.reduce_axes = tuple(call.axes)
        self.keepdim = call.keepdim
        self.strategy = "contiguous"
        self.N_padded = align_up(self.N, DEFAULT_ALIGNMENT)
        self._elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        self._combined_bytes = self._elem_bytes + 4
        # Reserve one alignment stripe of the dual shared buffers so tile_n
        # never consumes the full 65536-byte device budget (dtype-scaled).
        self._smem_budget = max(
            device_smem_budget() - DEFAULT_ALIGNMENT * self._combined_bytes,
            DEFAULT_ALIGNMENT * self._combined_bytes,
        )

        defaults = self.default_config
        requested = config or {}
        block_m = requested.get("block_m", defaults["block_m"])
        self._tile_n = requested.get("tile_n")
        if self._tile_n is None:
            self._tile_n = self._tile_n_for_block_m(block_m)
        config = {**defaults, **requested, "tile_n": self._tile_n}
        self.kernel = _argreduce_kernel_maca(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self._tile_n,
        )
        self.init_config(config)

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
        self._require_devices(x=x)
        in_shape = tuple(x.shape)
        # The TileLang program is specialized for a two-dimensional [M, N]
        # row layout; the Op hands kernels the original declared tensor.
        x_rows = x.reshape(self.M, self.N)
        result = _argreduce_fwd_wrapped_maca(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            self.config["tile_n"],
            x_rows,
        )
        if self.reduce_axes:
            return restore_reduced(result, in_shape, self.reduce_axes, self.keepdim)
        return result
