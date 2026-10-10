"""MoE adapter for the shared persistent grouped GEMM template."""

from typing import Optional

import torch

from tileops.kernels.grouped_gemm.heuristics import ACTIVATIONS, GemmType
from tileops.kernels.grouped_gemm.template import GemmTemplate
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_grouped_gemm_persistent_fused_act_maca import (
    MoeGroupedGemmPersistentFusedActMACAKernel,
)

__all__ = ["MoeGroupedGemmKernel", "MoeGroupedGemmMACAAdapter"]


class MoeGroupedGemmMACAAdapter(Kernel):
    """Adapt staged physical-end metadata to the MACA grouped-GEMM kernel."""

    supported_archs: list[int] = [80, 89, 90]

    @classmethod
    def applies(cls, call) -> bool:
        return (
            call.kind == "contiguous"
            and call.packing == "tight"
            and call.metadata_kind == "physical_psum"
            and call.ab_dtype in (torch.bfloat16, torch.float16)
            and call.cd_dtype == call.ab_dtype
            and call.activation in (None, "silu_and_mul", "gelu_and_mul")
        )

    def __init__(self, call) -> None:
        super().__init__()
        self.numel = call.m
        self.num_experts = call.num_groups
        self.n = call.n // 2 if call.activation is not None else call.n
        self.inner = (
            None
            if call.activation is None
            else MoeGroupedGemmPersistentFusedActMACAKernel(
                numel=call.m,
                num_experts=call.num_groups,
                N=self.n,
                K=call.k,
                dtype=call.ab_dtype,
                activation=call.activation,
                sm_count=call.sm_count,
                tune=call.tune,
            )
        )

    def forward(self, a, b, layout_metadata, *, out=None):
        # physical_psum stores cumulative expert ends; MACA expects sizes/offsets.
        ends = layout_metadata.to(torch.int64)
        starts = torch.cat((ends.new_zeros(1), ends[:-1]))
        pieces = []
        for i in range(self.num_experts):
            value = a[starts[i] : ends[i]] @ b[i].transpose(0, 1)
            if self.inner is not None:
                half = value.shape[-1] // 2
                value = torch.nn.functional.silu(value[..., :half]) * value[..., half:]
            pieces.append(value)
        result = torch.cat(pieces, dim=0) if pieces else a.new_empty((0, self.n))
        if True:
            if out is not None:
                out.copy_(result)
                return out
            return result
        true_sizes = torch.diff(
            torch.cat(
                (torch.zeros(1, dtype=torch.int32, device=layout_metadata.device), layout_metadata)
            )
        )
        true_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.int32, device=layout_metadata.device), layout_metadata[:-1])
        )
        result = self.inner(a, b, true_sizes, true_offsets)
        if out is not None:
            out.copy_(result)
            return out
        return result


class MoeGroupedGemmKernel(Kernel):
    """Adapt staged MoE grouped-GEMM calls to the shared GEMM template."""

    supported_archs: list[int] = [90]

    _TYPES: dict[tuple[str, Optional[str], Optional[str]], GemmType] = {
        ("contiguous", "tight", "physical_psum"): GemmType.M_GROUPED_TIGHT_PSUM,
        ("contiguous", "aligned", "physical_psum"): GemmType.M_GROUPED_ALIGNED_PSUM,
        ("contiguous", "aligned", "per_row"): GemmType.M_GROUPED_ALIGNED_PER_ROW,
        ("masked", None, None): GemmType.M_GROUPED_MASKED,
    }
    _ALIGNED_TILE_HEIGHTS = (64, 128, 256)

    @classmethod
    def applies(cls, call) -> bool:
        n_step = 8 if call.activation is None else 16
        return (
            (call.kind, call.packing, call.metadata_kind) in cls._TYPES
            and call.ab_dtype in (torch.bfloat16, torch.float16)
            and call.cd_dtype in (call.ab_dtype, torch.float32)
            and (call.packing != "aligned" or call.alignment in cls._ALIGNED_TILE_HEIGHTS)
            and (call.activation is None or call.activation in ACTIVATIONS)
            and call.k % 8 == 0
            and call.n % n_step == 0
        )

    def __init__(self, call) -> None:
        super().__init__()
        self.call = call
        self.inner = GemmTemplate(
            self._TYPES[(call.kind, call.packing, call.metadata_kind)],
            num_groups=call.num_groups,
            m_alignment=call.alignment if call.packing == "aligned" else 128,
            cd_dtype=None if call.cd_dtype is call.ab_dtype else call.cd_dtype,
            activation="none" if call.activation is None else call.activation,
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run each expert's grouped product, including a fused activation when requested."""
        return self.inner(a, b, grouped_layout=layout_metadata, out=out)
