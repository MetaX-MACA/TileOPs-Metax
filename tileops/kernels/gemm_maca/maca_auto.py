from __future__ import annotations

from typing import Optional

import torch

from tileops.kernels.gemm import GemmKernel
from tileops.kernels.kernel_base import Kernel


class MacaAutoGemmKernel(Kernel):
    """Compatibility wrapper for the TileLang compiler GEMM backend."""

    supported_archs: list[int] = GemmKernel.supported_archs

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
        self.inner = self._select_inner(m, n, k, dtype, config, tune, trans_a, trans_b)
        self.config = {
            "backend": "maca_auto",
            "selected_backend": self.inner.__class__.__name__,
            "selected_config": getattr(self.inner, "config", {}),
        }
        print(f"{self.__class__.__name__} initialized with config: {self.config}")

    def _select_inner(
            self,
            m: int,
            n: int,
            k: int,
            dtype: torch.dtype,
            config: Optional[dict],
            tune: bool,
            trans_a: bool,
            trans_b: bool,
    ) -> Kernel:
        return GemmKernel(m, n, k, dtype, config=config, tune=tune, trans_a=trans_a,
                          trans_b=trans_b)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.inner(a, b)

    def _inner_method(self, name: str, capability: str):
        try:
            return getattr(self.inner, name)
        except AttributeError:
            raise NotImplementedError(
                f"{self.inner.__class__.__name__} does not expose {capability}") from None

    def prepare_b(self, b: torch.Tensor) -> torch.Tensor:
        return self._inner_method("prepare_b", "prepared-B support")(b)

    def prepare_a(self, a: torch.Tensor) -> torch.Tensor:
        return self._inner_method("prepare_a", "prepared-A support")(a)

    def forward_with_prepared_b(self, a: torch.Tensor, b_prepared: torch.Tensor) -> torch.Tensor:
        return self._inner_method("forward_with_prepared_b", "prepared-B execution")(
            a, b_prepared)

    def forward_with_prepared_a_and_b(self, a_prepared: torch.Tensor,
                                      b_prepared: torch.Tensor) -> torch.Tensor:
        return self._inner_method(
            "forward_with_prepared_a_and_b", "prepared-A/B execution")(a_prepared, b_prepared)
