import sys
import types
from types import SimpleNamespace

import pytest
import torch

from tileops.kernels.gemm import GemmKernel
from tileops.kernels.gemm_maca import maca_hgemm as maca_hgemm_module
from tileops.kernels.gemm_maca.maca_hgemm import (
    MacaHGemmKernel,
    _compile_flags,
    _reference_layout_repo,
    _reference_muxi_layout_kernels,
)
from tileops.ops.gemm import GemmOp, _select_gemm_kernel
from tileops.utils import is_metax_c500


def _patch_metax_c500(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(name="MetaX C500"),
    )


def _install_fake_backend_module(
        monkeypatch: pytest.MonkeyPatch,
        module_name: str,
        class_name: str,
) -> type:
    module = types.ModuleType(module_name)

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.config = {"backend": class_name}

    fake_backend = type(
        class_name,
        (),
        {
            "supported_archs": [80],
            "__init__": __init__,
        },
    )
    setattr(module, class_name, fake_backend)
    monkeypatch.setitem(sys.modules, module_name, module)
    return fake_backend


def _install_fake_backend_module_with_prepacked_api(
        monkeypatch: pytest.MonkeyPatch,
        module_name: str,
        class_name: str,
) -> type:
    module = types.ModuleType(module_name)

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.config = {"backend": class_name}

    def prepare_b(self, b: torch.Tensor) -> torch.Tensor:
        self.prepared_b = b
        return b + 1

    def forward_with_prepared_b(self, a: torch.Tensor,
                                b_prepared: torch.Tensor) -> torch.Tensor:
        self.forward_args = (a, b_prepared)
        return b_prepared - 1

    fake_backend = type(
        class_name,
        (),
        {
            "supported_archs": [80],
            "__init__": __init__,
            "prepare_b": prepare_b,
            "forward_with_prepared_b": forward_with_prepared_b,
        },
    )
    setattr(module, class_name, fake_backend)
    monkeypatch.setitem(sys.modules, module_name, module)
    return fake_backend


def _install_fake_reference_layout_ab_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("muxi_layout_kernels")
    module.calls = []

    def gemm_layoutAB_ContinuousC(a: torch.Tensor, b: torch.Tensor,
                                  alpha: float, beta: float,
                                  bias: object = None) -> torch.Tensor:
        module.calls.append((tuple(a.shape), tuple(b.shape), alpha, beta, bias))
        m = a.shape[0] * 16
        n = b.shape[1] * 16
        return torch.full((n, m), 7.0, dtype=a.dtype, device=a.device)

    module.gemm_layoutAB_ContinuousC = gemm_layoutAB_ContinuousC
    monkeypatch.setitem(sys.modules, "muxi_layout_kernels", module)
    return module


def _install_fake_reference_layout_a_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("muxi_layout_kernels")
    module.calls = []

    def muxi_hgemm_layoutA(a: torch.Tensor, b: torch.Tensor,
                           alpha: float, beta: float) -> torch.Tensor:
        module.calls.append((tuple(a.shape), tuple(b.shape), alpha, beta))
        m = a.shape[0] * 16
        n = b.shape[0]
        return torch.full((n, m), 11.0, dtype=a.dtype, device=a.device)

    module.muxi_hgemm_layoutA = muxi_hgemm_layoutA
    monkeypatch.setitem(sys.modules, "muxi_layout_kernels", module)
    return module


@pytest.mark.smoke
def test_is_metax_c500(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)

    assert is_metax_c500()


@pytest.mark.smoke
def test_gemm_default_selector_uses_tilelang_compiler_backend_on_c500(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.delenv("TILEOPS_GEMM_BACKEND", raising=False)

    assert _select_gemm_kernel() is GemmKernel


@pytest.mark.smoke
@pytest.mark.parametrize("backend", ["maca_hgemm", "maca_auto"])
def test_gemm_selector_rejects_direct_hpp_backends(
        monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_GEMM_BACKEND", backend)

    with pytest.raises(RuntimeError, match="TileLang DSL/compiler"):
        _select_gemm_kernel()


@pytest.mark.smoke
def test_maca_hgemm_extra_compile_flags_are_appended(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_EXTRA_COMPILE_FLAGS", "-opt-info -lineinfo")

    flags = _compile_flags()

    assert flags[-2:] == ["-opt-info", "-lineinfo"]


@pytest.mark.smoke
def test_maca_hgemm_example_compile_profile_uses_minimal_flag_set(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_COMPILE_PROFILE", "example")

    flags = _compile_flags()

    assert flags == [
        "-O3",
        "-DENABLE_BF16",
        "-mllvm -metaxgpu-direct-address=disshared",
        "-mllvm -metaxgpu-force-global-saddr=1",
        "-gcc-version 11",
        f"-include {maca_hgemm_module._WRAPPER_HEADER}",
    ]


@pytest.mark.smoke
def test_maca_hgemm_mma_sched_profile_uses_mctriton_minreg_pipeline(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_COMPILE_PROFILE", "mma_sched")

    flags = _compile_flags()

    assert flags == [
        "-O3",
        "-DENABLE_BF16",
        "-mllvm -metaxgpu-direct-address=disshared",
        "-mllvm -metaxgpu-force-global-saddr=1",
        "-mllvm -metaxgpu-mma-sched=true",
        "-mllvm -metaxgpu-sched-select=metaxgpu-minreg",
        "-mllvm -map-use-pk-fma=1",
        "-gcc-version 11",
        f"-include {maca_hgemm_module._WRAPPER_HEADER}",
    ]


@pytest.mark.smoke
def test_maca_hgemm_async_a_macro_gate_is_disabled(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_USE_ASYNC_A_STAGING", "1")

    with pytest.raises(RuntimeError, match="register-resident body rewrite"):
        _compile_flags()


@pytest.mark.smoke
def test_maca_hgemm_example_async_a_staging_gate_is_disabled(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_USE_EXAMPLE_ASYNC_A_STAGING", "1")

    with pytest.raises(RuntimeError, match="register-resident body rewrite"):
        _compile_flags()


@pytest.mark.smoke
def test_maca_hgemm_reference_repo_prefers_workspace_sibling(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    repo_file = (
        tmp_path / "TileOPs-Metax" / "tileops" / "kernels" / "gemm_maca" / "maca_hgemm.py"
    )
    repo_file.parent.mkdir(parents=True)
    repo_file.touch()
    reference_repo = tmp_path / "muxi_native_layout_kernels"
    reference_repo.mkdir()
    monkeypatch.delenv("TILEOPS_MACA_HGEMM_REFERENCE_REPO", raising=False)
    monkeypatch.setattr(maca_hgemm_module, "__file__", str(repo_file))

    assert _reference_layout_repo() == reference_repo


@pytest.mark.smoke
def test_maca_hgemm_experimental_rowa_layout_b_gate_is_disabled(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_EXPERIMENTAL_ROWA_LAYOUT_B_BODY", "1")

    with pytest.raises(RuntimeError, match="disabled"):
        MacaHGemmKernel(128, 128, 128, dtype=torch.float16, tune=False)


@pytest.mark.smoke
def test_gemm_op_auto_routes_to_tilelang_compiler_backend_on_c500_fp16(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.delenv("TILEOPS_GEMM_BACKEND", raising=False)

    op = GemmOp(128, 128, 128, dtype=torch.float16, tune=False)

    assert isinstance(op.kernel, GemmKernel)


@pytest.mark.smoke
def test_gemm_op_auto_dispatch_exposes_compiler_prepared_b_path(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.delenv("TILEOPS_GEMM_BACKEND", raising=False)

    op = GemmOp(2, 3, 4, dtype=torch.float16, tune=False)
    b = torch.arange(12, dtype=torch.float16).reshape(4, 3)

    prepared_b = op.prepare_b(b)

    assert isinstance(op.kernel, GemmKernel)
    assert prepared_b.shape == (3, 4)
    assert torch.equal(prepared_b, b.transpose(0, 1).contiguous())


@pytest.mark.smoke
def test_gemm_kernel_prefers_maca_bsm_path_on_aligned_c500_fp16(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    kernel = GemmKernel(128, 128, 128, dtype=torch.float16, tune=False)

    assert kernel._use_maca_bsm_path is True
    assert kernel._use_col_major_output is False
    assert kernel.config == {
        "block_m": 128,
        "block_n": 128,
        "block_k": 128,
        "num_stages": 0,
        "threads": 256,
        "enable_rasterization": True,
    }


@pytest.mark.smoke
def test_gemm_kernel_prepare_b_reuses_native_cache(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    kernel = GemmKernel(2, 3, 4, dtype=torch.float16, tune=False)
    b = torch.arange(12, dtype=torch.float16).reshape(4, 3)

    prepared_first = kernel.prepare_b(b)
    prepared_second = kernel.prepare_b(b)

    assert prepared_first is prepared_second


@pytest.mark.smoke
def test_gemm_kernel_prepare_b_can_pack_bsm_tile_layout(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_GEMM_PACKED_B_TILE", "1")
    monkeypatch.delenv("TILEOPS_GEMM_SPLIT_K", raising=False)
    kernel = GemmKernel(128, 256, 128, dtype=torch.float16, tune=False)
    b = torch.arange(128 * 256, dtype=torch.float16).reshape(128, 256)

    prepared_b = kernel.prepare_b(b)
    expected = b.transpose(0, 1).contiguous().view(
        2,
        128,
        1,
        128,
    ).permute(0, 2, 1, 3).contiguous()

    assert kernel._use_maca_bsm_path is True
    assert kernel._use_packed_b_tile_path is True
    assert prepared_b.shape == (2, 1, 128, 128)
    assert torch.equal(prepared_b, expected)


@pytest.mark.smoke
def test_gemm_kernel_prepare_b_can_pack_splitk_bsm_tile_layout(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_GEMM_PACKED_B_TILE", "1")
    monkeypatch.setenv("TILEOPS_GEMM_SPLIT_K", "2")
    kernel = GemmKernel(128, 128, 256, dtype=torch.float16, tune=False)
    b = torch.arange(256 * 128, dtype=torch.float16).reshape(256, 128)

    prepared_b = kernel.prepare_b(b)
    expected = b.transpose(0, 1).contiguous().view(
        1,
        128,
        4,
        64,
    ).permute(0, 2, 1, 3).contiguous()

    assert kernel._use_split_k_path is True
    assert kernel._use_packed_b_tile_path is True
    assert prepared_b.shape == (1, 4, 128, 64)
    assert torch.equal(prepared_b, expected)


@pytest.mark.smoke
def test_gemm_kernel_can_select_packed_b_async_pipeline(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_GEMM_PACKED_B_TILE", "1")
    monkeypatch.setenv("TILEOPS_GEMM_PACKED_B_ASYNC_PIPELINE", "1")
    monkeypatch.setenv("TILEOPS_GEMM_SPLIT_K", "2")

    kernel = GemmKernel(128, 128, 256, dtype=torch.float16, tune=False)

    assert kernel._use_split_k_path is True
    assert kernel._use_packed_b_tile_path is True
    assert kernel._use_packed_b_async_pipeline_path is True


@pytest.mark.smoke
def test_gemm_kernel_rejects_incompatible_splitk_block_k_configuration(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_GEMM_SPLIT_K", "2")

    with pytest.raises(RuntimeError, match="block_k divisible by split_k"):
        GemmKernel(128, 128, 256, dtype=torch.float16, tune=False, config={"block_k": 33})


@pytest.mark.smoke
def test_gemm_kernel_packed_b_tile_leaves_transposed_b_layout_alone(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_GEMM_PACKED_B_TILE", "1")
    kernel = GemmKernel(128, 128, 128, dtype=torch.float16, tune=False, trans_b=True)
    b = torch.arange(128 * 128, dtype=torch.float16).reshape(128, 128)

    prepared_b = kernel.prepare_b(b)

    assert kernel._use_maca_bsm_path is True
    assert kernel._use_packed_b_tile_path is False
    assert prepared_b is b


@pytest.mark.smoke
def test_gemm_kernel_prepare_a_is_identity(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    kernel = GemmKernel(2, 3, 4, dtype=torch.float16, tune=False)
    a = torch.arange(8, dtype=torch.float16).reshape(2, 4)

    assert torch.equal(kernel.prepare_a(a), a)


@pytest.mark.smoke
def test_maca_hgemm_auto_launch_order_promotes_m_heavy_multi_wave_shapes(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.delenv("TILEOPS_MACA_HGEMM_USE_M_GROUP_SWIZZLE", raising=False)
    monkeypatch.delenv("TILEOPS_MACA_HGEMM_USE_N_GROUP_SWIZZLE", raising=False)

    m_heavy_kernel = MacaHGemmKernel(4096, 1024, 16384, dtype=torch.float16, tune=False)
    n_heavy_kernel = MacaHGemmKernel(1664, 2048, 16384, dtype=torch.float16, tune=False)
    single_wave_kernel = MacaHGemmKernel(1664, 1024, 16384, dtype=torch.float16, tune=False)

    assert m_heavy_kernel.config["launch_order"] == "m_group_swizzle"
    assert n_heavy_kernel.config["launch_order"] == "grid_xy"
    assert single_wave_kernel.config["launch_order"] == "grid_xy"


@pytest.mark.smoke
def test_maca_hgemm_explicit_launch_order_env_disables_auto_selection(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_USE_M_GROUP_SWIZZLE", "0")
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_USE_N_GROUP_SWIZZLE", "0")

    kernel = MacaHGemmKernel(4096, 1024, 16384, dtype=torch.float16, tune=False)

    assert kernel.config["launch_order"] == "grid_xy"


@pytest.mark.smoke
def test_maca_hgemm_reference_layout_ab_continuous_c_routes_through_external_entrypoint(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_AB_CONTINUOUS_C", "1")
    fake_reference = _install_fake_reference_layout_ab_module(monkeypatch)
    _reference_muxi_layout_kernels.cache_clear()

    kernel = MacaHGemmKernel(128, 16, 5120, dtype=torch.float16, tune=False)
    a = torch.ones((128, 5120), dtype=torch.float16)
    b = torch.ones((5120, 16), dtype=torch.float16)

    prepared_a = kernel.prepare_a(a)
    prepared_b = kernel.prepare_b(b)
    out = kernel.forward_with_prepared_a_and_b(prepared_a, prepared_b)
    expected_prepared_a = a.view(128 // 16, 16, 5120 // 8, 8).permute(0, 2, 1, 3).contiguous()
    expected_prepared_b = b.transpose(0, 1).contiguous().view(
        16 // 16,
        16,
        5120 // 32,
        4,
        8,
    ).permute(2, 0, 3, 1, 4).contiguous()

    assert kernel.use_reference_layout_ab_continuous_c is True
    assert prepared_a.shape == (8, 640, 16, 8)
    assert prepared_b.shape == (160, 1, 4, 16, 8)
    assert torch.equal(prepared_a, expected_prepared_a)
    assert torch.equal(prepared_b, expected_prepared_b)
    assert torch.equal(out, torch.full((128, 16), 7.0, dtype=torch.float16))
    assert fake_reference.calls == [((8, 640, 16, 8), (160, 1, 4, 16, 8), 1.0, 0.0, None)]


@pytest.mark.smoke
def test_maca_hgemm_reference_layout_a_routes_through_external_entrypoint(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_A_BODY", "1")
    fake_reference = _install_fake_reference_layout_a_module(monkeypatch)
    _reference_muxi_layout_kernels.cache_clear()

    kernel = MacaHGemmKernel(128, 64, 128, dtype=torch.float16, tune=False)
    a = torch.arange(128 * 128, dtype=torch.float16).reshape(128, 128)
    b = torch.arange(128 * 64, dtype=torch.float16).reshape(128, 64)

    prepared_a = kernel.prepare_a(a)
    prepared_b = kernel.prepare_b(b)
    out = kernel.forward_with_prepared_a_and_b(prepared_a, prepared_b)
    expected_prepared_a = a.view(128 // 16, 16, 128 // 8, 8).permute(0, 2, 1, 3).contiguous()
    expected_prepared_b = b.transpose(0, 1).contiguous()

    assert kernel.use_reference_layout_a is True
    assert kernel.config["backend"] == "maca_hgemm_reference_layout_a"
    assert prepared_a.shape == (8, 16, 16, 8)
    assert prepared_b.shape == (64, 128)
    assert torch.equal(prepared_a, expected_prepared_a)
    assert torch.equal(prepared_b, expected_prepared_b)
    assert torch.equal(out, torch.full((128, 64), 11.0, dtype=torch.float16))
    assert fake_reference.calls == [((8, 16, 16, 8), (64, 128), 1.0, 0.0)]


@pytest.mark.smoke
def test_maca_hgemm_rowa_layout_b_body_stays_disabled_after_failed_smoke(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_USE_ROWA_LAYOUT_B_BODY", "1")
    monkeypatch.delenv("TILEOPS_GEMM_BACKEND", raising=False)

    with pytest.raises(RuntimeError, match="disabled"):
        MacaHGemmKernel(128, 128, 128, dtype=torch.float16, tune=False)


@pytest.mark.smoke
def test_gemm_op_ignores_hpp_reference_layout_env_on_auto_dispatch(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metax_c500(monkeypatch)
    monkeypatch.setenv("TILEOPS_MACA_HGEMM_USE_REFERENCE_LAYOUT_AB_CONTINUOUS_C", "1")
    monkeypatch.delenv("TILEOPS_GEMM_BACKEND", raising=False)

    op = GemmOp(1664, 1024, 16384, dtype=torch.float16, tune=False)

    assert isinstance(op.kernel, GemmKernel)
