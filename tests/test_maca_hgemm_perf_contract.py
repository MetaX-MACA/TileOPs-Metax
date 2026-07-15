from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_GEMM = REPO_ROOT / "benchmarks" / "ops" / "bench_gemm.py"
BENCH_CONFTEST = REPO_ROOT / "benchmarks" / "conftest.py"
HGEMM_DRIVER = REPO_ROOT / "scripts" / "bench_maca_hgemm.py"
HGEMM_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "maca-hgemm-performance.yml"
GEMM_KERNEL = REPO_ROOT / "tileops" / "kernels" / "gemm.py"

TABLE_SHAPES = (
    (4096, 1024, 8192),
    (4096, 8192, 8192),
    (4096, 28672, 8192),
    (4096, 8192, 28672),
    (8192, 1024, 8192),
    (8192, 8192, 8192),
    (8192, 28672, 8192),
    (8192, 8192, 28672),
)
LONG_K_SHAPE = (1664, 1024, 262144)


def _load_driver_module():
    spec = importlib.util.spec_from_file_location("bench_maca_hgemm", HGEMM_DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _long_k_compiler_config(source: str) -> dict[str, object]:
    module = ast.parse(source)
    gemm_kernel = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "GemmKernel"
    )
    default_config = next(
        node
        for node in gemm_kernel.body
        if isinstance(node, ast.FunctionDef) and node.name == "default_config"
    )
    expected_terms = {
        "self._use_split_k_path",
        "self._use_packed_b_tile_path",
        "self.k >= 131072",
    }
    for node in ast.walk(default_config):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.BoolOp):
            continue
        if not isinstance(node.test.op, ast.And):
            continue
        if {ast.unparse(term) for term in node.test.values} != expected_terms:
            continue
        returned = next(
            (statement for statement in node.body if isinstance(statement, ast.Return)), None
        )
        assert returned is not None and isinstance(returned.value, ast.Dict)
        return {
            ast.literal_eval(key): ast.literal_eval(value)
            for key, value in zip(returned.value.keys, returned.value.values, strict=True)
        }
    raise AssertionError("missing long-K split-K packed-B compiler configuration")


def _splitk_packed_kernel_body(source: str) -> str:
    match = re.search(
        r"def _gemm_kernel_bsm_splitk_packed_b_tile\(.*?"
        r"(?=\n@functools\.lru_cache\(maxsize=32\)\n"
        r"def _gemm_kernel_bsm_splitk_packed_b_tile_async\()",
        source,
        re.DOTALL,
    )
    assert match is not None
    return match.group(0)


def _compiler_splitk_benchmark_body(source: str) -> str:
    match = re.search(
        r"def test_maca_hgemm_compiler_splitk_bench\(.*?"
        r"(?=\n\n@pytest\.mark\.parametrize\(\"m, n, k, dtype, trans_a, trans_b, tune\",\n"
        r"                         _HGEMM_LONG_K_SHAPES \+ _HGEMM_TABLE_SHAPES\)\n"
        r"def test_maca_hgemm_direct_hpp_bench\()",
        source,
        re.DOTALL,
    )
    assert match is not None
    return match.group(0)


def test_bench_gemm_includes_production_hgemm_shapes_and_prepacked_case() -> None:
    source = BENCH_GEMM.read_text()
    gemm_kernel = GEMM_KERNEL.read_text()
    compiler_splitk_body = _compiler_splitk_benchmark_body(source)

    assert "_HGEMM_TABLE_SHAPES" in source
    assert "_HGEMM_LONG_K_SHAPES" in source
    for shape in TABLE_SHAPES + (LONG_K_SHAPE,):
        assert ", ".join(str(v) for v in shape) in source

    assert "test_maca_hgemm_packed_b_bench" in source
    assert "TILEOPS_GEMM_PACKED_B_TILE" in source
    assert "forward_with_prepared_b" in source
    assert "tileops_packed_b" in source
    assert "test_maca_hgemm_compiler_splitk_bench" in source
    assert "TILEOPS_GEMM_SPLIT_K" in source
    assert "TILELANG_MACA_GEMM_USE_TEMPLATE" in source
    assert "TILELANG_MACA_GEMM_K_PACK" in source
    assert "tileops_compiler_splitk" in source
    assert '"backend": "compiler-splitk-packed"' in compiler_splitk_body
    assert "compiler split-K benchmark selected" in compiler_splitk_body
    assert compiler_splitk_body.index("prepared_b = op.prepare_b(b)") < compiler_splitk_body.index(
        'execution = getattr(op.kernel, "execution_info", None)'
    )
    assert "test_maca_hgemm_direct_hpp_bench" in source
    assert "MacaHGemmKernel" in source
    assert 'kernel_map={"gemm_kernel": MacaHGemmKernel}' in source
    assert "is_metax_c500()" in source
    assert "tileops_direct_hpp" in source
    assert "supported_archs: list[int] = [80, 90]" in gemm_kernel


def test_bench_gemm_runs_serial_nodes_for_large_hgemm_shapes() -> None:
    source = BENCH_CONFTEST.read_text()

    assert '"benchmarks/ops/bench_gemm.py"' in source


def test_hgemm_driver_declares_exact_table_and_long_k_shape_sets() -> None:
    driver = _load_driver_module()

    assert driver.TABLE_SHAPES == TABLE_SHAPES
    assert driver.LONG_K_SHAPES == (LONG_K_SHAPE,)
    assert driver.resolve_shapes("table") == TABLE_SHAPES
    assert driver.resolve_shapes("long-k") == (LONG_K_SHAPE,)
    assert driver.resolve_shapes("all") == (LONG_K_SHAPE,) + TABLE_SHAPES


def test_hgemm_driver_supports_direct_and_compiler_packed_backends(monkeypatch) -> None:
    driver = _load_driver_module()
    source = HGEMM_DRIVER.read_text()
    run_one_body = source[source.index("def run_one(") : source.index("\ndef metadata(")]

    assert driver.BACKENDS == ("direct-hpp", "compiler-packed-b", "compiler-splitk-packed")
    assert driver.resolve_backends("both") == driver.BACKENDS
    assert driver.compiler_packed_b_env() == {
        "TILEOPS_GEMM_SPLIT_K": "1",
        "TILEOPS_GEMM_PACKED_B_TILE": "1",
        "TILELANG_MACA_GEMM_USE_TEMPLATE": None,
        "TILELANG_MACA_GEMM_K_PACK": None,
        "TILELANG_MACA_GEMM_CONSUMER_SURFACE": None,
    }
    assert driver.compiler_splitk_packed_env() == {
        "TILEOPS_GEMM_SPLIT_K": "2",
        "TILEOPS_GEMM_PACKED_B_TILE": "1",
        "TILELANG_MACA_GEMM_USE_TEMPLATE": "1",
        "TILELANG_MACA_GEMM_K_PACK": "1",
        "TILELANG_MACA_GEMM_CONSUMER_SURFACE": None,
    }
    assert "_precondition_compiler_splitk_packed" in source
    assert "m, n, k = (128, 128, 256)" in source
    assert 'backend == "compiler-splitk-packed"' in source
    assert "def timed_fn():" in source
    assert "with torch_module.no_grad():" in source
    assert "CUPTI_BENCH_TRIALS = 7" in source
    assert "latency_ms = min(latency for latency in latencies if latency > 0)" in source
    assert "sys.path.insert" not in source
    assert "_validate_compiler_execution" in source
    assert "execution: dict | None" in source
    assert '"compiler-splitk-packed"' in source
    assert run_one_body.index("b_prepared = op.prepare_b(b)") < run_one_body.index(
        "execution = _validate_compiler_execution(backend, op)"
    )
    monkeypatch.setenv("TILELANG_MACA_GEMM_USE_TEMPLATE", "1")
    monkeypatch.setenv("TILELANG_MACA_GEMM_K_PACK", "1")
    monkeypatch.setenv("TILELANG_MACA_GEMM_CONSUMER_SURFACE", "wsm_aware")
    with driver._temporary_env(driver.compiler_packed_b_env()):
        assert os.environ.get("TILELANG_MACA_GEMM_USE_TEMPLATE") is None
        assert os.environ.get("TILELANG_MACA_GEMM_K_PACK") is None
        assert os.environ.get("TILELANG_MACA_GEMM_CONSUMER_SURFACE") is None


def test_gemm_kernel_has_first_class_long_k_compiler_fast_path() -> None:
    source = GEMM_KERNEL.read_text()
    long_k_config = _long_k_compiler_config(source)

    assert "_should_auto_use_maca_compiler_splitk_packed" in source
    assert "k >= 131072" in source
    assert "TILEOPS_GEMM_SPLIT_K" in source
    assert "TILEOPS_GEMM_PACKED_B_TILE" in source
    assert "TILELANG_MACA_GEMM_USE_TEMPLATE" in source
    assert "TILELANG_MACA_GEMM_K_PACK" in source
    assert "compiler split-K packed-B path" in source
    assert long_k_config == {
        "block_m": 128,
        "block_n": 128,
        "block_k": 128,
        "num_stages": 0,
        "threads": 256,
        "enable_rasterization": True,
    }
    assert "def execution_info" in source
    assert '"specialized_reduce": self._use_split_k_path' in source


def test_gemm_kernel_caches_compiled_hot_path_wrappers() -> None:
    source = GEMM_KERNEL.read_text()

    assert "_compiled_kernel_config" in source
    assert "def _get_compiled_kernel" in source
    assert "def _get_compiled_reduce_kernel" in source
    assert "kernel = self._get_compiled_kernel()" in source
    assert "return self._get_compiled_reduce_kernel()(partial_c)" in source


def test_splitk_reduce_has_specialized_two_way_sum() -> None:
    source = GEMM_KERNEL.read_text()

    assert "items_per_thread = 16" in source
    assert "if split_k == 2:" in source
    assert 'T.cast(partial_c[0, row, col], "float32")' in source
    assert 'T.cast(partial_c[1, row, col], "float32")' in source


def test_splitk_packed_compiler_kernel_preserves_opt_in_wsm_annotations() -> None:
    body = _splitk_packed_kernel_body(GEMM_KERNEL.read_text())

    assert "gemm_annotations = {" in body
    assert "maca_wsm_a_source_ptr" in body
    assert "maca_wsm_b_source_ptr" in body
    assert "maca_wsm_a_stride" in body
    assert "annotations=gemm_annotations" in body


def test_gemm_kernel_keeps_wsm_surface_out_of_default_compiler_env() -> None:
    source = GEMM_KERNEL.read_text()

    assert "TILELANG_MACA_GEMM_CONSUMER_SURFACE" not in source
    assert "_use_packed_b_wsm_path" not in source


def test_hgemm_workflow_runs_dedicated_driver() -> None:
    workflow = yaml.safe_load(HGEMM_WORKFLOW.read_text())

    assert workflow["name"] == "MACA HGEMM Performance"
    assert "workflow_dispatch" in workflow[True]
    job = workflow["jobs"]["maca-hgemm-performance"]
    assert job["runs-on"] == "tileops-metax-runner"

    run_blocks = "\n".join(step.get("run", "") for step in job["steps"] if isinstance(step, dict))
    assert "scripts/bench_maca_hgemm.py" in run_blocks
    assert "TILELANG_DISABLE_CACHE=1" in run_blocks
    assert "--backend both" in run_blocks
