#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

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
LONG_K_SHAPES = ((1664, 1024, 262144),)
BACKENDS = ("direct-hpp", "compiler-packed-b", "compiler-splitk-packed")
CUPTI_BENCH_TRIALS = 7


@dataclass(frozen=True)
class HgemmResult:
    backend: str
    m: int
    n: int
    k: int
    latency_ms: float
    tflops: float
    max_abs_diff: float | None
    mean_abs_diff: float | None
    config: dict | None
    execution: dict | None


def resolve_shapes(shape_set: str) -> tuple[tuple[int, int, int], ...]:
    if shape_set == "long-k":
        return LONG_K_SHAPES
    if shape_set == "table":
        return TABLE_SHAPES
    if shape_set == "all":
        return LONG_K_SHAPES + TABLE_SHAPES
    raise ValueError(f"unknown shape set: {shape_set}")


def resolve_backends(backend: str) -> tuple[str, ...]:
    if backend == "both":
        return BACKENDS
    if backend in BACKENDS:
        return (backend,)
    raise ValueError(f"unknown backend: {backend}")


def compiler_packed_b_env() -> dict[str, str | None]:
    return {
        "TILEOPS_GEMM_SPLIT_K": "1",
        "TILEOPS_GEMM_PACKED_B_TILE": "1",
        "TILELANG_MACA_GEMM_USE_TEMPLATE": None,
        "TILELANG_MACA_GEMM_K_PACK": None,
        "TILELANG_MACA_GEMM_CONSUMER_SURFACE": None,
    }


def compiler_splitk_packed_env() -> dict[str, str | None]:
    return {
        "TILEOPS_GEMM_SPLIT_K": "2",
        "TILEOPS_GEMM_PACKED_B_TILE": "1",
        "TILELANG_MACA_GEMM_USE_TEMPLATE": "1",
        "TILELANG_MACA_GEMM_K_PACK": "1",
        "TILELANG_MACA_GEMM_CONSUMER_SURFACE": None,
    }


@contextmanager
def _temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return (2.0 * m * n * k) / latency_ms * 1e-9


def _make_op(backend: str, m: int, n: int, k: int, torch_module):
    from tileops.ops import GemmOp

    if backend == "direct-hpp":
        from tileops.kernels.gemm_maca.maca_hgemm import MacaHGemmKernel

        return GemmOp(
            m,
            n,
            k,
            dtype=torch_module.float16,
            tune=False,
            kernel_map={"gemm_kernel": MacaHGemmKernel},
        )
    if backend in {"compiler-packed-b", "compiler-splitk-packed"}:
        return GemmOp(m, n, k, dtype=torch_module.float16, tune=False)
    raise ValueError(f"unknown backend: {backend}")


def _validate_compiler_execution(backend: str, op) -> dict | None:
    if backend == "direct-hpp":
        return None

    execution = getattr(op.kernel, "execution_info", None)
    if not isinstance(execution, dict):
        raise RuntimeError(f"{backend} did not expose compiler execution provenance")

    expected = {
        "compiler-packed-b": {
            "backend": "compiler-packed-b",
            "split_k": 1,
            "packed_b_tile": True,
            "template": False,
            "specialized_reduce": False,
        },
        "compiler-splitk-packed": {
            "backend": "compiler-splitk-packed",
            "split_k": 2,
            "packed_b_tile": True,
            "template": True,
            "specialized_reduce": True,
        },
    }[backend]
    mismatches = {
        key: (execution.get(key), value)
        for key, value in expected.items()
        if execution.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{backend} selected an unexpected execution route: {mismatches}")
    return dict(execution)


def _time_hot_path(torch_module, fn, warmup: int, repeat: int) -> tuple[float, object]:
    def timed_fn():
        with torch_module.no_grad():
            return fn()

    try:
        from tilelang.profiler import do_bench
    except Exception:
        do_bench = None

    if do_bench is not None:
        for backend in ("cupti", "event"):
            try:
                kwargs = {"warmup": warmup, "rep": repeat, "backend": backend}
                if backend == "event":
                    kwargs["return_mode"] = "median"
                trial_count = CUPTI_BENCH_TRIALS if backend == "cupti" else 1
                latencies = [float(do_bench(timed_fn, **kwargs)) for _ in range(trial_count)]
                latency_ms = min(latency for latency in latencies if latency > 0)
                if latency_ms > 0:
                    out = timed_fn()
                    torch_module.cuda.synchronize()
                    return latency_ms, out
            except Exception:
                pass

    out = None
    with torch_module.no_grad():
        for _ in range(warmup):
            out = timed_fn()
        torch_module.cuda.synchronize()

        times = []
        for _ in range(repeat):
            start = torch_module.cuda.Event(enable_timing=True)
            end = torch_module.cuda.Event(enable_timing=True)
            start.record()
            out = timed_fn()
            end.record()
            torch_module.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
    return statistics.median(times), out


def _compare(torch_module, actual, expected) -> tuple[float, float]:
    diff = (actual - expected).abs().float()
    return float(diff.max().item()), float(diff.mean().item())


def _precondition_compiler_splitk_packed(torch_module) -> None:
    from tileops.ops import GemmOp

    m, n, k = (128, 128, 256)
    a = torch_module.randint(
        0,
        2,
        (m, k),
        device="cuda",
        dtype=torch_module.int32,
    ).to(torch_module.float16)
    b = torch_module.randint(
        0,
        2,
        (k, n),
        device="cuda",
        dtype=torch_module.int32,
    ).to(torch_module.float16)
    op = GemmOp(m, n, k, dtype=torch_module.float16, tune=False)
    b_prepared = op.prepare_b(b)
    out = op.forward_with_prepared_b(a, b_prepared)
    expected = torch_module.matmul(a, b)
    torch_module.testing.assert_close(out, expected, rtol=0, atol=0)
    torch_module.cuda.synchronize()
    del a, b, b_prepared, out, expected


def run_one(
    backend: str,
    shape: tuple[int, int, int],
    *,
    warmup: int,
    repeat: int,
    check: bool,
    rtol: float,
    atol: float,
) -> HgemmResult:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/MACA device is required for MACA HGEMM benchmarking")

    m, n, k = shape
    if backend == "compiler-packed-b":
        env = compiler_packed_b_env()
    elif backend == "compiler-splitk-packed":
        env = compiler_splitk_packed_env()
    else:
        env = {}
    with _temporary_env(env):
        if backend == "compiler-splitk-packed":
            _precondition_compiler_splitk_packed(torch)

        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        op = _make_op(backend, m, n, k, torch)
        b_prepared = op.prepare_b(b)
        execution = _validate_compiler_execution(backend, op)
        torch.cuda.synchronize()
        run_hot_path = functools.partial(op.forward_with_prepared_b, a, b_prepared)

        latency_ms, out = _time_hot_path(
            torch,
            run_hot_path,
            warmup,
            repeat,
        )

        max_abs_diff = None
        mean_abs_diff = None
        if check:
            expected = torch.matmul(a, b)
            torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
            max_abs_diff, mean_abs_diff = _compare(torch, out, expected)

        config = getattr(op.kernel, "config", None)
        result = HgemmResult(
            backend=backend,
            m=m,
            n=n,
            k=k,
            latency_ms=latency_ms,
            tflops=_tflops(m, n, k, latency_ms),
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            config=config,
            execution=execution,
        )

        del a, b, b_prepared, out
        torch.cuda.empty_cache()
        return result


def metadata() -> dict[str, object]:
    import torch

    import tileops
    from tileops.kernels.gemm_maca import maca_hgemm

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "tileops_path": tileops.__file__,
        "maca_hgemm_path": maca_hgemm.__file__,
        "torch_version": torch.__version__,
        "device": device,
    }


def format_markdown(results: list[HgemmResult], meta: dict[str, object]) -> str:
    lines = [
        "# MACA HGEMM Performance",
        "",
        "## Environment",
        "",
    ]
    for key, value in meta.items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| backend | m | n | k | latency_ms | tflops | max_abs_diff | mean_abs_diff |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in results:
        lines.append(
            f"| {row.backend} | {row.m} | {row.n} | {row.k} | "
            f"{row.latency_ms:.6f} | {row.tflops:.6f} | "
            f"{row.max_abs_diff if row.max_abs_diff is not None else 'N/A'} | "
            f"{row.mean_abs_diff if row.mean_abs_diff is not None else 'N/A'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark MACA HGEMM production shapes.")
    parser.add_argument("--shape-set", choices=("long-k", "table", "all"), default="all")
    parser.add_argument(
        "--backend",
        choices=("direct-hpp", "compiler-packed-b", "compiler-splitk-packed", "both"),
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("maca_hgemm_perf.json"))
    parser.add_argument("--output-md", type=Path, default=Path("maca_hgemm_perf.md"))
    args = parser.parse_args(argv)

    meta = metadata()
    print(json.dumps({"metadata": meta}, indent=2, sort_keys=True))

    results: list[HgemmResult] = []
    for backend in resolve_backends(args.backend):
        for shape in resolve_shapes(args.shape_set):
            result = run_one(
                backend,
                shape,
                warmup=args.warmup,
                repeat=args.rep,
                check=not args.skip_check,
                rtol=args.rtol,
                atol=args.atol,
            )
            results.append(result)
            print(json.dumps(asdict(result), sort_keys=True), flush=True)

    payload = {
        "metadata": meta,
        "results": [asdict(row) for row in results],
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    args.output_md.write_text(format_markdown(results, meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
