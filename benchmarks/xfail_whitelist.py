"""Exact-node allowlist for known benchmark failures on MetaX MACA."""

_UNSUPPORTED_ARCHITECTURE = "kernel is not supported on the current MACA architecture"
_RUNTIME_LAUNCH_ERROR = "known MACA runtime launch error"
_COMPILATION_FAILURE = "known MACA benchmark compilation failure"
_BENCHMARK_API_MISMATCH = "benchmark uses an incompatible cumulative Op constructor"
_AUTOTUNE_FAILURE = "no benchmark configuration compiles and validates successfully"


# FIXME(staged-rollout): quarantine the current MetaX benchmark failures by exact node ID.
#
# Broken invariant: every collected TileOps benchmark runs on the MetaX test runner.
# Why: the backend and several benchmark call sites still have known compatibility gaps.
# Cleanup: remove each entry as soon as its node passes consistently on the MetaX runner.
_MACA_XFAIL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        _UNSUPPORTED_ARCHITECTURE,
        (
            "benchmarks/ops/attention/bench_deepseek_dsa_decode.py::test_dsa_decode_bench[single-batch-mainstream-float16]",
            "benchmarks/ops/attention/bench_deepseek_dsa_decode.py::test_dsa_decode_bench[longer-kv-lower-topk-float16]",
        ),
    ),
    (
        _RUNTIME_LAUNCH_ERROR,
        (
            "benchmarks/ops/attention/bench_gqa_decode.py::test_gqa_decode_bench[qwen3-30b-a3b-bs1-1k-float16]",
            "benchmarks/ops/bench_argreduce.py::test_argmin_bench[hidden-state-argmin-float16]",
            "benchmarks/ops/bench_argreduce.py::test_argmin_bench[hidden-state-argmin-bfloat16]",
            "benchmarks/ops/bench_reduce_multidim.py::test_argreduce_multidim_bench[argmin-7B-dim2-keepdim]",
        ),
    ),
    (
        _COMPILATION_FAILURE,
        (
            "benchmarks/ops/attention/bench_gqa_decode.py::test_gqa_decode_bench[qwen3-30b-a3b-bs1-4k-float16]",
            "benchmarks/ops/attention/bench_gqa_decode.py::test_gqa_decode_bench[qwen3-30b-a3b-bs1-8k-float16]",
            "benchmarks/ops/attention/bench_gqa_decode.py::test_gqa_decode_bench[qwen3-30b-a3b-bs1-16k-float16]",
            "benchmarks/ops/attention/bench_gqa_decode.py::test_gqa_decode_bench[qwen3-30b-a3b-bs1-32k-float16]",
            "benchmarks/ops/attention/bench_gqa_decode.py::test_gqa_decode_bench[qwen3-30b-a3b-bs1-64k-float16]",
            "benchmarks/ops/attention/bench_gqa_decode.py::test_gqa_decode_bench[qwen3-30b-a3b-bs1-128k-float16]",
            "benchmarks/ops/attention/bench_gqa_decode.py::test_gqa_decode_bench[qwen3-30b-a3b-bs1-256k-float16]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_logical_and_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_logical_or_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_and_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_or_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_xor_manifest_bench[cnn-feat-broadcast-bool]",
        ),
    ),
    (
        _BENCHMARK_API_MISMATCH,
        (
            "benchmarks/ops/bench_cumulative.py::test_cumsum_bench[hidden-state-scan-float16]",
            "benchmarks/ops/bench_cumulative.py::test_cumsum_bench[hidden-state-scan-bfloat16]",
            "benchmarks/ops/bench_cumulative.py::test_cumsum_bench[long-seq-scan-bfloat16]",
            "benchmarks/ops/bench_cumulative.py::test_cumprod_bench[hidden-state-scan-float16]",
            "benchmarks/ops/bench_cumulative.py::test_cumprod_bench[hidden-state-scan-bfloat16]",
            "benchmarks/ops/bench_cumulative.py::test_cumprod_bench[long-seq-scan-bfloat16]",
            "benchmarks/ops/bench_reduce_multidim.py::test_cumulative_multidim_bench[cumsum-7B-3D]",
            "benchmarks/ops/bench_reduce_multidim.py::test_cumulative_multidim_bench[cumsum-7B-3D-bf16]",
            "benchmarks/ops/bench_reduce_multidim.py::test_cumulative_multidim_bench[cumprod-7B-longctx-3D]",
        ),
    ),
    (
        _AUTOTUNE_FAILURE,
        (
        ),
    ),
)

MACA_XFAILS = {
    nodeid: reason
    for reason, nodeids in _MACA_XFAIL_GROUPS
    for nodeid in nodeids
}

if len(MACA_XFAILS) != sum(len(nodeids) for _, nodeids in _MACA_XFAIL_GROUPS):
    raise ValueError("duplicate node ID in the MACA benchmark xfail allowlist")
