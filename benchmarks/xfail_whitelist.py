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
            "benchmarks/ops/attention/bench_deepseek_mla_decode.py::test_mla_decode_bench[deepseek-v2-4k-float16]",
            "benchmarks/ops/attention/bench_deepseek_mla_decode.py::test_mla_decode_bench[deepseek-v2-4k-bfloat16]",
            "benchmarks/ops/attention/bench_deepseek_mla_decode.py::test_mla_decode_bench[deepseek-v2-32k-float16]",
            "benchmarks/ops/attention/bench_deepseek_mla_decode.py::test_mla_decode_bench[deepseek-v2-32k-bfloat16]",
            "benchmarks/ops/attention/bench_deepseek_mla_decode.py::test_mla_decode_bench[deepseek-v3-4k-bfloat16]",
            "benchmarks/ops/attention/bench_deepseek_mla_decode.py::test_mla_decode_bench[deepseek-v3-32k-bfloat16]",
        ),
    ),
    (
        _RUNTIME_LAUNCH_ERROR,
        (
        ),
    ),
    (
        _COMPILATION_FAILURE,
        (
            "benchmarks/ops/bench_elementwise_manifest.py::test_logical_and_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_logical_or_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_and_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_or_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_xor_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-4096-4-64-64-64-dtype0-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-4096-4-64-64-32-dtype1-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-4096-4-64-64-32-dtype2-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-2048-4-64-64-64-dtype3-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-8192-4-64-64-64-dtype4-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-16384-4-64-64-64-dtype5-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-32768-4-64-64-64-dtype6-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-2048-4-64-64-64-dtype7-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-4096-4-64-64-64-dtype8-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-8192-4-64-64-64-dtype9-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-16384-4-64-64-64-dtype10-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[2-32768-4-64-64-64-dtype11-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[2-4096-4-64-64-64-dtype0-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[2-4096-4-64-64-32-dtype1-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[2-4096-4-64-64-32-dtype2-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[2-2048-4-64-64-64-dtype3-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[2-8192-4-64-64-64-dtype4-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[2-16384-4-64-64-64-dtype5-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[2-2048-4-64-64-64-dtype6-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[2-4096-4-64-64-64-dtype7-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[2-8192-4-64-64-64-dtype8-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[2-16384-4-64-64-64-dtype9-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwdbwd[2-4096-4-64-64-64-dtype0-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwdbwd[2-4096-4-64-64-32-dtype1-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwdbwd[2-4096-4-64-64-32-dtype2-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwdbwd[2-2048-4-64-64-64-dtype3-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwdbwd[2-8192-4-64-64-64-dtype4-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwdbwd[2-16384-4-64-64-64-dtype5-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwdbwd[2-2048-4-64-64-64-dtype6-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwdbwd[2-4096-4-64-64-64-dtype7-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwdbwd[2-8192-4-64-64-64-dtype8-False]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwdbwd[2-16384-4-64-64-64-dtype9-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-4096-4-64-64-32-dtype0-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-4096-4-64-64-32-dtype1-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-2048-4-64-64-64-dtype2-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-4096-4-64-64-64-dtype3-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-8192-4-64-64-64-dtype4-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-16384-4-64-64-64-dtype5-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-32768-4-64-64-64-dtype6-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-2048-4-64-64-64-dtype7-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-4096-4-64-64-64-dtype8-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-8192-4-64-64-64-dtype9-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-16384-4-64-64-64-dtype10-False]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[2-32768-4-64-64-64-dtype11-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[2-2048-4-64-64-64-dtype0-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[2-4096-4-64-64-64-dtype1-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[2-8192-4-64-64-64-dtype2-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[2-16384-4-64-64-64-dtype3-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[2-2048-4-64-64-64-dtype4-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[2-4096-4-64-64-64-dtype5-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[2-8192-4-64-64-64-dtype6-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[2-16384-4-64-64-64-dtype7-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_fwdbwd_bench[2-2048-4-64-64-64-dtype0-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_fwdbwd_bench[2-4096-4-64-64-64-dtype1-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_fwdbwd_bench[2-8192-4-64-64-64-dtype2-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_fwdbwd_bench[2-16384-4-64-64-64-dtype3-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_fwdbwd_bench[2-2048-4-64-64-64-dtype4-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_fwdbwd_bench[2-4096-4-64-64-64-dtype5-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_fwdbwd_bench[2-8192-4-64-64-64-dtype6-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_fwdbwd_bench[2-16384-4-64-64-64-dtype7-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[2-4096-4-64-64-64-dtype4-False]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_fwdbwd_bench[2-4096-4-64-64-64-dtype4-False]",
            "benchmarks/ops/attention/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[serving-8b-p64-float16]",
            "benchmarks/ops/attention/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[serving-8b-long-p64-float16]",
            "benchmarks/ops/attention/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[throughput-8b-p64-float16]",
            "benchmarks/ops/attention/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[serving-70b-p64-float16]",
            "benchmarks/ops/attention/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[serving-8b-p256-float16]",
            "benchmarks/ops/attention/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[serving-70b-p256-float16]",
            "benchmarks/ops/attention/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[serving-405b-p256-float16]",
            "benchmarks/ops/attention/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[serving-8b-p64-softcap50-float16]",
            "benchmarks/ops/bench_ada_layer_norm.py::test_ada_layer_norm_bench[dit-xl-2-float16]",
            "benchmarks/ops/bench_ada_layer_norm.py::test_ada_layer_norm_bench[dit-xl-2-bfloat16]",
            "benchmarks/ops/bench_ada_layer_norm.py::test_ada_layer_norm_zero_bench[dit-xl-2-float16]",
            "benchmarks/ops/bench_ada_layer_norm.py::test_ada_layer_norm_zero_bench[dit-xl-2-bfloat16]",
        ),
    ),
    (
        _BENCHMARK_API_MISMATCH,
        (
        ),
    ),
    (
        _AUTOTUNE_FAILURE,
        (
            "benchmarks/ops/attention/bench_mha.py::test_mha_bwd_bench[llama-3.1-8b-short-float16]",
            "benchmarks/ops/attention/bench_mha.py::test_mha_bwd_bench[llama-3.1-8b-short-bfloat16]",
            "benchmarks/ops/attention/bench_mha.py::test_mha_bwd_bench[llama-3.1-8b-long-float16]",
            "benchmarks/ops/attention/bench_mha.py::test_mha_bwd_bench[llama-3.1-8b-long-bfloat16]",
            "benchmarks/ops/attention/bench_mha.py::test_mha_bwd_bench[llama-3.1-70b-short-float16]",
            "benchmarks/ops/attention/bench_mha.py::test_mha_bwd_bench[llama-3.1-70b-short-bfloat16]",
            "benchmarks/ops/attention/bench_mha.py::test_mha_bwd_bench[llama-3.1-70b-long-float16]",
            "benchmarks/ops/attention/bench_mha.py::test_mha_bwd_bench[llama-3.1-70b-long-bfloat16]",
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
