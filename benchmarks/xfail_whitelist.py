"""Exact-node allowlist for known benchmark failures on MetaX MACA."""

_UNSUPPORTED_ARCHITECTURE = "kernel is not supported on the current MACA architecture"
_RUNTIME_LAUNCH_ERROR = "known MACA runtime launch error"
_COMPILATION_FAILURE = "known MACA benchmark compilation failure"
_BENCHMARK_API_MISMATCH = "benchmark uses an incompatible cumulative Op constructor"
_AUTOTUNE_FAILURE = "no benchmark configuration compiles and validates successfully"
_NUMERICAL_MISMATCH = "known MACA numerical mismatch"
_OUT_OF_MEMORY = "known MACA out-of-memory on this workload"
_MISSING_BASELINE = "required baseline/package unavailable or incompatible on MACA"


# FIXME(staged-rollout): quarantine the current MetaX benchmark failures by exact node ID.
#
# Broken invariant: every collected TileOps benchmark runs on the MetaX test runner.
# Why: the backend and several benchmark call sites still have known compatibility gaps.
# Cleanup: remove each entry as soon as its node passes consistently on the MetaX runner.
_MACA_XFAIL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        _UNSUPPORTED_ARCHITECTURE,
        (
            # BTHD GDN needs Hopper WS; MACA reports sm80-class capability.
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b2-s2k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b2-s2k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b2-s4k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b2-s4k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b2-s8k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b2-s8k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b2-s16k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b2-s16k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b2-s32k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b2-s32k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b1-s4k-h16-d128-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_fwd[gdn-bthd-b1-s4k-h16-d128-bfloat16]",
            "benchmarks/ops/bench_grouped_gemm.py::test_grouped_gemm_bench[nt-batch16-m4096-n4096-k4096-bfloat16]",
        ),
    ),
    (
        _RUNTIME_LAUNCH_ERROR,
        (),
    ),
    (
        _COMPILATION_FAILURE,
        (
            "benchmarks/ops/bench_elementwise_manifest.py::test_logical_and_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_logical_or_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_and_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_or_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_xor_manifest_bench[cnn-feat-broadcast-bool]",
            # Legacy nodeids (pre-workload-name ids); keep until collection drops them.
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
            # Current named-id deltanet / gated-deltanet MACA gemm.h compile failures
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[dn-b2-s2k-h4-d64-float16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[dn-b2-s2k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[dn-b2-s4k-h4-d64-float16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[dn-b2-s4k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[dn-b2-s8k-h4-d64-float16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[dn-b2-s8k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[dn-b2-s16k-h4-d64-float16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[dn-b2-s16k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[dn-bwd-b2-s2k-h4-d64-float16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[dn-bwd-b2-s2k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[dn-bwd-b2-s4k-h4-d64-float16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[dn-bwd-b2-s4k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[dn-bwd-b2-s8k-h4-d64-float16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[dn-bwd-b2-s8k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[dn-bwd-b2-s16k-h4-d64-float16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[dn-bwd-b2-s16k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_bhtd_vs_fla_fwd[gdn-bhtd-b2-s2k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_bhtd_vs_fla_fwd[gdn-bhtd-b2-s2k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_bhtd_vs_fla_fwd[gdn-bhtd-b2-s4k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_bhtd_vs_fla_fwd[gdn-bhtd-b2-s4k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_bhtd_vs_fla_fwd[gdn-bhtd-b2-s8k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_bhtd_vs_fla_fwd[gdn-bhtd-b2-s8k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_bhtd_vs_fla_fwd[gdn-bhtd-b2-s16k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_bhtd_vs_fla_fwd[gdn-bhtd-b2-s16k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_bwd[gdn-bwd-b2-s2k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_bwd[gdn-bwd-b2-s2k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_bwd[gdn-bwd-b2-s4k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_bwd[gdn-bwd-b2-s4k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_bwd[gdn-bwd-b2-s8k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_bwd[gdn-bwd-b2-s8k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_bwd[gdn-bwd-b2-s16k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_bwd[gdn-bwd-b2-s16k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_mamba.py::test_da_cumsum_fwd_bench[mamba2-780m-b1-s4k-float16]",
            "benchmarks/ops/bench_mamba.py::test_da_cumsum_fwd_bench[mamba2-1p3b-b8-s2k-bfloat16]",
            "benchmarks/ops/bench_mamba.py::test_da_cumsum_fwd_bench[mamba2-780m-b1-s4k-dt-bias-float16]",
            "benchmarks/ops/bench_mamba.py::test_da_cumsum_fwd_bench[mamba2-1p3b-b8-s2k-dt-bias-bfloat16]",
            "benchmarks/ops/bench_mamba2_e2e.py::test_mamba2_fwd_bench[mamba2-2p7b-b1-s2k-bfloat16]",
            "benchmarks/ops/bench_mamba2_e2e.py::test_mamba2_fwd_bench[mamba2-1p3b-b1-s8k-float16]",
            "benchmarks/ops/bench_mamba2_e2e.py::test_mamba2_fwd_bench[mamba2-2p7b-b1-s2k-dt-bias-bfloat16]",
            "benchmarks/ops/bench_mamba2_e2e.py::test_mamba2_fwd_bench[mamba2-1p3b-b1-s8k-dt-bias-float16]",
            "benchmarks/ops/bench_mamba2_e2e.py::test_mamba2_fwd_bench[mamba2-2p7b-b1-s2k-init-states-bfloat16]",
            "benchmarks/ops/bench_mamba2_e2e.py::test_mamba2_fwd_bench[mamba2-1p3b-b1-s8k-init-states-float16]",
            "benchmarks/ops/bench_mamba2_e2e.py::test_mamba2_fwd_bench[mamba2-2p7b-b1-s2k-dt-bias-init-states-bfloat16]",
            "benchmarks/ops/bench_mamba2_e2e.py::test_mamba2_fwd_bench[mamba2-1p3b-b1-s8k-dt-bias-init-states-float16]",
            "benchmarks/ops/bench_softmax.py::test_softmax_bench[attn-weights-4k-float16]",
            "benchmarks/ops/bench_softmax.py::test_softmax_bench[attn-weights-4k-bfloat16]",
            "benchmarks/ops/bench_softmax.py::test_softmax_bench[attn-weights-4k-float32]",
            "benchmarks/ops/bench_softmax.py::test_log_softmax_bench[attn-weights-4k-float16]",
            "benchmarks/ops/bench_softmax.py::test_log_softmax_bench[attn-weights-4k-bfloat16]",
            "benchmarks/ops/bench_softmax.py::test_log_softmax_bench[attn-weights-4k-float32]",
        ),
    ),
    (
        _BENCHMARK_API_MISMATCH,
        (),
    ),
    (
        _AUTOTUNE_FAILURE,
        (),
    ),
    (
        _NUMERICAL_MISMATCH,
        (),
    ),
    (
        _OUT_OF_MEMORY,
        (
            "benchmarks/ops/bench_gated_deltanet_prefill.py::test_gated_deltanet_prefill_bhtd_bench[bhtd-qwen35-gdn-prefill-b1-s128k-h64-d128-float16]",
            "benchmarks/ops/bench_gated_deltanet_prefill.py::test_gated_deltanet_prefill_bhtd_bench[bhtd-qwen35-gdn-prefill-b1-s128k-h64-d128-bfloat16]",
            "benchmarks/ops/bench_moe_fused_moe.py::test_fused_moe_fwd_bench[deepseek-v3-prefill-bfloat16]",
            "benchmarks/ops/bench_moe_fused_moe.py::test_fused_moe_fwd_bench[kimi-k2-decode-bfloat16]",
            "benchmarks/ops/bench_moe_shared_fused_moe.py::test_shared_fused_moe_bench[1-384-8-7168-2048-18432-sigmoid-renormalize-correctionbias-2.827-bfloat16]",
            "benchmarks/ops/bench_moe_shared_fused_moe.py::test_shared_fused_moe_bench[32-384-8-7168-2048-18432-sigmoid-renormalize-correctionbias-2.827-bfloat16]",
            "benchmarks/ops/bench_moe_shared_fused_moe.py::test_shared_fused_moe_bench[512-384-8-7168-2048-18432-sigmoid-renormalize-correctionbias-2.827-bfloat16]",
            "benchmarks/ops/bench_moe_shared_fused_moe.py::test_shared_fused_moe_bench[2048-384-8-7168-2048-18432-sigmoid-renormalize-correctionbias-2.827-bfloat16]",
            "benchmarks/ops/bench_moe_shared_fused_moe.py::test_shared_fused_moe_bench[4096-384-8-7168-2048-18432-sigmoid-renormalize-correctionbias-2.827-bfloat16]",
        ),
    ),
    (
        _MISSING_BASELINE,
        (
            "benchmarks/ops/bench_pool.py::test_avg_pool3d_bench[video-2x2x2-float16]",
            "benchmarks/ops/bench_pool.py::test_avg_pool3d_bench[ceil-video-float16]",
            "benchmarks/ops/bench_pool.py::test_max_pool3d_bench[c3d-pool1-float16]",
            "benchmarks/ops/bench_pool.py::test_max_pool3d_bench[c3d-pool2-float16]",
            "benchmarks/ops/bench_pool.py::test_max_pool3d_bench[medicalnet-stem-bfloat16]",
            "benchmarks/ops/bench_rope.py::test_rope_neox_position_ids_bench[position-ids-s2k-h32-d128-float16]",
            "benchmarks/ops/bench_rope.py::test_rope_neox_position_ids_bench[position-ids-s4k-h32-d128-bfloat16]",
            "benchmarks/ops/bench_topk_selector.py::test_topk_selector_bench[topk1024-s32k-kv64k-float32]",
            "benchmarks/ops/bench_topk_selector.py::test_topk_selector_bench[topk2048-s32k-kv64k-float32]",
        ),
    ),
)

# Prefix rules are useful for parameterized benchmarks whose every case fails
# for the same backend/environment reason.  The prefix is matched against the
# normalized pytest nodeid, so it may target one function or an entire file.
# Keep exact nodeids above for partial failures where healthy cases must remain
# visible.
_MACA_XFAIL_PREFIX_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Example:
    # (
    #     _COMPILATION_FAILURE,
    #     ("benchmarks/ops/attention/bench_gqa.py::test_gqa_bwd",),
    # ),
)

MACA_XFAILS = {nodeid: reason for reason, nodeids in _MACA_XFAIL_GROUPS for nodeid in nodeids}

MACA_XFAIL_PREFIXES = {
    prefix: reason for reason, prefixes in _MACA_XFAIL_PREFIX_GROUPS for prefix in prefixes
}

if len(MACA_XFAILS) != sum(len(nodeids) for _, nodeids in _MACA_XFAIL_GROUPS):
    raise ValueError("duplicate node ID in the MACA benchmark xfail allowlist")

if len(MACA_XFAIL_PREFIXES) != sum(len(prefixes) for _, prefixes in _MACA_XFAIL_PREFIX_GROUPS):
    raise ValueError("duplicate node ID prefix in the MACA benchmark xfail allowlist")
