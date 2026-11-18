"""Exact-node allowlist for known benchmark failures on MetaX MACA."""

_UNSUPPORTED_ARCHITECTURE = "kernel is not supported on the current MACA architecture"
_RUNTIME_LAUNCH_ERROR = "known MACA runtime launch error"
_COMPILATION_FAILURE = "known MACA benchmark compilation failure"
_BENCHMARK_API_MISMATCH = "benchmark uses an incompatible cumulative Op constructor"
_AUTOTUNE_FAILURE = "no benchmark configuration compiles and validates successfully"
_NUMERICAL_MISMATCH = "known MACA numerical mismatch"
_OUT_OF_MEMORY = "known MACA out-of-memory on this workload"
_MISSING_BASELINE = "required baseline/package unavailable or incompatible on MACA"
_UNSUPPORTED_RUNTIME = "known MACA unsupported runtime operation"


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
        ),
    ),
    (
        _RUNTIME_LAUNCH_ERROR,
        (),
    ),
    (
        _COMPILATION_FAILURE,
        (
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
            "benchmarks/ops/bench_elementwise_manifest.py::test_logical_and_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_logical_or_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_and_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_or_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_xor_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_fp8_lightning_indexer.py::test_fp8_lightning_indexer_bench[lightning-indexer-s8k-h32-d64-bfloat16]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[gla-bwd-b2-s2k-h4-d64-float16]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[gla-bwd-b2-s2k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[gla-bwd-b2-s4k-h4-d64-float16]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[gla-bwd-b2-s4k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[gla-bwd-b2-s8k-h4-d64-float16]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[gla-bwd-b2-s8k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[gla-bwd-b2-s16k-h4-d64-float16]",
            "benchmarks/ops/bench_gla_chunkwise.py::test_gla_bwd_bench[gla-bwd-b2-s16k-h4-d64-bfloat16]",
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
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_autograd[dn-autograd-b2-s2k-h4-d64-float16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_autograd[dn-autograd-b2-s2k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_autograd[dn-autograd-b2-s8k-h4-d64-float16]",
            "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_autograd[dn-autograd-b2-s8k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_autograd[gdn-autograd-b2-s2k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_autograd[gdn-autograd-b2-s2k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_autograd[gdn-autograd-b2-s8k-h4-d64-float16]",
            "benchmarks/ops/bench_gated_deltanet.py::test_gated_deltanet_vs_fla_autograd[gdn-autograd-b2-s8k-h4-d64-bfloat16]",
            "benchmarks/ops/bench_group_norm.py::test_group_norm_bench[wider-channel-g32-affine-float16]",
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
        ("tests/ops/test_convolution.py::test_conv3d_dispatches_ndhwc_kernel_no_bias",),
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
            "benchmarks/ops/bench_moe_grouped_gemm_nopad.py::test_moe_grouped_gemm_nopad_bench[deepseek-v3-prefill-gate-up-bfloat16]",
        ),
    ),
    (
        _MISSING_BASELINE,
        (),
    ),
)

# Prefix rules are useful for parameterized benchmarks whose every case fails
# for the same backend/environment reason.  The prefix is matched against the
# normalized pytest nodeid, so it may target one function or an entire file.
# Keep exact nodeids above for partial failures where healthy cases must remain
# visible.
_MACA_XFAIL_PREFIX_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        _COMPILATION_FAILURE,
        (
            "benchmarks/ops/attention/bench_gqa.py::test_gqa_dense_decode_bench[",
            "benchmarks/ops/bench_gemm.py::test_gemm_fp8_bench[",
            "benchmarks/ops/bench_sampling_from_probs.py::test_sampling_from_probs_bench[",
            "benchmarks/ops/bench_softmax.py::test_softmax_bench[",
            "benchmarks/ops/bench_softmax.py::test_log_softmax_bench[",
            "benchmarks/ops/bench_top_p_mask.py::test_top_p_mask_bench[",
        ),
    ),
    (
        _UNSUPPORTED_ARCHITECTURE,
        (),
    ),
    (
        _OUT_OF_MEMORY,
        (
            "benchmarks/ops/bench_fused_moe.py::test_fused_moe_fwd_bench[ds-v3-grouped-prefill-bfloat16]",
            "benchmarks/ops/bench_fused_moe.py::test_fused_moe_fwd_bench[kimi-k2-prefill-bfloat16]",
            "benchmarks/ops/bench_fused_moe_experts.py::test_moe_experts_bench[ds-v3-decode-bfloat16]",
            "benchmarks/ops/bench_fused_moe_experts.py::test_moe_experts_bench[ds-v3-prefill-bfloat16]",
            "benchmarks/ops/bench_fused_moe_experts.py::test_indexed_expert_mlp_bench[ds-v3-t1-float16]",
            "benchmarks/ops/bench_fused_moe_experts.py::test_indexed_expert_mlp_bench[ds-v3-t32-float16]",
            "benchmarks/ops/bench_fused_moe_experts.py::test_indexed_expert_mlp_bench[ds-v3-t64-float16]",
            "benchmarks/ops/bench_mamba.py::test_ssd_chunk_state_fwd_bench[mamba2-1p3b-float16-float32]",
            "benchmarks/ops/bench_moe_fused_moe_shared_expert.py::test_fused_moe_shared_expert_bench[",
            "benchmarks/ops/bench_fused_moe.py::test_fused_moe_shared_expert_bench[kimi-k2-",
            "benchmarks/ops/bench_fused_moe.py::test_fused_moe_shared_expert_bench[ds-v3-grouped-t32-bfloat16]",
            "benchmarks/ops/bench_fused_moe.py::test_fused_moe_shared_expert_bench[ds-v3-grouped-t64-bfloat16]",
            "benchmarks/ops/bench_fused_moe.py::test_fused_moe_shared_expert_bench[ds-v3-grouped-t512-bfloat16]",
            "benchmarks/ops/bench_fused_moe.py::test_fused_moe_shared_expert_bench[ds-v3-grouped-t4096-bfloat16]",
            "benchmarks/ops/bench_moe_staged.py::test_moe_grouped_gemm_bench[deepseek-v3-prefill-gate-up-bfloat16]",
            "benchmarks/ops/bench_moe_staged.py::test_moe_grouped_gemm_bench[deepseek-v3-decode-gate-up-fused-bfloat16]",
            "benchmarks/ops/bench_moe_staged.py::test_moe_expert_mlp_bench[deepseek-v3-",
            "benchmarks/ops/bench_moe_staged.py::test_moe_expert_mlp_bench[qwen3-235b-",
        ),
    ),
    (
        _UNSUPPORTED_RUNTIME,
        (
            "benchmarks/ops/bench_dsa_decode.py::test_dsa_decode_bench[",
            "benchmarks/ops/bench_fused_topk.py::test_fused_topk_bench[",
        ),
    ),
)

# Latest MACA benchmark failures grouped by test function. These cases currently
# fail in the backend/runtime or in the benchmark reference contract.
_MACA_LOG_FAILURE = "known MACA benchmark failure from latest runner"
_MACA_LOG_XFAIL_PREFIXES = (
    "benchmarks/ops/bench_deltanet.py::test_deltanet_fwd_bench[",
    "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_bwd[",
    "benchmarks/ops/bench_deltanet.py::test_deltanet_vs_fla_fwd[",
    "benchmarks/ops/bench_elementwise.py::test_bitwise_and_bench[",
    "benchmarks/ops/bench_elementwise.py::test_bitwise_or_bench[",
    "benchmarks/ops/bench_elementwise.py::test_bitwise_xor_bench[",
    "benchmarks/ops/bench_elementwise.py::test_clamp_scalar_bench[",
    "benchmarks/ops/bench_elementwise.py::test_logical_and_bench[",
    "benchmarks/ops/bench_elementwise.py::test_logical_or_bench[",
    "benchmarks/ops/bench_elementwise.py::test_maximum_bench[",
    "benchmarks/ops/bench_elementwise.py::test_minimum_bench[",
    "benchmarks/ops/bench_engram.py::test_engram_gate_conv_bwd_bench[",
    "benchmarks/ops/bench_fft.py::test_fft_bench[",
    "benchmarks/ops/bench_fused_moe.py::test_fused_moe_fwd_bench[",
    "benchmarks/ops/bench_fused_moe.py::test_fused_moe_shared_expert_bench[",
    "benchmarks/ops/bench_fused_moe_experts.py::test_indexed_expert_mlp_bench[",
    "benchmarks/ops/bench_fused_moe_experts.py::test_moe_experts_bench[",
    "benchmarks/ops/bench_gdn.py::test_gdn_fwd_bench[",
    "benchmarks/ops/bench_gemm.py::test_gemm_w4a16_bench[",
    "benchmarks/ops/bench_gqa.py::test_gqa_dense_fwd_bench[",
    "benchmarks/ops/bench_gqa.py::test_gqa_varlen_fwd_bench[",
    "benchmarks/ops/bench_group_norm.py::test_group_norm_bench[",
    "benchmarks/ops/bench_hadamard_transform.py::test_hadamard_transform_bench[",
    "benchmarks/ops/bench_int4_quant_per_group.py::test_int4_quant_per_group_bench[",
    "benchmarks/ops/bench_mamba.py::test_causal_conv1d_bench[",
    "benchmarks/ops/bench_mamba.py::test_causal_conv1d_update_bench[",
    "benchmarks/ops/bench_mamba.py::test_selective_scan_bench[",
    "benchmarks/ops/bench_mamba.py::test_ssd_decode_bench[",
    "benchmarks/ops/bench_mla.py::test_mla_paged_bench[",
    "benchmarks/ops/bench_mla.py::test_mla_varlen_fwd_bench[",
    "benchmarks/ops/bench_moe_permute_align.py::test_permute_align_bench[",
    "benchmarks/ops/bench_moe_staged.py::test_moe_expert_mlp_bench[",
    "benchmarks/ops/bench_moe_staged.py::test_moe_grouped_gemm_bench[",
    "benchmarks/ops/bench_norm.py::test_fused_add_layer_norm_bench[",
    "benchmarks/ops/bench_norm.py::test_fused_add_rms_norm_per_token_quant_bench[",
    "benchmarks/ops/bench_pool.py::test_mean_pooling_bench[",
    "benchmarks/ops/bench_topk_select.py::test_topk_select_bench[",
    "benchmarks/ops/bench_vector_norm.py::test_vector_norm_bench[",
)


MACA_XFAILS = {nodeid: reason for reason, nodeids in _MACA_XFAIL_GROUPS for nodeid in nodeids}

MACA_XFAIL_PREFIXES = {
    prefix: reason for reason, prefixes in _MACA_XFAIL_PREFIX_GROUPS for prefix in prefixes
}
MACA_XFAIL_PREFIXES.update({prefix: _MACA_LOG_FAILURE for prefix in _MACA_LOG_XFAIL_PREFIXES})

if len(MACA_XFAILS) != sum(len(nodeids) for _, nodeids in _MACA_XFAIL_GROUPS):
    raise ValueError("duplicate node ID in the MACA benchmark xfail allowlist")

_all_maca_prefixes = [prefix for _, prefixes in _MACA_XFAIL_PREFIX_GROUPS for prefix in prefixes]
_all_maca_prefixes += list(_MACA_LOG_XFAIL_PREFIXES)
if len(MACA_XFAIL_PREFIXES) != len(set(_all_maca_prefixes)):
    raise ValueError("duplicate node ID prefix in the MACA benchmark xfail allowlist")
