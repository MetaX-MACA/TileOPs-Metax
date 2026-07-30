"""Exact-node allowlist for known failures on MetaX MACA."""

_UNSUPPORTED_ARCHITECTURE = "kernel is not supported on the current MACA architecture"
_COMPILATION_FAILURE = "known MACA xcore1000 compilation failure"
_NUMERICAL_MISMATCH = "known MACA numerical mismatch"
_RUNTIME_LAUNCH_ERROR = "known MACA runtime launch error"
_NON_INJECTIVE_LAYOUT = "TileLang rejects the generated loop layout as non-injective"
_CI_EXPECTATION_MISMATCH = "MetaX CI workflow does not match the upstream assertion"
_TRACE_RUNTIME_ERROR = "known MACA trace payload runtime error"


# FIXME(staged-rollout): quarantine the current MetaX backend failures by exact node ID.
#
# Broken invariant: every collected TileOps test passes on the MetaX test runner.
# Why: the backend still has architecture gaps, launch failures, and numerical mismatches.
# Cleanup: remove each entry as soon as its node passes consistently on the MetaX runner.
_MACA_XFAIL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        _COMPILATION_FAILURE,
        (
            "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[2-64-2-64-64-32-dtype0-False]",
            "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[2-64-2-64-64-32-dtype1-False]",
            "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[2-64-2-64-64-32-dtype2-False]",
            "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[2-64-2-64-64-32-dtype0-False]",
            "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[2-64-2-64-64-32-dtype1-False]",
            "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[2-64-2-64-64-32-dtype2-False]",
            "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[2-64-2-64-64-32-dtype0-False]",
            "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[2-64-2-64-64-32-dtype1-False]",
            "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[2-64-2-64-64-32-dtype2-False]",
            "tests/ops/test_gated_deltanet_prefill.py::test_gated_deltanet_prefill_fwd[1-64-2-64-64-32-dtype0-False]",
            "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[2-64-2-64-64-64-dtype0-False]",
        ),
    ),
    (
        _NUMERICAL_MISMATCH,
        (
            "tests/ops/attention/test_gqa.py::test_gqa_bwd[4-2048-64-4-128-False-dtype3-False]",
            "tests/ops/test_convolution.py::test_conv1d[full-padding-valid-fp16]",
            "tests/ops/test_convolution.py::test_conv1d_same_padding_even_kernel_matches_torch[bias]",
            "tests/ops/test_fused_moe_experts.py::TestFusedMoEExpertsNopadPersistent3WGFwdOp::test_forward_matches_torch_ref_activation[float16-gelu_and_mul]",
            "tests/ops/test_mamba.py::test_da_cumsum_fwd[2-4-128-16-True-True-dtype6-False]",
            "tests/ops/test_mamba.py::test_ssd_chunk_scan_fwd[1-2-64-4-64-32-1-dtype0-False]",
            "tests/ops/test_mamba.py::test_ssd_chunk_scan_fwd[1-2-128-4-128-32-1-dtype1-False]",
            "tests/ops/test_mamba.py::test_ssd_chunk_scan_fwd[2-4-64-8-64-64-2-dtype2-False]",
            "tests/ops/test_mamba.py::test_ssd_chunk_scan_fwd[2-2-64-4-64-32-2-dtype3-False]",
            "tests/ops/test_mamba.py::test_mamba2_fwd_e2e[1-256-4-64-32-1-256-dtype0]",
            "tests/ops/test_mamba.py::test_mamba2_fwd_e2e[1-256-4-64-32-1-256-dtype1]",
            "tests/ops/test_mamba.py::test_mamba2_fwd_e2e[2-512-8-64-64-2-256-dtype0]",
            "tests/ops/test_mamba.py::test_mamba2_fwd_e2e[2-512-8-64-64-2-256-dtype1]",
            "tests/ops/test_mamba.py::test_mamba2_fwd_e2e[1-512-4-64-128-1-256-dtype0]",
            "tests/ops/test_mamba.py::test_mamba2_fwd_e2e[1-512-4-64-128-1-256-dtype1]",
        ),
    ),
    (
        _RUNTIME_LAUNCH_ERROR,
        (
            "tests/ops/test_argreduce.py::test_argmin_op[256-4096-dtype3]",
            "tests/ops/test_argreduce.py::test_argmin_op[256-4096-dtype4]",
            "tests/ops/test_argreduce.py::test_argmin_dim_none[shape0-dtype0]",
            "tests/ops/test_argreduce.py::test_argmin_dim_none[shape1-dtype1]",
            "tests/ops/test_argreduce.py::test_argmin_dim_none[shape2-dtype2]",
            "tests/ops/test_argreduce.py::test_argmin_dim_none[shape4-dtype4]",
            "tests/ops/test_argreduce.py::test_argmin_dim_none[shape5-dtype5]",
            "tests/ops/test_engram.py::test_engram_decode[4-1024-512-20-4-5-dtype2-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[2-64-2-64-64-32-dtype0-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[2-64-2-64-64-32-dtype1-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[2-64-2-64-64-32-dtype2-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[1-128-4-64-64-32-dtype3-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[1-128-4-64-64-32-dtype4-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[1-128-4-64-64-32-dtype5-False]",
        ),
    ),
    (
        _NON_INJECTIVE_LAYOUT,
        (
            "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[2-64-2-64-64-64-dtype0-False]",
            "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[2-64-2-64-64-64-dtype1-False]",
            "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[2-64-2-64-64-64-dtype2-False]",
            "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[1-128-4-64-64-64-dtype3-False]",
            "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[1-128-4-64-64-64-dtype4-False]",
            "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[1-128-4-64-64-64-dtype5-False]",
        ),
    ),
    (
        _CI_EXPECTATION_MISMATCH,
        (
        ),
    ),
    (
        _TRACE_RUNTIME_ERROR,
        (
            "tests/trace/test_payload.py::test_payload_with_range_start_end",
            "tests/trace/test_payload.py::test_implicit_thread_blocks_with_payload_e2e",
        ),
    ),
)

MACA_XFAILS = {
    nodeid: reason
    for reason, nodeids in _MACA_XFAIL_GROUPS
    for nodeid in nodeids
}

if len(MACA_XFAILS) != sum(len(nodeids) for _, nodeids in _MACA_XFAIL_GROUPS):
    raise ValueError("duplicate node ID in the MACA xfail allowlist")
