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
        _UNSUPPORTED_ARCHITECTURE,
        (),
    ),
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
            "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[1-128-4-64-64-64-dtype3-False]",
            "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[2-64-2-64-64-64-dtype0-False]",
            "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[1-128-4-64-64-32-dtype3-False]",
            "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[1-128-4-64-64-32-dtype4-False]",
            "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[1-128-4-64-64-32-dtype5-False]",
            "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[full-bf16-tuned]",
            "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[1-128-4-64-64-32-dtype3-False]",
            "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[1-128-4-64-64-32-dtype4-False]",
            "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[1-128-4-64-64-32-dtype5-False]",
            "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[2-8192-4-64-64-64-dtype6-False]",
            "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[2-16384-4-64-64-64-dtype7-False]",
            "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[full-bf16-tuned]",
            "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[1-128-4-64-64-32-dtype3-False]",
            "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[1-128-4-64-64-32-dtype4-False]",
            "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[1-128-4-64-64-32-dtype5-False]",
            "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[2-8192-4-64-64-64-dtype6-False]",
            "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[2-16384-4-64-64-64-dtype7-False]",
            "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[full-bf16-tuned]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd_segmented_carry_matches_sequential_d128[dtype0-64]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd_segmented_carry_matches_sequential_d128[dtype1-128]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[1-128-2-128-128-64-dtype6-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[2-64-2-64-64-32-dtype0-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[2-64-2-64-64-32-dtype1-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[2-64-2-64-64-32-dtype2-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[1-128-4-64-64-32-dtype3-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[1-128-4-64-64-32-dtype4-False]",
            "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[1-128-4-64-64-32-dtype5-False]",
            "tests/ops/test_multi_dtype_instance.py::test_bitwise_alternates_between_bool_and_integer_storage",
            "tests/ops/test_multi_dtype_instance.py::test_logical_and_output_stays_bool_across_input_storage",
            "tests/ops/test_multi_dtype_instance.py::test_single_tensor_op_records_its_dtype[LogicalNotFwdOp]",
            "tests/ops/test_convolution.py::test_conv3d[smoke-video-depthwise3d-small-fp16]",
            "tests/ops/test_deltanet_autograd.py::test_deltanet_autograd_matches_the_ops_it_wraps",
            "tests/ops/test_deltanet_autograd.py::test_gated_deltanet_autograd_matches_the_ops_it_wraps",
            "tests/ops/test_pool.py::test_max_pool2d[smoke-3x3-s2-p1-bf16]",
            "tests/ops/test_pool.py::test_max_pool2d[full-nonsquare-ceil-bf16]",
            "tests/ops/test_softmax.py::test_log_softmax_op[shape6--1-dtype6-False]",
            "tests/ops/test_softmax.py::test_log_softmax_non_contiguous[shape5-dtype5]",
            "tests/ops/test_softmax.py::test_log_softmax_1d[300-dtype5]",
        ),
    ),
    (
        _NUMERICAL_MISMATCH,
        (
            "tests/ops/test_convolution.py::test_conv2d_batch_with_partial_tile_leaves_the_symmetric_kernel",
            "tests/ops/test_convolution.py::test_conv3d_does_not_dispatch_ndhwc_for_small_output",
            "tests/ops/test_vector_norm.py::test_vector_norm_long_sequence_tiled[inf]",
        ),
    ),
    (
        _RUNTIME_LAUNCH_ERROR,
        (
            "tests/ops/test_convolution.py::test_conv3d[smoke-3d-ndhwc-nonsymmetric-dilation-fp16]",
            "tests/ops/test_convolution.py::test_conv3d[full-video-stage-downsample-k3-s2-fp16]",
            "tests/ops/test_convolution.py::test_conv3d[full-unet-encoder-k3-s1-bf16]",
            "tests/ops/test_convolution.py::test_conv3d_dispatches_ndhwc_kernel_no_bias",
            "tests/ops/test_convolution.py::test_conv3d_ndhwc_kernel_roofline_counts_layout_traffic",
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
            "tests/ops/attention/test_gqa_fp8.py::test_gqa_prefill_fp8_tensor_core_rejects_unaligned_q_tiles[224]",
            "tests/ops/attention/test_gqa_fp8.py::test_gqa_prefill_fp8_tensor_core_rejects_unaligned_q_tiles[672]",
            "tests/ops/test_convolution.py::test_conv1d[smoke-tcn-k3-s1-bf16]",
            "tests/ops/test_convolution.py::test_conv2d[smoke-bf16-3x3]",
            "tests/ops/test_convolution.py::test_conv2d[full-bf16-3x3-s2]",
            "tests/ops/test_convolution.py::test_conv2d[full-bf16-1x1]",
            "tests/ops/test_family_dispatch.py::test_gemm_vector_on_a_transposed_operand_is_refused[lhs-row-trans-a]",
            "tests/ops/test_family_dispatch.py::test_gemm_vector_on_a_transposed_operand_is_refused[rhs-col-trans-a]",
            "tests/ops/test_gemm.py::test_gemm_kernel_tune_falls_back_to_default",
            "tests/ops/test_kernel_selection.py::test_paged_decode_dispatch_is_unchanged[bs1-fp16]",
            "tests/ops/test_moe_staged_contracts.py::test_public_ops_build_complete_calls_before_selection",
            "tests/ops/test_moe_staged_contracts.py::test_staged_wiring_builds_all_family_calls_without_an_executable_candidate",
            "tests/ops/test_moe_staged_contracts.py::test_injected_candidate_uses_common_selection_and_call_spec_cache",
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

MACA_XFAILS = {nodeid: reason for reason, nodeids in _MACA_XFAIL_GROUPS for nodeid in nodeids}

if len(MACA_XFAILS) != sum(len(nodeids) for _, nodeids in _MACA_XFAIL_GROUPS):
    raise ValueError("duplicate node ID in the MACA xfail allowlist")
