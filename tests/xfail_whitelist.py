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
            "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[1-128-4-64-64-64-dtype3-False]",
            "tests/ops/test_ada_layer_norm.py::test_ada_layer_norm_kernel_handles_natural_unaligned_shape[dtype0]",
            "tests/ops/test_ada_layer_norm.py::test_ada_layer_norm_kernel_handles_natural_unaligned_shape[dtype1]",
            "tests/ops/test_ada_layer_norm.py::test_ada_layer_norm_kernel_handles_natural_unaligned_shape[dtype2]",
            "tests/ops/test_ada_layer_norm.py::test_ada_layer_norm_async_copy_handles_row_tail",
            "tests/ops/test_ada_layer_norm.py::test_ada_layer_norm_async_policy_edge_correctness[fp16-lower-inside]",
            "tests/ops/test_ada_layer_norm.py::test_ada_layer_norm_op[1024-3000-dtype6]",
            "tests/ops/test_ada_layer_norm.py::test_ada_layer_norm_op[1024-3000-dtype7]",
            "tests/ops/test_ada_layer_norm.py::test_ada_layer_norm_op[1024-3000-dtype8]",
            "tests/ops/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero_kernel_handles_natural_unaligned_shape[dtype0]",
            "tests/ops/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero_kernel_handles_natural_unaligned_shape[dtype1]",
            "tests/ops/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero_kernel_handles_natural_unaligned_shape[dtype2]",
            "tests/ops/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero_async_copy_handles_row_tail",
            "tests/ops/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero_op[1024-3000-dtype6]",
            "tests/ops/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero_op[1024-3000-dtype7]",
            "tests/ops/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero_op[1024-3000-dtype8]",
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
        ),
    ),
    (
        _NUMERICAL_MISMATCH,
        (),
    ),
    (
        _RUNTIME_LAUNCH_ERROR,
        (
            # F.conv* reference / mcDNN cache miss on MACA
            "tests/ops/test_convolution.py::test_conv1d[full-convtasnet-pointwise-k1-s1-fp16]",
            "tests/ops/test_convolution.py::test_conv1d[full-sequence-downsample-k3-s2-bf16]",
            "tests/ops/test_convolution.py::test_conv1d[full-padding-valid-fp16]",
            "tests/ops/test_convolution.py::test_conv1d[full-padding-same-fp16]",
            "tests/ops/test_convolution.py::test_conv1d_no_bias_matches_torch",
            "tests/ops/test_convolution.py::test_conv1d_bias_matches_torch",
            "tests/ops/test_convolution.py::test_conv1d_dilation_matches_torch[no-bias]",
            "tests/ops/test_convolution.py::test_conv1d_dilation_matches_torch[bias]",
            "tests/ops/test_convolution.py::test_conv1d_dispatches_kernel[pointwise]",
            "tests/ops/test_convolution.py::test_conv2d[smoke-fp16-3x3]",
            "tests/ops/test_convolution.py::test_conv2d[smoke-bf16-3x3]",
            "tests/ops/test_convolution.py::test_conv2d[full-resblock-3x3-s1-fp16]",
            "tests/ops/test_convolution.py::test_conv2d[full-stage-transition-3x3-s2-fp16]",
            "tests/ops/test_convolution.py::test_conv2d[full-small-5x5-s1-fp16]",
            "tests/ops/test_convolution.py::test_conv2d[full-small-5x5-s2-fp16]",
            "tests/ops/test_convolution.py::test_conv2d[full-fp16-stride2]",
            "tests/ops/test_convolution.py::test_conv2d[full-bf16-3x3-s2]",
            "tests/ops/test_convolution.py::test_conv2d[full-deeplab-aspp-3x3-d2-fp16]",
            "tests/ops/test_convolution.py::test_conv2d_no_bias_matches_torch",
            "tests/ops/test_convolution.py::test_conv2d_dispatches_3x3_kernel",
            "tests/ops/test_convolution.py::test_conv2d_dispatches_5x5_kernel",
            "tests/ops/test_convolution.py::test_conv3d[smoke-3d-unet-k3-s1-fp16]",
            "tests/ops/test_convolution.py::test_conv3d[smoke-3d-unet-k3-s1-bf16]",
            "tests/ops/test_convolution.py::test_conv3d[full-video-stage-downsample-k3-s2-fp16]",
            "tests/ops/test_convolution.py::test_conv3d[full-3d-aspp-3x3x3-d2-fp16]",
            "tests/ops/test_convolution.py::test_conv3d[full-3d-resnext-grouped-k3-fp16]",
            "tests/ops/test_convolution.py::test_conv3d_no_bias_grouped_matches_torch",
            "tests/ops/test_convolution.py::test_conv2d_cold_traces_fullgraph_and_owns_its_graph_nodes[no-bias]",
            "tests/ops/test_convolution.py::test_conv2d_cold_traces_fullgraph_and_owns_its_graph_nodes[bias]",
            "tests/ops/test_convolution.py::test_conv1d_cold_traces_fullgraph_and_owns_its_graph_nodes",
            "tests/ops/test_convolution.py::test_a_non_contiguous_input_compiles_to_the_shape_the_fake_promised",
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
        (),
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
