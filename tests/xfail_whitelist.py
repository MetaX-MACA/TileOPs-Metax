"""Exact-node allowlist for known failures on MetaX MACA."""

_UNSUPPORTED_ARCHITECTURE = "kernel is not supported on the current MACA architecture"
_COMPILATION_FAILURE = "known MACA xcore1000 compilation failure"
_NUMERICAL_MISMATCH = "known MACA numerical mismatch"
_RUNTIME_LAUNCH_ERROR = "known MACA runtime launch error"
_NON_INJECTIVE_LAYOUT = "TileLang rejects the generated loop layout as non-injective"
_CI_EXPECTATION_MISMATCH = "MetaX CI workflow does not match the upstream assertion"
_TRACE_RUNTIME_ERROR = "known MACA trace payload runtime error"
_MANIFEST_VALIDATION = "known MACA manifest validation mismatch"


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
            "tests/ops/test_gemm.py::test_gemm_fp8[smoke-fp8-e4m3-per-tensor]",
            "tests/ops/test_gemm.py::test_gemm_fp8[smoke-fp8-e4m3-block128]",
            "tests/ops/test_gemm.py::test_gemm_fp8[full-fp8-e4m3-block128-large-m]",
            "tests/ops/test_gemm.py::test_gemm_fp8[full-fp8-e4m3-per-tensor-small-m-bias]",
            "tests/ops/test_gemm.py::test_gemm_fp8[full-fp8-e4m3-per-tensor-gemv]",
            "tests/ops/test_gemm.py::test_gemm_fp8[full-fp8-e4m3-block128-split-k]",
            "tests/ops/test_gemm.py::test_gemm_fp8[full-fp8-e4m3-per-tensor-split-k-bias]",
            "tests/ops/test_gemm.py::test_gemm_fp8_block128_single_k_block_uses_block_kernel",
            "tests/ops/test_gemm.py::test_gemm_fp8_revalidates_cached_signature_dtypes",
            "tests/ops/test_compile_boundary.py::test_a_cold_op_traces_fullgraph_and_matches_eager[gemm-fp8]",
            "tests/ops/test_compile_boundary.py::test_a_cold_op_traces_fullgraph_and_matches_eager[gla-bwd]",
            "tests/ops/test_compile_boundary.py::test_a_cold_op_traces_fullgraph_and_matches_eager[deltanet-fwd]",
            "tests/ops/test_compile_boundary.py::test_a_cold_op_traces_fullgraph_and_matches_eager[deltanet-bwd]",
            "tests/ops/test_compile_boundary.py::test_the_fake_reports_what_the_op_returns[gemm-fp8]",
            "tests/ops/test_compile_boundary.py::test_the_fake_reports_what_the_op_returns[gla-bwd]",
            "tests/ops/test_compile_boundary.py::test_the_fake_reports_what_the_op_returns[deltanet-fwd]",
            "tests/ops/test_compile_boundary.py::test_the_fake_reports_what_the_op_returns[deltanet-bwd]",
            "tests/ops/test_compile_boundary.py::test_the_traced_graph_holds_only_this_ops_operator[gemm-fp8]",
            "tests/ops/test_compile_boundary.py::test_the_traced_graph_holds_only_this_ops_operator[gla-bwd]",
            "tests/ops/test_compile_boundary.py::test_the_traced_graph_holds_only_this_ops_operator[deltanet-fwd]",
            "tests/ops/test_compile_boundary.py::test_the_traced_graph_holds_only_this_ops_operator[deltanet-bwd]",
            "tests/ops/test_moe_compile.py::test_leaf_op_owns_its_graph_nodes[staged_grouped_gemm_masked]",
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
            "tests/ops/test_deltanet_autograd.py::test_deltanet_autograd_matches_the_ops_it_wraps",
            "tests/ops/test_deltanet_autograd.py::test_gated_deltanet_autograd_matches_the_ops_it_wraps",
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
            "tests/ops/test_gemm.py::test_gemm_w4a16[full-w4a16-gemv-short-k-n-tail]",
            "tests/ops/test_gemm.py::test_gemm_w4a16[full-w4a16-gemv-n-tail]",
            "tests/ops/test_gemm.py::test_gemm_w4a16[full-w4a16-small-mn-tail]",
            "tests/ops/test_gemm.py::test_gemm_w4a16_gemv_preserves_fp32_scale[0]",
            "tests/ops/test_gemm.py::test_gemm_w4a16_gemv_preserves_fp32_scale[127]",
            "tests/ops/test_gemm.py::test_gemm_w4a16_gemv_preserves_fp32_scale[128]",
            "tests/ops/test_gemm.py::test_gemm_w4a16_gemv_preserves_fp32_scale[383]",
            "tests/ops/test_gemm.py::test_dense_splitk_interfaces_match_reference",
            "tests/ops/test_gemm.py::test_coop2_epilogue_chunking_matches_reference[3-0]",
            "tests/ops/test_gemm.py::test_coop2_epilogue_chunking_matches_reference[4-128]",
            "tests/ops/attention/test_gqa.py::test_gqa_decode_tuned_split_count_tracks_runtime_sequence",
            "tests/ops/test_fused_moe_experts.py::TestFusedMoEExpertsFwdOp::test_small_route_dispatch_replays_in_cuda_graph",
        ),
    ),
    (
        _NUMERICAL_MISMATCH,
        (
            "tests/ops/test_topk_selector.py::test_topk_selector_op[4-256-1024-1-32-float32-int32-False]",
            "tests/ops/test_topk_selector.py::test_topk_selector_op[8-512-2048-1-64-float32-int32-False]",
        ),
    ),
    (
        _CI_EXPECTATION_MISMATCH,
        (
            "tests/ops/test_family_dispatch.py::test_gemm_k_too_narrow_to_vectorize_is_refused_during_selection",
            "tests/ops/test_family_dispatch.py::test_gemm_uses_basic_mainloop_off_sm90",
        ),
    ),
    (
        _MANIFEST_VALIDATION,
        (
            "tests/test_validate_manifest.py::TestIntegration::test_validator_passes_on_current_codebase",
            "tests/test_validate_manifest.py::TestIntegration::test_schema_validation_no_errors_on_real_manifest",
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
            "tests/ops/test_kernel_selection.py::test_paged_decode_dispatch_is_unchanged[bs1-fp16]",
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
