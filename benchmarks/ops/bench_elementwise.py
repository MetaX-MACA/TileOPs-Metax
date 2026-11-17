"""Benchmarks for every elementwise manifest entry, one test per op over its manifest calls.

Each case is one workload row and dtype case; ``ElementwiseCall`` draws its inputs and holds
the op's reference. Every row is timed against the reference in torch eager and through
inductor. The fused gated ops are benchmarked in ``bench_fused_gated.py``, beside
flashinfer's kernels.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    compiled_reference,
)
from tileops._utils import is_maca
from tileops.elementwise import (
    AbsFwdOp,
    AddFwdOp,
    AlibiFwdOp,
    BitwiseAndFwdOp,
    BitwiseNotFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
    CeilFwdOp,
    ClampScalarFwdOp,
    ClampTensorFwdOp,
    CosFwdOp,
    DivFwdOp,
    DropoutFwdOp,
    ELUFwdOp,
    EqFwdOp,
    ErfFwdOp,
    ExpFwdOp,
    Expm1FwdOp,
    FloorDivideFwdOp,
    FloorFwdOp,
    GeFwdOp,
    GELUFwdOp,
    GtFwdOp,
    HardsigmoidFwdOp,
    HardswishFwdOp,
    HardtanhFwdOp,
    IsfiniteFwdOp,
    IsinfFwdOp,
    IsnanFwdOp,
    LeakyReLUFwdOp,
    LeFwdOp,
    LerpScalarFwdOp,
    LerpTensorFwdOp,
    Log1pFwdOp,
    LogFwdOp,
    LogicalAndFwdOp,
    LogicalNotFwdOp,
    LogicalOrFwdOp,
    LtFwdOp,
    MaskedFillScalarFwdOp,
    MaskedFillTensorFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MishFwdOp,
    MulFwdOp,
    NanToNumFwdOp,
    NeFwdOp,
    NegFwdOp,
    PowFwdOp,
    PReLUFwdOp,
    ReciprocalFwdOp,
    ReLUFwdOp,
    RemainderFwdOp,
    RoundFwdOp,
    RsqrtFwdOp,
    SELUFwdOp,
    SigmoidFwdOp,
    SignFwdOp,
    SiLUFwdOp,
    SinFwdOp,
    SinusoidalFwdOp,
    SoftplusFwdOp,
    SqrtFwdOp,
    SubFwdOp,
    TanhFwdOp,
    TruncFwdOp,
    WhereFwdOp,
)


def _bench(op_cls, case: bench.Case, *, torch_tag: str = "torch"):
    """Time the op against torch, in eager and through inductor."""
    op = op_cls(**case.arguments)
    bench.Runner(op, case).compare(
        {
            torch_tag: case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
        }
    )


@pytest.mark.parametrize("case", bench.cases(PReLUFwdOp), ids=lambda case: case.id)
def test_prelu_bench(case) -> None:
    if is_maca():
        op = PReLUFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(PReLUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MaskedFillTensorFwdOp), ids=lambda case: case.id)
def test_masked_fill_bench(case) -> None:
    if is_maca():
        op = MaskedFillTensorFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(MaskedFillTensorFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MaskedFillScalarFwdOp), ids=lambda case: case.id)
def test_masked_fill_scalar_bench(case) -> None:
    if is_maca():
        op = MaskedFillScalarFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(MaskedFillScalarFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(AddFwdOp), ids=lambda case: case.id)
def test_add_bench(case) -> None:
    if is_maca():
        op = AddFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(AddFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SubFwdOp), ids=lambda case: case.id)
def test_sub_bench(case) -> None:
    if is_maca():
        op = SubFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(SubFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MulFwdOp), ids=lambda case: case.id)
def test_mul_bench(case) -> None:
    if is_maca():
        op = MulFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(MulFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(DivFwdOp), ids=lambda case: case.id)
def test_div_bench(case) -> None:
    if is_maca():
        op = DivFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(DivFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(RemainderFwdOp), ids=lambda case: case.id)
def test_remainder_bench(case) -> None:
    if is_maca():
        op = RemainderFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(RemainderFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(PowFwdOp), ids=lambda case: case.id)
def test_pow_bench(case) -> None:
    if is_maca():
        op = PowFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(PowFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(FloorDivideFwdOp), ids=lambda case: case.id)
def test_floor_divide_bench(case) -> None:
    if is_maca():
        op = FloorDivideFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(FloorDivideFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LerpScalarFwdOp), ids=lambda case: case.id)
def test_lerp_bench(case) -> None:
    if is_maca():
        op = LerpScalarFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(LerpScalarFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MaximumFwdOp), ids=lambda case: case.id)
def test_maximum_bench(case) -> None:
    if is_maca():
        op = MaximumFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(MaximumFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MinimumFwdOp), ids=lambda case: case.id)
def test_minimum_bench(case) -> None:
    if is_maca():
        op = MinimumFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(MinimumFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(EqFwdOp), ids=lambda case: case.id)
def test_eq_bench(case) -> None:
    if is_maca():
        op = EqFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(EqFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(NeFwdOp), ids=lambda case: case.id)
def test_ne_bench(case) -> None:
    if is_maca():
        op = NeFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(NeFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(GtFwdOp), ids=lambda case: case.id)
def test_gt_bench(case) -> None:
    if is_maca():
        op = GtFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(GtFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LtFwdOp), ids=lambda case: case.id)
def test_lt_bench(case) -> None:
    if is_maca():
        op = LtFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(LtFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(GeFwdOp), ids=lambda case: case.id)
def test_ge_bench(case) -> None:
    if is_maca():
        op = GeFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(GeFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LeFwdOp), ids=lambda case: case.id)
def test_le_bench(case) -> None:
    if is_maca():
        op = LeFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(LeFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LogicalAndFwdOp), ids=lambda case: case.id)
def test_logical_and_bench(case) -> None:
    if is_maca():
        op = LogicalAndFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(LogicalAndFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LogicalOrFwdOp), ids=lambda case: case.id)
def test_logical_or_bench(case) -> None:
    if is_maca():
        op = LogicalOrFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(LogicalOrFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(BitwiseAndFwdOp), ids=lambda case: case.id)
def test_bitwise_and_bench(case) -> None:
    if is_maca():
        op = BitwiseAndFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(BitwiseAndFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(BitwiseOrFwdOp), ids=lambda case: case.id)
def test_bitwise_or_bench(case) -> None:
    if is_maca():
        op = BitwiseOrFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(BitwiseOrFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(BitwiseXorFwdOp), ids=lambda case: case.id)
def test_bitwise_xor_bench(case) -> None:
    if is_maca():
        op = BitwiseXorFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(BitwiseXorFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(AlibiFwdOp), ids=lambda case: case.id)
def test_alibi_bench(case) -> None:
    if is_maca():
        op = AlibiFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(AlibiFwdOp, case, torch_tag="torch-ref")


@pytest.mark.parametrize("case", bench.cases(SinusoidalFwdOp), ids=lambda case: case.id)
def test_sinusoidal_bench(case) -> None:
    if is_maca():
        op = SinusoidalFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(SinusoidalFwdOp, case, torch_tag="torch-ref")


@pytest.mark.parametrize("case", bench.cases(WhereFwdOp), ids=lambda case: case.id)
def test_where_bench(case) -> None:
    if is_maca():
        op = WhereFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(WhereFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LerpTensorFwdOp), ids=lambda case: case.id)
def test_lerp_tensor_bench(case) -> None:
    if is_maca():
        op = LerpTensorFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(LerpTensorFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ReLUFwdOp), ids=lambda case: case.id)
def test_relu_bench(case) -> None:
    if is_maca():
        op = ReLUFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(ReLUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(GELUFwdOp), ids=lambda case: case.id)
def test_gelu_bench(case) -> None:
    if is_maca():
        op = GELUFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(GELUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SiLUFwdOp), ids=lambda case: case.id)
def test_silu_bench(case) -> None:
    if is_maca():
        op = SiLUFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(SiLUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(HardswishFwdOp), ids=lambda case: case.id)
def test_hardswish_bench(case) -> None:
    if is_maca():
        op = HardswishFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(HardswishFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(HardsigmoidFwdOp), ids=lambda case: case.id)
def test_hardsigmoid_bench(case) -> None:
    if is_maca():
        op = HardsigmoidFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(HardsigmoidFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MishFwdOp), ids=lambda case: case.id)
def test_mish_bench(case) -> None:
    if is_maca():
        op = MishFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(MishFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SELUFwdOp), ids=lambda case: case.id)
def test_selu_bench(case) -> None:
    if is_maca():
        op = SELUFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(SELUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LeakyReLUFwdOp), ids=lambda case: case.id)
def test_leaky_relu_bench(case) -> None:
    if is_maca():
        op = LeakyReLUFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(LeakyReLUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ELUFwdOp), ids=lambda case: case.id)
def test_elu_bench(case) -> None:
    if is_maca():
        op = ELUFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(ELUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(HardtanhFwdOp), ids=lambda case: case.id)
def test_hardtanh_bench(case) -> None:
    if is_maca():
        op = HardtanhFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(HardtanhFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SoftplusFwdOp), ids=lambda case: case.id)
def test_softplus_bench(case) -> None:
    if is_maca():
        op = SoftplusFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(SoftplusFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ClampTensorFwdOp), ids=lambda case: case.id)
def test_clamp_bench(case) -> None:
    if is_maca():
        op = ClampTensorFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(ClampTensorFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ClampScalarFwdOp), ids=lambda case: case.id)
def test_clamp_scalar_bench(case) -> None:
    if is_maca():
        op = ClampScalarFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(ClampScalarFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(NanToNumFwdOp), ids=lambda case: case.id)
def test_nan_to_num_bench(case) -> None:
    if is_maca():
        op = NanToNumFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(NanToNumFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ExpFwdOp), ids=lambda case: case.id)
def test_exp_bench(case) -> None:
    if is_maca():
        op = ExpFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(ExpFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LogFwdOp), ids=lambda case: case.id)
def test_log_bench(case) -> None:
    if is_maca():
        op = LogFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(LogFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SqrtFwdOp), ids=lambda case: case.id)
def test_sqrt_bench(case) -> None:
    if is_maca():
        op = SqrtFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(SqrtFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(RsqrtFwdOp), ids=lambda case: case.id)
def test_rsqrt_bench(case) -> None:
    if is_maca():
        op = RsqrtFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(RsqrtFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(AbsFwdOp), ids=lambda case: case.id)
def test_abs_bench(case) -> None:
    if is_maca():
        op = AbsFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(AbsFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(NegFwdOp), ids=lambda case: case.id)
def test_neg_bench(case) -> None:
    if is_maca():
        op = NegFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(NegFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ReciprocalFwdOp), ids=lambda case: case.id)
def test_reciprocal_bench(case) -> None:
    if is_maca():
        op = ReciprocalFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(ReciprocalFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SignFwdOp), ids=lambda case: case.id)
def test_sign_bench(case) -> None:
    if is_maca():
        op = SignFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(SignFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SinFwdOp), ids=lambda case: case.id)
def test_sin_bench(case) -> None:
    if is_maca():
        op = SinFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(SinFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(CosFwdOp), ids=lambda case: case.id)
def test_cos_bench(case) -> None:
    if is_maca():
        op = CosFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(CosFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(FloorFwdOp), ids=lambda case: case.id)
def test_floor_bench(case) -> None:
    if is_maca():
        op = FloorFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(FloorFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(CeilFwdOp), ids=lambda case: case.id)
def test_ceil_bench(case) -> None:
    if is_maca():
        op = CeilFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(CeilFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(RoundFwdOp), ids=lambda case: case.id)
def test_round_bench(case) -> None:
    if is_maca():
        op = RoundFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(RoundFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(TruncFwdOp), ids=lambda case: case.id)
def test_trunc_bench(case) -> None:
    if is_maca():
        op = TruncFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(TruncFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ErfFwdOp), ids=lambda case: case.id)
def test_erf_bench(case) -> None:
    if is_maca():
        op = ErfFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(ErfFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(Log1pFwdOp), ids=lambda case: case.id)
def test_log1p_bench(case) -> None:
    if is_maca():
        op = Log1pFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(Log1pFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(Expm1FwdOp), ids=lambda case: case.id)
def test_expm1_bench(case) -> None:
    if is_maca():
        op = Expm1FwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(Expm1FwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SigmoidFwdOp), ids=lambda case: case.id)
def test_sigmoid_bench(case) -> None:
    if is_maca():
        op = SigmoidFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(SigmoidFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(TanhFwdOp), ids=lambda case: case.id)
def test_tanh_bench(case) -> None:
    if is_maca():
        op = TanhFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(TanhFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LogicalNotFwdOp), ids=lambda case: case.id)
def test_logical_not_bench(case) -> None:
    if is_maca():
        op = LogicalNotFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(LogicalNotFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(BitwiseNotFwdOp), ids=lambda case: case.id)
def test_bitwise_not_bench(case) -> None:
    if is_maca():
        op = BitwiseNotFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(BitwiseNotFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(IsnanFwdOp), ids=lambda case: case.id)
def test_isnan_bench(case) -> None:
    if is_maca():
        op = IsnanFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(IsnanFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(IsinfFwdOp), ids=lambda case: case.id)
def test_isinf_bench(case) -> None:
    if is_maca():
        op = IsinfFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(IsinfFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(IsfiniteFwdOp), ids=lambda case: case.id)
def test_isfinite_bench(case) -> None:
    if is_maca():
        op = IsfiniteFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(IsfiniteFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(DropoutFwdOp), ids=lambda case: case.id)
def test_dropout_bench(case) -> None:
    if is_maca():
        op = DropoutFwdOp(**case.arguments)
        bench.Runner(op, case).compare_torch()
        return
    _bench(DropoutFwdOp, case)
