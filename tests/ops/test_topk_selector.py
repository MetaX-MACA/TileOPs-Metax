import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import TopkSelectorFwdOp
from tileops.utils import str2dtype
from workloads.topk_selector import TopkSelectorWorkload


class TopkSelectorTest(TopkSelectorWorkload, TestBase):
    pass


class TopkSelectorFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, seq_len_kv, kv_group, topk, in_dtype_str, out_dtype_str, tune",
            [
                pytest.param(
                    4, 256, 1024, 1, 32, "float32", "int32", False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    8, 512, 2048, 1, 64, "float32", "int32", False, marks=pytest.mark.full
                ),
                pytest.param(
                    1,
                    32 * 1024,
                    64 * 1024,
                    1,
                    1024,
                    "float32",
                    "int32",
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1,
                    32 * 1024,
                    64 * 2048,
                    1,
                    2048,
                    "float32",
                    "int32",
                    False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


def _set_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare using set intersection (topk indices may be in different order)."""
    ref_np = output_ref.cpu().to(torch.int32).numpy()
    trt_np = output.cpu().to(torch.int32).numpy()

    set_ref = set(ref_np.flatten().tolist())
    set_trt = set(trt_np.flatten().tolist())
    intersection = set_ref & set_trt
    assert len(intersection) / len(set_ref) == 1.0, "output indices do not match reference indices"


@TopkSelectorFixture
def test_topk_selector_op(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    kv_group: int,
    topk: int,
    in_dtype_str: str,
    out_dtype_str: str,
    tune: bool,
) -> None:
    in_dtype = str2dtype[in_dtype_str]
    out_dtype = str2dtype[out_dtype_str]
    test = TopkSelectorTest(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)
    op = TopkSelectorFwdOp(topk=topk, tune=tune)
    test.check(op, *test.gen_inputs(), compare=_set_compare)


@pytest.mark.smoke
@pytest.mark.parametrize("valid_len", [4, 16, 17])
def test_topk_selector_op_short_row(valid_len: int) -> None:
    # starts/ends are per-row, so a row can hold fewer than topk valid elements.
    # The shared workload pins them to the full kv range, which never exercises this.
    batch, seq_len, seq_len_kv, kv_group, topk = 1, 1, 128, 1, 16

    torch.manual_seed(0)
    index_score = torch.randn(
        batch, seq_len, seq_len_kv, kv_group, dtype=torch.float32, device="cuda"
    )
    starts = torch.zeros(batch, seq_len, dtype=torch.int32, device="cuda")
    ends = torch.full((batch, seq_len), valid_len, dtype=torch.int32, device="cuda")

    op = TopkSelectorFwdOp(topk=topk)
    output = op(index_score, starts, ends)

    k = min(topk, valid_len)
    expected = torch.topk(index_score[0, 0, :valid_len, 0], k).indices.tolist()
    assert set(output[0, 0, 0, :k].tolist()) == set(expected)

    # The row cannot fill more than k slots, and the output buffer is uninitialised,
    # so the remainder has to be marked rather than left as whatever was in memory.
    assert (output[0, 0, 0, k:] == -1).all()
