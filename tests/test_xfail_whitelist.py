from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tests.conftest as test_config
from tests.xfail_whitelist import MACA_XFAILS

pytestmark = pytest.mark.smoke


def _make_item(nodeid: str) -> SimpleNamespace:
    return SimpleNamespace(nodeid=nodeid, add_marker=Mock())


def test_maca_xfail_allowlist_matches_exact_nodeids(monkeypatch) -> None:
    monkeypatch.setattr(test_config, "is_maca", lambda: True)
    known = _make_item(next(iter(MACA_XFAILS)))
    unknown = _make_item("tests/ops/test_unknown.py::test_unknown")

    test_config._apply_maca_xfails([known, unknown])

    known.add_marker.assert_called_once()
    marker = known.add_marker.call_args.args[0].mark
    assert marker.name == "xfail"
    assert marker.kwargs["strict"] is True
    assert marker.kwargs["reason"].startswith("MACA: ")
    unknown.add_marker.assert_not_called()


def test_maca_xfail_allowlist_is_inactive_off_maca(monkeypatch) -> None:
    monkeypatch.setattr(test_config, "is_maca", lambda: False)
    item = _make_item(next(iter(MACA_XFAILS)))

    test_config._apply_maca_xfails([item])

    item.add_marker.assert_not_called()
