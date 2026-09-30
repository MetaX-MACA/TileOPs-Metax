from pathlib import Path

from scripts.junit_summary_json import summarize


def test_junit_summary_counts_outcomes(tmp_path: Path):
    xml = tmp_path / "results.xml"
    xml.write_text(
        '<testsuite><testcase classname="a" name="ok"/>'
        '<testcase classname="b" name="bad"><failure/></testcase>'
        '<testcase classname="c" name="skip"><skipped/></testcase></testsuite>',
        encoding="utf-8",
    )

    assert summarize(xml) == {
        "total": 3,
        "passed": 1,
        "failed": 1,
        "skipped": 1,
        "failures": [{"classname": "b", "name": "bad"}],
    }
