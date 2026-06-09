from scripts import validate_manifest


def test_parse_json_output_accepts_equals_form():
    assert validate_manifest._parse_json_output(["x", "--json-output=out.json"]) == "out.json"


def test_parse_json_output_accepts_separate_value():
    assert validate_manifest._parse_json_output(["x", "--json-output", "out.json"]) == "out.json"
