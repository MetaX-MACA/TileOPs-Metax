from scripts.collect_env_json import collect


def test_collect_env_json_has_expected_sections():
    payload = collect()
    assert "platform" in payload
    assert "packages" in payload
    assert "torch_runtime" in payload
