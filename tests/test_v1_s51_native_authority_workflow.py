from pathlib import Path


def test_s51_native_authority_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s51-native-authority.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S51-ENABLE-NATIVE-AUTHORITY" in source
    assert "feat/hira-v1-s51-persisted-native-authority" in source
    assert "run-id: 37192065019" in source
    assert "hira-v1-s51-a0-persisted-native-authority" in source
    assert "scripts/hira_v1_s51_native_authority.py" in source
    assert "HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY" in source
    assert 'assert r["dev_generated"] is False' in source
    assert 'assert r["dev_encoded"] is False' in source
    assert 'assert r["dev_scored"] is False' in source
    assert 'assert r["private_correction_constructed"] is False' in source
    assert 'assert r["projection_parameter_count"]==32768' in source
    assert "native-authority.pt" in source
    assert "INTEGRITY.sha256" in source
