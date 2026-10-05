from pathlib import Path


def test_s63_a0_workflow_is_marker_gated_and_parent_pinned():
    source=Path(".github/workflows/hira-v1-s63-a0-learned-set-reliability-gate.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S63-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "run-id: 37307658289" in source
    assert "11344855774" in source
    assert "sha256:7812c348d707ea8675fc9370af573a87d9e5b1a0b10e55a1f335bf890ddb3305" in source
    assert "scripts/hira_v1_s63_a0_learned_set_reliability_gate.py" in source
    assert 'assert r["treatment_gate_parameter_count"]==61' in source
    assert 'assert r["added_treatment_parameter_count"]==56' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
