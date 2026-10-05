from pathlib import Path


def test_s64_a0_workflow_is_marker_gated_and_parent_pinned():
    source=Path(".github/workflows/hira-v1-s64-a0-contextual-reliability-gate.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S64-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "run-id: 37314015283" in source
    assert "11347706059" in source
    assert "sha256:dcb69358dfa97f3644198534faa1c2f5f214b4f1773d79008eec13cafaf2a97e" in source
    assert "scripts/hira_v1_s64_a0_contextual_reliability_gate.py" in source
    assert 'assert r["reference_gate_parameter_count"]==60' in source
    assert 'assert r["treatment_gate_parameter_count"]==60' in source
    assert 'assert r["added_treatment_parameter_count"]==0' in source
    assert 'assert r["parameter_initialization_bit_identical"] is True' in source
    assert 'assert r["context_projection_trainable"] is False' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
