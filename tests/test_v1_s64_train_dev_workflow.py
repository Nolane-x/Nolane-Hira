from pathlib import Path


def test_s64_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s64-contextual-reliability-gate-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S64-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37318082102" in source
    assert "11349331302" in source
    assert "sha256:462c4b845ba60b870a5ead4a26dbc9e4ba661426c6d2a23112484214d7c6d14e" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s64_train_dev.py" in source
    assert 'assert cv["reference_gate_trainable_parameters"]==60' in source
    assert 'assert cv["treatment_gate_trainable_parameters"]==60' in source
    assert 'assert cv["added_treatment_parameters"]==0' in source
    assert 'assert cv["parameter_initialization_bit_identical"] is True' in source
    assert 'assert cv["context_projection_seed"]==64064' in source
    assert 'assert cv["same_reliability_target"] is True' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
