from pathlib import Path


def test_s63_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s63-learned-set-reliability-gate-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S63-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37312740736" in source
    assert "11346084572" in source
    assert "sha256:39c9c84a93e5da0075b13018b7a5a4151dda4563476834478524e042b1cfac9f" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s63_train_dev.py" in source
    assert 'assert cv["reference_gate_trainable_parameters"]==5' in source
    assert 'assert cv["treatment_gate_trainable_parameters"]==61' in source
    assert 'assert cv["added_treatment_parameters"]==56' in source
    assert 'assert cv["same_reliability_target"] is True' in source
    assert 'assert cv["alpha_probe"]==0.35' in source
    assert 'assert cv["target_tolerance"]==1e-8' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
