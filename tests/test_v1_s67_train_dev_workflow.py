from pathlib import Path


def test_s67_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s67-safe-oracle-alpha-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S67-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37335108984" in source
    assert "11356196983" in source
    assert "sha256:14a77818c8e2158bdb3aff1b2cc864ef71fe47c1a451ce32d1d10cd234ffb793" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s67_train_dev.py" in source
    assert 'assert cv["reference_gate_trainable_parameters"]==60' in source
    assert 'assert cv["treatment_gate_trainable_parameters"]==60' in source
    assert 'assert cv["added_treatment_parameters"]==0' in source
    assert 'assert cv["oracle_alpha_lattice"]==[0.0,0.0875,0.175,0.2625,0.35]' in source
    assert 'assert cv["oracle_target_levels"]==[0.0,0.25,0.5,0.75,1.0]' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
