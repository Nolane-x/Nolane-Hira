from pathlib import Path


def test_s62_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s62-train-only-reliability-supervised-gate-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S62-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37305254725" in source
    assert "11342769947" in source
    assert "sha256:3db27b3a3a9fdc4c0b276c72c05649a6b77f59b868b2329914e94ccd7d483d0e" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s62_train_dev.py" in source
    assert 'assert cv["reference_gate_objective"]=="gold_ce"' in source
    assert 'assert cv["treatment_gate_objective"]=="train_only_reliability_bce"' in source
    assert 'assert cv["added_treatment_parameters"]==0' in source
    assert 'assert cv["alpha_probe"]==0.35' in source
    assert 'assert cv["target_tolerance"]==1e-8' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
