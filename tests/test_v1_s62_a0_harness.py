from pathlib import Path


def test_s62_a0_script_compiles_and_is_dev_target_free():
    path=Path("scripts/hira_v1_s62_a0_train_only_reliability_supervised_gate.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "S62_ALPHA_PROBE" in source
    assert "S62_TARGET_TOLERANCE" in source
    assert "adaptive_gate_gold_loss" in source
    assert "reliability_gate_loss" in source
    assert "added_treatment_parameter_count" in source
    assert "dev_target_dependency" in source
    assert "fresh_train_dev_exposed" in source


def test_s62_a0_workflow_is_marker_gated():
    source=Path(".github/workflows/hira-v1-s62-a0-train-only-reliability-supervised-gate.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S62-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s62_a0_train_only_reliability_supervised_gate.py" in source
    assert 'assert r["alpha_probe"]==0.35' in source
    assert 'assert r["target_tolerance"]==1e-8' in source
    assert 'assert r["added_treatment_parameter_count"]==0' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
