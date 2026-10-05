from pathlib import Path


def test_s60_a0_script_compiles_and_is_mechanical_only():
    path=Path("scripts/hira_v1_s60_a0_train_calibrated_bounded_hybrid.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "S60_ALPHA_INITIAL" in source
    assert "composer_gradient_to_upstream_zero" in source
    assert "fresh_train_dev_exposed" in source
    assert "pairwise_only_final_path" in source


def test_s60_a0_workflow_is_marker_gated():
    source=Path(".github/workflows/hira-v1-s60-a0-train-calibrated-bounded-hybrid.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S60-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s60_a0_train_calibrated_bounded_hybrid.py" in source
    assert 'assert r["composer_trainable_parameter_count"]==1' in source
    assert 'assert r["identity_override_max_abs_error"]==0.0' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
