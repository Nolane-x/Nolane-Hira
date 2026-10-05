from pathlib import Path


def test_s67_a0_workflow_is_marker_gated_and_parent_pinned():
    source=Path(".github/workflows/hira-v1-s67-a0-safe-oracle-alpha.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S67-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "run-id: 37331468679" in source
    assert "11354228532" in source
    assert "sha256:f3c6322eec8cce02a698807037a4c05e48c32b2addc5709a116a4384338379b1" in source
    assert "scripts/hira_v1_s67_a0_safe_oracle_alpha.py" in source
    assert 'assert r["reference_gate_parameter_count"]==60' in source
    assert 'assert r["treatment_gate_parameter_count"]==60' in source
    assert 'assert r["added_treatment_parameter_count"]==0' in source
    assert 'assert r["oracle_alpha_lattice"]==[0.0,0.0875,0.175,0.2625,0.35]' in source
    assert 'assert r["smaller_alpha_tie_break_max_abs_error"]==0.0' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
