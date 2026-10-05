from pathlib import Path


def test_s66_a0_workflow_is_marker_gated_and_parent_pinned():
    source=Path(".github/workflows/hira-v1-s66-a0-per-view-responsibility.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S66-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "run-id: 37325995163" in source
    assert "11352920320" in source
    assert "sha256:a44c560d304537bb7acaa1b5d569160ec744f897ae728edf95ceca7701be9dfc" in source
    assert "scripts/hira_v1_s66_a0_per_view_responsibility.py" in source
    assert 'assert r["reference_gate_parameter_count"]==60' in source
    assert 'assert r["treatment_gate_parameter_count"]==60' in source
    assert 'assert r["added_treatment_parameter_count"]==0' in source
    assert 'assert r["responsibility_states"]==[[0,0],[0,1],[1,0],[1,1]]' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
