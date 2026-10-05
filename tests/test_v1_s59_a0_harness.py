from pathlib import Path


def test_s59_a0_script_compiles_and_is_teacher_free():
    path=Path("scripts/hira_v1_s59_a0_explicit_learned_pairwise_decision_head.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "S59_PAIRWISE_PARAMETER_COUNT" in source
    assert "correction_received_pairwise_gradient" in source
    assert "fresh_train_dev_exposed" in source
    assert "teacher_dependency" in source
    assert "S57COURT" not in source
    assert "S58A0" not in source


def test_s59_a0_workflow_is_marker_gated():
    source=Path(".github/workflows/hira-v1-s59-a0-explicit-learned-pairwise-decision-head.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S59-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s59_a0_explicit_learned_pairwise_decision_head.py" in source
    assert 'assert r["pairwise_parameter_count"]==32832' in source
    assert 'assert r["teacher_dependency"] is False' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
