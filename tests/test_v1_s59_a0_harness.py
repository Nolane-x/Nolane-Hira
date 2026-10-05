from pathlib import Path


def test_s59_a0_script_compiles_and_is_mechanical_only():
    path=Path("scripts/hira_v1_s59_a0_matched_pairwise_decision_head.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "HIRA_V1_S59_A0_MATCHED_ANTISYMMETRIC_PAIRWISE_HEAD_READY" in source
    assert "correction_parameter_count_per_arm" in source
    assert "decision_head_parameter_count_per_arm" in source
    assert "matched_initialization_exact" in source
    assert "initial_b_gradient_l1" in source
    assert "post_b_update_a_gradient_l1" in source
    assert "teacher_artifact_dependency" in source
    assert "used_for_model_selection" in source


def test_s59_a0_workflow_is_marker_gated():
    source=Path(".github/workflows/hira-v1-s59-a0-matched-pairwise-decision-head.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S59-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s59_a0_matched_pairwise_decision_head.py" in source
    assert 'assert r["decision_head_parameter_count_per_arm"]==16384' in source
    assert 'assert r["total_private_parameter_count_per_arm"]==131072' in source
    assert 'assert r["teacher_artifact_dependency"] is False' in source
