from pathlib import Path


def test_s58_a0_script_compiles_and_pins_teacher_authority():
    path=Path("scripts/hira_v1_s58_a0_consensus_teacher_pairwise_ranking.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "37271509208" in source
    assert "11327849211" in source
    assert "804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa" in source
    assert "TEACHER_SELECTED_EPOCH=19" in source
    assert 'payload.get("branch")!="reference"' in source
    assert "teacher.load_correction_state_dict" in source
    assert "freeze=True" in source
    assert "weighted_teacher_consensus_auxiliary" in source
    assert "used_for_model_selection" in source


def test_s58_a0_workflow_is_marker_gated_and_teacher_pinned():
    source=Path(".github/workflows/hira-v1-s58-a0-consensus-teacher-pairwise-ranking.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S58-ENABLE-A0" in source
    assert "run-id: 37271509208" in source
    assert "hira-v1-s57-discrete-pairwise-ranking-consistency" in source
    assert "reference-private-candidate.pt" in source
    assert "804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa" in source
    assert "scripts/hira_v1_s58_a0_consensus_teacher_pairwise_ranking.py" in source
    assert 'assert r["teacher_trainable_parameter_count"]==0' in source
    assert 'assert r["treatment_auxiliary_gradient_l1"]>0.0' in source
    assert 'assert r["used_for_model_selection"] is False' in source
