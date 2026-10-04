from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))

from hira_v1_s56_a0_cross_view_decision_consistency import cases


def test_s56_a0_script_compiles_and_is_mechanical_only():
    path=Path("scripts/hira_v1_s56_a0_cross_view_decision_consistency.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert len(cases())==16
    assert "HIRA_V1_S56_A0_CROSS_VIEW_DECISION_CONSISTENCY_READY" in source
    assert "decision_coefficient=0.0" in source
    assert "ordering_coefficient=0.0" in source
    assert "decision_coefficient=0.10" in source
    assert "ordering_coefficient=0.05" in source
    assert "treatment_auxiliary_gradient_l1" in source
    assert "used_for_model_selection" in source


def test_s56_a0_workflow_is_marker_gated():
    source=Path(".github/workflows/hira-v1-s56-a0-cross-view-decision-consistency.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S56-ENABLE-A0" in source
    assert "feat/hira-v1-s56-cross-view-decision-consistency" in source
    assert "scripts/hira_v1_s56_a0_cross_view_decision_consistency.py" in source
    assert "tests/test_v1_cross_view_decision_consistency.py" in source
    assert 'assert r["added_trainable_parameter_count"]==0' in source
    assert 'assert r["treatment_auxiliary_gradient_l1"]>0.0' in source
    assert 'assert r["uniform_top1_top2_probability_gap"]==0.0' in source
