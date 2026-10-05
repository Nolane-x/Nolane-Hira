from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))

from hira_v1_s57_a0_discrete_pairwise_ranking_consistency import cases


def test_s57_a0_script_compiles_and_is_mechanical_only():
    path=Path("scripts/hira_v1_s57_a0_discrete_pairwise_ranking_consistency.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert len(cases())==16
    assert "HIRA_V1_S57_A0_DISCRETE_PAIRWISE_RANKING_CONSISTENCY_READY" in source
    assert "coefficient=0.0" in source
    assert "coefficient=0.05" in source
    assert "anchor_sign_detached" in source
    assert "wrong_gold_filtered_direction_count" in source
    assert "treatment_auxiliary_gradient_l1" in source
    assert "used_for_model_selection" in source


def test_s57_a0_workflow_is_marker_gated():
    source=Path(".github/workflows/hira-v1-s57-a0-discrete-pairwise-ranking-consistency.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S57-ENABLE-A0" in source
    assert "feat/hira-v1-s57-discrete-pairwise-ranking-consistency" in source
    assert "scripts/hira_v1_s57_a0_discrete_pairwise_ranking_consistency.py" in source
    assert "tests/test_v1_pairwise_ranking_consistency.py" in source
    assert 'assert r["anchor_sign_detached"] is True' in source
    assert 'assert r["wrong_gold_filtered_direction_count"]>0' in source
    assert 'assert r["treatment_auxiliary_gradient_l1"]>0.0' in source
