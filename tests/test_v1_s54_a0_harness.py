from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))

from hira_v1_s54_a0_joint_state_query_option_interaction import cases


def test_s54_a0_script_compiles_and_is_mechanical_only():
    path=Path("scripts/hira_v1_s54_a0_joint_state_query_option_interaction.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "HIRA_V1_S54_A0_JOINT_STATE_QUERY_OPTION_INTERACTION_READY" in source
    assert "SEED=75_001" in source
    assert "JointStateQueryOptionPrivateCorrectionFork" in source
    assert "TokenLateInteractionQueryFreePrivateCorrectionFork" in source
    assert "s53_query_option_bypass_absent" in source
    assert "state_context_sensitivity" in source
    assert "query_context_sensitivity" in source
    assert "option_context_sensitivity" in source
    assert "used_for_model_selection" in source
    assert len(cases())==16


def test_s54_a0_workflow_is_marker_gated():
    source=Path(".github/workflows/hira-v1-s54-a0-joint-state-query-option-interaction.yml").read_text(encoding="utf-8")
    assert "feat/hira-v1-s54-joint-state-query-option-interaction" in source
    assert "research/HIRA-V1-S54-ENABLE-A0" in source
    assert "scripts/hira_v1_s54_a0_joint_state_query_option_interaction.py" in source
    assert "tests/test_v1_joint_state_query_option_interaction.py" in source
    assert "run-id: 37192490832" in source
    assert 'assert r["joint_interaction_parameter_count"]==0' in source
    assert 'assert r["s53_query_option_bypass_absent"] is True' in source
    assert 'assert r["state_logit_sensitivity"]>0.0' in source
    assert 'assert r["query_logit_sensitivity"]>0.0' in source
    assert 'assert r["option_logit_sensitivity"]>0.0' in source
