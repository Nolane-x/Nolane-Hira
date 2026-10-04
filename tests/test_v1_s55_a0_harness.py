from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
from hira_v1_s55_a0_learned_joint_relation_interaction import cases


def test_s55_a0_script_compiles_and_is_mechanical_only():
    path=Path("scripts/hira_v1_s55_a0_learned_joint_relation_interaction.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "HIRA_V1_S55_A0_LEARNED_JOINT_RELATION_INTERACTION_READY" in source
    assert "reference_zero_state_channel_invariant_error" in source
    assert "treatment_state_channel_sensitivity" in source
    assert "learned_joint_gradient_l1" in source
    assert "used_for_model_selection" in source
    assert len(cases())==16


def test_s55_a0_workflow_is_marker_gated():
    source=Path(".github/workflows/hira-v1-s55-a0-learned-joint-relation-interaction.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S55-ENABLE-A0" in source
    assert "feat/hira-v1-s55-learned-joint-relation-interaction" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s55_a0_learned_joint_relation_interaction.py" in source
    assert 'assert r["reference_learned_joint_parameter_count"]==65536' in source
    assert 'assert r["treatment_private_trainable_parameter_count"]==180224' in source
    assert 'assert r["learned_joint_gradient_l1"]>0.0' in source
