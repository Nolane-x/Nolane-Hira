from pathlib import Path


def test_s54_trainer_compiles_and_binds_only_joint_context():
    path=Path("scripts/hira_v1_s54_train_dev.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "SEED=75_001" in source
    assert "PRIVATE_TRAINABLE=114_688" in source
    assert "TokenLateInteractionQueryFreePrivateCorrectionFork" in source
    assert "JointStateQueryOptionPrivateCorrectionFork" in source
    assert "option_conditioned_token_level_query_context" in source
    assert "joint_state_query_option_context" in source
    assert "joint_interaction_trainable_parameters" in source
    assert "native_training_performed" in source
    assert "second_dev_run_performed" in source


def test_s54_trainer_is_a0_and_native_authority_gated():
    source=Path("scripts/hira_v1_s54_train_dev.py").read_text(encoding="utf-8")
    assert "HIRA_V1_S54_A0_JOINT_STATE_QUERY_OPTION_INTERACTION_READY" in source
    assert "ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628" in source
    assert "19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916" in source
    assert "state_logit_sensitivity" in source
    assert "query_logit_sensitivity" in source
    assert "option_logit_sensitivity" in source


def test_s54_trainer_has_no_native_training_path():
    source=Path("scripts/hira_v1_s54_train_dev.py").read_text(encoding="utf-8")
    assert "_train_shared_native" not in source
    assert "norm_balanced_gradient_update" not in source
    assert "build_hira_v1_s17_norm_balanced_core" not in source
