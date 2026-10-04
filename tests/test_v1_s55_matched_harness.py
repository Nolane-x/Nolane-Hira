from pathlib import Path


def test_s55_trainer_is_matched_and_artifact_pinned():
    source=Path("scripts/hira_v1_s55_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s55_train_dev.py","exec")
    assert "SEED=76_001" in source
    assert "PRIVATE_TRAINABLE=180_224" in source
    assert "CORRECTION_PARAMS=114_688" in source
    assert "LEARNED_PARAMS=65_536" in source
    assert "reference=LearnedJointRelationPrivateCorrectionFork" in source
    assert "treatment=LearnedJointRelationPrivateCorrectionFork" in source
    assert "use_state_joint_context=False" in source
    assert "use_state_joint_context=True" in source
    assert "params=op.private_parameters()" in source
    assert "op.private_state_dict()" in source
    assert "op.load_private_state_dict(best_state,freeze=True)" in source
    assert "native_optimizer_constructed" in source
    assert "native_training_performed" in source
    assert "second_dev_run_performed" in source


def test_s55_module_supports_full_private_checkpoint_roundtrip():
    source=Path("src/nmd/v1_learned_joint_relation_interaction.py").read_text(encoding="utf-8")
    assert "def learned_joint_state_dict" in source
    assert "def load_learned_joint_state_dict" in source
    assert "def private_state_dict" in source
    assert "def load_private_state_dict" in source


def test_s55_zero_init_warm_start_guard_is_epoch_scoped_before_dev():
    source=Path("scripts/hira_v1_s55_train_dev.py").read_text(encoding="utf-8")
    assert "learned_gradient_live=False" in source
    assert "learned_gradient_live=True" in source
    assert "learned-joint gradient never became live in epoch" in source
    assert source.index("learned-joint gradient never became live in epoch") < source.index('metrics=s50._private_metrics(op,"treatment",dev_cache)')
    assert "S55 {name} learned-joint gradient vanished" not in source


def test_s55_mechanical_replay_has_separate_marker():
    source=Path(".github/workflows/hira-v1-s55-learned-joint-relation-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S55-ENABLE-TRAIN-DEV" in source
    assert "research/HIRA-V1-S55-ENABLE-MECHANICAL-REPLAY" in source
