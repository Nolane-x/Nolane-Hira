from pathlib import Path


def test_s46_trainer_compiles_and_preserves_s45_training_mechanics():
    path=Path("scripts/hira_v1_s46_train_dev.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")

    assert "SEED = 67_001" in source
    assert "s45._train_arm" in source
    assert "s45.SEED = SEED" in source
    assert "generate_s46_cases" in source
    assert "_assert_s46_fresh" in source
    assert "RobustThreeExpertMedianFusion" in source
    assert "legacy_s45_shell" in source
    assert "robust_s46_shell" in source
    assert "delta_robust_minus_legacy" in source
    assert "checkpoint_selection" in source
    assert "second_dev_run_performed" in source
    assert "external_laya_jev_evaluation_opened" in source


def test_s46_trainer_same_checkpoint_shell_has_zero_new_parameters():
    source=Path("scripts/hira_v1_s46_train_dev.py").read_text(encoding="utf-8")
    assert '"treatment_trainable_parameters": 0' in source
    assert '"learned_gate": False' in source
    assert '"temperature": None' in source
    assert '"threshold": None' in source


def test_s46_train_dev_marker_not_armed_during_staging():
    assert not Path("research/HIRA-V1-S46-ENABLE-TRAIN-DEV").exists()


def test_s46_matched_workflow_is_one_shot_a0_bound_and_same_checkpoint():
    path=Path(".github/workflows/hira-v1-s46-matched-robust-three-expert-consensus-train-dev.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s46-robust-three-expert-consensus" in source
    assert "research/HIRA-V1-S46-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37134236894" in source
    assert "hira-v1-s46-a0-robust-three-expert-consensus" in source
    assert "scripts/hira_v1_s46_train_dev.py" in source
    assert "tests/test_v1_s46_authority.py" in source
    assert "tests/test_v1_s46_matched_harness.py" in source
    assert 'assert r["seed"]==67001' in source
    assert "legacy_s45_shell" in source
    assert "robust_s46_shell" in source
    assert "delta_robust_minus_legacy" in source
    assert "reference-native-candidate.pt" in source
    assert "treatment-private-correction-candidate.pt" in source
    assert 'assert r["second_dev_run_performed"] is False' in source
