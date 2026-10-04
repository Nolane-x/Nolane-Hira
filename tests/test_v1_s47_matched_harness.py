from pathlib import Path


def test_s47_trainer_compiles_and_preserves_s45_training_mechanics():
    path=Path("scripts/hira_v1_s47_train_dev.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "SEED=68_001" in source
    assert 's45._train_arm("reference"' in source
    assert 's45._train_arm("treatment"' in source
    assert "s45.SEED=SEED" in source
    assert "generate_s47_cases" in source
    assert "_assert_s47_fresh" in source
    assert "OrdinalPairwiseConsensus" in source
    assert "legacy_s45_shell" in source
    assert "ordinal_s47_shell" in source
    assert "delta_ordinal_minus_legacy" in source
    assert "ordinal_diagnostics" in source
    assert "second_dev_run_performed" in source


def test_s47_shell_is_zero_parameter_and_magnitude_free():
    source=Path("scripts/hira_v1_s47_train_dev.py").read_text(encoding="utf-8")
    assert '"treatment_trainable_parameters":0' in source
    assert '"uses_raw_magnitude_after_ranking":False' in source
    assert '"lexicographic_base_rule":"2K-1"' in source
    assert '"learned_gate":False' in source
    assert '"temperature":None' in source
    assert '"threshold":None' in source


def test_s47_train_dev_marker_is_well_formed_when_authorized():
    marker=Path("research/HIRA-V1-S47-ENABLE-TRAIN-DEV")
    if not marker.exists():
        return
    source=marker.read_text(encoding="utf-8")
    assert "seed 68001" in source
    assert "TRAIN 768" in source
    assert "DEV 192" in source
    assert "exact S45 training mechanics" in source
    assert "same-checkpoint legacy S45 shell vs ordinal S47 shell" in source
    assert "No second S47 DEV." in source


def test_s47_matched_workflow_is_one_shot_a0_bound_and_same_checkpoint():
    path=Path(".github/workflows/hira-v1-s47-matched-ordinal-pairwise-consensus-train-dev.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s47-ordinal-pairwise-consensus" in source
    assert "research/HIRA-V1-S47-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37167755900" in source
    assert "hira-v1-s47-a0-ordinal-pairwise-consensus" in source
    assert "scripts/hira_v1_s47_train_dev.py" in source
    assert "tests/test_v1_s47_authority.py" in source
    assert "tests/test_v1_s47_matched_harness.py" in source
    assert 'assert r["seed"]==68001' in source
    assert "legacy_s45_shell" in source
    assert "ordinal_s47_shell" in source
    assert "delta_ordinal_minus_legacy" in source
    assert "reference-native-candidate.pt" in source
    assert "treatment-private-correction-candidate.pt" in source
    assert 'assert r["second_dev_run_performed"] is False' in source
