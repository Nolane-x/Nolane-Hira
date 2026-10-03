from pathlib import Path


def test_s44_matched_trainer_compiles_and_binds_private_fork():
    path = Path("scripts/hira_v1_s44_train_dev.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert "SEED = 65001" in source
    assert "generate_s44_cases" in source
    assert "validate_s44_partitions" in source
    assert "PrivateCorrectionRepresentationFork" in source
    assert "from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion" in source
    assert "v1_private_correction_representation_fusion" not in source
    assert "CORRECTION_PARAMETER_COUNT = 114_688" in source
    assert "TREATMENT_TOTAL = HIRA_V1_S17_TOTAL_PARAMETER_COUNT + CORRECTION_PARAMETER_COUNT" in source
    assert "correction_optimizer =" in source
    assert "native_optimizer =" in source
    assert 'correction_coefficient": 0.10' in source
    assert 'correction_optimizer_separate_from_native": True' in source
    assert 'second_encoder_pass": False' in source
    assert '"runtime_trajectory_identity"' in source
    assert "private correction representation evaluation effect" in source

    assert "NativeGradientIsolatedBilinearCorrectnessReadout" not in source
    assert "SEED = 60001" not in source


def test_s44_matched_workflow_is_marker_gated_and_a0_bound():
    path = Path(
        ".github/workflows/"
        "hira-v1-s44-matched-private-correction-fork-train-dev.yml"
    )
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s44-private-correction-fork" in source
    assert "research/HIRA-V1-S44-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37120084360" in source
    assert "hira-v1-s44-a0-private-correction-fork" in source
    assert "scripts/hira_v1_s44_train_dev.py" in source
    assert "tests/test_v1_s44_authority.py" in source
    assert "tests/test_v1_s44_matched_harness.py" in source
    assert 'assert r["seed"]==65001' in source
    assert "V1_S44_FRESH_MATCHED_NATIVE_REFERENCE_VS_PRIVATE_CORRECTION_REPRESENTATION_FORK" in source
    assert "treatment-private-correction-candidate.pt" in source
    assert 'assert t["all_epoch_runtime_state_sha256_equal"] is True' in source
    assert 'assert a["gates"]["runtime_trajectory_identity"] is True' in source
