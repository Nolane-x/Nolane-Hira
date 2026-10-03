from pathlib import Path


def test_s45_matched_trainer_compiles_and_binds_cross_view_consistency():
    path = Path("scripts/hira_v1_s45_train_dev.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert "SEED = 66001" in source
    assert "generate_s45_cases" in source
    assert "validate_s45_partitions" in source
    assert "PrivateCorrectionRepresentationFork" in source
    assert "symmetric_js_divergence" in source
    assert "relation_cross_view_mean_js" in source
    assert "s35.BINDING_COEFFICIENT * ce + s35.INVARIANCE_COEFFICIENT * js" in source
    assert "CORRECTION_PARAMETER_COUNT = 114_688" in source
    assert "TREATMENT_TOTAL = HIRA_V1_S17_TOTAL_PARAMETER_COUNT + CORRECTION_PARAMETER_COUNT" in source
    assert "correction_optimizer =" in source
    assert "native_optimizer =" in source
    assert 'correction_coefficient": 0.10' in source
    assert 'correction_cross_view_js_coefficient": s35.INVARIANCE_COEFFICIENT' in source
    assert 'correction_optimizer_separate_from_native": True' in source
    assert 'second_encoder_pass": False' in source
    assert '"runtime_trajectory_identity"' in source
    assert "cross-view consistent private correction evaluation effect" in source

    assert "SEED = 65001" not in source
    assert "v1_private_correction_representation_fusion" not in source


def test_s45_matched_workflow_is_one_shot_a0_bound_and_consistency_verified():
    path = Path(
        ".github/workflows/"
        "hira-v1-s45-matched-cross-view-consistent-private-correction-train-dev.yml"
    )
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s45-cross-view-consistent-private-correction" in source
    assert "research/HIRA-V1-S45-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37128007831" in source
    assert "hira-v1-s45-a0-cross-view-consistent-private-correction" in source
    assert "scripts/hira_v1_s45_train_dev.py" in source
    assert "tests/test_v1_s45_authority.py" in source
    assert "tests/test_v1_s45_matched_harness.py" in source
    assert 'assert r["seed"]==66001' in source
    assert "V1_S45_FRESH_MATCHED_NATIVE_REFERENCE_VS_CROSS_VIEW_CONSISTENT_PRIVATE_CORRECTION" in source
    assert 'assert loss["correction_cross_view_js_coefficient"]==0.25' in source
    assert 'assert "relation_cross_view_mean_js" in a["selected_dev"]' in source
    assert 'assert "relation_cross_view_mean_js" in r["matched_selected_dev_delta_treatment_minus_reference"]' in source
    assert 'assert t["all_epoch_runtime_state_sha256_equal"] is True' in source
    assert 'assert a["gates"]["runtime_trajectory_identity"] is True' in source
