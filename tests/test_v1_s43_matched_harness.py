from pathlib import Path


def test_s43_matched_trainer_compiles_and_binds_relative_gap_authority():
    path = Path("scripts/hira_v1_s43_train_dev.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert "SEED = 64001" in source
    assert "generate_s43_cases" in source
    assert "validate_s43_partitions" in source
    assert "relative_gap_signature_geometry_anchor" in source
    assert "HIRA_V1_S43_A0_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_READY" in source
    assert "S43_A0_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_FULL_BILINEAR_ONLY" in source
    assert "relative_gap_geometry_shape" in source
    assert "treatment_adamw_state = next_adamw_state" in source
    assert "actual_stateful_adamw_runtime_parameter_delta" in source
    assert "treatment-relative-gap-candidate.pt" in source

    # Parent S42 appears only as freshness ancestry.
    assert "s42._assert_s42_fresh" in source
    assert 's42.generate_s42_cases("train")' in source

    assert "relational_signature_geometry_anchor" not in source
    assert "SEED = 63001" not in source


def test_s43_matched_trainer_freezes_optimizer_and_quantized_guard():
    source = Path("scripts/hira_v1_s43_train_dev.py").read_text(encoding="utf-8")

    assert "BETAS = (0.9, 0.999)" in source
    assert "ADAM_EPS = 1e-8" in source
    assert "WEIGHT_DECAY = 0.01" in source
    assert "PROJECTION_EPS = 1e-12" in source
    assert "adamw_candidate_deltas" in source
    assert "project_adamw_runtime_delta_against_anchor" in source
    assert "apply_parameter_deltas" in source
    assert "runtime movement exceeds float rounding bound" in source
    assert "W movement exceeds float rounding bound" in source
    assert "quantized actual step became anchor-increasing" in source
    assert '"relative_gap_anchor"' in source


def test_s43_matched_workflow_is_marker_gated_and_receipt_bound():
    path = Path(
        ".github/workflows/"
        "hira-v1-s43-matched-cross-view-relative-gap-geometry-train-dev.yml"
    )
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s43-cross-view-relative-gap" in source
    assert "research/HIRA-V1-S43-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37112131898" in source
    assert "hira-v1-s43-a0-cross-view-relative-gap-geometry" in source
    assert "scripts/hira_v1_s43_train_dev.py" in source
    assert "tests/test_v1_s43_matched_harness.py" in source
    assert 'assert r["seed"]==64001' in source
    assert "V1_S43_FRESH_MATCHED_NATIVE_REFERENCE_VS_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_FULL_BILINEAR" in source
    assert "treatment-relative-gap-candidate.pt" in source
    assert "full_cross_view_same_option_vs_wrong_relative_gap_mse_to_detached_matched_reference" in source
