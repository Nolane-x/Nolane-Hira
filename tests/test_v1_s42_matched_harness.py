from pathlib import Path


def test_s42_matched_trainer_compiles_and_binds_relational_authority():
    path = Path("scripts/hira_v1_s42_train_dev.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert "SEED = 63001" in source
    assert "generate_s42_cases" in source
    assert "validate_s42_partitions" in source
    assert "HIRA_V1_S42_A0_CROSS_VIEW_RELATIONAL_GEOMETRY_READY" in source
    assert "relational_signature_geometry_anchor" in source
    assert "full_cross_view_k_by_k_signature_similarity_mse_to_detached_matched_reference" in source
    assert "V1_S42_FRESH_MATCHED_NATIVE_REFERENCE_VS_CROSS_VIEW_RELATIONAL_GEOMETRY_FULL_BILINEAR" in source
    assert "treatment-relational-geometry-candidate.pt" in source

    assert "signature_anchor_loss" not in source
    assert "SEED = 62001" not in source
    assert "OPTIMIZER_STEP_ANCHORED" not in source


def test_s42_freshness_chain_explicitly_excludes_s41_and_s42_a0():
    source = Path("scripts/hira_v1_s42_train_dev.py").read_text(encoding="utf-8")

    assert "import hira_v1_s41_train_dev as s41" in source
    assert "s41._assert_s41_fresh(train_rows, dev_rows)" in source
    assert 's41.generate_s41_cases("train")' in source
    assert 's41.generate_s41_cases("dev")' in source
    assert "s42_a0_cases()" in source
    assert "S42 exact state overlap with exposed S41 rows" in source
    assert "S42 TRAIN DEV state overlap with S42-A0" in source


def test_s42_matched_trainer_keeps_s41_optimizer_step_engine():
    source = Path("scripts/hira_v1_s42_train_dev.py").read_text(encoding="utf-8")

    assert "adamw_candidate_deltas" in source
    assert "project_adamw_runtime_delta_against_anchor" in source
    assert "apply_parameter_deltas" in source
    assert "treatment_adamw_state = next_adamw_state" in source
    assert "BETAS = (0.9, 0.999)" in source
    assert "ADAM_EPS = 1e-8" in source
    assert "WEIGHT_DECAY = 0.01" in source
    assert "PROJECTION_EPS = 1e-12" in source
    assert "runtime movement exceeds float rounding bound" in source
    assert "quantized actual step became anchor-increasing" in source


def test_s42_matched_workflow_is_marker_gated_and_a0_bound():
    path = Path(
        ".github/workflows/"
        "hira-v1-s42-matched-cross-view-relational-geometry-train-dev.yml"
    )
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s42-cross-view-relational-geometry" in source
    assert "research/HIRA-V1-S42-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37105795175" in source
    assert "hira-v1-s42-a0-cross-view-relational-geometry" in source
    assert "scripts/hira_v1_s42_train_dev.py" in source
    assert "tests/test_v1_s42_matched_harness.py" in source
    assert 'assert r["seed"]==63001' in source
    assert "hira-v1-s42-matched-cross-view-relational-geometry-train-dev-v1" in source
    assert "V1_S42_FRESH_MATCHED_NATIVE_REFERENCE_VS_CROSS_VIEW_RELATIONAL_GEOMETRY_FULL_BILINEAR" in source
    assert "treatment-relational-geometry-candidate.pt" in source
