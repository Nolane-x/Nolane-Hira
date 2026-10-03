from pathlib import Path


def test_s41_matched_trainer_compiles_and_binds_optimizer_step_authority():
    path = Path("scripts/hira_v1_s41_train_dev.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert "SEED = 62001" in source
    assert "generate_s41_cases" in source
    assert "HIRA_V1_S41_A0_OPTIMIZER_STEP_ANCHORED_COADAPTATION_READY" in source
    assert "adamw_candidate_deltas" in source
    assert "project_adamw_runtime_delta_against_anchor" in source
    assert "apply_parameter_deltas" in source
    assert "treatment_adamw_state = next_adamw_state" in source
    assert "actual_stateful_adamw_runtime_parameter_delta" in source

    assert "project_runtime_gradient_against_reference_anchor" not in source
    assert "tr_opt.step()" not in source
    assert "SEED = 61001" not in source


def test_s41_matched_trainer_freezes_optimizer_semantics_and_quantized_guard():
    source = Path("scripts/hira_v1_s41_train_dev.py").read_text(encoding="utf-8")

    assert "BETAS = (0.9, 0.999)" in source
    assert "ADAM_EPS = 1e-8" in source
    assert "WEIGHT_DECAY = 0.01" in source
    assert "PROJECTION_EPS = 1e-12" in source
    assert "foreach=False" in source
    assert "fused=False" in source
    assert "amsgrad=False" in source
    assert "capturable=False" in source
    assert "differentiable=False" in source

    assert "runtime movement exceeds float rounding bound" in source
    assert "W movement exceeds float rounding bound" in source
    assert "quantized actual step became anchor-increasing" in source
    assert '"max_runtime_rounding_ratio"' in source
    assert '"max_w_rounding_ratio"' in source


def test_s41_matched_workflow_is_marker_gated_and_receipt_bound():
    path = Path(
        ".github/workflows/"
        "hira-v1-s41-matched-optimizer-step-anchored-train-dev.yml"
    )
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s41-optimizer-step-anchor" in source
    assert "research/HIRA-V1-S41-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37101438505" in source
    assert "hira-v1-s41-a0-optimizer-step-anchor" in source
    assert "scripts/hira_v1_s41_train_dev.py" in source
    assert 'assert r["seed"]==62001' in source
    assert "V1_S41_FRESH_MATCHED_NATIVE_REFERENCE_VS_OPTIMIZER_STEP_ANCHORED_FULL_BILINEAR" in source
    assert "treatment-optimizer-step-anchored-candidate.pt" in source
    assert "hira-v1-s41-matched-optimizer-step-anchored-coadaptation-train-dev-v1" in source
    assert "hira-v1-s41-matched-optimizer-step-anchored-train-dev-v1" not in source
