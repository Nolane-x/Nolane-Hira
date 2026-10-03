from pathlib import Path


def test_s41_a0_script_compiles_and_binds_optimizer_step_engine():
    path = Path("scripts/hira_v1_s41_a0_optimizer_step_anchor.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert 'SCHEMA_VERSION = "hira-v1-s41-a0-optimizer-step-anchored-coadaptation-v1"' in source
    assert 'OUTCOME = "HIRA_V1_S41_A0_OPTIMIZER_STEP_ANCHORED_COADAPTATION_READY"' in source
    assert "adamw_candidate_deltas" in source
    assert "project_adamw_runtime_delta_against_anchor" in source
    assert "foreach=False" in source
    assert "fused=False" in source
    assert "seed=62001" in source
    assert "hira_v1_s40_a0" not in source


def test_s41_a0_workflow_is_marker_gated_and_verifies_optimizer_faithfulness():
    path = Path(".github/workflows/hira-v1-s41-a0-optimizer-step-anchor.yml")
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s41-optimizer-step-anchor" in source
    assert "research/HIRA-V1-S41-ENABLE-A0" in source
    assert "scripts/hira_v1_s41_a0_optimizer_step_anchor.py" in source
    assert '"adamw_candidate_parameter_max_abs_error"' in source
    assert '"adamw_exp_avg_max_abs_error"' in source
    assert 'assert r[key]==0.0' in source
    assert 'r["conflict_actual_step_pre_dot"]>0.0' in source
    assert 'r["conflict_actual_step_projected"] is True' in source
    assert '"applied_runtime_delta_max_abs_error"' in source
    assert '"applied_w_delta_max_abs_error"' in source
