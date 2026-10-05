from pathlib import Path


def test_s64_a0_harness_freezes_matched_capacity_context_variable():
    source=Path("scripts/hira_v1_s64_a0_context_projected_reliability_gate.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s64_a0_context_projected_reliability_gate.py","exec")
    assert "SEED=85_001" in source
    assert "LearnedSetReliabilityGate" in source
    assert "ContextProjectedReliabilityGate" in source
    assert "reference.parameter_count!=S64_GATE_PARAMETER_COUNT" in source
    assert "treatment.parameter_count!=S64_GATE_PARAMETER_COUNT" in source
    assert "learned_initialization_bit_identical" in source
    assert "projection_seed" in source
    assert "projection_digest" in source
    assert "context_affine_alpha_max_abs_error" in source
    assert "gradient_to_fused_pairwise_context_zero" in source
    assert '"fresh_train_dev_exposed":False' in source
    assert '"used_for_model_selection":False' in source
