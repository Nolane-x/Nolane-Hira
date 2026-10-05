from pathlib import Path


def test_s64_a0_harness_freezes_matched_contextual_mechanics():
    source=Path("scripts/hira_v1_s64_a0_contextual_reliability_gate.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s64_a0_contextual_reliability_gate.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s64-a0-contextual-reliability-gate-v1"' in source
    assert "SEED=85_001" in source
    assert "ContextInjectedReliabilityGate" in source
    assert "use_context=False" in source
    assert "use_context=True" in source
    assert 'reference.parameter_count!=60 or treatment.parameter_count!=60' in source
    assert "_parameter_state_equal(reference,treatment)" in source
    assert "treatment_context_sensitivity_max_abs" in source
    assert "reference_context_max_abs" in source
    assert "added_treatment_parameter_count" in source
    assert '"fresh_train_dev_exposed":False' in source
    assert '"external_laya_jev_evaluation_opened"' not in source
