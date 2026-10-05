from pathlib import Path


def test_s64_trainer_changes_only_context_information():
    source=Path("scripts/hira_v1_s64_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s64_train_dev.py","exec")
    assert "SEED=85_001" in source
    assert "REFERENCE_GATE_PARAMS=60" in source
    assert "TREATMENT_GATE_PARAMS=60" in source
    assert "ContextInjectedReliabilityGate(use_context=False" in source
    assert "ContextInjectedReliabilityGate(use_context=True" in source
    assert "contextual_reliability_loss" in source
    assert '"reference_gate_family":"s64_surface_zero_context"' in source
    assert '"treatment_gate_family":"s64_context_injected"' in source
    assert '"reference_gate_objective":"train_only_reliability_bce"' in source
    assert '"treatment_gate_objective":"train_only_reliability_bce"' in source
    assert '"added_treatment_parameters":0' in source
    assert '"parameter_initialization_bit_identical":True' in source
    assert '"same_reliability_target":True' in source
    assert '"shared_correction_head_trajectory":True' in source
    assert '"fused_shadow_selection_arm":False' in source
    assert "reference_phi_gradient_ever_live" in source
    assert "treatment_phi_gradient_ever_live" in source
    assert "s17._selection_key" in source
    assert "second_dev_run_performed" in source
