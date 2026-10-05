from pathlib import Path


def test_s65_trainer_changes_only_train_objective():
    source=Path("scripts/hira_v1_s65_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s65_train_dev.py","exec")
    assert "SEED=86_001" in source
    assert "REFERENCE_GATE_PARAMS=60" in source
    assert "TREATMENT_GATE_PARAMS=60" in source
    assert "ContextInjectedReliabilityGate(use_context=True" in source
    assert "contextual_reliability_loss" in source
    assert "cross_view_soft_and_reliability_loss" in source
    assert '"reference_gate_objective":"independent_view_bce"' in source
    assert '"treatment_gate_objective":"cross_view_exact_min_soft_and_bce"' in source
    assert '"added_treatment_parameters":0' in source
    assert '"same_context_path":True' in source
    assert '"cross_view_interaction_train_only":True' in source
    assert '"single_view_inference_both_arms":True' in source
    assert "s17._selection_key" in source
    assert "second_dev_run_performed" in source
