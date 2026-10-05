from pathlib import Path


def test_s61_trainer_is_shared_trajectory_adaptive_gate_court():
    source=Path("scripts/hira_v1_s61_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s61_train_dev.py","exec")
    assert "SEED=82_001" in source
    assert "CORRECTION_PARAMS=114_688" in source
    assert "PAIRWISE_PARAMS=32_832" in source
    assert "GATE_PARAMS=5" in source
    assert "ConfidenceAdaptiveBoundedHybridGate" in source
    assert "adaptive_gate_gold_loss" in source
    assert '"reference_decision_shell":"existing_fused"' in source
    assert '"treatment_decision_shell":"confidence_adaptive_bounded_hybrid"' in source
    assert '"gate_feature_dimension":4' in source
    assert '"gate_gradient_enters_correction":False' in source
    assert '"gate_gradient_enters_pairwise":False' in source
    assert '"pairwise_only_final_path":False' in source
    assert '"teacher_dependency":False' in source
    assert "s17._selection_key" in source
    assert "feature_set_variant_performed" in source
    assert "second_dev_run_performed" in source
