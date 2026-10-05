from pathlib import Path


def test_s66_trainer_changes_only_responsibility_supervision():
    source=Path("scripts/hira_v1_s66_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s66_train_dev.py","exec")
    assert "SEED=87_001" in source
    assert "REFERENCE_GATE_PARAMS=60" in source
    assert "TREATMENT_GATE_PARAMS=60" in source
    assert "contextual_reliability_loss" in source
    assert "per_view_responsibility_loss" in source
    assert '"reference_gate_objective":"shared_pair_target_independent_bce"' in source
    assert '"treatment_gate_objective":"per_view_counterfactual_responsibility_bce"' in source
    assert '"added_treatment_parameters":0' in source
    assert '"reference_shared_pair_target":True' in source
    assert '"treatment_per_view_responsibility":True' in source
    assert '"single_view_inference_both_arms":True' in source
    assert '"same_context_path":True' in source
    assert '"same_probe_and_tolerance":True' in source
    assert "train_responsibility_disagreement_fraction" in source
    assert 'ref_diag["target_positive_count"]' in source
    assert 'trt_diag["pair_target_positive_count"]' in source
    assert "s17._selection_key" in source
    assert "second_dev_run_performed" in source
    assert "cross_view_soft_and_reliability_loss" not in source
