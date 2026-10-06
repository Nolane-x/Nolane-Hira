from pathlib import Path


def test_s70_matched_trainer_freezes_only_aggregation_variable():
    source=Path("scripts/hira_v1_s70_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s70_train_dev.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s70-fused-anchored-pairwise-train-dev-v1"' in source
    assert "SEED=91_001" in source
    assert "generate_s70_cases" in source
    assert "validate_s70_partitions" in source
    assert "build_query_gated_pairwise_representation" in source
    assert "build_reference_pairwise_representation" in source
    assert "uniform_pairwise_aggregate" in source
    assert "fused_anchored_pairwise_aggregate" in source
    assert "head=ExplicitPairwiseDecisionHead" in source
    assert "reference_head=" not in source
    assert "treatment_head=" not in source
    assert '"shared_pairwise_head":True' in source
    assert '"shared_pairwise_training_trajectory":True' in source
    assert '"shared_selection_epoch":True' in source
    assert "key=s17._selection_key(epoch,fused_shadow)" in source
    assert '"second_dev_run_performed":False' in source
    assert '"external_laya_jev_evaluation_opened":False' in source


def test_s70_trainer_keeps_gate_context_reference_representation():
    source=Path("scripts/hira_v1_s70_train_dev.py").read_text(encoding="utf-8")
    assert '"gate_context_source_both_arms":"exact_s59_reference_representation"' in source
    assert '"representation_family_both_arms":"s69_query_gated_identity_interaction"' in source
