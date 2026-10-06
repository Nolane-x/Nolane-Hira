from pathlib import Path


def test_s71_matched_trainer_freezes_only_row_stat_exposure():
    source=Path("scripts/hira_v1_s71_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s71_train_dev.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s71-multistat-row-composer-train-dev-v1"' in source
    assert "SEED=92_001" in source
    assert "generate_s71_cases" in source
    assert "validate_s71_partitions" in source
    assert "build_query_gated_pairwise_representation" in source
    assert "build_reference_pairwise_representation" in source
    assert "MultiStatPairwiseRowComposer(use_multistat=False" in source
    assert "MultiStatPairwiseRowComposer(use_multistat=True" in source
    assert "head=ExplicitPairwiseDecisionHead" in source
    assert "reference_head=" not in source
    assert "treatment_head=" not in source
    assert '"shared_pairwise_head":True' in source
    assert '"shared_pairwise_training_trajectory":True' in source
    assert '"shared_correction_trajectory":True' in source
    assert '"shared_selection_epoch":True' in source
    assert "key=s17._selection_key(epoch,fused_shadow)" in source
    assert '"residual_direction_both_arms":"uniform_row_mean"' in source
    assert '"second_dev_run_performed":False' in source
    assert '"external_laya_jev_evaluation_opened":False' in source


def test_s71_matched_trainer_uses_exact_controlled_row_channels():
    source=Path("scripts/hira_v1_s71_train_dev.py").read_text(encoding="utf-8")
    assert '"reference_row_channels":"normalized_mean_plus_three_zero_channels"' in source
    assert '"treatment_row_channels":"normalized_mean_max_min_rms"' in source
    assert '"composer_trainable_parameters_per_arm":COMPOSER_PARAMS' in source
    assert '"composer_context_source_both_arms":"exact_s59_reference_representation"' in source
    assert '"composer_objective_both_arms":"per_view_counterfactual_responsibility_bce"' in source
