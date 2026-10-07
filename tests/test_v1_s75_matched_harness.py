from pathlib import Path


def test_s75_matched_trainer_freezes_controlled_variable():
    source=Path("scripts/hira_v1_s75_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s75_train_dev.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s75-token-level-bidirectional-binding-train-dev-v1"' in source
    assert "SEED=96_001" in source
    assert "generate_s75_cases" in source
    assert "validate_s75_partitions" in source
    assert "build_query_gated_pairwise_representation" in source
    assert "build_token_level_bidirectional_pairwise_representation" in source
    assert '"reference_representation_family":"s69_query_gated_identity_interaction"' in source
    assert '"treatment_representation_family":' in source
    assert '"s75_token_level_bidirectional_binding"' in source
    assert '"exact_s74_state_question_option_overlap":0' in source
    assert '"second_dev_run_performed":False' in source
    assert '"external_laya_jev_evaluation_opened":False' in source


def test_s75_trainer_has_no_post_dev_rescue_knobs():
    source=Path("scripts/hira_v1_s75_train_dev.py").read_text(encoding="utf-8")
    assert '"token_weight_formula_changed":False' in source
    assert '"pooling_or_normalization_changed":False' in source
    assert '"representation_dimension_changed":False' in source
    assert '"learned_representation_adapter_added":False' in source
    assert '"pairwise_width_rank_changed":False' in source
    assert '"loss_optimizer_changed":False' in source
    assert '"reliability_target_reopened":False' in source
