from pathlib import Path


def test_s73_matched_trainer_freezes_controlled_variable():
    source=Path("scripts/hira_v1_s73_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s73_train_dev.py","exec")

    assert 'SCHEMA_VERSION="hira-v1-s73-counterfactual-safety-veto-train-dev-v1"' in source
    assert "SEED=94_001" in source
    assert "generate_s73_cases" in source
    assert "validate_s73_partitions" in source
    assert "CounterfactualSafetyVeto" in source
    assert "counterfactual_safety_targets" in source
    assert "counterfactual_safety_loss" in source
    assert "use_opponent_profile_direction=True" in source
    assert '"reference_veto_usage":"ignored_always_candidate"' in source
    assert '"treatment_veto_usage":"hard_candidate_or_fused"' in source
    assert '"hard_accept_threshold":S73_ACCEPT_THRESHOLD' in source
    assert '"veto_output_mode":"endpoint_only"' in source
    assert '"second_dev_run_performed":False' in source
    assert '"external_laya_jev_evaluation_opened":False' in source


def test_s73_trainer_has_no_post_dev_rescue_knobs():
    source=Path("scripts/hira_v1_s73_train_dev.py").read_text(encoding="utf-8")
    assert '"threshold_sweep_performed":False' in source
    assert '"feature_sweep_performed":False' in source
    assert '"class_weight_or_smoothing_changed":False' in source
    assert '"safety_target_changed":False' in source
    assert '"residual_or_representation_or_head_changed":False' in source
