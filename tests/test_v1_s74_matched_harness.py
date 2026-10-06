from pathlib import Path


def test_s74_matched_trainer_freezes_direct_core_variable():
    source=Path("scripts/hira_v1_s74_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s74_train_dev.py","exec")

    assert 'SCHEMA_VERSION="hira-v1-s74-direct-set-arbitration-core-train-dev-v1"' in source
    assert "SEED=95_001" in source
    assert "generate_s74_cases" in source
    assert "validate_s74_partitions" in source
    assert "DirectSetArbitrationCore" in source
    assert "direct_set_arbitration_loss" in source
    assert "use_relational=False" in source
    assert "use_relational=True" in source
    assert '"reference_input":"fused_native_plus_zero_relational"' in source
    assert '"treatment_input":"fused_native_plus_live_s69_relational"' in source
    assert '"final_logit_semantics":"direct_dsac_scores"' in source
    assert '"second_dev_run_performed":False' in source
    assert '"external_laya_jev_evaluation_opened":False' in source


def test_s74_trainer_has_no_post_dev_rescue_knobs():
    source=Path("scripts/hira_v1_s74_train_dev.py").read_text(encoding="utf-8")
    assert '"hidden_width_sweep_performed":False' in source
    assert '"channel_sweep_performed":False' in source
    assert '"loss_weight_sweep_performed":False' in source
    assert '"activation_sweep_performed":False' in source
    assert '"fused_residual_fallback_added":False' in source
    assert '"target_changed":False' in source
