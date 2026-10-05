from pathlib import Path


def test_s56_trainer_is_objective_only_matched_court():
    source=Path("scripts/hira_v1_s56_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s56_train_dev.py","exec")
    assert "SEED=77_001" in source
    assert "PRIVATE_TRAINABLE=114_688" in source
    assert "REFERENCE_DECISION_COEFFICIENT=0.0" in source
    assert "REFERENCE_ORDER_COEFFICIENT=0.0" in source
    assert "TREATMENT_DECISION_COEFFICIENT=0.10" in source
    assert "TREATMENT_ORDER_COEFFICIENT=0.05" in source
    assert source.count("JointStateQueryOptionPrivateCorrectionFork(")>=2
    assert 'added_trainable_parameters":0' in source
    assert "weighted_cross_view_decision_auxiliary" in source
    assert "native_optimizer_constructed" in source
    assert "second_dev_run_performed" in source


def test_s56_trainer_reports_anti_collapse_diagnostics():
    source=Path("scripts/hira_v1_s56_train_dev.py").read_text(encoding="utf-8")
    assert "mean_fused_entropy" in source
    assert "mean_top1_top2_probability_gap" in source
    assert "mean_standardized_logit_rms" in source
    assert "mean_standardized_pair_decision_js" in source
    assert "mean_pairwise_ordering_loss" in source
