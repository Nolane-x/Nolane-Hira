from pathlib import Path


def test_s60_trainer_is_shared_trajectory_bounded_hybrid_court():
    source=Path("scripts/hira_v1_s60_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s60_train_dev.py","exec")
    assert "SEED=81_001" in source
    assert "CORRECTION_PARAMS=114_688" in source
    assert "PAIRWISE_PARAMS=32_832" in source
    assert "COMPOSER_PARAMS=1" in source
    assert "TrainCalibratedBoundedHybridComposer" in source
    assert "composer_gold_loss" in source
    assert '"reference_decision_shell":"existing_fused"' in source
    assert '"treatment_decision_shell":"train_calibrated_bounded_hybrid"' in source
    assert '"composer_gradient_enters_correction":False' in source
    assert '"composer_gradient_enters_pairwise":False' in source
    assert '"pairwise_only_final_path":False' in source
    assert '"teacher_dependency":False' in source
    assert "s17._selection_key" in source
    assert "alpha_max_sweep_performed" in source
    assert "second_dev_run_performed" in source
