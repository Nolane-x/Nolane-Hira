from pathlib import Path


def test_s59_trainer_is_shared_trajectory_decision_shell_court():
    source=Path("scripts/hira_v1_s59_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s59_train_dev.py","exec")
    assert "SEED=80_001" in source
    assert "CORRECTION_PARAMS=114_688" in source
    assert "PAIRWISE_PARAMS=32_832" in source
    assert "build_pairwise_representation" in source
    assert "shared_training_trajectory" in source
    assert '"reference_decision_shell":"existing_fused"' in source
    assert '"treatment_decision_shell":"explicit_learned_pairwise"' in source
    assert '"pairwise_gradient_enters_correction":False' in source
    assert '"teacher_dependency":False' in source
    assert "s17._selection_key" in source
    assert "width_sweep_performed" in source
    assert "second_dev_run_performed" in source
