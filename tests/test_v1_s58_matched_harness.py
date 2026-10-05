from pathlib import Path


def test_s58_trainer_is_matched_teacher_consensus_court():
    source=Path("scripts/hira_v1_s58_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s58_train_dev.py","exec")
    assert "SEED=79_001" in source
    assert "PRIVATE_TRAINABLE=114_688" in source
    assert "REFERENCE_COEFFICIENT=0.0" in source
    assert "TREATMENT_COEFFICIENT=0.05" in source
    assert "804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa" in source
    assert "weighted_teacher_consensus_auxiliary" in source
    assert "precomputed_once_before_student_training" in source
    assert "reference_treatment_same_teacher_target_bytes" in source
    assert "selected_using_s58_data" in source
    assert "teacher_swap_performed" in source
    assert "second_dev_run_performed" in source
