from pathlib import Path


def test_s48_trainer_compiles_and_is_capacity_matched():
    path=Path("scripts/hira_v1_s48_train_dev.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "SEED=69_001" in source
    assert "PrivateCorrectionRepresentationFork" in source
    assert "QueryQuotientPrivateCorrectionFork" in source
    assert "_state_equal" in source
    assert "reference_raw_query" in source
    assert "treatment_query_quotient" in source
    assert "runtime_trajectory_sha256" in source
    assert "correction_initialization_identical" in source
    assert "delta_treatment_minus_reference" in source
    assert "treatment_quotient_diagnostics" in source
    assert "second_dev_run_performed" in source


def test_s48_trainer_preserves_equal_correction_capacity():
    source=Path("scripts/hira_v1_s48_train_dev.py").read_text(encoding="utf-8")
    assert '"reference_correction_parameters":114_688' in source
    assert '"treatment_correction_parameters":114_688' in source
    assert '"reference_total_trainable_parameters":163_840' in source
    assert '"treatment_total_trainable_parameters":163_840' in source
    assert '"quotient_trainable_parameters":0' in source
    assert '"second_encoder_pass":False' in source


def test_s48_train_dev_marker_is_well_formed_when_authorized():
    marker=Path("research/HIRA-V1-S48-ENABLE-TRAIN-DEV")
    if not marker.exists():
        return
    source=marker.read_text(encoding="utf-8")
    assert "seed 69001" in source
    assert "TRAIN 768" in source
    assert "DEV 192" in source
    assert "raw-query correction vs query-quotient correction" in source
    assert "No second S48 DEV." in source
