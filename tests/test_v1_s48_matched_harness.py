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


def test_s48_matched_workflow_is_one_shot_a0_bound_and_capacity_matched():
    path=Path(".github/workflows/hira-v1-s48-matched-query-quotient-option-evidence-train-dev.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s48-query-quotient-option-evidence" in source
    assert "research/HIRA-V1-S48-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37171579714" in source
    assert "hira-v1-s48-a0-query-quotient-option-evidence" in source
    assert "scripts/hira_v1_s48_train_dev.py" in source
    assert "tests/test_v1_s48_authority.py" in source
    assert "tests/test_v1_s48_matched_harness.py" in source
    assert 'assert r["seed"]==69001' in source
    assert "reference_raw_query" in source
    assert "treatment_query_quotient" in source
    assert "delta_treatment_minus_reference" in source
    assert "reference-raw-query" in source
    assert "treatment-query-quotient" in source
    assert 'assert r["second_dev_run_performed"] is False' in source
