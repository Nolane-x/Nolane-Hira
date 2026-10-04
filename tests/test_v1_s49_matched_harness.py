from pathlib import Path


def test_s49_trainer_compiles_and_uses_matched_private_signature_arms():
    path=Path("scripts/hira_v1_s49_train_dev.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "SEED=70_001" in source
    assert "reference_question_conditioned_signature" in source
    assert "treatment_query_free_identity" in source
    assert "QueryFreeIdentityPrivateCorrectionFork" in source
    assert "_s49_correction_block" in source
    assert "_s49_treatment_relation_outputs" in source
    assert "correction_initialization_exact" in source
    assert "matched_selected_dev_delta_treatment_minus_reference" in source
    assert "treatment_identity_diagnostics" in source


def test_s49_trainer_keeps_capacity_and_native_trajectory_guard():
    source=Path("scripts/hira_v1_s49_train_dev.py").read_text(encoding="utf-8")
    assert '"reference_correction_parameters":CORRECTION' in source
    assert '"treatment_correction_parameters":CORRECTION' in source
    assert '"identity_trainable_parameters":0' in source
    assert '"second_encoder_pass":False' in source
    assert 'reference["runtime_trajectory_sha256"]!=treatment["runtime_trajectory_sha256"]' in source


def test_s49_train_dev_marker_is_well_formed_when_authorized():
    marker=Path("research/HIRA-V1-S49-ENABLE-TRAIN-DEV")
    if not marker.exists(): return
    source=marker.read_text(encoding="utf-8")
    assert "seed 70001" in source
    assert "TRAIN 768" in source
    assert "DEV 192" in source
    assert "question-conditioned signature vs query-free identity" in source
    assert "No second S49 DEV." in source
