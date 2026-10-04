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


def test_s49_matched_workflow_is_a0_bound_and_one_shot():
    path=Path(".github/workflows/hira-v1-s49-matched-query-free-option-identity-train-dev.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s49-private-query-free-option-identity" in source
    assert "research/HIRA-V1-S49-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37178164136" in source
    assert "hira-v1-s49-a0-query-free-option-identity" in source
    assert "scripts/hira_v1_s49_train_dev.py" in source
    assert "tests/test_v1_s49_authority.py" in source
    assert "tests/test_v1_s49_matched_harness.py" in source
    assert 'assert r["seed"]==70001' in source
    assert "reference-question-conditioned-private-candidate.pt" in source
    assert "treatment-query-free-identity-private-candidate.pt" in source
    assert 'assert r["second_dev_run_performed"] is False' in source
