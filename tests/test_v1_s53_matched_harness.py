from pathlib import Path


def test_s53_trainer_compiles_and_has_single_controlled_variable():
    path=Path("scripts/hira_v1_s53_train_dev.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "PRIVATE_TRAINABLE=114_688" in source
    assert 'reference=QueryFreeIdentityPrivateCorrectionFork' in source
    assert 'treatment=TokenLateInteractionQueryFreePrivateCorrectionFork' in source
    assert 's50._private_loss(' in source
    assert '"treatment",canonical,paraphrase' in source
    assert "correction_state_dict()" in source
    assert "load_correction_state_dict(best_state,freeze=True)" in source
    assert '"late_interaction_trainable_parameters":0' in source
    assert "native_training_performed" in source


def test_s53_workflow_is_marker_gated_and_exact_authority_pinned():
    source=Path(".github/workflows/hira-v1-s53-token-query-option-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S53-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37205099128" in source
    assert "run-id: 37192490832" in source
    assert "hira-v1-s53-a0-token-query-option-late-interaction" in source
    assert "hira-v1-s51-native-authority" in source
    assert "scripts/hira_v1_s53_train_dev.py" in source
    assert 'assert cv["token_temperature"]==0.10' in source
    assert 'assert cv["late_interaction_trainable_parameters"]==0' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
