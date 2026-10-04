from pathlib import Path


def test_s52_trainer_compiles_and_binds_only_auxiliary_coefficient():
    path=Path("scripts/hira_v1_s52_train_dev.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "REFERENCE_AUX_COEFFICIENT=0.0" in source
    assert "TREATMENT_AUX_COEFFICIENT=0.10" in source
    assert "PRIVATE_TRAINABLE=147_456" in source
    assert "CORRECTION=114_688" in source
    assert "CANONICALIZER=32_768" in source
    assert "weighted_relation_code_auxiliary" in source
    assert "op.private_state_dict()" in source
    assert "op.load_private_state_dict(best_state,freeze=True)" in source
    assert "native_training_performed" in source


def test_s52_workflow_is_marker_gated_and_exact_authority_pinned():
    source=Path(".github/workflows/hira-v1-s52-query-relation-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S52-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37199183308" in source
    assert "run-id: 37192490832" in source
    assert "hira-v1-s52-a0-query-relation-canonicalization" in source
    assert "hira-v1-s51-native-authority" in source
    assert "scripts/hira_v1_s52_train_dev.py" in source
    assert 'cv=r["controlled_variable"]' in source
    assert 'assert cv["reference_auxiliary_coefficient"]==0.0' in source
    assert 'assert cv["treatment_auxiliary_coefficient"]==0.10' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
