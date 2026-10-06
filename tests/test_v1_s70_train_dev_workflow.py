from pathlib import Path


def test_s70_train_dev_workflow_is_marker_gated_and_authorities_pinned():
    source=Path(".github/workflows/hira-v1-s70-fused-anchored-pairwise-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S70-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37422000871" in source
    assert "11393387610" in source
    assert "sha256:6e021d9fe8807eb6ff097bfe16b9bb464a8ee87e949d064883ec43d6cfe5a9ff" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s70_train_dev.py" in source
    assert 'assert cv["shared_pairwise_head"] is True' in source
    assert 'assert cv["shared_pairwise_training_trajectory"] is True' in source
    assert 'assert cv["shared_selection_epoch"] is True' in source
    assert 'assert cv["aggregation_trainable_parameters_per_arm"]==0' in source
    assert 'assert cv["softmax_temperature"]==1.0' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
