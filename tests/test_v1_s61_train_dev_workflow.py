from pathlib import Path


def test_s61_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s61-confidence-adaptive-bounded-hybrid-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S61-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37299903484" in source
    assert "11341361300" in source
    assert "sha256:2313d461197d8a635842a86220b55cf249600f58e0d044ba6d016edf3d9ca4fb" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s61_train_dev.py" in source
    assert 'assert cv["gate_trainable_parameters"]==5' in source
    assert 'assert cv["gate_feature_dimension"]==4' in source
    assert 'assert cv["pairwise_only_final_path"] is False' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
