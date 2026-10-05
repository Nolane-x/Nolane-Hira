from pathlib import Path


def test_s59_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s59-explicit-learned-pairwise-decision-head-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S59-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37278474963" in source
    assert "11331042601" in source
    assert "sha256:5271081c4bcacbced9ca85c146ddd36b5dc9ecdc0c6dc9038c9b3fb6219a11b6" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s59_train_dev.py" in source
    assert 'assert cv["pairwise_trainable_parameters"]==32832' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
