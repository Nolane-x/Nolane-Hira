from pathlib import Path


def test_s58_train_dev_workflow_is_marker_gated_and_authorities_pinned():
    source=Path(".github/workflows/hira-v1-s58-consensus-teacher-pairwise-ranking-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S58-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37274963437" in source
    assert "11330280320" in source
    assert "sha256:f4785071bafcf3674cecefbba98468c9d2dcb42a5c1212f0c446177bde5f5ebc" in source
    assert "run-id: 37271509208" in source
    assert "804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s58_train_dev.py" in source
    assert 'assert cv["treatment_teacher_consensus_coefficient"]==0.05' in source
    assert 'assert r["teacher_swap_performed"] is False' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
