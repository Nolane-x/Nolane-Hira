from pathlib import Path


def test_s56_train_dev_workflow_is_marker_gated_and_exact_a0_pinned():
    source=Path(".github/workflows/hira-v1-s56-cross-view-decision-consistency-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S56-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37215444292" in source
    assert "11307997189" in source
    assert "sha256:48ee087d9a5fdd76790fbbdb625ebbd381ad44581a0df04bb047772baeff49cd" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s56_train_dev.py" in source
    assert 'assert branch["parameter_surface"]["private_trainable_parameters"]==114688' in source
    assert 'assert cv["treatment_decision_consistency_coefficient"]==0.10' in source
    assert 'assert cv["treatment_ordering_consistency_coefficient"]==0.05' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
