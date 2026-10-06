from pathlib import Path


def test_s69_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s69-query-gated-interaction-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S69-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37417409926" in source
    assert "11391117254" in source
    assert "sha256:40c2e13e6a40f9e0b2a51a4cdeb7a6819511da4dd29d9ce19f3d688ad0617383" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s69_train_dev.py" in source
    assert 'assert cv["reference_representation_parameters"]==0' in source
    assert 'assert cv["treatment_representation_parameters"]==0' in source
    assert 'assert cv["reference_pairwise_trainable_parameters"]==32832' in source
    assert 'assert cv["treatment_pairwise_trainable_parameters"]==32832' in source
    assert 'assert cv["interaction_scale"]==16.0' in source
    assert 'assert cv["gate_context_source_both_arms"]=="exact_s59_reference_representation"' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
