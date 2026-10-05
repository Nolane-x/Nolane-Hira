from pathlib import Path


def test_s65_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s65-cross-view-soft-and-reliability-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S65-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37322871071" in source
    assert "11349983389" in source
    assert "sha256:1abecf58d75be991cd89dd3bb6732a859b1159426023a209758b1d5a7d459dea" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s65_train_dev.py" in source
    assert 'assert cv["reference_gate_objective"]=="independent_view_bce"' in source
    assert 'assert cv["treatment_gate_objective"]=="cross_view_exact_min_soft_and_bce"' in source
    assert 'assert cv["reference_gate_trainable_parameters"]==60' in source
    assert 'assert cv["treatment_gate_trainable_parameters"]==60' in source
    assert 'assert cv["added_treatment_parameters"]==0' in source
    assert 'assert cv["single_view_inference_both_arms"] is True' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
