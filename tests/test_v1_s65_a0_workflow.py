from pathlib import Path


def test_s65_a0_workflow_is_marker_gated_and_parent_pinned():
    source=Path(".github/workflows/hira-v1-s65-a0-cross-view-soft-and-reliability.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S65-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "run-id: 37319541951" in source
    assert "11348594503" in source
    assert "sha256:f3d04d452a51b7358a23a9391579bca6ce1ab2c16fa107dc9f7f79740dba90ea" in source
    assert "scripts/hira_v1_s65_a0_cross_view_soft_and_reliability.py" in source
    assert 'assert r["reference_gate_parameter_count"]==60' in source
    assert 'assert r["treatment_gate_parameter_count"]==60' in source
    assert 'assert r["added_treatment_parameter_count"]==0' in source
    assert 'assert r["soft_and_exact_min_max_abs_error"]<=1e-7' in source
    assert 'assert r["same_single_view_inference_path"] is True' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
