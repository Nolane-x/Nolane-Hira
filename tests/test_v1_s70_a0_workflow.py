from pathlib import Path


def test_s70_a0_workflow_is_marker_gated_and_parent_pinned():
    source=Path(".github/workflows/hira-v1-s70-a0-fused-anchored-pairwise.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S70-ENABLE-A0" in source
    assert "run-id: 37420104460" in source
    assert "11392672918" in source
    assert "sha256:c24c00e9b79d23b3e7e3fe4d17013270e0410d822b85c62230bb85e972a2f94d" in source
    assert "scripts/hira_v1_s70_a0_fused_anchored_pairwise.py" in source
    assert 'assert r["aggregation_parameter_count"]==0' in source
    assert 'assert r["reference_gate_parameter_count"]==60' in source
    assert 'assert r["treatment_gate_parameter_count"]==60' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
