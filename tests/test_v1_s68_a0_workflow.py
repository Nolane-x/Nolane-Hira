from pathlib import Path


def test_s68_a0_workflow_is_marker_gated_and_parent_pinned():
    source=Path(".github/workflows/hira-v1-s68-a0-context-modulated-pairwise.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S68-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s68_a0_context_modulated_pairwise.py" in source
    assert 'assert r["parent_s67"]["merged_main"]=="942871bb44bbef3e28e74698f161b02c6796cdcb"' in source
    assert 'assert r["parent_s67"]["scientific_run"]==37336779415' in source
    assert 'assert r["parent_s67"]["actions_artifact_available"] is False' in source
    assert 'assert r["reference_pairwise_parameter_count"]==33344' in source
    assert 'assert r["treatment_pairwise_parameter_count"]==33344' in source
    assert 'assert r["added_treatment_parameter_count"]==0' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
