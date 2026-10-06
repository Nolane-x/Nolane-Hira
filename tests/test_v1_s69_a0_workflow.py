from pathlib import Path


def test_s69_a0_workflow_is_marker_gated_and_parent_pinned():
    source=Path(".github/workflows/hira-v1-s69-a0-query-gated-interaction.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S69-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "run-id: 37389861004" in source
    assert "11381470793" in source
    assert "sha256:0156bb57125f6b8c4f380649505803ed76cf49b8edf1c5bd4774fd8790386eaa" in source
    assert "scripts/hira_v1_s69_a0_query_gated_interaction.py" in source
    assert 'assert r["reference_representation_parameter_count"]==0' in source
    assert 'assert r["treatment_representation_parameter_count"]==0' in source
    assert 'assert r["reference_pairwise_parameter_count"]==32832' in source
    assert 'assert r["treatment_pairwise_parameter_count"]==32832' in source
    assert 'assert r["interaction_scale"]==16.0' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
