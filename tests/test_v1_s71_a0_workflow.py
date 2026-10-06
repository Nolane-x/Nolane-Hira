from pathlib import Path


def test_s71_a0_workflow_is_marker_gated_and_parent_pinned():
    source=Path(".github/workflows/hira-v1-s71-a0-multistat-row-composer.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S71-ENABLE-A0" in source
    assert "run-id: 37423476142" in source
    assert "11394610000" in source
    assert "sha256:8af711f789efb9717eaed0eff882fade7bab37b546f86a35502db82242ad221f" in source
    assert "scripts/hira_v1_s71_a0_multistat_row_composer.py" in source
    assert 'assert r["reference_parameter_count"]==80' in source
    assert 'assert r["treatment_parameter_count"]==80' in source
    assert 'assert r["added_treatment_parameter_count"]==0' in source
    assert 'assert r["reference_extra_row_channels_max_abs"]==0.0' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
