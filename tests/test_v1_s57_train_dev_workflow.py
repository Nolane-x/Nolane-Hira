from pathlib import Path


def test_s57_train_dev_workflow_is_marker_gated_and_exact_a0_pinned():
    source=Path(".github/workflows/hira-v1-s57-discrete-pairwise-ranking-consistency-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S57-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37266891469" in source
    assert "11326757222" in source
    assert "sha256:fcfbececa46f5bac0e1fe75ae51348d1b4c543b5b7dc3166410c148a35dd75ce" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s57_train_dev.py" in source
    assert 'assert cv["treatment_ordinal_consistency_coefficient"]==0.05' in source
    assert 'assert cv["gold_order_protection"] is True' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
