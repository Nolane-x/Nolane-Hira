from pathlib import Path


def test_s60_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s60-train-calibrated-bounded-hybrid-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S60-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37291693731" in source
    assert "11337255933" in source
    assert "sha256:0a8f44e150932c0df4393cb74af94a380f93680e7b3fde2e0ea642295356c182" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s60_train_dev.py" in source
    assert 'assert cv["composer_trainable_parameters"]==1' in source
    assert 'assert cv["alpha_max"]==0.35' in source
    assert 'assert cv["pairwise_only_final_path"] is False' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
