from pathlib import Path


def test_s70_a0_harness_freezes_fused_anchored_aggregation():
    source=Path("scripts/hira_v1_s70_a0_fused_anchored_pairwise.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s70_a0_fused_anchored_pairwise.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s70-a0-fused-anchored-pairwise-aggregation-v1"' in source
    assert "SEED=91_001" in source
    assert "uniform_pairwise_aggregate" in source
    assert "fused_anchored_pairwise_aggregate" in source
    assert '"uniform_prior_collapse_max_abs_error"' in source
    assert '"fresh_train_dev_exposed":False' in source
