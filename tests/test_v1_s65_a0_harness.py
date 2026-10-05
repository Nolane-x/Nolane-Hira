from pathlib import Path


def test_s65_a0_harness_freezes_cross_view_soft_and_only():
    source=Path("scripts/hira_v1_s65_a0_cross_view_soft_and_reliability.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s65_a0_cross_view_soft_and_reliability.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s65-a0-cross-view-soft-and-reliability-v1"' in source
    assert "SEED=86_001" in source
    assert '"reference_objective":"independent_view_bce"' in source
    assert '"treatment_objective":"cross_view_exact_min_soft_and_bce"' in source
    assert "cross_view_soft_and_logit" in source
    assert "cross_view_soft_and_reliability_loss" in source
    assert "soft_and_exact_min_max_abs_error" in source
    assert "lower_view_gradient_route_max_abs_error" in source
    assert '"same_single_view_inference_path":True' in source
    assert '"fresh_train_dev_exposed":False' in source
