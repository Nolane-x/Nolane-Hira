from pathlib import Path


def test_s66_a0_harness_freezes_per_view_responsibility():
    source=Path("scripts/hira_v1_s66_a0_per_view_responsibility.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s66_a0_per_view_responsibility.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s66-a0-per-view-responsibility-v1"' in source
    assert "SEED=87_001" in source
    assert "per_view_responsibility_targets" in source
    assert "per_view_responsibility_loss" in source
    assert 'responsibility_states' in source
    assert 'treatment_target_distinct_from_pair_target' in source
    assert '"fresh_train_dev_exposed":False' in source
