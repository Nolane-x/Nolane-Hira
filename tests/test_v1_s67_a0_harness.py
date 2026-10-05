from pathlib import Path


def test_s67_a0_harness_freezes_safe_oracle_alpha():
    source=Path("scripts/hira_v1_s67_a0_safe_oracle_alpha.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s67_a0_safe_oracle_alpha.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s67-a0-safe-oracle-alpha-v1"' in source
    assert "SEED=88_001" in source
    assert "S67_ALPHA_LATTICE" in source
    assert "safe_oracle_alpha_targets" in source
    assert "safe_oracle_alpha_loss" in source
    assert "select_safe_oracle_alpha" in source
    assert '"oracle_target_levels"' in source
    assert '"smaller_alpha_tie_break_max_abs_error"' in source
    assert '"fresh_train_dev_exposed":False' in source
