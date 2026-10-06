from pathlib import Path


def test_s71_a0_harness_freezes_multistat_composer():
    source=Path("scripts/hira_v1_s71_a0_multistat_row_composer.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s71_a0_multistat_row_composer.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s71-a0-multistat-row-composer-v1"' in source
    assert "SEED=92_001" in source
    assert "MultiStatPairwiseRowComposer" in source
    assert "S71_PARAMETER_COUNT" in source
    assert '"reference_extra_row_channels_max_abs"' in source
    assert '"fixed_alpha_residual_direction_max_abs_error"' in source
    assert '"fresh_train_dev_exposed":False' in source
