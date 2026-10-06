from pathlib import Path


def test_s74_a0_harness_contract():
    source=Path("scripts/hira_v1_s74_a0_direct_set_arbitration_core.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s74_a0_direct_set_arbitration_core.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s74-a0-direct-set-arbitration-core-v1"' in source
    assert 'OUTCOME="HIRA_V1_S74_A0_DIRECT_SET_ARBITRATION_CORE_READY"' in source
    assert "SEED=95_001" in source
    assert "reference_relational_channels_max_abs" in source
    assert "treatment_relational_channels_max_abs" in source
    assert "fresh_train_dev_exposed" in source
    assert '"fresh_train_dev_exposed":False' in source
