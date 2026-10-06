from pathlib import Path


def test_s69_a0_harness_freezes_interaction_representation():
    source=Path("scripts/hira_v1_s69_a0_query_gated_interaction.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s69_a0_query_gated_interaction.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s69-a0-query-gated-interaction-rep-v1"' in source
    assert "SEED=90_001" in source
    assert "build_reference_pairwise_representation" in source
    assert "build_query_gated_pairwise_representation" in source
    assert "S69_INTERACTION_SCALE" in source
    assert '"fresh_train_dev_exposed":False' in source
