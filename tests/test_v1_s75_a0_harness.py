from pathlib import Path


def test_s75_a0_harness_is_mechanical_only():
    source=Path("scripts/hira_v1_s75_a0_token_level_bidirectional_binding.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s75_a0_token_level_bidirectional_binding.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s75-a0-token-level-bidirectional-binding-v1"' in source
    assert "SEED=96_001" in source
    assert "build_query_gated_pairwise_representation" in source
    assert "build_token_level_bidirectional_pairwise_representation" in source
    assert "S59_PAIRWISE_PARAMETER_COUNT" in source
    assert '"fresh_train_dev_exposed":False' in source
    assert '"used_for_model_selection":False' in source
    assert '"external_laya_jev_evaluation_opened"' not in source
