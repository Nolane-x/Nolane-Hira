from pathlib import Path


def test_s72_a0_harness_freezes_vector_residual_direction():
    source=Path("scripts/hira_v1_s72_a0_opponent_profile_vector_residual.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s72_a0_opponent_profile_vector_residual.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s72-a0-opponent-profile-vector-residual-v1"' in source
    assert "SEED=93_001" in source
    assert "opponent_profile_vector_direction" in source
    assert "reference_pairwise_mean_direction" in source
    assert '"alpha_match_max_abs_error"' in source
    assert '"neutral_direction_collapse_max_abs_error"' in source
    assert '"nontrivial_direction_max_abs_difference"' in source
    assert '"fresh_train_dev_exposed":False' in source
