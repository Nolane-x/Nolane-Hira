from pathlib import Path


def test_s72_a0_workflow_is_marker_gated_and_parent_pinned():
    source=Path(".github/workflows/hira-v1-s72-a0-opponent-profile-vector-residual.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S72-ENABLE-A0" in source
    assert "run-id: 37459447097" in source
    assert "11412505173" in source
    assert "sha256:aa116237d5508131db18f88a23893d9107d05ae67f46ccffe638520fe8098a50" in source
    assert "scripts/hira_v1_s72_a0_opponent_profile_vector_residual.py" in source
    assert 'assert r["reference_composer_parameter_count"]==80' in source
    assert 'assert r["treatment_composer_parameter_count"]==80' in source
    assert 'assert r["alpha_match_max_abs_error"]==0.0' in source
    assert 'assert r["neutral_direction_collapse_max_abs_error"]<=1e-6' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
