from pathlib import Path


def test_s54_workflow_is_marker_gated_and_exact_authority_pinned():
    source=Path(".github/workflows/hira-v1-s54-joint-state-query-option-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S54-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37208186642" in source
    assert "run-id: 37192490832" in source
    assert "hira-v1-s54-a0-joint-state-query-option-interaction" in source
    assert "hira-v1-s51-native-authority" in source
    assert "scripts/hira_v1_s54_train_dev.py" in source
    assert "11305528035" in source
    assert "sha256:b83d5a8a11ed656fa67ad3b3c9d0b0bbb8a4e858a4ba2300555d932a168effde" in source
    assert 'assert cv["reference_query_context"]=="option_conditioned_token_level_query_context"' in source
    assert 'assert cv["treatment_query_context"]=="joint_state_query_option_context"' in source
    assert 'assert r["second_dev_run_performed"] is False' in source


def test_s54_workflow_keeps_matched_zero_parameter_interaction_surface():
    source=Path(".github/workflows/hira-v1-s54-joint-state-query-option-train-dev.yml").read_text(encoding="utf-8")
    assert 'assert cv["reference_private_trainable_parameters"]==114688' in source
    assert 'assert cv["treatment_private_trainable_parameters"]==114688' in source
    assert 'assert cv["joint_interaction_trainable_parameters"]==0' in source
    assert 'assert cv["token_temperature"]==0.10' in source
