from pathlib import Path


def test_s72_train_dev_workflow_is_one_shot_and_pinned():
    source=Path(".github/workflows/hira-v1-s72-opponent-profile-vector-residual-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S72-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37192490832" in source
    assert "run-id: 37463175304" in source
    assert "11413616350" in source
    assert "sha256:a4d8052a0e76160310d4a99e86ef71f07810a96e77681cb34d747b85c98ef52a" in source
    assert "scripts/hira_v1_s72_train_dev.py" in source
    assert "HIRA_V1_S72_OPPONENT_PROFILE_VECTOR_RESIDUAL_DEV_COMPLETE" in source
    assert 'assert ref["composer_state_sha256"]==trt["composer_state_sha256"]' in source
    assert 'assert ref["selected_dev"]["pairwise_gold_pair_accuracy"]==trt["selected_dev"]["pairwise_gold_pair_accuracy"]' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
