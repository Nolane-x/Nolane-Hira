from pathlib import Path


def test_s72_matched_trainer_freezes_direction_only_comparison():
    source=Path("scripts/hira_v1_s72_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s72_train_dev.py","exec")

    assert 'SEED=93_001' in source
    assert 'A0_RUN=37463175304' in source
    assert 'A0_ARTIFACT_ID=11413616350' in source
    assert 'sha256:a4d8052a0e76160310d4a99e86ef71f07810a96e77681cb34d747b85c98ef52a' in source

    assert 'reference=MultiStatPairwiseRowComposer(use_multistat=False' in source
    assert 'treatment=MultiStatPairwiseRowComposer(use_multistat=False' in source
    assert 'use_multistat=True' not in source

    assert 'use_opponent_profile_direction=False' in source
    assert 'use_opponent_profile_direction=True' in source
    assert 'reference_residual_direction' in source
    assert 'treatment_residual_direction' in source
    assert 'fused_opponent_profile_confidence' in source

    assert 'if ref_digest!=trt_digest:' in source
    assert 'selected composer states diverged' in source
    assert 'selected alpha policy diverged' in source

    assert '"second_dev_run_performed":False' in source
    assert '"external_laya_jev_evaluation_opened":False' in source
    assert '"production_ready_claimed":False' in source
