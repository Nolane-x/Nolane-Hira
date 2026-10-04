from pathlib import Path


def test_s51_private_court_workflow_is_exact_authority_pinned():
    source=Path(".github/workflows/hira-v1-s51-artifact-pinned-private-court.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S51-ENABLE-PRIVATE-COURT" in source
    assert "run-id: 37192065019" in source
    assert "run-id: 37192490832" in source
    assert "hira-v1-s51-native-authority" in source
    assert "11299783210" in source
    assert "sha256:d2c2ec63ef6d7c197051ff357ed3ea21c93e8a903b4c5bb7ab8c01e216d91c59" in source
    assert "ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628" in source
    assert "19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916" in source
    assert "590aa9464a5d7925eb1028130f0f66957955437f2b62c6338fcd252b7de56c51" in source
    assert "scripts/hira_v1_s51_private_court.py" in source
    assert 'assert b["native_optimizer_constructed"] is False' in source
    assert 'assert b["native_training_performed"] is False' in source
    assert 'assert cache["reference_treatment_same_cache_bytes"] is True' in source
    assert 'assert r["second_dev_run_performed"] is False' in source


def test_s51_private_court_uses_frozen_private_optimizer_constants():
    source=Path("scripts/hira_v1_s51_private_court.py").read_text(encoding="utf-8")
    assert "lr=s35.LR" in source
    assert "weight_decay=s35.WEIGHT_DECAY" in source
    assert "lr=2e-4" not in source
    assert "weight_decay=0.01" not in source
