import copy

import pytest
import torch

from nmd.hira import HIRACore
from nmd.mainline import HiraV0Mainline, HiraV0Manifest
from nmd.mainline_m1_authority import generate_m1_authority
from nmd.mainline_m1_cache import (
    OOD_FEATURE_NAMES,
    SEMANTIC_OOD_FEATURE_COUNT,
    compile_m1_frozen_cache,
    validate_m1_frozen_cache,
)
from nmd.mainline_reliability import HiraV0ReliabilityPolicy
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.w33_coevidence_semantic import CoEvidenceSemanticScorer


def _model():
    torch.manual_seed(16301)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=96,
    )
    scorer = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    scorer.freeze_candidate()
    runtime = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        coevidence_symmetric_semantic_scorer=scorer,
    )
    model = HiraV0Mainline(
        runtime,
        manifest=HiraV0Manifest.m1_mechanism_provisional(),
        reliability_policy=HiraV0ReliabilityPolicy.m1_mechanism_fail_closed(),
    )
    # Synthetic unit models do not carry real checkpoint bytes, but the
    # machine-readable manifest preserves exact mainline provenance.
    return model


def test_m1_id_cache_is_state_once_frozen_and_precalibration():
    model = _model()
    rows = generate_m1_authority("cal_dev")[:6]
    before = model.runtime.state_encode_calls
    cache = compile_m1_frozen_cache(model, rows, partition="cal_dev")

    assert model.runtime.state_encode_calls - before == len(rows)
    metadata = cache["metadata"]
    assert metadata["case_count"] == len(rows)
    assert metadata["state_encodes_per_case"] == 1.0
    assert metadata["decision_core_frozen"] is True
    assert metadata["calibration_applied"] is False
    assert metadata["production_ready"] is False
    assert tuple(metadata["ood_feature_names"]) == OOD_FEATURE_NAMES
    assert metadata["semantic_ood_feature_count"] == SEMANTIC_OOD_FEATURE_COUNT

    for case in cache["cases"]:
        assert case["is_ood"] is False
        assert case["gold_index"] is not None
        assert isinstance(case["gold_probabilities"], torch.Tensor)
        assert case["logits"].requires_grad is False
        assert case["logits"].is_inference() is False
        assert case["ood_features"].shape == (len(OOD_FEATURE_NAMES),)
        assert torch.isfinite(case["ood_features"]).all()


def test_m1_ood_cache_has_no_fake_task_gold():
    model = _model()
    rows = generate_m1_authority("ood_dev")[:6]
    cache = compile_m1_frozen_cache(model, rows, partition="ood_dev")

    for case in cache["cases"]:
        assert case["is_ood"] is True
        assert case["gold_index"] is None
        assert case["gold_probabilities"] is None
        assert case["ood_kind"] is not None


def test_m1_ood_feature_contract_cannot_be_confidence_only():
    model = _model()
    rows = generate_m1_authority("ood_dev")[:3]
    cache = compile_m1_frozen_cache(model, rows, partition="ood_dev")

    broken = copy.deepcopy(cache)
    broken["metadata"]["semantic_ood_feature_count"] = 0
    with pytest.raises(ValueError, match="confidence-only"):
        validate_m1_frozen_cache(broken, expected_partition="ood_dev")


def test_m1_cache_rejects_calibrated_runtime():
    model = _model()
    # Any non-None calibrator is forbidden at cache-compilation time.
    model.runtime.reliability_calibrator = object()
    rows = generate_m1_authority("cal_dev")[:2]
    with pytest.raises(RuntimeError, match="before calibration"):
        compile_m1_frozen_cache(model, rows, partition="cal_dev")


def test_m1_cache_rejects_trainable_decision_core():
    model = _model()
    parameter = next(model.runtime.encoder.parameters())
    parameter.requires_grad_(True)
    rows = generate_m1_authority("cal_dev")[:2]
    with pytest.raises(RuntimeError, match="completely frozen"):
        compile_m1_frozen_cache(model, rows, partition="cal_dev")


def test_m1_cache_partition_mismatch_fails_closed():
    model = _model()
    rows = generate_m1_authority("cal_dev")[:2]
    with pytest.raises(ValueError, match="do not match"):
        compile_m1_frozen_cache(model, rows, partition="cal_train")
