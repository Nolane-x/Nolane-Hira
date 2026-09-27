from types import SimpleNamespace

import pytest
import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.mainline import HiraV0Mainline, HiraV0Manifest
from nmd.mainline_reliability import (
    HiraV0ReliabilityPolicy,
    confidence_diagnostics,
)
from nmd.ood import OODAction, OODGate
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.w33_coevidence_semantic import CoEvidenceSemanticScorer


def _decision(probabilities):
    return SimpleNamespace(probabilities=torch.tensor(probabilities, dtype=torch.float32))


def _qualified_policy(*, confidence_threshold=0.70, ood_threshold=0.60):
    return HiraV0ReliabilityPolicy(
        policy_id="test-qualified-policy",
        calibration_status="qualified",
        calibration_authority="fresh-calibration-authority",
        selective_policy_status="qualified",
        confidence_threshold=confidence_threshold,
        ood_status="qualified",
        ood_gate=OODGate(
            threshold=ood_threshold,
            calibrator_id="fresh-ood-calibrator",
            authority="fresh-ood-authority",
        ),
    )


def _options():
    return (
        LogicalOption(
            "wait",
            "a short wait is acceptable",
            aliases=("brief delay remains permitted",),
            value=0.0,
        ),
        LogicalOption(
            "now",
            "action must begin immediately",
            aliases=("no brief delay is permitted",),
            value=1.0,
        ),
    )


def _synthetic_m1():
    torch.manual_seed(16101)
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    scorer.freeze_candidate()
    runtime = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        coevidence_symmetric_semantic_scorer=scorer,
    )
    return HiraV0Mainline(
        runtime,
        manifest=HiraV0Manifest.m1_mechanism_provisional(),
        reliability_policy=HiraV0ReliabilityPolicy.m1_mechanism_fail_closed(),
    )


def test_confidence_diagnostics_are_bounded_and_separate():
    x = confidence_diagnostics(torch.tensor([0.7, 0.2, 0.1]))
    assert x.option_count == 3
    assert x.max_probability == pytest.approx(0.7)
    assert x.top_margin == pytest.approx(0.5)
    assert 0.0 <= x.normalized_entropy <= 1.0


def test_high_raw_confidence_can_never_substitute_for_ood_authority():
    policy = HiraV0ReliabilityPolicy.m1_mechanism_fail_closed()
    out = policy.evaluate(
        _decision([0.999, 0.001]),
        probabilities_calibrated=False,
    )
    assert out.reliability.action == OODAction.ESCALATE
    assert out.reliability.reason == "ood_authority_not_qualified"
    assert out.reliability.confidence.max_probability > 0.99


def test_qualified_ood_still_cannot_accept_without_qualified_calibration():
    policy = HiraV0ReliabilityPolicy(
        policy_id="ood-only",
        calibration_status="provisional",
        selective_policy_status="qualified",
        confidence_threshold=0.70,
        ood_status="qualified",
        ood_gate=OODGate(
            threshold=0.60,
            calibrator_id="ood-a",
            authority="fresh-ood",
        ),
    )
    out = policy.evaluate(
        _decision([0.95, 0.05]),
        probabilities_calibrated=False,
        ood_score=0.10,
    )
    assert out.reliability.ood.action == OODAction.ACCEPT
    assert out.reliability.action == OODAction.ESCALATE
    assert out.reliability.reason == "calibration_authority_not_qualified"


def test_ood_abstention_precedes_confidence_acceptance():
    policy = _qualified_policy()
    out = policy.evaluate(
        _decision([0.99, 0.01]),
        probabilities_calibrated=True,
        ood_score=0.90,
    )
    assert out.reliability.ood.action == OODAction.ABSTAIN
    assert out.reliability.action == OODAction.ABSTAIN
    assert out.reliability.reason == "ood:ood_score_above_threshold"


def test_distribution_shift_escalates_even_with_high_confidence():
    policy = _qualified_policy()
    out = policy.evaluate(
        _decision([0.99, 0.01]),
        probabilities_calibrated=True,
        ood_score=0.05,
        distribution_shift=True,
    )
    assert out.reliability.action == OODAction.ESCALATE
    assert out.reliability.reason == "ood:distribution_shift"


def test_all_qualified_authorities_can_accept_only_calibrated_probabilities():
    policy = _qualified_policy(confidence_threshold=0.70)

    raw = policy.evaluate(
        _decision([0.90, 0.10]),
        probabilities_calibrated=False,
        ood_score=0.10,
    )
    assert raw.reliability.action == OODAction.ESCALATE
    assert raw.reliability.reason == "decision_probabilities_not_qualified_calibrated"

    calibrated = policy.evaluate(
        _decision([0.90, 0.10]),
        probabilities_calibrated=True,
        ood_score=0.10,
    )
    assert calibrated.reliability.action == OODAction.ACCEPT
    assert calibrated.accepted is True


def test_qualified_low_confidence_abstains():
    policy = _qualified_policy(confidence_threshold=0.80)
    out = policy.evaluate(
        _decision([0.65, 0.35]),
        probabilities_calibrated=True,
        ood_score=0.10,
    )
    assert out.reliability.action == OODAction.ABSTAIN
    assert out.reliability.reason == "qualified_confidence_below_threshold"


def test_session_reliability_is_state_once_and_separate_from_noul():
    model = _synthetic_m1()
    before = model.runtime.state_encode_calls
    session = model.open_session(
        "The request may wait briefly and no immediate handling is required."
    )
    assert model.runtime.state_encode_calls == before + 1

    out = session.decide_reliable(
        primitive="noul",
        question_text="Must action begin immediately?",
        options=_options(),
    )

    assert model.runtime.state_encode_calls == before + 1
    assert session.query_count == 1
    assert out.decision.primitive == "noul"
    assert out.reliability.action == OODAction.ESCALATE
    assert out.reliability.reason == "ood_authority_not_qualified"
    assert out.decision.probabilities.numel() == 2


def test_m1_manifest_is_provisional_not_available():
    model = _synthetic_m1()
    manifest = model.manifest.to_dict()
    assert manifest["version"] == "0.0-m1a"
    assert manifest["reliability_ood_abstention"] == "provisional"
    assert manifest["production_ready"] is False
    assert manifest["transfer_core"] == "provisional"


def test_qualified_policy_requires_explicit_authority_material():
    with pytest.raises(ValueError, match="authority id"):
        HiraV0ReliabilityPolicy(
            policy_id="bad-calibration",
            calibration_status="qualified",
        )

    with pytest.raises(ValueError, match="confidence threshold"):
        HiraV0ReliabilityPolicy(
            policy_id="bad-selective",
            selective_policy_status="qualified",
        )

    with pytest.raises(ValueError, match="OOD gate"):
        HiraV0ReliabilityPolicy(
            policy_id="bad-ood",
            ood_status="qualified",
        )
