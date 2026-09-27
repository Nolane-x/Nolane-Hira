from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Literal

import torch
from torch import Tensor

from .ood import OODAction, OODGate, OODReceipt
from .runtime import DecisionOutput

AuthorityMaturity = Literal["missing", "provisional", "qualified"]


@dataclass(frozen=True)
class ConfidenceDiagnostics:
    max_probability: float
    normalized_entropy: float
    top_margin: float
    option_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HiraV0ReliabilityReceipt:
    action: OODAction
    reason: str
    confidence: ConfidenceDiagnostics
    probabilities_calibrated: bool
    calibration_status: AuthorityMaturity
    calibration_authority: str | None
    selective_policy_status: AuthorityMaturity
    confidence_threshold: float | None
    ood_status: AuthorityMaturity
    ood: OODReceipt
    policy_id: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["action"] = self.action.value
        payload["ood"]["action"] = self.ood.action.value
        return payload


@dataclass(frozen=True)
class HiraV0ReliableDecision:
    decision: DecisionOutput
    reliability: HiraV0ReliabilityReceipt

    @property
    def accepted(self) -> bool:
        return self.reliability.action == OODAction.ACCEPT

    @property
    def abstained(self) -> bool:
        return self.reliability.action == OODAction.ABSTAIN

    @property
    def escalated(self) -> bool:
        return self.reliability.action == OODAction.ESCALATE


def confidence_diagnostics(probabilities: Tensor) -> ConfidenceDiagnostics:
    if probabilities.ndim != 1 or probabilities.numel() < 2:
        raise ValueError("reliability probabilities must be [K] with K>=2")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("reliability probabilities must be finite")
    if bool((probabilities < 0).any()):
        raise ValueError("reliability probabilities must be non-negative")

    total = float(probabilities.sum())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("reliability probability mass must sum to one")

    p = probabilities.detach().float()
    top = torch.topk(p, k=2).values
    max_probability = float(top[0])
    top_margin = float(top[0] - top[1])

    safe = p.clamp_min(1e-12)
    entropy = float(-(safe * safe.log()).sum())
    denom = log(float(p.numel()))
    normalized_entropy = entropy / denom if denom > 0.0 else 0.0
    normalized_entropy = min(1.0, max(0.0, normalized_entropy))

    return ConfidenceDiagnostics(
        max_probability=max_probability,
        normalized_entropy=normalized_entropy,
        top_margin=top_margin,
        option_count=int(p.numel()),
    )


@dataclass(frozen=True)
class HiraV0ReliabilityPolicy:
    """Fail-closed M1 reliability policy.

    Calibration and OOD are separate authorities. Decision confidence metrics
    are diagnostics only. ACCEPT is impossible unless calibration, OOD and the
    selective policy are all independently marked qualified.
    """

    policy_id: str
    calibration_status: AuthorityMaturity = "missing"
    calibration_authority: str | None = None
    selective_policy_status: AuthorityMaturity = "missing"
    confidence_threshold: float | None = None
    ood_status: AuthorityMaturity = "missing"
    ood_gate: OODGate | None = None

    def __post_init__(self) -> None:
        allowed = {"missing", "provisional", "qualified"}
        for field_name in (
            "calibration_status",
            "selective_policy_status",
            "ood_status",
        ):
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(f"invalid reliability authority maturity: {value}")

        if not self.policy_id:
            raise ValueError("reliability policy_id must be non-empty")

        if self.confidence_threshold is not None and not (
            0.0 <= float(self.confidence_threshold) <= 1.0
        ):
            raise ValueError("confidence_threshold must be in [0,1]")

        if self.calibration_status == "qualified" and not self.calibration_authority:
            raise ValueError("qualified calibration requires an authority id")
        if (
            self.selective_policy_status == "qualified"
            and self.confidence_threshold is None
        ):
            raise ValueError(
                "qualified selective policy requires confidence threshold"
            )
        if self.ood_status == "qualified" and self.ood_gate is None:
            raise ValueError("qualified OOD status requires an OOD gate")

    @classmethod
    def m1_mechanism_fail_closed(cls) -> "HiraV0ReliabilityPolicy":
        return cls(
            policy_id="hira-v0-m1-mechanism-unqualified",
            calibration_status="provisional",
            calibration_authority=None,
            selective_policy_status="provisional",
            confidence_threshold=None,
            ood_status="provisional",
            ood_gate=None,
        )

    @staticmethod
    def _missing_ood_receipt(reason: str) -> OODReceipt:
        return OODReceipt(
            action=OODAction.ESCALATE,
            ood_score=None,
            threshold=None,
            authority="missing_ood_authority",
            calibrator_id=None,
            reason=reason,
        )

    def evaluate(
        self,
        decision: DecisionOutput,
        *,
        probabilities_calibrated: bool = False,
        ood_score: float | None = None,
        distribution_shift: bool = False,
        ood_calibrator_valid: bool = True,
    ) -> HiraV0ReliableDecision:
        diagnostics = confidence_diagnostics(decision.probabilities)

        # OOD is independent from decision confidence and must fail closed.
        if self.ood_status != "qualified" or self.ood_gate is None:
            ood = self._missing_ood_receipt("ood_authority_not_qualified")
            receipt = HiraV0ReliabilityReceipt(
                action=OODAction.ESCALATE,
                reason="ood_authority_not_qualified",
                confidence=diagnostics,
                probabilities_calibrated=bool(probabilities_calibrated),
                calibration_status=self.calibration_status,
                calibration_authority=self.calibration_authority,
                selective_policy_status=self.selective_policy_status,
                confidence_threshold=self.confidence_threshold,
                ood_status=self.ood_status,
                ood=ood,
                policy_id=self.policy_id,
            )
            return HiraV0ReliableDecision(decision=decision, reliability=receipt)

        ood = self.ood_gate.decide(
            ood_score,
            distribution_shift=distribution_shift,
            calibrator_valid=ood_calibrator_valid,
        )
        if ood.action == OODAction.ESCALATE:
            receipt = HiraV0ReliabilityReceipt(
                action=OODAction.ESCALATE,
                reason=f"ood:{ood.reason}",
                confidence=diagnostics,
                probabilities_calibrated=bool(probabilities_calibrated),
                calibration_status=self.calibration_status,
                calibration_authority=self.calibration_authority,
                selective_policy_status=self.selective_policy_status,
                confidence_threshold=self.confidence_threshold,
                ood_status=self.ood_status,
                ood=ood,
                policy_id=self.policy_id,
            )
            return HiraV0ReliableDecision(decision=decision, reliability=receipt)

        if ood.action == OODAction.ABSTAIN:
            receipt = HiraV0ReliabilityReceipt(
                action=OODAction.ABSTAIN,
                reason=f"ood:{ood.reason}",
                confidence=diagnostics,
                probabilities_calibrated=bool(probabilities_calibrated),
                calibration_status=self.calibration_status,
                calibration_authority=self.calibration_authority,
                selective_policy_status=self.selective_policy_status,
                confidence_threshold=self.confidence_threshold,
                ood_status=self.ood_status,
                ood=ood,
                policy_id=self.policy_id,
            )
            return HiraV0ReliableDecision(decision=decision, reliability=receipt)

        if self.calibration_status != "qualified":
            reason = "calibration_authority_not_qualified"
            action = OODAction.ESCALATE
        elif not probabilities_calibrated:
            reason = "decision_probabilities_not_qualified_calibrated"
            action = OODAction.ESCALATE
        elif self.selective_policy_status != "qualified":
            reason = "selective_policy_not_qualified"
            action = OODAction.ESCALATE
        elif self.confidence_threshold is None:
            reason = "selective_confidence_threshold_missing"
            action = OODAction.ESCALATE
        elif diagnostics.max_probability < float(self.confidence_threshold):
            reason = "qualified_confidence_below_threshold"
            action = OODAction.ABSTAIN
        else:
            reason = "qualified_calibration_ood_and_selective_policy_accept"
            action = OODAction.ACCEPT

        receipt = HiraV0ReliabilityReceipt(
            action=action,
            reason=reason,
            confidence=diagnostics,
            probabilities_calibrated=bool(probabilities_calibrated),
            calibration_status=self.calibration_status,
            calibration_authority=self.calibration_authority,
            selective_policy_status=self.selective_policy_status,
            confidence_threshold=self.confidence_threshold,
            ood_status=self.ood_status,
            ood=ood,
            policy_id=self.policy_id,
        )
        return HiraV0ReliableDecision(decision=decision, reliability=receipt)


__all__ = [
    "AuthorityMaturity",
    "ConfidenceDiagnostics",
    "HiraV0ReliableDecision",
    "HiraV0ReliabilityPolicy",
    "HiraV0ReliabilityReceipt",
    "confidence_diagnostics",
]
