from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class ReferenceAnchorProjectionDiagnostics:
    runtime_gradient_norm: float
    anchor_gradient_norm: float
    pre_dot: float
    post_dot: float
    projected: bool
    projection_coefficient: float

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "runtime_gradient_norm": self.runtime_gradient_norm,
            "anchor_gradient_norm": self.anchor_gradient_norm,
            "pre_dot": self.pre_dot,
            "post_dot": self.post_dot,
            "projected": self.projected,
            "projection_coefficient": self.projection_coefficient,
        }


def _validate_pair(a: Sequence[Tensor], b: Sequence[Tensor]) -> None:
    if not a or len(a) != len(b):
        raise ValueError("S40 gradient lists must be nonempty and equal length")
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape or x.device != y.device or x.dtype != y.dtype:
            raise ValueError(f"S40 gradient mismatch at {i}")
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError(f"S40 non-finite gradient at {i}")


def _dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    total = a[0].new_zeros(())
    for x, y in zip(a, b):
        total = total + torch.sum(x * y)
    return total


def _sq(a: Sequence[Tensor]) -> Tensor:
    return _dot(a, a)


def signature_anchor_loss(
    treatment_signatures: Tensor,
    reference_signatures: Tensor,
    *,
    valid_mask: Tensor | None = None,
    epsilon: float = 1e-12,
) -> Tensor:
    if epsilon <= 0:
        raise ValueError("S40 anchor epsilon must be positive")
    if treatment_signatures.shape != reference_signatures.shape:
        raise ValueError("S40 treatment/reference signature shape mismatch")
    if treatment_signatures.ndim < 2:
        raise ValueError("S40 signatures must include a feature dimension")
    if not bool(
        torch.isfinite(treatment_signatures).all()
        and torch.isfinite(reference_signatures).all()
    ):
        raise ValueError("S40 non-finite signature input")

    reference = reference_signatures.detach()
    treatment = F.normalize(treatment_signatures, dim=-1, eps=epsilon)
    reference = F.normalize(reference, dim=-1, eps=epsilon)
    drift = 1.0 - torch.sum(treatment * reference, dim=-1)

    if valid_mask is None:
        return drift.mean()

    if valid_mask.shape != drift.shape or valid_mask.dtype != torch.bool:
        raise ValueError("S40 anchor valid_mask mismatch")
    count = valid_mask.sum()
    if int(count.item()) < 1:
        raise ValueError("S40 anchor requires at least one valid signature")
    return drift.masked_select(valid_mask).mean()


def project_runtime_gradient_against_reference_anchor(
    runtime_gradient: Sequence[Tensor],
    anchor_gradient: Sequence[Tensor],
    *,
    epsilon: float = 1e-12,
) -> tuple[list[Tensor], ReferenceAnchorProjectionDiagnostics]:
    if epsilon <= 0:
        raise ValueError("S40 projection epsilon must be positive")
    _validate_pair(runtime_gradient, anchor_gradient)

    g = [x.detach().clone() for x in runtime_gradient]
    a = [x.detach().clone() for x in anchor_gradient]

    g_norm_t = torch.sqrt(_sq(g))
    a_sq = _sq(a)
    a_norm_t = torch.sqrt(a_sq)
    pre = _dot(a, g)

    projected = bool(pre.item() < 0.0 and float(a_norm_t.item()) > 0.0)
    coeff = pre.new_zeros(())
    out = g
    if projected:
        coeff = pre / (a_sq + epsilon)
        out = [x - coeff * y for x, y in zip(g, a)]

    post = _dot(a, out)
    return out, ReferenceAnchorProjectionDiagnostics(
        runtime_gradient_norm=float(g_norm_t.item()),
        anchor_gradient_norm=float(a_norm_t.item()),
        pre_dot=float(pre.item()),
        post_dot=float(post.item()),
        projected=projected,
        projection_coefficient=float(coeff.item()),
    )


__all__ = [
    "ReferenceAnchorProjectionDiagnostics",
    "project_runtime_gradient_against_reference_anchor",
    "signature_anchor_loss",
]
