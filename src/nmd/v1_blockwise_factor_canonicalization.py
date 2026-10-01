from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
import torch
import torch.nn.functional as F

from .v1_relation_canonicalization import cross_view_relation_signature_loss


FACTORIZED_SIGNATURE_DIMENSION = 256
FACTORIZED_BLOCK_DIMENSION = 128
ROLE_BLOCK_WEIGHT = 0.50
VALUE_BLOCK_WEIGHT = 0.50


@dataclass(frozen=True)
class BlockwiseFactorCanonicalizationLoss:
    total: Tensor
    role_total: Tensor
    value_total: Tensor
    role_alignment: Tensor
    role_separation: Tensor
    value_alignment: Tensor
    value_separation: Tensor


def split_factorized_role_value_signature(signature: Tensor) -> tuple[Tensor, Tensor]:
    if signature.ndim != 3:
        raise ValueError("S27 factorized signature must be [B,K,D]")
    if signature.shape[-1] != FACTORIZED_SIGNATURE_DIMENSION:
        raise ValueError("S27 factorized signature width must be 256")
    role = F.normalize(signature[..., :FACTORIZED_BLOCK_DIMENSION], dim=-1)
    value = F.normalize(signature[..., FACTORIZED_BLOCK_DIMENSION:], dim=-1)
    if not bool(torch.isfinite(role).all() and torch.isfinite(value).all()):
        raise ValueError("S27 normalized factor blocks became non-finite")
    return role, value


def blockwise_cross_view_factor_canonicalization_loss(
    canonical: Tensor,
    paraphrase: Tensor,
    *,
    separation_margin: float = 0.20,
    role_weight: float = ROLE_BLOCK_WEIGHT,
    value_weight: float = VALUE_BLOCK_WEIGHT,
) -> BlockwiseFactorCanonicalizationLoss:
    if canonical.shape != paraphrase.shape:
        raise ValueError("S27 canonical/paraphrase signatures must match")
    if canonical.ndim != 3 or canonical.shape[1] < 2:
        raise ValueError("S27 signatures must be [B,K,256] with K>=2")
    if separation_margin < 0:
        raise ValueError("S27 separation margin must be non-negative")
    if role_weight != ROLE_BLOCK_WEIGHT or value_weight != VALUE_BLOCK_WEIGHT:
        raise ValueError("S27 block weights are frozen at 0.50 / 0.50")

    canonical_role, canonical_value = split_factorized_role_value_signature(canonical)
    paraphrase_role, paraphrase_value = split_factorized_role_value_signature(paraphrase)

    role_total, role_alignment, role_separation = cross_view_relation_signature_loss(
        canonical_role,
        paraphrase_role,
        separation_margin=separation_margin,
    )
    value_total, value_alignment, value_separation = cross_view_relation_signature_loss(
        canonical_value,
        paraphrase_value,
        separation_margin=separation_margin,
    )
    total = role_weight * role_total + value_weight * value_total

    values = (
        total,
        role_total,
        value_total,
        role_alignment,
        role_separation,
        value_alignment,
        value_separation,
    )
    if not all(bool(torch.isfinite(v)) for v in values):
        raise ValueError("S27 blockwise canonicalization loss became non-finite")

    return BlockwiseFactorCanonicalizationLoss(
        total=total,
        role_total=role_total,
        value_total=value_total,
        role_alignment=role_alignment,
        role_separation=role_separation,
        value_alignment=value_alignment,
        value_separation=value_separation,
    )


__all__ = [
    "FACTORIZED_SIGNATURE_DIMENSION",
    "FACTORIZED_BLOCK_DIMENSION",
    "ROLE_BLOCK_WEIGHT",
    "VALUE_BLOCK_WEIGHT",
    "BlockwiseFactorCanonicalizationLoss",
    "split_factorized_role_value_signature",
    "blockwise_cross_view_factor_canonicalization_loss",
]
