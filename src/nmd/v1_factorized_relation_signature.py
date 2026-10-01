from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_relation_canonicalization import CrossViewRelationCanonicalizer


@dataclass(frozen=True)
class FactorizedRoleValueDiagnostics:
    state_role_normalized_entropy: Tensor
    state_role_max_weight: Tensor
    option_role_normalized_entropy: Tensor
    option_role_max_weight: Tensor
    mean_pair_entropy: Tensor
    mean_role_compatibility: Tensor
    mean_value_compatibility: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "state_role_normalized_entropy": float(
                self.state_role_normalized_entropy.detach().cpu()
            ),
            "state_role_max_weight": float(
                self.state_role_max_weight.detach().cpu()
            ),
            "option_role_normalized_entropy": float(
                self.option_role_normalized_entropy.detach().cpu()
            ),
            "option_role_max_weight": float(
                self.option_role_max_weight.detach().cpu()
            ),
            "mean_pair_entropy": float(self.mean_pair_entropy.detach().cpu()),
            "mean_role_compatibility": float(
                self.mean_role_compatibility.detach().cpu()
            ),
            "mean_value_compatibility": float(
                self.mean_value_compatibility.detach().cpu()
            ),
        }


class FactorizedRoleValueRelationCanonicalizer(CrossViewRelationCanonicalizer):
    """S26 zero-parameter factorized role/value relation representation.

    S13 adds role and pair-derived vectors into one signature. S26 keeps role
    and value/content relation blocks distinct until the final normalized
    signature. Relation logits use a frozen symmetric role/value combination.
    """

    role_weight = 0.50
    value_weight = 0.50

    @property
    def factorization_added_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _orthogonal_residual(tokens: Tensor, anchor: Tensor) -> Tensor:
        if tokens.ndim == 3:
            coefficient = torch.einsum("bsd,bd->bs", tokens, anchor)
            residual = tokens - coefficient[..., None] * anchor[:, None, :]
        elif tokens.ndim == 5:
            coefficient = torch.einsum("bkvtd,bkvd->bkvt", tokens, anchor)
            residual = tokens - coefficient[..., None] * anchor[:, :, :, None, :]
        else:
            raise ValueError("S26 unsupported residual rank")
        return F.normalize(residual, dim=-1)

    def forward(
        self,
        *,
        projection: nn.Linear,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> tuple[Tensor, Tensor, FactorizedRoleValueDiagnostics]:
        self._validate(
            projection=projection,
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        state = self._project(projection, state_tokens)
        question = self._project(projection, question_tokens)
        options = self._project(projection, option_view_tokens)

        state_q = torch.einsum("bqd,bsd->bqs", question, state)
        state_q = state_q.masked_fill(~question_mask[..., None], -1e4)
        state_role_score = state_q.amax(dim=1).masked_fill(~state_mask, -1e4)
        state_role = torch.softmax(
            state_role_score / self.role_temperature,
            dim=-1,
        )
        state_role = state_role * state_mask.to(state_role.dtype)
        state_role = state_role / state_role.sum(-1, keepdim=True).clamp_min(1e-12)
        state_anchor = F.normalize(
            torch.einsum("bs,bsd->bd", state_role, state),
            dim=-1,
        )

        option_q = torch.einsum("bqd,bkvtd->bkvqt", question, options)
        option_q = option_q.masked_fill(
            ~question_mask[:, None, None, :, None],
            -1e4,
        )
        option_role_score = option_q.amax(dim=-2).masked_fill(
            ~option_view_token_mask,
            -1e4,
        )
        option_role = torch.softmax(
            option_role_score / self.role_temperature,
            dim=-1,
        )
        option_role = option_role * option_view_token_mask.to(option_role.dtype)
        option_role = option_role / option_role.sum(-1, keepdim=True).clamp_min(1e-12)
        option_anchor = F.normalize(
            torch.einsum("bkvt,bkvtd->bkvd", option_role, options),
            dim=-1,
        )

        role_compatibility = torch.einsum(
            "bd,bkvd->bkv",
            state_anchor,
            option_anchor,
        )

        state_content = self._orthogonal_residual(state, state_anchor)
        option_content = self._orthogonal_residual(options, option_anchor)

        content_pair_score = torch.einsum(
            "bsd,bkvtd->bkvst",
            state_content,
            option_content,
        )
        pair_mask = (
            state_mask[:, None, None, :, None]
            & option_view_token_mask[:, :, :, None, :]
        )
        content_pair_score = content_pair_score.masked_fill(~pair_mask, -1e4)

        b, k, v, s, t = content_pair_score.shape
        flat_score = content_pair_score.reshape(b, k, v, s * t)
        flat_mask = pair_mask.reshape(b, k, v, s * t)
        pair_weight = torch.softmax(flat_score / self.pair_temperature, dim=-1)
        pair_weight = pair_weight * flat_mask.to(pair_weight.dtype)
        pair_weight = pair_weight / pair_weight.sum(-1, keepdim=True).clamp_min(1e-12)
        pair_weight_5d = pair_weight.reshape(b, k, v, s, t)

        state_value = F.normalize(
            torch.einsum(
                "bkvst,bsd->bkvd",
                pair_weight_5d,
                state_content,
            ),
            dim=-1,
        )
        option_value = F.normalize(
            torch.einsum(
                "bkvst,bkvtd->bkvd",
                pair_weight_5d,
                option_content,
            ),
            dim=-1,
        )

        value_compatibility = torch.einsum(
            "bkvd,bkvd->bkv",
            state_value,
            option_value,
        )

        role_delta = F.normalize(
            option_anchor - state_anchor[:, None, None, :],
            dim=-1,
        )
        value_delta = F.normalize(
            option_value - state_value,
            dim=-1,
        )
        view_signature = F.normalize(
            torch.cat((role_delta, value_delta), dim=-1),
            dim=-1,
        )

        active = option_view_mask.to(view_signature.dtype)[..., None]
        signatures = (view_signature * active).sum(-2)
        signatures = signatures / active.sum(-2).clamp_min(1.0)
        signatures = F.normalize(signatures, dim=-1)

        view_score = (
            self.role_weight * role_compatibility
            + self.value_weight * value_compatibility
        )
        view_score = view_score.masked_fill(~option_view_mask, 0.0)
        view_count = option_view_mask.sum(-1).clamp_min(1).to(view_score.dtype)
        logits = view_score.sum(-1) / view_count
        logits = logits / self.contrastive_temperature

        if not bool(torch.isfinite(signatures).all() and torch.isfinite(logits).all()):
            raise ValueError("S26 factorized relation output became non-finite")

        state_entropy = self._normalized_entropy(state_role, state_mask)
        option_entropy = self._normalized_entropy(
            option_role,
            option_view_token_mask,
        )
        option_active = option_view_mask.to(option_entropy.dtype)
        option_entropy_mean = (
            option_entropy * option_active
        ).sum() / option_active.sum().clamp_min(1.0)
        option_max_mean = (
            option_role.max(-1).values * option_active
        ).sum() / option_active.sum().clamp_min(1.0)

        pair_eps = torch.finfo(pair_weight.dtype).eps
        pair_count = flat_mask.sum(-1).to(pair_weight.dtype)
        pair_den = torch.where(
            pair_count > 1,
            pair_count.log(),
            torch.ones_like(pair_count),
        )
        pair_entropy = -(
            pair_weight.clamp_min(pair_eps).log() * pair_weight
        ).sum(-1) / pair_den
        pair_entropy_mean = (
            pair_entropy * option_active
        ).sum() / option_active.sum().clamp_min(1.0)

        diagnostics = FactorizedRoleValueDiagnostics(
            state_role_normalized_entropy=state_entropy.mean(),
            state_role_max_weight=state_role.max(-1).values.mean(),
            option_role_normalized_entropy=option_entropy_mean,
            option_role_max_weight=option_max_mean,
            mean_pair_entropy=pair_entropy_mean,
            mean_role_compatibility=(
                role_compatibility * option_active
            ).sum() / option_active.sum().clamp_min(1.0),
            mean_value_compatibility=(
                value_compatibility * option_active
            ).sum() / option_active.sum().clamp_min(1.0),
        )
        return logits, signatures, diagnostics


__all__ = [
    "FactorizedRoleValueDiagnostics",
    "FactorizedRoleValueRelationCanonicalizer",
]
