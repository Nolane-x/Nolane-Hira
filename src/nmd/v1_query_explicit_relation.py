from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_relation_canonicalization import (
    CanonicalRelationDiagnostics,
    CrossViewRelationCanonicalizer,
)


@dataclass(frozen=True)
class QueryExplicitRelationDiagnostics:
    base: CanonicalRelationDiagnostics
    mean_query_option_compatibility: Tensor
    max_query_option_compatibility: Tensor
    mean_query_option_signature_norm: Tensor

    def to_dict(self) -> dict[str, float]:
        out = {f"base_{k}": v for k, v in self.base.to_dict().items()}
        out.update(
            {
                "mean_query_option_compatibility": float(
                    self.mean_query_option_compatibility.detach().cpu()
                ),
                "max_query_option_compatibility": float(
                    self.max_query_option_compatibility.detach().cpu()
                ),
                "mean_query_option_signature_norm": float(
                    self.mean_query_option_signature_norm.detach().cpu()
                ),
            }
        )
        return out


class QueryExplicitRelationCanonicalizer(nn.Module):
    """S33 zero-parameter query-explicit relation coordinates.

    The exact S13 canonicalizer remains the base relation path. S33 adds one
    explicit query-option coordinate block and one explicit query-option
    compatibility term without adding learned state.

    Final signature:
        normalize(concat(base_S13_signature, query_option_signature))

    Final logits:
        0.5 * base_S13_logits + 0.5 * query_option_logits
    """

    def __init__(
        self,
        *,
        role_temperature: float = 0.10,
        pair_temperature: float = 0.10,
        contrastive_temperature: float = 0.10,
        base_weight: float = 0.50,
        query_option_weight: float = 0.50,
    ):
        super().__init__()
        for name, value in (
            ("role", role_temperature),
            ("pair", pair_temperature),
            ("contrastive", contrastive_temperature),
        ):
            if value <= 0:
                raise ValueError(f"S33 {name} temperature must be positive")
        if base_weight < 0 or query_option_weight < 0:
            raise ValueError("S33 score weights must be non-negative")
        if abs((base_weight + query_option_weight) - 1.0) > 1e-12:
            raise ValueError("S33 score weights must sum to 1")

        self.role_temperature = float(role_temperature)
        self.pair_temperature = float(pair_temperature)
        self.contrastive_temperature = float(contrastive_temperature)
        self.base_weight = float(base_weight)
        self.query_option_weight = float(query_option_weight)
        self.base = CrossViewRelationCanonicalizer(
            role_temperature=role_temperature,
            pair_temperature=pair_temperature,
            contrastive_temperature=contrastive_temperature,
        )

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _project(projection: nn.Linear, x: Tensor) -> Tensor:
        if not isinstance(projection, nn.Linear):
            raise TypeError("S33 requires a shared nn.Linear projection")
        if projection.bias is not None:
            raise ValueError("S33 requires bias-free shared projection")
        if x.shape[-1] != projection.in_features:
            raise ValueError("S33 projection input mismatch")
        y = F.normalize(projection(x), dim=-1)
        if not bool(torch.isfinite(y).all()):
            raise ValueError("S33 projection produced non-finite values")
        return y

    @staticmethod
    def _masked_query_anchor(question: Tensor, question_mask: Tensor) -> Tensor:
        weight = question_mask.to(question.dtype)[..., None]
        pooled = (question * weight).sum(dim=1)
        pooled = pooled / weight.sum(dim=1).clamp_min(1.0)
        anchor = F.normalize(pooled, dim=-1)
        if not bool(torch.isfinite(anchor).all()):
            raise ValueError("S33 query anchor became non-finite")
        return anchor

    def forward_with_components(
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
    ) -> tuple[
        Tensor,
        Tensor,
        QueryExplicitRelationDiagnostics,
        dict[str, Tensor],
    ]:
        # Let the inherited S13 implementation own all validation and the
        # complete base relation path. This also makes base-identity testable.
        base_logits, base_signatures, base_diag = self.base(
            projection=projection,
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        question = self._project(projection, question_tokens)
        options = self._project(projection, option_view_tokens)
        query_anchor = self._masked_query_anchor(question, question_mask)

        # Recompute exactly the S13 option role anchor. The only new
        # representation is downstream of this inherited role selection.
        option_q = torch.einsum("bqd,bkvtd->bkvqt", question, options)
        option_q = option_q.masked_fill(
            ~question_mask[:, None, None, :, None],
            -1e4,
        )
        option_role_score = option_q.amax(dim=-2)
        option_role_score = option_role_score.masked_fill(
            ~option_view_token_mask,
            -1e4,
        )
        option_role = torch.softmax(
            option_role_score / self.role_temperature,
            dim=-1,
        )
        option_role = (
            option_role * option_view_token_mask.to(option_role.dtype)
        )
        option_role = option_role / option_role.sum(
            -1, keepdim=True
        ).clamp_min(1e-12)
        option_anchor = F.normalize(
            torch.einsum("bkvt,bkvtd->bkvd", option_role, options),
            dim=-1,
        )

        query_option_delta_view = F.normalize(
            option_anchor - query_anchor[:, None, None, :],
            dim=-1,
        )

        active = option_view_mask.to(
            query_option_delta_view.dtype
        )[..., None]
        query_option_signature = (
            query_option_delta_view * active
        ).sum(dim=-2)
        query_option_signature = query_option_signature / active.sum(
            dim=-2
        ).clamp_min(1.0)
        query_option_signature = F.normalize(
            query_option_signature,
            dim=-1,
        )

        signatures = F.normalize(
            torch.cat((base_signatures, query_option_signature), dim=-1),
            dim=-1,
        )

        query_option_compatibility_view = torch.einsum(
            "bd,bkvd->bkv",
            query_anchor,
            option_anchor,
        )
        query_option_compatibility_view = (
            query_option_compatibility_view
            * option_view_mask.to(query_option_compatibility_view.dtype)
        )
        view_count = option_view_mask.sum(-1).clamp_min(1).to(
            query_option_compatibility_view.dtype
        )
        query_option_score = (
            query_option_compatibility_view.sum(-1) / view_count
        )
        query_option_logits = (
            query_option_score / self.contrastive_temperature
        )

        logits = (
            self.base_weight * base_logits
            + self.query_option_weight * query_option_logits
        )

        if not bool(
            torch.isfinite(logits).all()
            and torch.isfinite(signatures).all()
            and torch.isfinite(query_option_signature).all()
        ):
            raise ValueError("S33 query-explicit relation became non-finite")

        option_active = option_view_mask.to(
            query_option_compatibility_view.dtype
        )
        active_count = option_active.sum().clamp_min(1.0)
        mean_compat = (
            query_option_compatibility_view * option_active
        ).sum() / active_count
        max_compat = query_option_compatibility_view.masked_fill(
            ~option_view_mask,
            float("-inf"),
        ).amax(-1)
        max_compat = max_compat.mean()

        diagnostics = QueryExplicitRelationDiagnostics(
            base=base_diag,
            mean_query_option_compatibility=mean_compat,
            max_query_option_compatibility=max_compat,
            mean_query_option_signature_norm=(
                query_option_signature.norm(dim=-1).mean()
            ),
        )
        components = {
            "base_logits": base_logits,
            "base_signatures": base_signatures,
            "query_anchor": query_anchor,
            "option_anchor": option_anchor,
            "query_option_delta_view": query_option_delta_view,
            "query_option_signature": query_option_signature,
            "query_option_compatibility_view": (
                query_option_compatibility_view
            ),
            "query_option_logits": query_option_logits,
        }
        return logits, signatures, diagnostics, components

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
    ) -> tuple[Tensor, Tensor, QueryExplicitRelationDiagnostics]:
        logits, signatures, diagnostics, _ = self.forward_with_components(
            projection=projection,
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )
        return logits, signatures, diagnostics


__all__ = [
    "QueryExplicitRelationDiagnostics",
    "QueryExplicitRelationCanonicalizer",
]
