from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CanonicalRelationDiagnostics:
    state_role_normalized_entropy: Tensor
    state_role_max_weight: Tensor
    option_role_normalized_entropy: Tensor
    option_role_max_weight: Tensor
    mean_pair_entropy: Tensor
    mean_best_pair_score: Tensor

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
            "mean_best_pair_score": float(
                self.mean_best_pair_score.detach().cpu()
            ),
        }


class CrossViewRelationCanonicalizer(nn.Module):
    """Parameter-free S13 relation-signature extractor.

    The only learned geometry is the inherited shared projection. This module
    turns a state/question/option relation into one normalized signature per
    logical option. Semantically equivalent wording views can then be aligned
    directly in relation space without adding a downstream head.
    """

    def __init__(
        self,
        *,
        role_temperature: float = 0.10,
        pair_temperature: float = 0.10,
        contrastive_temperature: float = 0.10,
    ):
        super().__init__()
        for name, value in (
            ("role", role_temperature),
            ("pair", pair_temperature),
            ("contrastive", contrastive_temperature),
        ):
            if value <= 0:
                raise ValueError(f"S13 {name} temperature must be positive")
        self.role_temperature = float(role_temperature)
        self.pair_temperature = float(pair_temperature)
        self.contrastive_temperature = float(contrastive_temperature)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _project(self, projection: nn.Linear, x: Tensor) -> Tensor:
        if not isinstance(projection, nn.Linear):
            raise TypeError("S13 canonicalizer requires a shared nn.Linear projection")
        if projection.bias is not None:
            raise ValueError("S13 canonicalizer requires bias-free shared projection")
        if x.shape[-1] != projection.in_features:
            raise ValueError("S13 projection input mismatch")
        y = F.normalize(projection(x), dim=-1)
        if not bool(torch.isfinite(y).all()):
            raise ValueError("S13 projection produced non-finite values")
        return y

    def _validate(
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
    ) -> None:
        if state_tokens.ndim != 3:
            raise ValueError("S13 state_tokens must be [B,S,D]")
        if question_tokens.ndim != 3:
            raise ValueError("S13 question_tokens must be [B,Q,D]")
        if option_view_tokens.ndim != 5:
            raise ValueError("S13 option_view_tokens must be [B,K,V,T,D]")
        if (
            state_tokens.shape[-1] != projection.in_features
            or question_tokens.shape[-1] != projection.in_features
            or option_view_tokens.shape[-1] != projection.in_features
        ):
            raise ValueError("S13 d_model mismatch")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("S13 state_mask mismatch")
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("S13 question_mask mismatch")
        if (
            option_view_token_mask.shape != option_view_tokens.shape[:4]
            or option_view_token_mask.dtype != torch.bool
        ):
            raise ValueError("S13 option token mask mismatch")
        if option_view_mask.shape != option_view_tokens.shape[:3] or option_view_mask.dtype != torch.bool:
            raise ValueError("S13 option view mask mismatch")
        if not (
            state_tokens.shape[0]
            == question_tokens.shape[0]
            == option_view_tokens.shape[0]
        ):
            raise ValueError("S13 batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("S13 canonicalization requires K>=2")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("S13 requires state content")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("S13 requires question content")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("S13 requires active option views")
        if bool(((option_view_token_mask.sum(-1) < 1) & option_view_mask).any()):
            raise ValueError("S13 active option views require content tokens")

    @staticmethod
    def _normalized_entropy(weights: Tensor, mask: Tensor) -> Tensor:
        eps = torch.finfo(weights.dtype).eps
        count = mask.sum(-1).to(weights.dtype)
        den = torch.where(count > 1, count.log(), torch.ones_like(count))
        entropy = -(weights.clamp_min(eps).log() * weights).sum(-1)
        return entropy / den

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
    ) -> tuple[Tensor, Tensor, CanonicalRelationDiagnostics]:
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
        state_role = torch.softmax(state_role_score / self.role_temperature, dim=-1)
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
        option_role_score = option_q.amax(dim=-2)
        option_role_score = option_role_score.masked_fill(
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

        state_rel = F.normalize(state - state_anchor[:, None, :], dim=-1)
        option_rel = F.normalize(
            options - option_anchor[:, :, :, None, :],
            dim=-1,
        )

        direct = torch.einsum("bsd,bkvtd->bkvst", state, options)
        relative = torch.einsum("bsd,bkvtd->bkvst", state_rel, option_rel)
        pair_score = 0.5 * (direct + relative)
        pair_mask = (
            state_mask[:, None, None, :, None]
            & option_view_token_mask[:, :, :, None, :]
        )
        pair_score = pair_score.masked_fill(~pair_mask, -1e4)

        b, k, v, s, t = pair_score.shape
        flat_score = pair_score.reshape(b, k, v, s * t)
        flat_mask = pair_mask.reshape(b, k, v, s * t)
        pair_weight = torch.softmax(flat_score / self.pair_temperature, dim=-1)
        pair_weight = pair_weight * flat_mask.to(pair_weight.dtype)
        pair_weight = pair_weight / pair_weight.sum(-1, keepdim=True).clamp_min(1e-12)
        pair_weight_5d = pair_weight.reshape(b, k, v, s, t)

        state_sig = torch.einsum(
            "bkvst,bsd->bkvd",
            pair_weight_5d,
            state_rel,
        )
        option_sig = torch.einsum(
            "bkvst,bkvtd->bkvd",
            pair_weight_5d,
            option_rel,
        )
        role_delta = F.normalize(
            option_anchor - state_anchor[:, None, None, :],
            dim=-1,
        )
        view_signature = F.normalize(
            state_sig + option_sig + role_delta,
            dim=-1,
        )
        active = option_view_mask.to(view_signature.dtype)[..., None]
        signatures = (view_signature * active).sum(-2)
        signatures = signatures / active.sum(-2).clamp_min(1.0)
        signatures = F.normalize(signatures, dim=-1)

        best_pair = pair_score.amax(dim=(-1, -2))
        role_compatibility = torch.einsum(
            "bd,bkvd->bkv",
            state_anchor,
            option_anchor,
        )
        view_score = 0.5 * (best_pair + role_compatibility)
        view_score = view_score.masked_fill(~option_view_mask, 0.0)
        view_count = option_view_mask.sum(-1).clamp_min(1).to(view_score.dtype)
        logits = view_score.sum(-1) / view_count
        logits = logits / self.contrastive_temperature

        if not bool(torch.isfinite(signatures).all() and torch.isfinite(logits).all()):
            raise ValueError("S13 canonicalization produced non-finite values")

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

        diagnostics = CanonicalRelationDiagnostics(
            state_role_normalized_entropy=state_entropy.mean(),
            state_role_max_weight=state_role.max(-1).values.mean(),
            option_role_normalized_entropy=option_entropy_mean,
            option_role_max_weight=option_max_mean,
            mean_pair_entropy=pair_entropy_mean,
            mean_best_pair_score=(
                best_pair * option_active
            ).sum() / option_active.sum().clamp_min(1.0),
        )
        return logits, signatures, diagnostics


def cross_view_relation_signature_loss(
    canonical: Tensor,
    paraphrase: Tensor,
    *,
    separation_margin: float = 0.20,
) -> tuple[Tensor, Tensor, Tensor]:
    if canonical.ndim != 3 or paraphrase.shape != canonical.shape:
        raise ValueError("S13 relation signatures must share [B,K,D]")
    if canonical.shape[1] < 2:
        raise ValueError("S13 relation signature loss requires K>=2")
    if separation_margin < 0:
        raise ValueError("S13 separation margin must be non-negative")

    canonical = F.normalize(canonical, dim=-1)
    paraphrase = F.normalize(paraphrase, dim=-1)
    same = (canonical * paraphrase).sum(-1)
    alignment = (1.0 - same).mean()

    cross = torch.einsum("bkd,bjd->bkj", canonical, paraphrase)
    k = canonical.shape[1]
    eye = torch.eye(k, dtype=torch.bool, device=cross.device)[None, :, :]
    wrong = cross.masked_fill(eye, float("-inf")).amax(-1)
    separation = F.relu(
        cross.diagonal(dim1=-2, dim2=-1).new_tensor(separation_margin)
        - (same - wrong)
    ).mean()

    total = alignment + separation
    if not bool(torch.isfinite(total)):
        raise ValueError("S13 signature loss became non-finite")
    return total, alignment, separation


def relation_signature_same_option_cosine(
    canonical: Tensor,
    paraphrase: Tensor,
) -> Tensor:
    if canonical.shape != paraphrase.shape or canonical.ndim != 3:
        raise ValueError("S13 relation signatures must share [B,K,D]")
    return (
        F.normalize(canonical, dim=-1)
        * F.normalize(paraphrase, dim=-1)
    ).sum(-1)


__all__ = [
    "CanonicalRelationDiagnostics",
    "CrossViewRelationCanonicalizer",
    "cross_view_relation_signature_loss",
    "relation_signature_same_option_cosine",
]
