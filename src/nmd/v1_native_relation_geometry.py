from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_relation_canonicalization import CanonicalRelationDiagnostics


class NativeA13RelationCanonicalizer(nn.Module):
    """S35 S13-equivalent relation geometry in native adapted A13 256D space.

    This operator deliberately has no reference to the shared W28-style
    projection. The primary S17 scorer may still use that projection, but the
    treatment relation path does not.
    """

    def __init__(
        self,
        *,
        role_temperature: float = 0.10,
        pair_temperature: float = 0.10,
        contrastive_temperature: float = 0.10,
        native_dimension: int = 256,
    ):
        super().__init__()
        for name, value in (
            ("role", role_temperature),
            ("pair", pair_temperature),
            ("contrastive", contrastive_temperature),
        ):
            if value <= 0:
                raise ValueError(f"S35 {name} temperature must be positive")
        if native_dimension != 256:
            raise ValueError("S35 native dimension is frozen at 256")
        self.role_temperature = float(role_temperature)
        self.pair_temperature = float(pair_temperature)
        self.contrastive_temperature = float(contrastive_temperature)
        self.native_dimension = int(native_dimension)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _native(x: Tensor) -> Tensor:
        y = F.normalize(x, dim=-1)
        if not bool(torch.isfinite(y).all()):
            raise ValueError("S35 native normalization produced non-finite values")
        return y

    def _validate(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> None:
        if state_tokens.ndim != 3:
            raise ValueError("S35 state_tokens must be [B,S,D]")
        if question_tokens.ndim != 3:
            raise ValueError("S35 question_tokens must be [B,Q,D]")
        if option_view_tokens.ndim != 5:
            raise ValueError("S35 option_view_tokens must be [B,K,V,T,D]")
        if not (
            state_tokens.shape[-1]
            == question_tokens.shape[-1]
            == option_view_tokens.shape[-1]
            == self.native_dimension
        ):
            raise ValueError("S35 native dimension mismatch")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("S35 state_mask mismatch")
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("S35 question_mask mismatch")
        if (
            option_view_token_mask.shape != option_view_tokens.shape[:4]
            or option_view_token_mask.dtype != torch.bool
        ):
            raise ValueError("S35 option token mask mismatch")
        if option_view_mask.shape != option_view_tokens.shape[:3] or option_view_mask.dtype != torch.bool:
            raise ValueError("S35 option view mask mismatch")
        if not (
            state_tokens.shape[0]
            == question_tokens.shape[0]
            == option_view_tokens.shape[0]
        ):
            raise ValueError("S35 batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("S35 requires K>=2")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("S35 requires state content")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("S35 requires question content")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("S35 requires active option views")
        if bool(((option_view_token_mask.sum(-1) < 1) & option_view_mask).any()):
            raise ValueError("S35 active option views require content tokens")

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
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> tuple[Tensor, Tensor, CanonicalRelationDiagnostics]:
        self._validate(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        state = self._native(state_tokens)
        question = self._native(question_tokens)
        options = self._native(option_view_tokens)

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
            raise ValueError("S35 native relation produced non-finite values")

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


__all__ = ["NativeA13RelationCanonicalizer"]
