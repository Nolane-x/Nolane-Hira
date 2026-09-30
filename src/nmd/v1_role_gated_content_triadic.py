from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_projection_relearning import TrainableProjectionTriadicScorer


@dataclass(frozen=True)
class RoleGatedContentDiagnostics:
    state_role_entropy: Tensor
    state_role_max_weight: Tensor
    option_role_entropy: Tensor
    option_role_max_weight: Tensor
    mean_role_compatibility: Tensor
    mean_content_score: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "state_role_entropy": float(self.state_role_entropy.detach().cpu()),
            "state_role_max_weight": float(self.state_role_max_weight.detach().cpu()),
            "option_role_entropy": float(self.option_role_entropy.detach().cpu()),
            "option_role_max_weight": float(self.option_role_max_weight.detach().cpu()),
            "mean_role_compatibility": float(
                self.mean_role_compatibility.detach().cpu()
            ),
            "mean_content_score": float(self.mean_content_score.detach().cpu()),
        }


class RoleGatedContentTriadicScorer(TrainableProjectionTriadicScorer):
    """S21 parameter-free role/content factorization over the shared projection."""

    role_temperature = 0.10
    role_weight = 0.50
    content_weight = 0.50

    @property
    def factorization_added_parameter_count(self) -> int:
        return 0

    @staticmethod
    def _normalized_entropy(weights: Tensor, mask: Tensor) -> Tensor:
        eps = torch.finfo(weights.dtype).eps
        count = mask.sum(-1).to(weights.dtype)
        den = torch.where(count > 1, count.log(), torch.ones_like(count))
        entropy = -(weights.clamp_min(eps).log() * weights).sum(-1)
        return entropy / den

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
        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.d_model:
            raise ValueError("S21 state_tokens must be [B,S,D]")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("S21 state_mask mismatch")
        if question_tokens.ndim != 3 or question_tokens.shape[-1] != self.d_model:
            raise ValueError("S21 question_tokens must be [B,Q,D]")
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("S21 question_mask mismatch")
        if option_view_tokens.ndim != 5 or option_view_tokens.shape[-1] != self.d_model:
            raise ValueError("S21 option_view_tokens must be [B,K,V,T,D]")
        if option_view_token_mask.shape != option_view_tokens.shape[:4]:
            raise ValueError("S21 option token mask mismatch")
        if option_view_token_mask.dtype != torch.bool:
            raise ValueError("S21 option token mask must be bool")
        if option_view_mask.shape != option_view_tokens.shape[:3]:
            raise ValueError("S21 option view mask mismatch")
        if option_view_mask.dtype != torch.bool:
            raise ValueError("S21 option view mask must be bool")
        if not (
            state_tokens.shape[0]
            == question_tokens.shape[0]
            == option_view_tokens.shape[0]
        ):
            raise ValueError("S21 batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("S21 requires K>=2")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("S21 requires state content")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("S21 requires question content")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("S21 requires active option views")
        if bool(((option_view_token_mask.sum(-1) < 1) & option_view_mask).any()):
            raise ValueError("S21 active views require content tokens")
        if bool((option_view_token_mask & ~option_view_mask[..., None]).any()):
            raise ValueError("S21 inactive views cannot expose tokens")

    @staticmethod
    def _orthogonal_residual(tokens: Tensor, anchor: Tensor) -> Tensor:
        # anchor shape is tokens shape without the token axis.
        if tokens.ndim == 3:
            coefficient = torch.einsum("bsd,bd->bs", tokens, anchor)
            residual = tokens - coefficient[..., None] * anchor[:, None, :]
        elif tokens.ndim == 5:
            coefficient = torch.einsum("bkvtd,bkvd->bkvt", tokens, anchor)
            residual = tokens - coefficient[..., None] * anchor[:, :, :, None, :]
        else:
            raise ValueError("S21 unsupported residual rank")
        return F.normalize(residual, dim=-1)

    def forward_with_diagnostics(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> tuple[Tensor, RoleGatedContentDiagnostics]:
        self._validate(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        state = self._project(state_tokens)
        question = self._project(question_tokens)
        options = self._project(option_view_tokens)

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

        similarity = torch.einsum(
            "bkvtd,bsd->bkvts",
            option_content,
            state_content,
        )

        option_to_state = similarity.masked_fill(
            ~state_mask[:, None, None, None, :],
            -1e4,
        ).max(dim=-1).values
        option_to_state = option_to_state.masked_fill(
            ~option_view_token_mask,
            0.0,
        )
        option_count = option_view_token_mask.sum(-1).clamp_min(1).to(
            option_to_state.dtype
        )
        option_mean = option_to_state.sum(-1) / option_count

        state_to_option = similarity.masked_fill(
            ~option_view_token_mask[..., None],
            -1e4,
        ).max(dim=-2).values
        state_to_option = state_to_option.masked_fill(
            ~state_mask[:, None, None, :],
            0.0,
        )
        state_count = state_mask.sum(-1).clamp_min(1).to(state_to_option.dtype)
        state_mean = state_to_option.sum(-1) / state_count[:, None, None]

        content_score = 0.5 * (option_mean + state_mean)
        view_score = (
            self.role_weight * role_compatibility
            + self.content_weight * content_score
        )
        view_score = view_score.masked_fill(~option_view_mask, 0.0)
        view_count = option_view_mask.sum(-1).clamp_min(1).to(view_score.dtype)
        logits = view_score.sum(-1) / view_count

        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S21 role-gated content scorer produced non-finite logits")

        state_entropy = self._normalized_entropy(state_role, state_mask)
        option_entropy = self._normalized_entropy(
            option_role,
            option_view_token_mask,
        )
        active = option_view_mask.to(option_entropy.dtype)
        diagnostics = RoleGatedContentDiagnostics(
            state_role_entropy=state_entropy.mean(),
            state_role_max_weight=state_role.max(-1).values.mean(),
            option_role_entropy=(option_entropy * active).sum()
            / active.sum().clamp_min(1.0),
            option_role_max_weight=(option_role.max(-1).values * active).sum()
            / active.sum().clamp_min(1.0),
            mean_role_compatibility=(role_compatibility * active).sum()
            / active.sum().clamp_min(1.0),
            mean_content_score=(content_score * active).sum()
            / active.sum().clamp_min(1.0),
        )
        return logits, diagnostics

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
    ) -> Tensor:
        logits, _ = self.forward_with_diagnostics(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )
        return logits


__all__ = [
    "RoleGatedContentDiagnostics",
    "RoleGatedContentTriadicScorer",
]
