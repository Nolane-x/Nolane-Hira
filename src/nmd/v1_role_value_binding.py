from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RoleValueBindingDiagnostics:
    role_normalized_entropy: Tensor
    state_normalized_entropy: Tensor
    expert_choice_agreement: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "role_normalized_entropy": float(
                self.role_normalized_entropy.detach().cpu()
            ),
            "state_normalized_entropy": float(
                self.state_normalized_entropy.detach().cpu()
            ),
            "expert_choice_agreement": float(
                self.expert_choice_agreement.detach().cpu()
            ),
        }


class RoleValueFactorizedBinding(nn.Module):
    """Parameter-free S11 role/value product-of-experts binding.

    The role expert asks whether an option matches the query-selected role.
    The state expert asks whether the option's semantic content is supported
    by the already-encoded state. The experts remain separate until the
    final K-way product-of-experts fusion, so no single pooled evidence vector
    must simultaneously preserve role and value identity.
    """

    def __init__(
        self,
        *,
        role_temperature: float = 0.10,
        state_temperature: float = 0.10,
    ):
        super().__init__()
        if role_temperature <= 0:
            raise ValueError("S11 role temperature must be positive")
        if state_temperature <= 0:
            raise ValueError("S11 state temperature must be positive")
        self.role_temperature = float(role_temperature)
        self.state_temperature = float(state_temperature)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _project(self, projection: nn.Linear, x: Tensor) -> Tensor:
        if not isinstance(projection, nn.Linear):
            raise TypeError("S11 binding requires a shared nn.Linear projection")
        if projection.bias is not None:
            raise ValueError("S11 binding requires bias-free shared projection")
        if x.shape[-1] != projection.in_features:
            raise ValueError("S11 binding projection input mismatch")
        y = F.normalize(projection(x), dim=-1)
        if not bool(torch.isfinite(y).all()):
            raise ValueError("S11 binding projection produced non-finite values")
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
            raise ValueError("S11 state_tokens must be [B,S,D]")
        if question_tokens.ndim != 3:
            raise ValueError("S11 question_tokens must be [B,Q,D]")
        if option_view_tokens.ndim != 5:
            raise ValueError("S11 option_view_tokens must be [B,K,V,T,D]")
        if (
            state_tokens.shape[-1] != projection.in_features
            or question_tokens.shape[-1] != projection.in_features
            or option_view_tokens.shape[-1] != projection.in_features
        ):
            raise ValueError("S11 binding d_model mismatch")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("S11 state_mask mismatch")
        if (
            question_mask.shape != question_tokens.shape[:2]
            or question_mask.dtype != torch.bool
        ):
            raise ValueError("S11 question_mask mismatch")
        if (
            option_view_token_mask.shape != option_view_tokens.shape[:4]
            or option_view_token_mask.dtype != torch.bool
        ):
            raise ValueError("S11 option token mask mismatch")
        if (
            option_view_mask.shape != option_view_tokens.shape[:3]
            or option_view_mask.dtype != torch.bool
        ):
            raise ValueError("S11 option view mask mismatch")
        if not (
            state_tokens.shape[0]
            == question_tokens.shape[0]
            == option_view_tokens.shape[0]
        ):
            raise ValueError("S11 binding batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("S11 binding requires K>=2")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("S11 binding requires state content")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("S11 binding requires question content")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("S11 binding requires active option views")
        active_tokens = option_view_token_mask.sum(-1)
        if bool(((active_tokens < 1) & option_view_mask).any()):
            raise ValueError("S11 active option views require content")
        if bool((option_view_token_mask & ~option_view_mask[..., None]).any()):
            raise ValueError("S11 inactive views cannot expose token positions")

    @staticmethod
    def _average_views(scores: Tensor, option_view_mask: Tensor) -> Tensor:
        weights = option_view_mask.to(scores.dtype)
        return (scores * weights).sum(-1) / weights.sum(-1).clamp_min(1.0)

    @staticmethod
    def _normalized_entropy(logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        eps = torch.finfo(probs.dtype).eps
        entropy = -(probs.clamp_min(eps).log() * probs).sum(-1)
        k = logits.shape[-1]
        if k <= 1:
            return entropy.new_zeros(())
        denominator = logits.new_tensor(float(k)).log()
        return (entropy / denominator).mean()

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
    ) -> tuple[Tensor, Tensor, Tensor, RoleValueBindingDiagnostics]:
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

        # Role expert: every query token retrieves its strongest supporting
        # option token. Generic query tokens contribute nearly equally across K;
        # role-bearing tokens provide the discriminative signal.
        role_pair = torch.einsum("bqd,bkvtd->bkvqt", question, options)
        role_pair = role_pair.masked_fill(
            ~option_view_token_mask[:, :, :, None, :],
            -1e4,
        )
        role_per_question = role_pair.amax(-1)
        role_per_question = role_per_question.masked_fill(
            ~question_mask[:, None, None, :],
            0.0,
        )
        q_count = question_mask.sum(-1).clamp_min(1).to(role_per_question.dtype)
        role_view = role_per_question.sum(-1) / q_count[:, None, None]
        role_logits = self._average_views(role_view, option_view_mask)

        # State-presence expert: every option token must retrieve support from
        # the state. A same-role distractor can match the field token but its
        # absent value token lowers this expert; an option for another true
        # state fact can score well here but must still pass the role expert.
        state_pair = torch.einsum("bkvtd,bsd->bkvts", options, state)
        state_pair = state_pair.masked_fill(
            ~state_mask[:, None, None, None, :],
            -1e4,
        )
        state_per_option_token = state_pair.amax(-1)
        state_per_option_token = state_per_option_token.masked_fill(
            ~option_view_token_mask,
            0.0,
        )
        option_count = option_view_token_mask.sum(-1).clamp_min(1).to(
            state_per_option_token.dtype
        )
        state_view = state_per_option_token.sum(-1) / option_count
        state_logits = self._average_views(state_view, option_view_mask)

        role_logp = F.log_softmax(
            role_logits / self.role_temperature,
            dim=-1,
        )
        state_logp = F.log_softmax(
            state_logits / self.state_temperature,
            dim=-1,
        )
        binding_logits = role_logp + state_logp

        if not bool(torch.isfinite(binding_logits).all()):
            raise ValueError("S11 binding logits became non-finite")

        diagnostics = RoleValueBindingDiagnostics(
            role_normalized_entropy=self._normalized_entropy(role_logp),
            state_normalized_entropy=self._normalized_entropy(state_logp),
            expert_choice_agreement=(
                role_logits.argmax(-1) == state_logits.argmax(-1)
            ).to(binding_logits.dtype).mean(),
        )
        return binding_logits, role_logits, state_logits, diagnostics


def binding_gold_vs_max_wrong_margin(logits: Tensor, gold: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError("S11 binding logits must be [B,K>=2]")
    if gold.shape != logits.shape[:1] or gold.dtype != torch.long:
        raise ValueError("S11 gold shape/dtype mismatch")
    if gold.numel() and (
        int(gold.min().item()) < 0 or int(gold.max().item()) >= logits.shape[-1]
    ):
        raise ValueError("S11 gold index out of range")
    gold_logits = logits.gather(-1, gold[:, None]).squeeze(-1)
    indices = torch.arange(logits.shape[-1], device=logits.device)
    mask = indices[None, :].eq(gold[:, None])
    strongest_wrong = logits.masked_fill(mask, float("-inf")).max(-1).values
    return gold_logits - strongest_wrong


__all__ = [
    "RoleValueBindingDiagnostics",
    "RoleValueFactorizedBinding",
    "binding_gold_vs_max_wrong_margin",
]
