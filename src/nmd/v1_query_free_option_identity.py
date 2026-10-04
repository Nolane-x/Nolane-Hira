from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_private_correction_fork import PrivateCorrectionRepresentationFork


class QueryFreeStateOptionIdentity(nn.Module):
    """S49 zero-parameter query-free state↔option identity geometry.

    Identity is constructed only from adapted state and option token embeddings.
    No question tensor is accepted by the API.
    """

    def __init__(self, *, pair_temperature: float = 0.10, native_dimension: int = 256):
        super().__init__()
        if pair_temperature != 0.10:
            raise ValueError("S49 pair temperature is frozen at 0.10")
        if native_dimension != 256:
            raise ValueError("S49 native dimension is frozen at 256")
        self.pair_temperature = float(pair_temperature)
        self.native_dimension = int(native_dimension)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _masked_mean(tokens: Tensor, mask: Tensor, dim: int) -> Tensor:
        weight = mask.to(tokens.dtype)
        while weight.ndim < tokens.ndim:
            weight = weight.unsqueeze(-1)
        numer = (tokens * weight).sum(dim=dim)
        denom = weight.sum(dim=dim).clamp_min(1.0)
        return numer / denom

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> Tensor:
        if state_tokens.ndim != 3:
            raise ValueError("S49 state_tokens must be [B,S,D]")
        if option_view_tokens.ndim != 5:
            raise ValueError("S49 option_view_tokens must be [B,K,V,T,D]")
        if state_tokens.shape[-1] != self.native_dimension:
            raise ValueError("S49 state dimension changed")
        if option_view_tokens.shape[-1] != self.native_dimension:
            raise ValueError("S49 option dimension changed")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("S49 state mask mismatch")
        if (
            option_view_token_mask.shape != option_view_tokens.shape[:4]
            or option_view_token_mask.dtype != torch.bool
        ):
            raise ValueError("S49 option token mask mismatch")
        if (
            option_view_mask.shape != option_view_tokens.shape[:3]
            or option_view_mask.dtype != torch.bool
        ):
            raise ValueError("S49 option view mask mismatch")
        if state_tokens.shape[0] != option_view_tokens.shape[0]:
            raise ValueError("S49 batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("S49 requires K >= 2")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("S49 requires state content")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("S49 requires active option views")
        if bool(((option_view_token_mask.sum(-1) < 1) & option_view_mask).any()):
            raise ValueError("S49 active option views require content tokens")

        # The private identity path is detached from the native runtime.
        state = F.normalize(state_tokens.detach(), dim=-1)
        options = F.normalize(option_view_tokens.detach(), dim=-1)

        state_center = F.normalize(
            self._masked_mean(state, state_mask, dim=1),
            dim=-1,
        )
        option_center = F.normalize(
            self._masked_mean(options, option_view_token_mask, dim=3),
            dim=-1,
        )

        state_relative = F.normalize(
            state - state_center[:, None, :],
            dim=-1,
        )
        option_relative = F.normalize(
            options - option_center[:, :, :, None, :],
            dim=-1,
        )

        direct = torch.einsum("bsd,bkvtd->bkvst", state, options)
        relative = torch.einsum(
            "bsd,bkvtd->bkvst",
            state_relative,
            option_relative,
        )
        pair_score = 0.5 * (direct + relative)
        pair_mask = (
            state_mask[:, None, None, :, None]
            & option_view_token_mask[:, :, :, None, :]
        )
        pair_score = pair_score.masked_fill(~pair_mask, -1e4)

        b, k, v, s, t = pair_score.shape
        flat_score = pair_score.reshape(b, k, v, s * t)
        flat_mask = pair_mask.reshape(b, k, v, s * t)
        pair_weight = torch.softmax(
            flat_score / self.pair_temperature,
            dim=-1,
        )
        pair_weight = pair_weight * flat_mask.to(pair_weight.dtype)
        pair_weight = pair_weight / pair_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        pair_weight_5d = pair_weight.reshape(b, k, v, s, t)

        matched_state = torch.einsum(
            "bkvst,bsd->bkvd",
            pair_weight_5d,
            state_relative,
        )
        matched_option = torch.einsum(
            "bkvst,bkvtd->bkvd",
            pair_weight_5d,
            option_relative,
        )
        center_direction = F.normalize(
            option_center - state_center[:, None, None, :],
            dim=-1,
        )
        view_identity = F.normalize(
            matched_state + matched_option + center_direction,
            dim=-1,
        )

        active = option_view_mask.to(view_identity.dtype)[..., None]
        identity = (view_identity * active).sum(dim=-2)
        identity = identity / active.sum(dim=-2).clamp_min(1.0)
        identity = F.normalize(identity, dim=-1)

        if not bool(torch.isfinite(identity).all()):
            raise ValueError("S49 identity produced non-finite values")
        return identity


class QueryFreeIdentityPrivateCorrectionFork(PrivateCorrectionRepresentationFork):
    """S49 private correction using query-free option identity plus raw query."""

    def __init__(
        self,
        *,
        train_correction: bool = False,
        pair_temperature: float = 0.10,
    ):
        super().__init__(train_correction=train_correction)
        self.identity = QueryFreeStateOptionIdentity(
            pair_temperature=pair_temperature,
            native_dimension=self.native_dimension,
        )

    @property
    def identity_parameter_count(self) -> int:
        return self.identity.parameter_count

    def identity_signatures(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> Tensor:
        return self.identity(
            state_tokens=state_tokens,
            state_mask=state_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

    def correction_logits_from_state_option(
        self,
        *,
        native_logits: Tensor,
        state_tokens: Tensor,
        state_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        identity = self.identity_signatures(
            state_tokens=state_tokens,
            state_mask=state_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )
        corrected = self.correction_logits(
            native_logits=native_logits,
            signatures=identity,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        return corrected, identity


__all__ = [
    "QueryFreeStateOptionIdentity",
    "QueryFreeIdentityPrivateCorrectionFork",
]
