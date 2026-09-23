from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MIN_COVERAGE_WEIGHT = 0.5
SALIENCE_THRESHOLD = 0.5
INITIAL_LOGIT_SCALE = 10.0


def candidate_relative_idf(
    token_ids: Tensor,
    token_mask: Tensor,
) -> Tensor:
    """Candidate-relative IDF salience with valid-token mean normalized to one.

    Args:
        token_ids: [B,K,T] integer token IDs.
        token_mask: [B,K,T] bool content-token mask.
    """
    if token_ids.ndim != 3 or token_mask.shape != token_ids.shape:
        raise ValueError("token_ids/token_mask must both be [B,K,T]")
    if token_ids.dtype != torch.long:
        raise ValueError("token_ids must be torch.long")
    if token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be bool")
    if (token_mask.sum(-1) < 1).any():
        raise ValueError("every option requires at least one content token")

    batch, k, _ = token_ids.shape
    weights = torch.zeros_like(token_ids, dtype=torch.float32)

    # Token identity is discrete metadata. The fixed IDF calculation is not
    # part of autograd, while all projected similarities remain differentiable.
    ids_cpu = token_ids.detach().cpu()
    mask_cpu = token_mask.detach().cpu()
    for b in range(batch):
        df: dict[int, int] = {}
        for row in range(k):
            ids = set(ids_cpu[b, row][mask_cpu[b, row]].tolist())
            for token_id in ids:
                key = int(token_id)
                df[key] = df.get(key, 0) + 1
        for row in range(k):
            valid_positions = mask_cpu[b, row].nonzero(
                as_tuple=False
            ).flatten().tolist()
            for pos in valid_positions:
                token_id = int(ids_cpu[b, row, pos])
                weights[b, row, pos] = (
                    math.log((k + 1) / (df[token_id] + 1)) + 1.0
                )

    weights = weights.to(device=token_ids.device)
    for b in range(batch):
        for row in range(k):
            valid = token_mask[b, row]
            mean = weights[b, row][valid].mean()
            if not torch.isfinite(mean) or float(mean) <= 0.0:
                raise ValueError("invalid competitive salience mean")
            weights[b, row][valid] /= mean
    return weights


class CompetitiveCoarseScorer(nn.Module):
    """Production port of the promoted W5h forward competitive binder."""

    def __init__(self, d_model: int = 256, d_rel: int = 128):
        super().__init__()
        self.d_model = int(d_model)
        self.d_rel = int(d_rel)
        self.projection = nn.Linear(self.d_model, self.d_rel, bias=False)
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(INITIAL_LOGIT_SCALE))
        )

    def scale(self) -> Tensor:
        return self.log_scale.clamp(
            math.log(0.1),
            math.log(100.0),
        ).exp()

    def _project(self, x: Tensor) -> Tensor:
        return F.normalize(self.projection(x), dim=-1)

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_tokens: Tensor,
        option_token_ids: Tensor,
        option_mask: Tensor,
    ) -> Tensor:
        """Return full-K competitive coarse logits.

        Shapes:
            state_tokens: [B,S,D]
            state_mask: [B,S]
            question_tokens: [B,Q,D]
            question_mask: [B,Q]
            option_tokens: [B,K,T,D]
            option_token_ids: [B,K,T]
            option_mask: [B,K,T]
        """
        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.d_model:
            raise ValueError("state_tokens must be [B,S,D]")
        if question_tokens.ndim != 3 or question_tokens.shape[-1] != self.d_model:
            raise ValueError("question_tokens must be [B,Q,D]")
        if option_tokens.ndim != 4 or option_tokens.shape[-1] != self.d_model:
            raise ValueError("option_tokens must be [B,K,T,D]")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("state_mask mismatch")
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("question_mask mismatch")
        if option_mask.shape != option_tokens.shape[:3] or option_mask.dtype != torch.bool:
            raise ValueError("option_mask mismatch")
        if option_token_ids.shape != option_mask.shape:
            raise ValueError("option_token_ids mismatch")
        if option_token_ids.dtype != torch.long:
            raise ValueError("option_token_ids must be torch.long")
        if not (
            state_tokens.shape[0]
            == question_tokens.shape[0]
            == option_tokens.shape[0]
        ):
            raise ValueError("competitive coarse batch mismatch")
        if option_tokens.shape[1] < 2:
            raise ValueError("competitive coarse requires at least two options")
        if (state_mask.sum(-1) < 1).any():
            raise ValueError("competitive coarse requires state content tokens")
        if (question_mask.sum(-1) < 1).any():
            raise ValueError("competitive coarse requires question content tokens")
        if (option_mask.sum(-1) < 1).any():
            raise ValueError("competitive coarse requires option content tokens")

        context = torch.cat([state_tokens, question_tokens], dim=1)
        context_mask = torch.cat([state_mask, question_mask], dim=1)
        projected_context = self._project(context)
        projected_options = self._project(option_tokens)

        similarity = torch.einsum(
            "bktd,bcd->bktc",
            projected_options,
            projected_context,
        )
        similarity = similarity.masked_fill(
            ~context_mask[:, None, None, :],
            -1e4,
        )

        salience = candidate_relative_idf(option_token_ids, option_mask)
        salience = salience.to(
            device=similarity.device,
            dtype=similarity.dtype,
        )
        weights = salience * option_mask.to(salience.dtype)
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-8)

        # W5h soft anti-collapse operator: a context token is useful for an
        # option token only to the extent it explains it better than the
        # option's sibling tokens on average.
        common = (
            similarity * weights[..., None]
        ).sum(dim=2) / denom
        adjusted = similarity - common[:, :, None, :]
        adjusted = adjusted.masked_fill(
            ~context_mask[:, None, None, :],
            -1e4,
        )
        coverage = adjusted.max(dim=-1).values

        salient_mask = option_mask & (salience >= SALIENCE_THRESHOLD)
        empty = salient_mask.sum(-1) == 0
        if empty.any():
            salient_mask = salient_mask.clone()
            salient_mask[empty] = option_mask[empty]

        weighted_mean = (
            (coverage * weights).sum(-1)
            / weights.sum(-1).clamp_min(1e-8)
        )
        min_coverage = coverage.masked_fill(
            ~salient_mask,
            1e4,
        ).min(dim=-1).values
        raw = weighted_mean + MIN_COVERAGE_WEIGHT * min_coverage
        logits = raw * self.scale().to(raw.device, raw.dtype)
        if not torch.isfinite(logits).all():
            raise ValueError("competitive coarse produced non-finite logits")
        return logits


def count_competitive_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
