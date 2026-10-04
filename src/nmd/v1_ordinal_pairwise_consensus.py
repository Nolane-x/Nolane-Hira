from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OrdinalPairwiseDiagnostics:
    unanimous_pair_fraction: Tensor
    majority_pair_fraction: Tensor
    tied_pair_fraction: Tensor
    private_tiebreak_topset_fraction: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "unanimous_pair_fraction": float(self.unanimous_pair_fraction.detach().cpu()),
            "majority_pair_fraction": float(self.majority_pair_fraction.detach().cpu()),
            "tied_pair_fraction": float(self.tied_pair_fraction.detach().cpu()),
            "private_tiebreak_topset_fraction": float(
                self.private_tiebreak_topset_fraction.detach().cpu()
            ),
        }


class OrdinalPairwiseConsensus(nn.Module):
    """S47 zero-parameter ordinal decision shell.

    Logit magnitudes are discarded after per-expert pairwise comparisons.
    Copeland majority determines the primary ordering. Corrected-private ordinal
    score is used only as an exact lexicographic tie-break with a K-derived base.
    """

    def __init__(self):
        super().__init__()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _validate(primary: Tensor, native: Tensor, corrected: Tensor) -> None:
        if primary.ndim != 2 or native.ndim != 2 or corrected.ndim != 2:
            raise ValueError("S47 experts must be [B,K]")
        if not (primary.shape == native.shape == corrected.shape):
            raise ValueError("S47 experts must share exact [B,K] shape")
        if primary.shape[-1] < 2:
            raise ValueError("S47 requires K >= 2")
        if not (
            bool(torch.isfinite(primary).all())
            and bool(torch.isfinite(native).all())
            and bool(torch.isfinite(corrected).all())
        ):
            raise ValueError("S47 expert logits must be finite")

    @staticmethod
    def _pairwise_rank(logits: Tensor) -> Tensor:
        # [B,K,K], +1 when i>j, -1 when i<j, 0 on exact ties.
        return torch.sign(logits[:, :, None] - logits[:, None, :])

    def components(
        self,
        primary_logits: Tensor,
        native_relation_logits: Tensor,
        corrected_relation_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._validate(primary_logits, native_relation_logits, corrected_relation_logits)
        p = self._pairwise_rank(primary_logits)
        n = self._pairwise_rank(native_relation_logits)
        c = self._pairwise_rank(corrected_relation_logits)

        vote_sum = p + n + c
        majority = torch.sign(vote_sum)
        copeland = majority.sum(dim=-1)
        private = c.sum(dim=-1)
        return copeland, private, vote_sum

    def forward(
        self,
        primary_logits: Tensor,
        native_relation_logits: Tensor,
        corrected_relation_logits: Tensor,
    ) -> tuple[Tensor, OrdinalPairwiseDiagnostics]:
        copeland, private, vote_sum = self.components(
            primary_logits,
            native_relation_logits,
            corrected_relation_logits,
        )
        k = primary_logits.shape[-1]
        base = 2 * k - 1
        fused = copeland * base + private
        if not bool(torch.isfinite(fused).all()):
            raise ValueError("S47 fused ordinal evidence became non-finite")

        # Count unique unordered pairs only.
        upper = torch.triu(
            torch.ones(k, k, dtype=torch.bool, device=fused.device),
            diagonal=1,
        )[None]
        abs_vote = vote_sum.abs()
        pair_count = float(primary_logits.shape[0] * k * (k - 1) / 2)
        unanimous = ((abs_vote == 3) & upper).sum().to(fused.dtype) / pair_count
        majority = ((abs_vote == 1) & upper).sum().to(fused.dtype) / pair_count
        tied = ((abs_vote == 0) & upper).sum().to(fused.dtype) / pair_count

        max_copeland = copeland.max(dim=-1, keepdim=True).values
        top_mask = copeland == max_copeland
        top_count = top_mask.sum(dim=-1)
        private_masked = private.masked_fill(~top_mask, float("-inf"))
        private_best = private_masked.max(dim=-1, keepdim=True).values
        private_winner_count = ((private == private_best) & top_mask).sum(dim=-1)
        changed = (top_count > 1) & (private_winner_count < top_count)

        diagnostics = OrdinalPairwiseDiagnostics(
            unanimous_pair_fraction=unanimous,
            majority_pair_fraction=majority,
            tied_pair_fraction=tied,
            private_tiebreak_topset_fraction=changed.to(fused.dtype).mean(),
        )
        return fused, diagnostics


__all__ = ["OrdinalPairwiseConsensus", "OrdinalPairwiseDiagnostics"]
