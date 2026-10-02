from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def global_cross_case_relation_contrastive_loss(
    canonical: Tensor,
    paraphrase: Tensor,
    gold: Tensor,
    *,
    temperature: float = 0.10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Symmetric in-batch contrastive loss over query-conditioned gold signatures.

    canonical/paraphrase: [Q,K,D] relation signatures for the same Q semantic
    queries in two wording views.
    gold: [Q] logical gold option index for each semantic query.

    The positive for query i is its gold signature in the other wording view.
    Every other semantic query in the batch is a negative.
    """

    if canonical.ndim != 3 or paraphrase.shape != canonical.shape:
        raise ValueError("S31 relation signatures must share [Q,K,D]")
    q, k, _d = canonical.shape
    if q < 2:
        raise ValueError("S31 global contrastive loss requires Q>=2")
    if k < 2:
        raise ValueError("S31 global contrastive loss requires K>=2")
    if gold.ndim != 1 or gold.shape[0] != q:
        raise ValueError("S31 gold must be [Q]")
    if gold.dtype != torch.long:
        raise ValueError("S31 gold must be torch.long")
    if temperature <= 0:
        raise ValueError("S31 temperature must be positive")
    if bool(((gold < 0) | (gold >= k)).any()):
        raise ValueError("S31 gold index out of range")

    row = torch.arange(q, device=canonical.device)
    c = F.normalize(canonical[row, gold], dim=-1)
    p = F.normalize(paraphrase[row, gold], dim=-1)

    logits_cp = torch.einsum("id,jd->ij", c, p) / float(temperature)
    logits_pc = logits_cp.transpose(0, 1)
    labels = torch.arange(q, device=canonical.device)

    c2p = F.cross_entropy(logits_cp, labels)
    p2c = F.cross_entropy(logits_pc, labels)
    total = 0.5 * (c2p + p2c)

    if not bool(
        torch.isfinite(total)
        and torch.isfinite(c2p)
        and torch.isfinite(p2c)
    ):
        raise ValueError("S31 global contrastive loss became non-finite")
    return total, c2p, p2c


__all__=["global_cross_case_relation_contrastive_loss"]
