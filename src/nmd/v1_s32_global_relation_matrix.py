from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def global_query_option_relation_contrastive_loss(
    canonical: Tensor,
    paraphrase: Tensor,
    *,
    temperature: float = 0.10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Symmetric global contrastive loss over every query-option relation.

    canonical/paraphrase: [Q,K,D] relation signatures for the same Q semantic
    queries and the same K logical options in two wording views.

    Positive pair:
      (q,k)_canonical <-> (q,k)_paraphrase

    Negatives:
      every other (q',k') relation in the current batch, including
      same-query wrong options and cross-query options.

    The operator is training-only and adds no learned state.
    """
    if canonical.ndim != 3 or paraphrase.shape != canonical.shape:
        raise ValueError("S32 relation signatures must share [Q,K,D]")
    q, k, d = canonical.shape
    if q < 2:
        raise ValueError("S32 all-option contrastive loss requires Q>=2")
    if k < 2:
        raise ValueError("S32 all-option contrastive loss requires K>=2")
    if d < 1:
        raise ValueError("S32 relation signature width must be positive")
    if temperature <= 0:
        raise ValueError("S32 temperature must be positive")

    c = F.normalize(canonical.reshape(q * k, d), dim=-1)
    p = F.normalize(paraphrase.reshape(q * k, d), dim=-1)

    logits_cp = torch.einsum("id,jd->ij", c, p) / float(temperature)
    logits_pc = logits_cp.transpose(0, 1)
    labels = torch.arange(q * k, device=canonical.device)

    c2p = F.cross_entropy(logits_cp, labels)
    p2c = F.cross_entropy(logits_pc, labels)
    total = 0.5 * (c2p + p2c)

    if not bool(
        torch.isfinite(total)
        and torch.isfinite(c2p)
        and torch.isfinite(p2c)
    ):
        raise ValueError("S32 all-option contrastive loss became non-finite")
    return total, c2p, p2c


__all__=["global_query_option_relation_contrastive_loss"]
