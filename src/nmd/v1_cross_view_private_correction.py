from __future__ import annotations

import torch
from torch import Tensor


def symmetric_cross_view_js(
    canonical_logits: Tensor,
    paraphrase_logits: Tensor,
    *,
    epsilon: float = 1e-12,
) -> Tensor:
    """Symmetric JS divergence for paired corrected relation distributions.

    Inputs are paired [B,K] logits for semantically equivalent canonical and
    paraphrase views. No temperature or learned parameter is permitted.
    """
    if canonical_logits.ndim != 2 or paraphrase_logits.ndim != 2:
        raise ValueError("S45 corrected logits must be [B,K]")
    if canonical_logits.shape != paraphrase_logits.shape:
        raise ValueError("S45 canonical/paraphrase shapes must match")
    if canonical_logits.shape[-1] < 2:
        raise ValueError("S45 requires K >= 2")
    if epsilon != 1e-12:
        raise ValueError("S45 JS epsilon is frozen at 1e-12")

    p = torch.softmax(canonical_logits, dim=-1)
    q = torch.softmax(paraphrase_logits, dim=-1)
    m = 0.5 * (p + q)

    p_safe = p.clamp_min(epsilon)
    q_safe = q.clamp_min(epsilon)
    m_safe = m.clamp_min(epsilon)

    kl_pm = (p * (p_safe.log() - m_safe.log())).sum(dim=-1)
    kl_qm = (q * (q_safe.log() - m_safe.log())).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    value = js.mean()

    if not bool(torch.isfinite(value)):
        raise ValueError("S45 JS produced non-finite value")
    return value


__all__ = ["symmetric_cross_view_js"]
