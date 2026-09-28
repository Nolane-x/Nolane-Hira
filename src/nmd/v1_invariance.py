from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def symmetric_js_divergence(
    logits_a: Tensor,
    logits_b: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Jensen-Shannon divergence between equivalent decision views.

    The function consumes unnormalized K-way logits, has no learned
    parameters, and is symmetric by construction.
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError("S8 consistency logits must have identical shape")
    if logits_a.ndim < 2:
        raise ValueError("S8 consistency logits must include candidate axis")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("unsupported S8 consistency reduction")
    if not bool(torch.isfinite(logits_a).all() and torch.isfinite(logits_b).all()):
        raise ValueError("S8 consistency received non-finite logits")

    log_pa = F.log_softmax(logits_a, dim=-1)
    log_pb = F.log_softmax(logits_b, dim=-1)
    pa = log_pa.exp()
    pb = log_pb.exp()
    mixture = 0.5 * (pa + pb)
    log_m = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()

    kl_a = (pa * (log_pa - log_m)).sum(-1)
    kl_b = (pb * (log_pb - log_m)).sum(-1)
    js = 0.5 * (kl_a + kl_b)

    # Numerical roundoff can produce tiny negative values around zero.
    js = js.clamp_min(0.0)
    if reduction == "none":
        return js
    if reduction == "sum":
        return js.sum()
    return js.mean()


def selected_choice_agreement(logits_a: Tensor, logits_b: Tensor) -> Tensor:
    if logits_a.shape != logits_b.shape:
        raise ValueError("S8 agreement logits must have identical shape")
    return logits_a.argmax(-1).eq(logits_b.argmax(-1)).to(torch.float32).mean()


__all__ = ["selected_choice_agreement", "symmetric_js_divergence"]
