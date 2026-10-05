from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_contextual_reliability_gate import ContextInjectedReliabilityGate
from .v1_reliability_supervised_adaptive_gate import train_only_reliability_target


def cross_view_soft_and_logit(
    canonical_logit:Tensor,
    paraphrase_logit:Tensor,
)->Tensor:
    if canonical_logit.shape!=paraphrase_logit.shape:
        raise ValueError("S65 paired gate logit shape mismatch")
    if canonical_logit.ndim!=1:
        raise ValueError("S65 paired gate logits must be [B]")
    if not bool(torch.isfinite(canonical_logit).all()):
        raise ValueError("S65 canonical gate logits non-finite")
    if not bool(torch.isfinite(paraphrase_logit).all()):
        raise ValueError("S65 paraphrase gate logits non-finite")

    # Parameter-free exact soft-AND in logit space: algebraically min(zc,zp).
    out=0.5*(
        canonical_logit
        +paraphrase_logit
        -(canonical_logit-paraphrase_logit).abs()
    )
    if not bool(torch.isfinite(out).all()):
        raise ValueError("S65 soft-AND logit non-finite")
    return out


def cross_view_soft_and_reliability_loss(
    gate:ContextInjectedReliabilityGate,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    context_canonical:Tensor,
    context_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    target,target_diag=train_only_reliability_target(
        fused_canonical,
        fused_paraphrase,
        pairwise_canonical,
        pairwise_paraphrase,
        gold,
    )
    zc=gate.pre_sigmoid_logit(
        fused_canonical.detach(),
        pairwise_canonical.detach(),
        context_canonical.detach(),
    )
    zp=gate.pre_sigmoid_logit(
        fused_paraphrase.detach(),
        pairwise_paraphrase.detach(),
        context_paraphrase.detach(),
    )
    if zc.shape!=target.shape or zp.shape!=target.shape:
        raise RuntimeError("S65 gate logit/target shape changed")

    z_and=cross_view_soft_and_logit(zc,zp)
    loss=F.binary_cross_entropy_with_logits(z_and,target)
    if not bool(torch.isfinite(loss)):
        raise ValueError("S65 soft-AND reliability BCE non-finite")

    with torch.no_grad():
        ac=gate.alpha(
            fused_canonical,
            pairwise_canonical,
            context_canonical,
        ).reshape(-1)
        ap=gate.alpha(
            fused_paraphrase,
            pairwise_paraphrase,
            context_paraphrase,
        ).reshape(-1)
        all_alpha=torch.cat([ac,ap])
        lower_is_c=zc.lt(zp)
        lower_is_p=zp.lt(zc)
        ties=zc.eq(zp)

    return loss,{
        **target_diag,
        "target_positive_count":target.sum().detach(),
        "target_count":torch.tensor(
            target.numel(),device=target.device,dtype=torch.long
        ),
        "mean_alpha":all_alpha.mean().detach(),
        "min_alpha":all_alpha.min().detach(),
        "max_alpha":all_alpha.max().detach(),
        "std_alpha":all_alpha.std(unbiased=False).detach(),
        "mean_canonical_logit":zc.detach().mean(),
        "mean_paraphrase_logit":zp.detach().mean(),
        "mean_soft_and_logit":z_and.detach().mean(),
        "canonical_lower_count":lower_is_c.sum().detach(),
        "paraphrase_lower_count":lower_is_p.sum().detach(),
        "tie_count":ties.sum().detach(),
    }


__all__=[
    "cross_view_soft_and_logit",
    "cross_view_soft_and_reliability_loss",
]
