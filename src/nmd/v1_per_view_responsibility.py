from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_contextual_reliability_gate import ContextInjectedReliabilityGate
from .v1_invariance import symmetric_js_divergence
from .v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    fixed_bounded_probe,
    train_only_reliability_target,
)


def per_view_responsibility_targets(
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    gold:Tensor,
    *,
    tolerance:float=S62_TARGET_TOLERANCE,
)->tuple[Tensor,Tensor,dict[str,Tensor]]:
    if tolerance!=S62_TARGET_TOLERANCE:
        raise ValueError("S66 target tolerance frozen at 1e-8")
    if fused_canonical.shape!=fused_paraphrase.shape:
        raise ValueError("S66 fused view shape mismatch")
    if pairwise_canonical.shape!=pairwise_paraphrase.shape:
        raise ValueError("S66 pairwise view shape mismatch")
    if fused_canonical.shape!=pairwise_canonical.shape:
        raise ValueError("S66 decision surface shape mismatch")
    b,k=fused_canonical.shape
    if gold.shape!=(b,) or gold.dtype not in (torch.int64,torch.long):
        raise ValueError("S66 gold must be int64 [B]")
    if bool(((gold<0)|(gold>=k)).any()):
        raise ValueError("S66 gold index out of range")

    fc=fused_canonical.detach()
    fp=fused_paraphrase.detach()
    pc=pairwise_canonical.detach()
    pp=pairwise_paraphrase.detach()
    g=gold.detach()

    qc=fixed_bounded_probe(fc,pc)
    qp=fixed_bounded_probe(fp,pp)

    ce_c0=F.cross_entropy(fc,g,reduction="none")
    ce_p0=F.cross_entropy(fp,g,reduction="none")
    ce_c1=F.cross_entropy(qc,g,reduction="none")
    ce_p1=F.cross_entropy(qp,g,reduction="none")

    js0=symmetric_js_divergence(fc,fp,reduction="none")
    js_c1=symmetric_js_divergence(qc,fp,reduction="none")
    js_p1=symmetric_js_divergence(fc,qp,reduction="none")

    c_safe=ce_c1.le(ce_c0+S62_TARGET_TOLERANCE)
    p_safe=ce_p1.le(ce_p0+S62_TARGET_TOLERANCE)
    c_stable=js_c1.lt(js0-S62_TARGET_TOLERANCE)
    p_stable=js_p1.lt(js0-S62_TARGET_TOLERANCE)

    yc=(c_safe & c_stable).to(fc.dtype).detach()
    yp=(p_safe & p_stable).to(fc.dtype).detach()

    if yc.requires_grad or yp.requires_grad:
        raise RuntimeError("S66 per-view targets retained gradient")

    return yc,yp,{
        "ce_canonical_base":ce_c0.detach(),
        "ce_paraphrase_base":ce_p0.detach(),
        "ce_canonical_probe":ce_c1.detach(),
        "ce_paraphrase_probe":ce_p1.detach(),
        "js_base":js0.detach(),
        "js_canonical_only_probe":js_c1.detach(),
        "js_paraphrase_only_probe":js_p1.detach(),
        "canonical_correctness_safe":c_safe.detach(),
        "paraphrase_correctness_safe":p_safe.detach(),
        "canonical_stability_better":c_stable.detach(),
        "paraphrase_stability_better":p_stable.detach(),
        "canonical_positive_fraction":yc.mean().detach(),
        "paraphrase_positive_fraction":yp.mean().detach(),
        "target_disagreement_fraction":yc.ne(yp).to(fc.dtype).mean().detach(),
    }


def per_view_responsibility_loss(
    gate:ContextInjectedReliabilityGate,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    context_canonical:Tensor,
    context_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    yc,yp,target_diag=per_view_responsibility_targets(
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
    if zc.shape!=yc.shape or zp.shape!=yp.shape:
        raise RuntimeError("S66 gate logit/target shape changed")

    loss=0.5*(
        F.binary_cross_entropy_with_logits(zc,yc)
        +F.binary_cross_entropy_with_logits(zp,yp)
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("S66 responsibility BCE non-finite")

    pair_target,pair_diag=train_only_reliability_target(
        fused_canonical,
        fused_paraphrase,
        pairwise_canonical,
        pairwise_paraphrase,
        gold,
    )

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

    return loss,{
        **target_diag,
        "pair_target_positive_fraction":pair_diag["positive_fraction"],
        "pair_target_positive_count":pair_target.sum().detach(),
        "canonical_target_positive_count":yc.sum().detach(),
        "paraphrase_target_positive_count":yp.sum().detach(),
        "target_count":torch.tensor(
            yc.numel(),device=yc.device,dtype=torch.long
        ),
        "mean_alpha":all_alpha.mean().detach(),
        "min_alpha":all_alpha.min().detach(),
        "max_alpha":all_alpha.max().detach(),
        "std_alpha":all_alpha.std(unbiased=False).detach(),
    }


__all__=[
    "per_view_responsibility_targets",
    "per_view_responsibility_loss",
]
