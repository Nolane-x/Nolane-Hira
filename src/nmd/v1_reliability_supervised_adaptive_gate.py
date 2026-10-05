from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_confidence_adaptive_bounded_hybrid import (
    S61_ALPHA_MAX,
    S61_EPSILON,
    ConfidenceAdaptiveBoundedHybridGate,
    adaptive_gate_features,
)
from .v1_invariance import symmetric_js_divergence


S62_ALPHA_PROBE=0.35
S62_TARGET_TOLERANCE=1e-8


def _centered_rms(logits:Tensor)->Tensor:
    if logits.ndim!=2 or logits.shape[-1]<2:
        raise ValueError("S62 logits must be [B,K] with K>=2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S62 logits contain non-finite values")
    centered=logits-logits.mean(dim=-1,keepdim=True)
    return centered.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)


def fixed_bounded_probe(
    fused_logits:Tensor,
    pairwise_logits:Tensor,
    *,
    alpha_probe:float=S62_ALPHA_PROBE,
)->Tensor:
    if alpha_probe!=S62_ALPHA_PROBE:
        raise ValueError("S62 alpha_probe frozen at 0.35")
    if fused_logits.shape!=pairwise_logits.shape:
        raise ValueError("S62 probe fused/pairwise shape mismatch")
    fused=fused_logits.detach()
    pairwise=pairwise_logits.detach()
    if fused.ndim!=2 or fused.shape[-1]<2:
        raise ValueError("S62 probe logits must be [B,K]")
    if not bool(torch.isfinite(fused).all() and torch.isfinite(pairwise).all()):
        raise ValueError("S62 probe received non-finite logits")

    pair_centered=pairwise-pairwise.mean(dim=-1,keepdim=True)
    pair_z=pair_centered/_centered_rms(pairwise)
    bounded=torch.tanh(pair_z)
    residual=S62_ALPHA_PROBE*_centered_rms(fused)*bounded
    out=fused+residual
    if not bool(torch.isfinite(out).all()):
        raise ValueError("S62 probe output non-finite")
    bound=S62_ALPHA_PROBE*_centered_rms(fused)
    if bool((residual.abs()>bound+1e-7).any()):
        raise RuntimeError("S62 probe bounded residual contract violated")
    return out.detach()


def train_only_reliability_target(
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    gold:Tensor,
    *,
    tolerance:float=S62_TARGET_TOLERANCE,
)->tuple[Tensor,dict[str,Tensor]]:
    if tolerance!=S62_TARGET_TOLERANCE:
        raise ValueError("S62 target tolerance frozen at 1e-8")
    if fused_canonical.shape!=fused_paraphrase.shape:
        raise ValueError("S62 fused view shape mismatch")
    if pairwise_canonical.shape!=pairwise_paraphrase.shape:
        raise ValueError("S62 pairwise view shape mismatch")
    if fused_canonical.shape!=pairwise_canonical.shape:
        raise ValueError("S62 decision surface shape mismatch")
    b,k=fused_canonical.shape
    if gold.shape!=(b,) or gold.dtype not in (torch.int64,torch.long):
        raise ValueError("S62 gold must be int64 [B]")
    if bool(((gold<0)|(gold>=k)).any()):
        raise ValueError("S62 gold index out of range")

    fc=fused_canonical.detach()
    fp=fused_paraphrase.detach()
    pc=pairwise_canonical.detach()
    pp=pairwise_paraphrase.detach()
    g=gold.detach()

    qc=fixed_bounded_probe(fc,pc)
    qp=fixed_bounded_probe(fp,pp)

    ce_base=0.5*(
        F.cross_entropy(fc,g,reduction="none")
        +F.cross_entropy(fp,g,reduction="none")
    )
    ce_probe=0.5*(
        F.cross_entropy(qc,g,reduction="none")
        +F.cross_entropy(qp,g,reduction="none")
    )
    js_base=symmetric_js_divergence(fc,fp,reduction="none")
    js_probe=symmetric_js_divergence(qc,qp,reduction="none")

    correctness_safe=ce_probe.le(ce_base+S62_TARGET_TOLERANCE)
    stability_better=js_probe.lt(js_base-S62_TARGET_TOLERANCE)
    target=(correctness_safe & stability_better).to(fc.dtype).detach()

    if target.requires_grad:
        raise RuntimeError("S62 reliability target retained gradient")
    return target,{
        "ce_base":ce_base.detach(),
        "ce_probe":ce_probe.detach(),
        "js_base":js_base.detach(),
        "js_probe":js_probe.detach(),
        "correctness_safe":correctness_safe.detach(),
        "stability_better":stability_better.detach(),
        "positive_fraction":target.mean().detach(),
    }


def adaptive_gate_pre_sigmoid_logit(
    gate:ConfidenceAdaptiveBoundedHybridGate,
    fused_logits:Tensor,
    pairwise_logits:Tensor,
)->Tensor:
    features=adaptive_gate_features(
        fused_logits.detach(),pairwise_logits.detach()
    )
    logits=F.linear(
        features.detach(),
        gate.w.unsqueeze(0),
        gate.b.reshape(1),
    ).squeeze(-1)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S62 gate logits non-finite")
    return logits


def reliability_gate_loss(
    gate:ConfidenceAdaptiveBoundedHybridGate,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    target,target_diag=train_only_reliability_target(
        fused_canonical,fused_paraphrase,
        pairwise_canonical,pairwise_paraphrase,gold,
    )
    zc=adaptive_gate_pre_sigmoid_logit(
        gate,fused_canonical,pairwise_canonical
    )
    zp=adaptive_gate_pre_sigmoid_logit(
        gate,fused_paraphrase,pairwise_paraphrase
    )
    if zc.shape!=target.shape or zp.shape!=target.shape:
        raise RuntimeError("S62 gate logit/target shape changed")
    loss=0.5*(
        F.binary_cross_entropy_with_logits(zc,target)
        +F.binary_cross_entropy_with_logits(zp,target)
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("S62 reliability BCE non-finite")

    with torch.no_grad():
        ac=gate.alpha(fused_canonical,pairwise_canonical).reshape(-1)
        ap=gate.alpha(fused_paraphrase,pairwise_paraphrase).reshape(-1)
        all_alpha=torch.cat([ac,ap])
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
    }


__all__=[
    "S62_ALPHA_PROBE",
    "S62_TARGET_TOLERANCE",
    "fixed_bounded_probe",
    "train_only_reliability_target",
    "adaptive_gate_pre_sigmoid_logit",
    "reliability_gate_loss",
]
