from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_confidence_adaptive_bounded_hybrid import S61_EPSILON
from .v1_contextual_reliability_gate import ContextInjectedReliabilityGate
from .v1_invariance import symmetric_js_divergence
from .v1_per_view_responsibility import per_view_responsibility_targets
from .v1_reliability_supervised_adaptive_gate import S62_TARGET_TOLERANCE


S67_ALPHA_MAX=0.35
S67_ALPHA_LATTICE=(0.0,0.0875,0.175,0.2625,0.35)
S67_TARGET_LEVELS=(0.0,0.25,0.5,0.75,1.0)


def _centered_rms(logits:Tensor)->Tensor:
    if logits.ndim!=2 or logits.shape[-1]<2:
        raise ValueError("S67 logits must be [B,K] with K>=2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S67 logits non-finite")
    centered=logits-logits.mean(dim=-1,keepdim=True)
    return centered.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)


def bounded_probe_at_alpha(
    fused_logits:Tensor,
    pairwise_logits:Tensor,
    alpha:float,
)->Tensor:
    value=float(alpha)
    if value not in S67_ALPHA_LATTICE:
        raise ValueError("S67 alpha must come from frozen lattice")
    if fused_logits.shape!=pairwise_logits.shape:
        raise ValueError("S67 fused/pairwise shape mismatch")

    fused=fused_logits.detach()
    pairwise=pairwise_logits.detach()
    if value==0.0:
        return fused

    pair_centered=pairwise-pairwise.mean(dim=-1,keepdim=True)
    pair_z=pair_centered/_centered_rms(pairwise)
    bounded=torch.tanh(pair_z)
    residual=value*_centered_rms(fused)*bounded
    out=fused+residual
    bound=value*_centered_rms(fused)
    if bool((residual.abs()>bound+1e-7).any()):
        raise RuntimeError("S67 bounded residual contract violated")
    if not bool(torch.isfinite(out).all()):
        raise ValueError("S67 probe output non-finite")
    return out.detach()



def select_safe_oracle_alpha(
    baseline_ce:Tensor,
    baseline_js:Tensor,
    candidate_ce:Tensor,
    candidate_js:Tensor,
)->tuple[Tensor,Tensor,Tensor]:
    if candidate_ce.ndim!=2 or candidate_js.ndim!=2:
        raise ValueError("S67 candidate tables must be [L-1,B]")
    if candidate_ce.shape!=candidate_js.shape:
        raise ValueError("S67 candidate CE/JS table mismatch")
    if candidate_ce.shape[0]!=len(S67_ALPHA_LATTICE)-1:
        raise ValueError("S67 candidate table must match frozen lattice")
    if baseline_ce.shape!=baseline_js.shape or baseline_ce.ndim!=1:
        raise ValueError("S67 baseline CE/JS must be [B]")
    if candidate_ce.shape[1]!=baseline_ce.shape[0]:
        raise ValueError("S67 candidate/baseline batch mismatch")

    best_alpha=torch.zeros_like(baseline_ce)
    best_js=baseline_js.clone()
    best_ce=baseline_ce.clone()
    for row,alpha in enumerate(S67_ALPHA_LATTICE[1:]):
        ce=candidate_ce[row]
        js=candidate_js[row]
        safe=ce.le(baseline_ce+S62_TARGET_TOLERANCE)
        better=js.lt(best_js-S62_TARGET_TOLERANCE)
        update=safe & better
        best_alpha=torch.where(
            update,
            torch.full_like(best_alpha,float(alpha)),
            best_alpha,
        )
        best_js=torch.where(update,js,best_js)
        best_ce=torch.where(update,ce,best_ce)
    return best_alpha.detach(),best_ce.detach(),best_js.detach()


def _safe_oracle_for_view(
    *,
    own_fused:Tensor,
    other_fused:Tensor,
    own_pairwise:Tensor,
    gold:Tensor,
    canonical_side:bool,
)->tuple[Tensor,dict[str,Tensor]]:
    own=own_fused.detach()
    other=other_fused.detach()
    pair=own_pairwise.detach()
    g=gold.detach()

    ce0=F.cross_entropy(own,g,reduction="none")
    if canonical_side:
        js0=symmetric_js_divergence(own,other,reduction="none")
    else:
        js0=symmetric_js_divergence(other,own,reduction="none")

    b=own.shape[0]
    candidate_ce=[]
    candidate_js=[]
    for alpha in S67_ALPHA_LATTICE[1:]:
        q=bounded_probe_at_alpha(own,pair,alpha)
        candidate_ce.append(F.cross_entropy(q,g,reduction="none"))
        candidate_js.append(
            symmetric_js_divergence(q,other,reduction="none")
            if canonical_side
            else symmetric_js_divergence(other,q,reduction="none")
        )
    best_alpha,best_ce,best_js=select_safe_oracle_alpha(
        ce0,
        js0,
        torch.stack(candidate_ce,dim=0),
        torch.stack(candidate_js,dim=0),
    )

    target=(best_alpha/S67_ALPHA_MAX).detach()
    if target.requires_grad:
        raise RuntimeError("S67 oracle target retained gradient")

    allowed=torch.tensor(S67_TARGET_LEVELS,device=target.device,dtype=target.dtype)
    membership=(target.unsqueeze(-1)-allowed).abs().amin(dim=-1)
    if bool((membership>1e-7).any()):
        raise RuntimeError("S67 target escaped frozen levels")

    nonzero=best_alpha.gt(0)
    if bool((nonzero & best_ce.gt(ce0+S62_TARGET_TOLERANCE)).any()):
        raise RuntimeError("S67 selected unsafe alpha")
    if bool((nonzero & best_js.ge(js0-S62_TARGET_TOLERANCE)).any()):
        raise RuntimeError("S67 selected non-improving alpha")

    return target,{
        "baseline_ce":ce0.detach(),
        "selected_ce":best_ce.detach(),
        "baseline_js":js0.detach(),
        "selected_js":best_js.detach(),
        "selected_alpha":best_alpha.detach(),
        "positive_fraction":nonzero.to(own.dtype).mean().detach(),
        "mean_target":target.mean().detach(),
    }


def safe_oracle_alpha_targets(
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,Tensor,dict[str,Tensor]]:
    if fused_canonical.shape!=fused_paraphrase.shape:
        raise ValueError("S67 fused view shape mismatch")
    if pairwise_canonical.shape!=pairwise_paraphrase.shape:
        raise ValueError("S67 pairwise view shape mismatch")
    if fused_canonical.shape!=pairwise_canonical.shape:
        raise ValueError("S67 decision surface shape mismatch")
    b,k=fused_canonical.shape
    if gold.shape!=(b,) or gold.dtype not in (torch.int64,torch.long):
        raise ValueError("S67 gold must be int64 [B]")
    if bool(((gold<0)|(gold>=k)).any()):
        raise ValueError("S67 gold out of range")

    yc,dc=_safe_oracle_for_view(
        own_fused=fused_canonical,
        other_fused=fused_paraphrase,
        own_pairwise=pairwise_canonical,
        gold=gold,
        canonical_side=True,
    )
    yp,dp=_safe_oracle_for_view(
        own_fused=fused_paraphrase,
        other_fused=fused_canonical,
        own_pairwise=pairwise_paraphrase,
        gold=gold,
        canonical_side=False,
    )

    return yc,yp,{
        "canonical_selected_alpha":dc["selected_alpha"],
        "paraphrase_selected_alpha":dp["selected_alpha"],
        "canonical_baseline_ce":dc["baseline_ce"],
        "paraphrase_baseline_ce":dp["baseline_ce"],
        "canonical_selected_ce":dc["selected_ce"],
        "paraphrase_selected_ce":dp["selected_ce"],
        "js_base":dc["baseline_js"],
        "canonical_selected_js":dc["selected_js"],
        "paraphrase_selected_js":dp["selected_js"],
        "canonical_positive_fraction":dc["positive_fraction"],
        "paraphrase_positive_fraction":dp["positive_fraction"],
        "canonical_mean_target":dc["mean_target"],
        "paraphrase_mean_target":dp["mean_target"],
        "target_disagreement_fraction":yc.ne(yp).to(yc.dtype).mean().detach(),
    }


def safe_oracle_alpha_loss(
    gate:ContextInjectedReliabilityGate,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    context_canonical:Tensor,
    context_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    yc,yp,target_diag=safe_oracle_alpha_targets(
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
        raise RuntimeError("S67 gate logit/target shape changed")

    loss=0.5*(
        F.binary_cross_entropy_with_logits(zc,yc)
        +F.binary_cross_entropy_with_logits(zp,yp)
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("S67 oracle-alpha BCE non-finite")

    binary_c,binary_p,_=per_view_responsibility_targets(
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
        "binary_canonical_positive_fraction":binary_c.mean().detach(),
        "binary_paraphrase_positive_fraction":binary_p.mean().detach(),
        "oracle_vs_binary_canonical_difference_fraction":
            yc.ne(binary_c).to(yc.dtype).mean().detach(),
        "oracle_vs_binary_paraphrase_difference_fraction":
            yp.ne(binary_p).to(yp.dtype).mean().detach(),
        "mean_alpha":all_alpha.mean().detach(),
        "min_alpha":all_alpha.min().detach(),
        "max_alpha":all_alpha.max().detach(),
        "std_alpha":all_alpha.std(unbiased=False).detach(),
    }


__all__=[
    "S67_ALPHA_MAX",
    "S67_ALPHA_LATTICE",
    "S67_TARGET_LEVELS",
    "bounded_probe_at_alpha",
    "select_safe_oracle_alpha",
    "safe_oracle_alpha_targets",
    "safe_oracle_alpha_loss",
]
