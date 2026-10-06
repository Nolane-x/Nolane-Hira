from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_confidence_adaptive_bounded_hybrid import S61_ALPHA_MAX, S61_EPSILON
from .v1_invariance import symmetric_js_divergence
from .v1_multistat_pairwise_row_composer import pairwise_row_mean


S73_FEATURE_DIM=8
S73_PARAMETER_COUNT=9
S73_ACCEPT_THRESHOLD=0.5
S73_TARGET_TOLERANCE=1e-8


def _centered_rms(x:Tensor)->Tensor:
    if x.ndim!=2 or x.shape[-1]<2:
        raise ValueError("S73 logits must be [B,K], K>=2")
    centered=x-x.mean(dim=-1,keepdim=True)
    return centered.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)


def _normalized_top2_margin(x:Tensor)->Tensor:
    top2=torch.topk(x,k=2,dim=-1).values
    margin=top2[...,0]-top2[...,1]
    return margin/_centered_rms(x).squeeze(-1)


def _normalized_entropy(logits:Tensor)->Tensor:
    probs=torch.softmax(logits,dim=-1)
    entropy=-(probs*probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)
    return entropy/math.log(float(logits.shape[-1]))


def safety_features(
    fused_logits:Tensor,
    candidate_logits:Tensor,
    pair_matrix:Tensor,
    direction:Tensor,
    alpha:Tensor,
)->Tensor:
    if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
        raise ValueError("S73 fused logits must be [B,K], K>=2")
    if candidate_logits.shape!=fused_logits.shape:
        raise ValueError("S73 candidate shape mismatch")
    b,k=fused_logits.shape
    if pair_matrix.shape!=(b,k,k):
        raise ValueError("S73 pair matrix shape mismatch")
    if direction.shape!=fused_logits.shape:
        raise ValueError("S73 direction shape mismatch")
    if alpha.shape not in {(b,),(b,1)}:
        raise ValueError("S73 alpha shape mismatch")

    fused=fused_logits.detach()
    candidate=candidate_logits.detach()
    pair=pair_matrix.detach()
    d=direction.detach()
    a=alpha.detach().reshape(b)

    if not bool(
        torch.isfinite(fused).all()
        and torch.isfinite(candidate).all()
        and torch.isfinite(pair).all()
        and torch.isfinite(d).all()
        and torch.isfinite(a).all()
    ):
        raise ValueError("S73 feature source non-finite")

    fused_margin=_normalized_top2_margin(fused)
    entropy=_normalized_entropy(fused)
    alpha_feature=(a/float(S61_ALPHA_MAX)).clamp(0.0,1.0)

    top_idx=fused.argmax(dim=-1,keepdim=True)
    top_change=(candidate-fused).gather(-1,top_idx).abs().squeeze(-1)
    top_change=top_change/_centered_rms(fused).squeeze(-1)

    conflict=candidate.argmax(dim=-1).ne(fused.argmax(dim=-1)).to(fused.dtype)

    row_mean=pairwise_row_mean(pair)
    row_margin=_normalized_top2_margin(row_mean)

    direction_max=d.abs().amax(dim=-1)

    centered_f=fused-fused.mean(dim=-1,keepdim=True)
    centered_d=d-d.mean(dim=-1,keepdim=True)
    numerator=(centered_f*centered_d).sum(dim=-1)
    denominator=(
        centered_f.square().sum(dim=-1).sqrt()
        *centered_d.square().sum(dim=-1).sqrt()
    ).clamp_min(S61_EPSILON)
    cosine=(numerator/denominator).clamp(-1.0,1.0)

    features=torch.stack(
        [
            fused_margin,
            entropy,
            alpha_feature,
            top_change,
            conflict,
            row_margin,
            direction_max,
            cosine,
        ],
        dim=-1,
    )
    if features.shape!=(b,S73_FEATURE_DIM):
        raise RuntimeError("S73 feature dimension changed")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("S73 features non-finite")
    return features.detach()


def counterfactual_safety_targets(
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    candidate_canonical:Tensor,
    candidate_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,Tensor,dict[str,Tensor]]:
    if fused_canonical.shape!=fused_paraphrase.shape:
        raise ValueError("S73 fused view shape mismatch")
    if candidate_canonical.shape!=fused_canonical.shape:
        raise ValueError("S73 canonical candidate shape mismatch")
    if candidate_paraphrase.shape!=fused_paraphrase.shape:
        raise ValueError("S73 paraphrase candidate shape mismatch")
    if gold.ndim!=1 or gold.shape[0]!=fused_canonical.shape[0]:
        raise ValueError("S73 gold shape mismatch")

    fc=fused_canonical.detach()
    fp=fused_paraphrase.detach()
    cc=candidate_canonical.detach()
    cp=candidate_paraphrase.detach()
    g=gold.detach()

    base_ce_c=F.cross_entropy(fc,g,reduction="none")
    base_ce_p=F.cross_entropy(fp,g,reduction="none")
    cand_ce_c=F.cross_entropy(cc,g,reduction="none")
    cand_ce_p=F.cross_entropy(cp,g,reduction="none")

    base_js=symmetric_js_divergence(fc,fp,reduction="none")
    cand_js_c=symmetric_js_divergence(cc,fp,reduction="none")
    cand_js_p=symmetric_js_divergence(fc,cp,reduction="none")

    safe_c=(
        (cand_ce_c<=base_ce_c+S73_TARGET_TOLERANCE)
        &(cand_js_c<=base_js+S73_TARGET_TOLERANCE)
    )
    safe_p=(
        (cand_ce_p<=base_ce_p+S73_TARGET_TOLERANCE)
        &(cand_js_p<=base_js+S73_TARGET_TOLERANCE)
    )

    yc=safe_c.to(fc.dtype).detach()
    yp=safe_p.to(fp.dtype).detach()
    return yc,yp,{
        "canonical_safe_count":yc.sum().detach(),
        "paraphrase_safe_count":yp.sum().detach(),
        "canonical_ce_nonworse_count":(
            cand_ce_c<=base_ce_c+S73_TARGET_TOLERANCE
        ).to(fc.dtype).sum().detach(),
        "paraphrase_ce_nonworse_count":(
            cand_ce_p<=base_ce_p+S73_TARGET_TOLERANCE
        ).to(fp.dtype).sum().detach(),
        "canonical_js_nonworse_count":(
            cand_js_c<=base_js+S73_TARGET_TOLERANCE
        ).to(fc.dtype).sum().detach(),
        "paraphrase_js_nonworse_count":(
            cand_js_p<=base_js+S73_TARGET_TOLERANCE
        ).to(fp.dtype).sum().detach(),
    }


class CounterfactualSafetyVeto(nn.Module):
    """Nine-parameter TRAIN-only safety predictor for the frozen S72 candidate."""

    def __init__(self,*,trainable:bool=True):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(S73_FEATURE_DIM,dtype=torch.float32))
        self.bias=nn.Parameter(torch.zeros((),dtype=torch.float32))
        for p in self.parameters():
            p.requires_grad_(bool(trainable))
        if self.parameter_count!=S73_PARAMETER_COUNT:
            raise RuntimeError("S73 parameter count changed")

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def logit(self,features:Tensor)->Tensor:
        if features.ndim!=2 or features.shape[-1]!=S73_FEATURE_DIM:
            raise ValueError("S73 features must be [B,8]")
        x=features.detach()
        out=F.linear(x,self.weight.unsqueeze(0),self.bias.reshape(1)).squeeze(-1)
        if not bool(torch.isfinite(out).all()):
            raise ValueError("S73 predictor logit non-finite")
        return out

    def probability(self,features:Tensor)->Tensor:
        return torch.sigmoid(self.logit(features))

    def accept(self,features:Tensor)->Tensor:
        return self.probability(features)>=S73_ACCEPT_THRESHOLD

    def apply(
        self,
        fused_logits:Tensor,
        candidate_logits:Tensor,
        features:Tensor,
        *,
        force_accept:bool|None=None,
        return_diagnostics:bool=False,
    ):
        if fused_logits.shape!=candidate_logits.shape:
            raise ValueError("S73 fused/candidate shape mismatch")
        if features.shape[0]!=fused_logits.shape[0]:
            raise ValueError("S73 feature batch mismatch")

        fused=fused_logits.detach()
        candidate=candidate_logits.detach()
        if force_accept is None:
            prob=self.probability(features)
            accept=prob>=S73_ACCEPT_THRESHOLD
        else:
            accept=torch.full(
                (fused.shape[0],),
                bool(force_accept),
                device=fused.device,
                dtype=torch.bool,
            )
            prob=accept.to(fused.dtype)

        out=torch.where(accept.unsqueeze(-1),candidate,fused)
        if not bool(torch.isfinite(out).all()):
            raise ValueError("S73 veto output non-finite")

        if return_diagnostics:
            return out,{
                "accept":accept.detach(),
                "accept_probability":prob.detach(),
                "accept_rate":accept.to(fused.dtype).mean().detach(),
            }
        return out

    def state_dict_exact(self)->dict[str,Tensor]:
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()}

    def load_state_dict_exact(self,state:dict[str,Tensor],*,freeze:bool=False)->None:
        self.load_state_dict(state,strict=True)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)


def counterfactual_safety_loss(
    predictor:CounterfactualSafetyVeto,
    features_canonical:Tensor,
    features_paraphrase:Tensor,
    target_canonical:Tensor,
    target_paraphrase:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    zc=predictor.logit(features_canonical)
    zp=predictor.logit(features_paraphrase)
    if zc.shape!=target_canonical.shape or zp.shape!=target_paraphrase.shape:
        raise RuntimeError("S73 predictor target shape changed")

    loss=0.5*(
        F.binary_cross_entropy_with_logits(zc,target_canonical.detach())
        +F.binary_cross_entropy_with_logits(zp,target_paraphrase.detach())
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("S73 predictor loss non-finite")

    with torch.no_grad():
        pc=torch.sigmoid(zc)
        pp=torch.sigmoid(zp)
        ac=pc>=S73_ACCEPT_THRESHOLD
        ap=pp>=S73_ACCEPT_THRESHOLD

    return loss,{
        "canonical_predicted_accept_count":ac.to(zc.dtype).sum().detach(),
        "paraphrase_predicted_accept_count":ap.to(zp.dtype).sum().detach(),
        "canonical_mean_accept_probability":pc.mean().detach(),
        "paraphrase_mean_accept_probability":pp.mean().detach(),
    }


__all__=[
    "S73_FEATURE_DIM",
    "S73_PARAMETER_COUNT",
    "S73_ACCEPT_THRESHOLD",
    "S73_TARGET_TOLERANCE",
    "safety_features",
    "counterfactual_safety_targets",
    "CounterfactualSafetyVeto",
    "counterfactual_safety_loss",
]
