from __future__ import annotations

import torch
from torch import Tensor


S70_SOFTMAX_TEMPERATURE=1.0
S70_AGGREGATION_PARAMETER_COUNT=0


def _validate_inputs(pair_logits:Tensor,fused_logits:Tensor)->tuple[int,int]:
    if pair_logits.ndim!=3 or pair_logits.shape[1]!=pair_logits.shape[2]:
        raise ValueError("S70 pair logits must be [B,K,K]")
    b,k,_=pair_logits.shape
    if k<2:
        raise ValueError("S70 requires K>=2")
    if fused_logits.shape!=(b,k):
        raise ValueError("S70 fused logits must be [B,K]")
    if not bool(torch.isfinite(pair_logits).all()):
        raise ValueError("S70 pair logits non-finite")
    if not bool(torch.isfinite(fused_logits).all()):
        raise ValueError("S70 fused logits non-finite")
    return b,k


def uniform_pairwise_aggregate(
    pair_logits:Tensor,
    fused_logits:Tensor,
)->Tensor:
    """Exact current S59 row-mean aggregation on valid antisymmetric matrices.

    fused_logits is accepted only to keep the reference/treatment call surface
    matched. Both inputs are detached at the composition boundary.
    """
    _,k=_validate_inputs(pair_logits,fused_logits)
    pair=pair_logits.detach()
    eye=torch.eye(k,dtype=torch.bool,device=pair.device).unsqueeze(0)
    masked=pair.masked_fill(eye,0.0)
    score=masked.sum(dim=-1)/float(k-1)
    if not bool(torch.isfinite(score).all()):
        raise ValueError("S70 uniform aggregate non-finite")
    return score.detach()


def fused_anchored_pairwise_weights(fused_logits:Tensor)->Tensor:
    if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
        raise ValueError("S70 fused logits must be [B,K]")
    if not bool(torch.isfinite(fused_logits).all()):
        raise ValueError("S70 fused logits non-finite")

    fused=fused_logits.detach()
    if S70_SOFTMAX_TEMPERATURE!=1.0:
        raise RuntimeError("S70 temperature contract changed")
    prior=torch.softmax(fused,dim=-1)

    b,k=prior.shape
    weights=prior[:,None,:].expand(b,k,k).clone()
    eye=torch.eye(k,dtype=torch.bool,device=prior.device).unsqueeze(0)
    weights=weights.masked_fill(eye,0.0)
    denom=weights.sum(dim=-1,keepdim=True)
    if bool((denom<=0).any()):
        raise RuntimeError("S70 opponent normalization collapsed")
    weights=weights/denom

    if bool((weights<0).any()) or not bool(torch.isfinite(weights).all()):
        raise RuntimeError("S70 opponent weights invalid")
    row_sum=weights.sum(dim=-1)
    if float((row_sum-1.0).abs().max())>1e-6:
        raise RuntimeError("S70 opponent weights do not sum to one")
    if bool(weights.masked_select(eye.expand_as(weights)).ne(0).any()):
        raise RuntimeError("S70 diagonal opponent weight nonzero")
    return weights.detach()


def fused_anchored_pairwise_aggregate(
    pair_logits:Tensor,
    fused_logits:Tensor,
)->Tensor:
    _validate_inputs(pair_logits,fused_logits)
    pair=pair_logits.detach()
    weights=fused_anchored_pairwise_weights(fused_logits.detach())
    score=(pair*weights).sum(dim=-1)
    if not bool(torch.isfinite(score).all()):
        raise ValueError("S70 fused-anchored aggregate non-finite")
    return score.detach()


__all__=[
    "S70_SOFTMAX_TEMPERATURE",
    "S70_AGGREGATION_PARAMETER_COUNT",
    "uniform_pairwise_aggregate",
    "fused_anchored_pairwise_weights",
    "fused_anchored_pairwise_aggregate",
]
