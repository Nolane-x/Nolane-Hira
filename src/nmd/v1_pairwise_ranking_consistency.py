from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_cross_view_decision_consistency import (
    decision_discrimination_diagnostics,
    standardize_full_k_logits,
)


S57_ACTIVE_THRESHOLD=0.25
S57_MARGIN_FLOOR=0.05
S57_ORDINAL_COEFFICIENT=0.05


def _validate_inputs(
    logits_a: Tensor,
    logits_b: Tensor,
    gold: Tensor,
)->tuple[int,int]:
    if logits_a.shape!=logits_b.shape:
        raise ValueError("S57 paired logits shape mismatch")
    if logits_a.ndim!=2 or logits_a.shape[1]<2:
        raise ValueError("S57 logits must be [B,K] with K>=2")
    if not bool(torch.isfinite(logits_a).all()) or not bool(torch.isfinite(logits_b).all()):
        raise ValueError("S57 logits contain non-finite values")
    b,k=logits_a.shape
    if gold.shape!=(b,):
        raise ValueError("S57 gold must be [B]")
    if gold.dtype not in (torch.int64,torch.long):
        raise ValueError("S57 gold must be int64")
    if int(gold.min())<0 or int(gold.max())>=k:
        raise ValueError("S57 gold index out of range")
    return b,k


def detached_anchor_sign(margin: Tensor)->Tensor:
    """Return a sign target with no gradient path into the anchor margin."""
    return torch.sign(margin.detach())


def pairwise_ordinal_consistency(
    logits_a: Tensor,
    logits_b: Tensor,
    gold: Tensor,
    *,
    active_threshold: float = S57_ACTIVE_THRESHOLD,
    margin_floor: float = S57_MARGIN_FLOOR,
)->tuple[Tensor,dict[str,Tensor]]:
    b,k=_validate_inputs(logits_a,logits_b,gold)
    if active_threshold!=S57_ACTIVE_THRESHOLD:
        raise ValueError("S57 active threshold is frozen at 0.25")
    if margin_floor!=S57_MARGIN_FLOOR:
        raise ValueError("S57 margin floor is frozen at 0.05")

    za=standardize_full_k_logits(logits_a)
    zb=standardize_full_k_logits(logits_b)
    ii,jj=torch.triu_indices(k,k,offset=1,device=za.device)

    ma=za[:,ii]-za[:,jj]
    mb=zb[:,ii]-zb[:,jj]

    pair_eligible=torch.maximum(ma.abs(),mb.abs())>=active_threshold
    a_strong=ma.abs()>=active_threshold
    b_strong=mb.abs()>=active_threshold

    g=gold.to(device=za.device)
    gold_i=ii.unsqueeze(0).eq(g.unsqueeze(1))
    gold_j=jj.unsqueeze(0).eq(g.unsqueeze(1))
    gold_involved=gold_i|gold_j
    non_gold=~gold_involved

    a_gold_correct=(gold_i & (ma>0)) | (gold_j & (ma<0))
    b_gold_correct=(gold_i & (mb>0)) | (gold_j & (mb<0))

    raw_a=pair_eligible & a_strong
    raw_b=pair_eligible & b_strong

    valid_a=raw_a & (non_gold | a_gold_correct)
    valid_b=raw_b & (non_gold | b_gold_correct)

    sign_a=detached_anchor_sign(ma)
    sign_b=detached_anchor_sign(mb)

    penalty_a_to_b=F.softplus(margin_floor-sign_a*mb)
    penalty_b_to_a=F.softplus(margin_floor-sign_b*ma)

    zero=torch.zeros((),device=za.device,dtype=za.dtype)
    sum_loss=torch.where(valid_a,penalty_a_to_b,torch.zeros_like(penalty_a_to_b)).sum()
    sum_loss=sum_loss+torch.where(valid_b,penalty_b_to_a,torch.zeros_like(penalty_b_to_a)).sum()

    active_count=valid_a.sum()+valid_b.sum()
    if int(active_count.detach().cpu())==0:
        loss=(ma.sum()+mb.sum())*0.0
    else:
        loss=sum_loss/active_count.to(sum_loss.dtype)

    if not bool(torch.isfinite(loss)):
        raise ValueError("S57 ordinal loss is non-finite")

    raw_count=raw_a.sum()+raw_b.sum()
    filtered_a=raw_a & gold_involved & (~a_gold_correct)
    filtered_b=raw_b & gold_involved & (~b_gold_correct)
    filtered_count=filtered_a.sum()+filtered_b.sum()

    disagree_a=valid_a & ((sign_a*mb)<0)
    disagree_b=valid_b & ((sign_b*ma)<0)
    disagree_count=disagree_a.sum()+disagree_b.sum()

    non_gold_a=valid_a & non_gold
    non_gold_b=valid_b & non_gold
    non_gold_count=non_gold_a.sum()+non_gold_b.sum()

    denom_all=torch.tensor(
        float(2*b*ii.numel()),device=za.device,dtype=torch.float32
    )
    diagnostics={
        "pair_eligible_fraction":pair_eligible.to(torch.float32).mean(),
        "active_directional_anchor_fraction":active_count.to(torch.float32)/denom_all,
        "gold_filtered_direction_fraction":(
            filtered_count.to(torch.float32)/raw_count.clamp_min(1).to(torch.float32)
        ),
        "sign_disagreement_fraction":(
            disagree_count.to(torch.float32)/active_count.clamp_min(1).to(torch.float32)
        ),
        "non_gold_active_direction_fraction":(
            non_gold_count.to(torch.float32)/active_count.clamp_min(1).to(torch.float32)
        ),
        "active_directional_anchor_count":active_count.detach(),
        "raw_directional_anchor_count":raw_count.detach(),
        "gold_filtered_direction_count":filtered_count.detach(),
        "mean_retained_penalty":loss.detach(),
    }
    return loss,diagnostics


def weighted_pairwise_ordinal_auxiliary(
    logits_a: Tensor,
    logits_b: Tensor,
    gold: Tensor,
    *,
    coefficient: float,
)->tuple[Tensor,dict[str,Tensor]]:
    if coefficient not in (0.0,S57_ORDINAL_COEFFICIENT):
        raise ValueError("S57 ordinal coefficient must be 0.0 or 0.05")
    ordinal,diag=pairwise_ordinal_consistency(logits_a,logits_b,gold)
    if coefficient==0.0:
        weighted=(logits_a.sum()+logits_b.sum())*0.0
    else:
        weighted=coefficient*ordinal
    return weighted,{
        "coefficient":torch.tensor(
            float(coefficient),device=logits_a.device,dtype=logits_a.dtype
        ),
        "ordinal_loss":ordinal.detach(),
        **{key:value.detach() for key,value in diag.items()},
    }


__all__=[
    "S57_ACTIVE_THRESHOLD",
    "S57_MARGIN_FLOOR",
    "S57_ORDINAL_COEFFICIENT",
    "detached_anchor_sign",
    "pairwise_ordinal_consistency",
    "weighted_pairwise_ordinal_auxiliary",
    "decision_discrimination_diagnostics",
]
