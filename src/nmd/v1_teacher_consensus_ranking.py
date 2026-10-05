from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_cross_view_decision_consistency import (
    decision_discrimination_diagnostics,
    standardize_full_k_logits,
)


S58_TEACHER_ACTIVE_THRESHOLD=0.25
S58_STUDENT_MARGIN_FLOOR=0.05
S58_CONSENSUS_COEFFICIENT=0.05


def _validate(
    student_a: Tensor,
    student_b: Tensor,
    teacher_a: Tensor,
    teacher_b: Tensor,
    gold: Tensor,
)->tuple[int,int]:
    if student_a.shape!=student_b.shape:
        raise ValueError("S58 student paired logits shape mismatch")
    if teacher_a.shape!=teacher_b.shape:
        raise ValueError("S58 teacher paired logits shape mismatch")
    if student_a.shape!=teacher_a.shape:
        raise ValueError("S58 teacher/student logits shape mismatch")
    if student_a.ndim!=2 or student_a.shape[1]<2:
        raise ValueError("S58 logits must be [B,K] with K>=2")
    if not all(bool(torch.isfinite(x).all()) for x in (student_a,student_b,teacher_a,teacher_b)):
        raise ValueError("S58 logits contain non-finite values")
    b,k=student_a.shape
    if gold.shape!=(b,) or gold.dtype not in (torch.int64,torch.long):
        raise ValueError("S58 gold must be int64 [B]")
    if int(gold.min())<0 or int(gold.max())>=k:
        raise ValueError("S58 gold index out of range")
    return b,k


def teacher_consensus_targets(
    teacher_a: Tensor,
    teacher_b: Tensor,
    gold: Tensor,
    *,
    active_threshold: float = S58_TEACHER_ACTIVE_THRESHOLD,
)->tuple[Tensor,Tensor,dict[str,Tensor]]:
    if active_threshold!=S58_TEACHER_ACTIVE_THRESHOLD:
        raise ValueError("S58 teacher threshold is frozen at 0.25")
    if teacher_a.shape!=teacher_b.shape:
        raise ValueError("S58 teacher paired logits shape mismatch")
    if teacher_a.ndim!=2 or teacher_a.shape[1]<2:
        raise ValueError("S58 teacher logits must be [B,K] with K>=2")
    if not bool(torch.isfinite(teacher_a).all()) or not bool(torch.isfinite(teacher_b).all()):
        raise ValueError("S58 teacher logits contain non-finite values")
    b,k=teacher_a.shape
    if gold.shape!=(b,) or gold.dtype not in (torch.int64,torch.long):
        raise ValueError("S58 gold must be int64 [B]")

    # Teacher is always detached. Standardization makes eligibility invariant
    # to shared offsets and positive per-view scaling.
    ta=standardize_full_k_logits(teacher_a.detach())
    tb=standardize_full_k_logits(teacher_b.detach())
    ii,jj=torch.triu_indices(k,k,offset=1,device=ta.device)
    ma=ta[:,ii]-ta[:,jj]
    mb=tb[:,ii]-tb[:,jj]

    strong_a=ma.abs()>=active_threshold
    strong_b=mb.abs()>=active_threshold
    strong_both=strong_a&strong_b
    sign_a=torch.sign(ma)
    sign_b=torch.sign(mb)
    agree=sign_a.eq(sign_b)&sign_a.ne(0)

    g=gold.to(device=ta.device)
    gold_i=ii.unsqueeze(0).eq(g.unsqueeze(1))
    gold_j=jj.unsqueeze(0).eq(g.unsqueeze(1))
    gold_involved=gold_i|gold_j
    non_gold=~gold_involved

    consensus_sign=sign_a.detach()
    gold_correct=(gold_i&(consensus_sign>0)) | (gold_j&(consensus_sign<0))
    raw_consensus=strong_both&agree
    active=raw_consensus&(non_gold|gold_correct)

    weak=~strong_both
    disagreement=strong_both&(~agree)
    wrong_gold=raw_consensus&gold_involved&(~gold_correct)

    denom_pairs=torch.tensor(float(max(1,b*ii.numel())),device=ta.device,dtype=torch.float32)
    raw_count=raw_consensus.sum()
    active_count=active.sum()
    non_gold_active=(active&non_gold).sum()

    diagnostics={
        "teacher_strong_both_fraction":strong_both.to(torch.float32).mean(),
        "teacher_weak_pair_fraction":weak.to(torch.float32).mean(),
        "teacher_sign_disagreement_fraction":disagreement.to(torch.float32).sum()/denom_pairs,
        "teacher_raw_consensus_fraction":raw_consensus.to(torch.float32).mean(),
        "teacher_active_consensus_fraction":active.to(torch.float32).mean(),
        "teacher_wrong_gold_filtered_fraction":(
            wrong_gold.to(torch.float32).sum()/raw_count.clamp_min(1).to(torch.float32)
        ),
        "teacher_non_gold_active_fraction":(
            non_gold_active.to(torch.float32)/active_count.clamp_min(1).to(torch.float32)
        ),
        "teacher_active_consensus_count":active_count.detach(),
        "teacher_wrong_gold_filtered_count":wrong_gold.sum().detach(),
    }
    return active.detach(),consensus_sign.detach(),diagnostics


def teacher_consensus_pairwise_loss(
    student_a: Tensor,
    student_b: Tensor,
    teacher_a: Tensor,
    teacher_b: Tensor,
    gold: Tensor,
    *,
    active_threshold: float = S58_TEACHER_ACTIVE_THRESHOLD,
    margin_floor: float = S58_STUDENT_MARGIN_FLOOR,
)->tuple[Tensor,dict[str,Tensor]]:
    b,k=_validate(student_a,student_b,teacher_a,teacher_b,gold)
    if active_threshold!=S58_TEACHER_ACTIVE_THRESHOLD:
        raise ValueError("S58 teacher threshold is frozen at 0.25")
    if margin_floor!=S58_STUDENT_MARGIN_FLOOR:
        raise ValueError("S58 student margin floor is frozen at 0.05")

    active,target_sign,teacher_diag=teacher_consensus_targets(
        teacher_a,teacher_b,gold,active_threshold=active_threshold
    )

    sa=standardize_full_k_logits(student_a)
    sb=standardize_full_k_logits(student_b)
    ii,jj=torch.triu_indices(k,k,offset=1,device=sa.device)
    ma=sa[:,ii]-sa[:,jj]
    mb=sb[:,ii]-sb[:,jj]

    penalty_a=F.softplus(margin_floor-target_sign*ma)
    penalty_b=F.softplus(margin_floor-target_sign*mb)
    weighted_a=torch.where(active,penalty_a,torch.zeros_like(penalty_a))
    weighted_b=torch.where(active,penalty_b,torch.zeros_like(penalty_b))

    active_count=active.sum()
    if int(active_count.detach().cpu())==0:
        loss=(ma.sum()+mb.sum())*0.0
    else:
        loss=(weighted_a.sum()+weighted_b.sum())/(2*active_count).to(ma.dtype)

    if not bool(torch.isfinite(loss)):
        raise ValueError("S58 consensus loss is non-finite")

    violation_a=active&((target_sign*ma)<margin_floor)
    violation_b=active&((target_sign*mb)<margin_floor)
    denom=(2*active_count).clamp_min(1).to(torch.float32)

    return loss,{
        **teacher_diag,
        "student_violation_fraction":(
            violation_a.to(torch.float32).sum()+violation_b.to(torch.float32).sum()
        )/denom,
        "mean_active_student_penalty":loss.detach(),
    }


def weighted_teacher_consensus_auxiliary(
    student_a: Tensor,
    student_b: Tensor,
    teacher_a: Tensor,
    teacher_b: Tensor,
    gold: Tensor,
    *,
    coefficient: float,
)->tuple[Tensor,dict[str,Tensor]]:
    if coefficient not in (0.0,S58_CONSENSUS_COEFFICIENT):
        raise ValueError("S58 consensus coefficient must be 0.0 or 0.05")

    loss,diag=teacher_consensus_pairwise_loss(
        student_a,student_b,teacher_a,teacher_b,gold
    )
    if coefficient==0.0:
        weighted=(student_a.sum()+student_b.sum())*0.0
    else:
        weighted=coefficient*loss

    return weighted,{
        "coefficient":torch.tensor(
            float(coefficient),device=student_a.device,dtype=student_a.dtype
        ),
        "consensus_pairwise_loss":loss.detach(),
        **{key:value.detach() for key,value in diag.items()},
    }


__all__=[
    "S58_TEACHER_ACTIVE_THRESHOLD",
    "S58_STUDENT_MARGIN_FLOOR",
    "S58_CONSENSUS_COEFFICIENT",
    "teacher_consensus_targets",
    "teacher_consensus_pairwise_loss",
    "weighted_teacher_consensus_auxiliary",
    "decision_discrimination_diagnostics",
]
