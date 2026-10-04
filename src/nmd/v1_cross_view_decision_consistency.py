from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


S56_STANDARDIZE_EPSILON=1e-6
S56_ORDER_THRESHOLD=0.25
S56_ORDER_MARGIN_FLOOR=0.05
S56_DECISION_COEFFICIENT=0.10
S56_ORDER_COEFFICIENT=0.05


def standardize_full_k_logits(
    logits: Tensor,
    *,
    epsilon: float = S56_STANDARDIZE_EPSILON,
)->Tensor:
    if logits.ndim<2 or logits.shape[-1]<2:
        raise ValueError("S56 logits must end in K>=2")
    if epsilon!=S56_STANDARDIZE_EPSILON:
        raise ValueError("S56 standardization epsilon is frozen at 1e-6")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S56 logits contain non-finite values")
    centered=logits-logits.mean(dim=-1,keepdim=True)
    rms=torch.sqrt(centered.square().mean(dim=-1,keepdim=True)+epsilon)
    standardized=centered/rms
    if not bool(torch.isfinite(standardized).all()):
        raise ValueError("S56 standardized logits are non-finite")
    return standardized


def symmetric_js_from_standardized_logits(
    logits_a: Tensor,
    logits_b: Tensor,
)->Tensor:
    if logits_a.shape!=logits_b.shape:
        raise ValueError("S56 paired logits shape mismatch")
    za=standardize_full_k_logits(logits_a)
    zb=standardize_full_k_logits(logits_b)
    log_pa=F.log_softmax(za,dim=-1)
    log_pb=F.log_softmax(zb,dim=-1)
    pa=log_pa.exp()
    pb=log_pb.exp()
    m=0.5*(pa+pb)
    log_m=torch.log(m.clamp_min(1e-12))
    js=0.5*(
        F.kl_div(log_m,log_pa,log_target=True,reduction="none").sum(-1)
        +F.kl_div(log_m,log_pb,log_target=True,reduction="none").sum(-1)
    )
    # F.kl_div input/target direction is easy to misuse; explicitly compute
    # KL(pa||m)+KL(pb||m) below for contract clarity.
    js_explicit=0.5*(
        (pa*(log_pa-log_m)).sum(-1)
        +(pb*(log_pb-log_m)).sum(-1)
    )
    if not bool(torch.isfinite(js_explicit).all()):
        raise ValueError("S56 JS is non-finite")
    return js_explicit.mean()


def pairwise_ordering_consistency(
    logits_a: Tensor,
    logits_b: Tensor,
    *,
    active_threshold: float = S56_ORDER_THRESHOLD,
    margin_floor: float = S56_ORDER_MARGIN_FLOOR,
)->tuple[Tensor,dict[str,Tensor]]:
    if logits_a.shape!=logits_b.shape:
        raise ValueError("S56 paired logits shape mismatch")
    if active_threshold!=S56_ORDER_THRESHOLD:
        raise ValueError("S56 order threshold is frozen at 0.25")
    if margin_floor!=S56_ORDER_MARGIN_FLOOR:
        raise ValueError("S56 order margin floor is frozen at 0.05")

    za=standardize_full_k_logits(logits_a)
    zb=standardize_full_k_logits(logits_b)
    k=za.shape[-1]
    ii,jj=torch.triu_indices(k,k,offset=1,device=za.device)

    ma=za[...,ii]-za[...,jj]
    mb=zb[...,ii]-zb[...,jj]
    active=torch.maximum(ma.abs(),mb.abs())>=active_threshold
    signed_strength=torch.sign(ma*mb)*torch.minimum(ma.abs(),mb.abs())
    penalty=F.relu(margin_floor-signed_strength)
    weighted=torch.where(active,penalty,torch.zeros_like(penalty))

    active_count=active.sum()
    if int(active_count.detach().cpu())==0:
        loss=(ma.sum()+mb.sum())*0.0
    else:
        loss=weighted.sum()/active_count.to(weighted.dtype)

    if not bool(torch.isfinite(loss)):
        raise ValueError("S56 ordering loss is non-finite")

    diagnostics={
        "active_pair_fraction":active.to(torch.float32).mean(),
        "mean_active_penalty":loss.detach(),
        "sign_disagreement_fraction":(
            (active & ((ma*mb)<0)).to(torch.float32).sum()
            / active_count.clamp_min(1).to(torch.float32)
        ),
    }
    return loss,diagnostics


def weighted_cross_view_decision_auxiliary(
    logits_a: Tensor,
    logits_b: Tensor,
    *,
    decision_coefficient: float,
    ordering_coefficient: float,
)->tuple[Tensor,dict[str,Tensor]]:
    allowed=(
        (0.0,0.0),
        (S56_DECISION_COEFFICIENT,S56_ORDER_COEFFICIENT),
    )
    if (decision_coefficient,ordering_coefficient) not in allowed:
        raise ValueError("S56 auxiliary coefficients are frozen")

    decision=symmetric_js_from_standardized_logits(logits_a,logits_b)
    ordering,order_diag=pairwise_ordering_consistency(logits_a,logits_b)
    weighted=decision_coefficient*decision+ordering_coefficient*ordering
    if decision_coefficient==0.0 and ordering_coefficient==0.0:
        # Preserve a valid autograd graph while guaranteeing exact zero.
        weighted=(logits_a.sum()+logits_b.sum())*0.0

    return weighted,{
        "decision_js":decision.detach(),
        "ordering_loss":ordering.detach(),
        "active_pair_fraction":order_diag["active_pair_fraction"].detach(),
        "sign_disagreement_fraction":order_diag["sign_disagreement_fraction"].detach(),
    }


def decision_discrimination_diagnostics(logits: Tensor)->dict[str,Tensor]:
    if logits.ndim<2 or logits.shape[-1]<2:
        raise ValueError("S56 diagnostics require K>=2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S56 diagnostic logits are non-finite")

    probs=torch.softmax(logits,dim=-1)
    entropy=-(probs*probs.clamp_min(1e-12).log()).sum(-1)
    top2=torch.topk(probs,k=2,dim=-1).values
    top_gap=top2[...,0]-top2[...,1]

    centered=logits-logits.mean(dim=-1,keepdim=True)
    rms=torch.sqrt(centered.square().mean(dim=-1))

    return {
        "mean_entropy":entropy.mean(),
        "mean_top1_top2_probability_gap":top_gap.mean(),
        "mean_unregularized_logit_rms":rms.mean(),
    }


__all__=[
    "S56_STANDARDIZE_EPSILON",
    "S56_ORDER_THRESHOLD",
    "S56_ORDER_MARGIN_FLOOR",
    "S56_DECISION_COEFFICIENT",
    "S56_ORDER_COEFFICIENT",
    "standardize_full_k_logits",
    "symmetric_js_from_standardized_logits",
    "pairwise_ordering_consistency",
    "weighted_cross_view_decision_auxiliary",
    "decision_discrimination_diagnostics",
]
