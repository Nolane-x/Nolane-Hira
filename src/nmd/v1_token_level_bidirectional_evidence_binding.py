from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_explicit_pairwise_decision_head import S59_REPRESENTATION_DIMENSION
from .v1_query_gated_identity_interaction import (
    S69_NATIVE_DIMENSION,
    S69_NORM_EPSILON,
)


S75_NATIVE_DIMENSION=S69_NATIVE_DIMENSION
S75_REPRESENTATION_DIMENSION=S59_REPRESENTATION_DIMENSION
S75_REPRESENTATION_PARAMETER_COUNT=0
S75_TEMPERATURE=0.10
S75_INTERACTION_SCALE=math.sqrt(float(S75_NATIVE_DIMENSION))
S75_NORM_EPSILON=S69_NORM_EPSILON


def _validate(
    identity:Tensor,
    state_tokens:Tensor,
    state_mask:Tensor,
    option_view_tokens:Tensor,
    option_view_token_mask:Tensor,
    option_view_mask:Tensor,
    question_tokens:Tensor,
    question_mask:Tensor,
)->tuple[int,int,int,int,int]:
    if identity.ndim!=3 or identity.shape[-1]!=S75_NATIVE_DIMENSION:
        raise ValueError("S75 identity must be [B,K,256]")
    b,k,d=identity.shape
    if k<2:
        raise ValueError("S75 requires K>=2")
    if state_tokens.ndim!=3 or state_tokens.shape[0]!=b or state_tokens.shape[-1]!=d:
        raise ValueError("S75 state_tokens must be [B,S,256]")
    if state_mask.shape!=state_tokens.shape[:2] or state_mask.dtype!=torch.bool:
        raise ValueError("S75 state_mask mismatch")
    if option_view_tokens.ndim!=5 or option_view_tokens.shape[0]!=b or option_view_tokens.shape[1]!=k or option_view_tokens.shape[-1]!=d:
        raise ValueError("S75 option_view_tokens must be [B,K,V,T,256]")
    _b,_k,v,t,_d=option_view_tokens.shape
    if option_view_token_mask.shape!=(b,k,v,t) or option_view_token_mask.dtype!=torch.bool:
        raise ValueError("S75 option token mask mismatch")
    if option_view_mask.shape!=(b,k,v) or option_view_mask.dtype!=torch.bool:
        raise ValueError("S75 option view mask mismatch")
    if question_tokens.ndim!=3 or question_tokens.shape[0]!=b or question_tokens.shape[-1]!=d:
        raise ValueError("S75 question_tokens must be [B,Q,256]")
    if question_mask.shape!=question_tokens.shape[:2] or question_mask.dtype!=torch.bool:
        raise ValueError("S75 question_mask mismatch")
    if bool((state_mask.sum(-1)<1).any()):
        raise ValueError("S75 requires one active state token")
    if bool((question_mask.sum(-1)<1).any()):
        raise ValueError("S75 requires one active query token")
    option_mask=option_view_token_mask & option_view_mask.unsqueeze(-1)
    if bool((option_mask.reshape(b,k,-1).sum(-1)<1).any()):
        raise ValueError("S75 requires one active option token per option")
    for name,value in (
        ("identity",identity),
        ("state_tokens",state_tokens),
        ("option_view_tokens",option_view_tokens),
        ("question_tokens",question_tokens),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"S75 {name} non-finite")
    return b,k,v,t,d


def token_level_bidirectional_pool(
    query:Tensor,
    state_support:Tensor,
    option_support:Tensor,
    question_mask:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    if query.ndim!=3 or query.shape[-1]!=S75_NATIVE_DIMENSION:
        raise ValueError("S75 query must be [B,Q,256]")
    b,q,d=query.shape
    if state_support.shape!=(b,q):
        raise ValueError("S75 state support mismatch")
    if option_support.ndim!=3 or option_support.shape[0]!=b or option_support.shape[-1]!=q:
        raise ValueError("S75 option support mismatch")
    k=option_support.shape[1]
    if question_mask.shape!=(b,q) or question_mask.dtype!=torch.bool:
        raise ValueError("S75 question mask mismatch")
    if not bool(torch.isfinite(state_support).all()) or not bool(torch.isfinite(option_support).all()):
        raise ValueError("S75 supports non-finite")

    qmask=question_mask[:,None,:]
    joint=(state_support[:,None,:]+option_support)/S75_TEMPERATURE
    joint=joint.masked_fill(~qmask,-1e4)
    attention=torch.softmax(joint,dim=-1)
    attention=attention*qmask.to(attention.dtype)
    attention=attention/attention.sum(dim=-1,keepdim=True).clamp_min(S75_NORM_EPSILON)

    state_gate=torch.tanh(state_support[:,None,:]/S75_TEMPERATURE)
    option_gate=torch.tanh(option_support/S75_TEMPERATURE)
    compatibility=state_gate*option_gate

    signed=attention*compatibility*qmask.to(attention.dtype)
    denom=signed.abs().sum(dim=-1,keepdim=True).clamp_min(S75_NORM_EPSILON)
    weight=signed/denom

    evidence=torch.einsum("bkq,bqd->bkd",weight,query)
    evidence=F.normalize(evidence,dim=-1,eps=S75_NORM_EPSILON)

    if evidence.shape!=(b,k,d):
        raise RuntimeError("S75 evidence shape changed")
    for name,value in (
        ("attention",attention),
        ("compatibility",compatibility),
        ("weight",weight),
        ("evidence",evidence),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"S75 {name} non-finite")
    return evidence.detach(),{
        "joint_attention":attention.detach(),
        "bidirectional_compatibility":compatibility.detach(),
        "signed_weight":weight.detach(),
        "signed_weight_l1":weight.abs().sum(dim=-1).detach(),
    }


def build_token_level_bidirectional_context(
    *,
    identity:Tensor,
    state_tokens:Tensor,
    state_mask:Tensor,
    option_view_tokens:Tensor,
    option_view_token_mask:Tensor,
    option_view_mask:Tensor,
    question_tokens:Tensor,
    question_mask:Tensor,
    return_diagnostics:bool=False,
):
    b,k,v,t,d=_validate(
        identity,state_tokens,state_mask,
        option_view_tokens,option_view_token_mask,option_view_mask,
        question_tokens,question_mask,
    )
    identity_detached=identity.detach()
    state=F.normalize(state_tokens.detach(),dim=-1,eps=S75_NORM_EPSILON)
    option=F.normalize(option_view_tokens.detach(),dim=-1,eps=S75_NORM_EPSILON)
    query=F.normalize(question_tokens.detach(),dim=-1,eps=S75_NORM_EPSILON)

    state_score=torch.einsum("bqd,bsd->bqs",query,state)
    state_score=state_score.masked_fill(~state_mask[:,None,:],-1e4)
    state_support=state_score.max(dim=-1).values

    option_mask=option_view_token_mask & option_view_mask.unsqueeze(-1)
    option_flat=option.reshape(b,k,v*t,d)
    option_mask_flat=option_mask.reshape(b,k,v*t)
    option_score=torch.einsum("bqd,bknd->bkqn",query,option_flat)
    option_score=option_score.masked_fill(~option_mask_flat[:,:,None,:],-1e4)
    option_support=option_score.max(dim=-1).values

    evidence,pool_diag=token_level_bidirectional_pool(
        query,state_support,option_support,question_mask
    )
    bound=evidence+S75_INTERACTION_SCALE*(identity_detached*evidence)
    context=F.normalize(bound,dim=-1,eps=S75_NORM_EPSILON)
    if not bool(torch.isfinite(context).all()):
        raise ValueError("S75 context non-finite")

    if return_diagnostics:
        return context.detach(),{
            **pool_diag,
            "state_support":state_support.detach(),
            "option_support":option_support.detach(),
            "pre_identity_evidence":evidence.detach(),
        }
    return context.detach()


def build_token_level_bidirectional_pairwise_representation(
    *,
    identity:Tensor,
    state_tokens:Tensor,
    state_mask:Tensor,
    option_view_tokens:Tensor,
    option_view_token_mask:Tensor,
    option_view_mask:Tensor,
    question_tokens:Tensor,
    question_mask:Tensor,
    return_diagnostics:bool=False,
):
    context,diag=build_token_level_bidirectional_context(
        identity=identity,
        state_tokens=state_tokens,
        state_mask=state_mask,
        option_view_tokens=option_view_tokens,
        option_view_token_mask=option_view_token_mask,
        option_view_mask=option_view_mask,
        question_tokens=question_tokens,
        question_mask=question_mask,
        return_diagnostics=True,
    )
    rep=torch.cat([identity.detach(),context],dim=-1)
    rep=F.normalize(rep,dim=-1,eps=S75_NORM_EPSILON)
    if rep.shape[-1]!=S75_REPRESENTATION_DIMENSION:
        raise RuntimeError("S75 representation dimension changed")
    if not bool(torch.isfinite(rep).all()):
        raise ValueError("S75 representation non-finite")
    rep=rep.detach()
    if return_diagnostics:
        return rep,diag
    return rep


__all__=[
    "S75_NATIVE_DIMENSION",
    "S75_REPRESENTATION_DIMENSION",
    "S75_REPRESENTATION_PARAMETER_COUNT",
    "S75_TEMPERATURE",
    "S75_INTERACTION_SCALE",
    "S75_NORM_EPSILON",
    "token_level_bidirectional_pool",
    "build_token_level_bidirectional_context",
    "build_token_level_bidirectional_pairwise_representation",
]
