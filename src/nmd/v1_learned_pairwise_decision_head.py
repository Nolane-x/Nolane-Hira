from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_joint_state_query_option_interaction import (
    JointStateQueryOptionPrivateCorrectionFork,
)


S59_HEAD_RANK=32
S59_HEAD_SEED=80590
S59_HEAD_SCALE=1.0


class MatchedLearnedDecisionHeadFork(JointStateQueryOptionPrivateCorrectionFork):
    """S59 equal-capacity pointwise or antisymmetric-pairwise decision head."""

    def __init__(
        self,
        *,
        mode: str,
        train_correction: bool = False,
        train_head: bool = False,
        head_rank: int = S59_HEAD_RANK,
        head_seed: int = S59_HEAD_SEED,
        head_scale: float = S59_HEAD_SCALE,
    ):
        super().__init__(train_correction=train_correction)
        if mode not in {"pointwise","pairwise"}:
            raise ValueError("S59 mode must be pointwise or pairwise")
        if head_rank!=S59_HEAD_RANK:
            raise ValueError("S59 head rank is frozen at 32")
        if head_seed!=S59_HEAD_SEED:
            raise ValueError("S59 head seed is frozen at 80590")
        if head_scale!=S59_HEAD_SCALE:
            raise ValueError("S59 head scale is frozen at 1.0")

        self.mode=mode
        self.head_rank=int(head_rank)
        self.head_seed=int(head_seed)
        self.head_scale=float(head_scale)

        generator=torch.Generator(device="cpu")
        generator.manual_seed(self.head_seed)
        std=math.sqrt(2.0/self.native_dimension)
        a=torch.randn(
            self.head_rank,self.native_dimension,
            generator=generator,dtype=torch.float32,
        )*std
        self.decision_a=nn.Parameter(a,requires_grad=bool(train_head))
        self.decision_b=nn.Parameter(
            torch.zeros(self.head_rank,self.native_dimension,dtype=torch.float32),
            requires_grad=bool(train_head),
        )

    @property
    def decision_head_parameter_count(self)->int:
        return int(self.decision_a.numel()+self.decision_b.numel())

    @property
    def total_private_parameter_count(self)->int:
        return int(self.correction_parameter_count+self.decision_head_parameter_count)

    def decision_head_parameters(self)->list[nn.Parameter]:
        return [self.decision_a,self.decision_b]

    def private_parameters(self)->list[nn.Parameter]:
        return [*self.correction_parameters(),*self.decision_head_parameters()]

    def decision_head_state_dict(self)->dict[str,Tensor]:
        return {
            "decision.a":self.decision_a.detach().cpu().clone(),
            "decision.b":self.decision_b.detach().cpu().clone(),
        }

    def load_decision_head_state_dict(
        self,state_dict:dict[str,Tensor],*,freeze:bool=True
    )->None:
        expected={"decision.a","decision.b"}
        if set(state_dict)!=expected:
            raise ValueError("S59 decision head checkpoint keys changed")
        shapes={
            "decision.a":(self.head_rank,self.native_dimension),
            "decision.b":(self.head_rank,self.native_dimension),
        }
        targets={"decision.a":self.decision_a,"decision.b":self.decision_b}
        with torch.no_grad():
            for key in sorted(expected):
                value=state_dict[key]
                if tuple(value.shape)!=shapes[key]:
                    raise ValueError(f"S59 decision head shape changed: {key}")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"S59 decision head tensor non-finite: {key}")
                targets[key].copy_(
                    value.to(device=targets[key].device,dtype=targets[key].dtype)
                )
        for p in self.decision_head_parameters():
            p.requires_grad_(not freeze)

    def _validate_inputs(self,identity:Tensor,context:Tensor)->tuple[int,int]:
        if identity.shape!=context.shape:
            raise ValueError("S59 identity/context shape mismatch")
        if identity.ndim!=3 or identity.shape[-1]!=self.native_dimension:
            raise ValueError("S59 identity/context must be [B,K,256]")
        if identity.shape[1]<2:
            raise ValueError("S59 requires K>=2")
        if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(context).all()):
            raise ValueError("S59 identity/context contains non-finite values")
        return int(identity.shape[0]),int(identity.shape[1])

    def pointwise_residual(self,identity:Tensor,context:Tensor)->Tensor:
        self._validate_inputs(identity,context)
        base=identity.detach()
        ctx=context.detach()
        a=self.decision_a.to(dtype=base.dtype)
        b=self.decision_b.to(dtype=base.dtype)
        left=F.linear(base,a,bias=None)
        right=F.linear(ctx,b,bias=None)
        score=(left*right).sum(-1)/math.sqrt(self.head_rank)
        score=self.head_scale*score
        if not bool(torch.isfinite(score).all()):
            raise ValueError("S59 pointwise residual non-finite")
        return score

    def pairwise_matrix(self,identity:Tensor,context:Tensor)->Tensor:
        _batch,k=self._validate_inputs(identity,context)
        base=identity.detach()
        ctx=context.detach()
        a=self.decision_a.to(dtype=base.dtype)
        b=self.decision_b.to(dtype=base.dtype)

        projected=F.linear(base,a,bias=None)
        difference=projected[:,:,None,:]-projected[:,None,:,:]

        pair_context=F.normalize(
            ctx[:,:,None,:]+ctx[:,None,:,:],
            dim=-1,
            eps=1e-12,
        )
        projected_context=F.linear(pair_context,b,bias=None)
        raw=(difference*projected_context).sum(-1)/math.sqrt(self.head_rank)

        pair=0.5*(raw-raw.transpose(-1,-2))
        diagonal=torch.eye(k,dtype=torch.bool,device=pair.device)[None]
        pair=pair.masked_fill(diagonal,0.0)
        pair=self.head_scale*pair
        if not bool(torch.isfinite(pair).all()):
            raise ValueError("S59 pairwise matrix non-finite")
        return pair

    def pairwise_residual(self,identity:Tensor,context:Tensor)->Tensor:
        pair=self.pairwise_matrix(identity,context)
        k=pair.shape[-1]
        residual=pair.sum(-1)/float(k-1)
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("S59 pairwise residual non-finite")
        return residual

    def decision_residual(self,identity:Tensor,context:Tensor)->Tensor:
        if self.mode=="pointwise":
            return self.pointwise_residual(identity,context)
        return self.pairwise_residual(identity,context)

    def correction_logits_from_state_option(
        self,
        *,
        native_logits: Tensor,
        state_tokens: Tensor,
        state_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        return_context: bool = False,
    ):
        (
            base_logits,
            identity,
            context,
            weights,
            state_support,
            option_support,
        )=super().correction_logits_from_state_option(
            native_logits=native_logits,
            state_tokens=state_tokens,
            state_mask=state_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            return_context=True,
        )
        head=self.decision_residual(identity,context)
        logits=base_logits+self.head_scale*0.0+head
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S59 relation logits non-finite")
        if return_context:
            return (
                logits,identity,context,weights,state_support,option_support,head
            )
        return logits,identity


def matched_head_initialization_exact(
    reference:MatchedLearnedDecisionHeadFork,
    treatment:MatchedLearnedDecisionHeadFork,
)->bool:
    if reference.mode!="pointwise" or treatment.mode!="pairwise":
        return False
    rc=reference.correction_state_dict()
    tc=treatment.correction_state_dict()
    rh=reference.decision_head_state_dict()
    th=treatment.decision_head_state_dict()
    return (
        rc.keys()==tc.keys()
        and all(torch.equal(rc[k],tc[k]) for k in rc)
        and rh.keys()==th.keys()
        and all(torch.equal(rh[k],th[k]) for k in rh)
    )


__all__=[
    "S59_HEAD_RANK",
    "S59_HEAD_SEED",
    "S59_HEAD_SCALE",
    "MatchedLearnedDecisionHeadFork",
    "matched_head_initialization_exact",
]
