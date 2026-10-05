from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_explicit_pairwise_decision_head import (
    S59_HEAD_SEED,
    S59_HIDDEN_DIMENSION,
    S59_PAIRWISE_PARAMETER_COUNT,
    S59_REPRESENTATION_DIMENSION,
    gold_pairwise_loss,
)


S68_CONTEXT_DIMENSION=8
S68_CONTEXT_PROJECTION_SEED=68_068
S68_MODULATION_SCALE=0.5
S68_MODULATION_PARAMETER_COUNT=S59_HIDDEN_DIMENSION*S68_CONTEXT_DIMENSION
S68_PAIRWISE_PARAMETER_COUNT=S59_PAIRWISE_PARAMETER_COUNT+S68_MODULATION_PARAMETER_COUNT


def build_fixed_context_projection()->Tensor:
    g=torch.Generator(device="cpu").manual_seed(S68_CONTEXT_PROJECTION_SEED)
    p=torch.randn(
        S68_CONTEXT_DIMENSION,
        S59_REPRESENTATION_DIMENSION,
        generator=g,
        dtype=torch.float32,
    )
    p=F.normalize(p,dim=-1)
    if p.shape!=(S68_CONTEXT_DIMENSION,S59_REPRESENTATION_DIMENSION):
        raise RuntimeError("S68 context projection shape changed")
    if not bool(torch.isfinite(p).all()):
        raise ValueError("S68 context projection non-finite")
    return p


def pairwise_set_context_feature(
    representation:Tensor,
    projection:Tensor,
    *,
    use_joint_context:bool,
)->Tensor:
    if representation.ndim!=3:
        raise ValueError("S68 representation must be [B,K,512]")
    if representation.shape[1]<2 or representation.shape[-1]!=S59_REPRESENTATION_DIMENSION:
        raise ValueError("S68 representation shape changed")
    if projection.shape!=(S68_CONTEXT_DIMENSION,S59_REPRESENTATION_DIMENSION):
        raise ValueError("S68 context projection shape changed")
    if not bool(torch.isfinite(representation).all()):
        raise ValueError("S68 representation contains non-finite values")

    rep=representation.detach()
    if use_joint_context:
        source=rep
    else:
        source=torch.cat(
            [rep[...,:256],torch.zeros_like(rep[...,256:])],
            dim=-1,
        )

    mean=source.mean(dim=1)
    mean=F.normalize(mean,dim=-1)
    z=torch.tanh(F.linear(mean,projection.detach(),bias=None))
    if z.shape[-1]!=S68_CONTEXT_DIMENSION:
        raise RuntimeError("S68 context feature dimension changed")
    if not bool(torch.isfinite(z).all()):
        raise ValueError("S68 context feature non-finite")
    return z.detach()


class ContextModulatedPairwiseHead(nn.Module):
    """S68 parameter-matched context-modulated antisymmetric comparator.

    Reference and treatment have the same trainable tensors. The controlled
    variable is whether the set context includes the joint state/query/option
    half of the detached S59 representation.
    """

    def __init__(
        self,
        *,
        use_joint_context:bool,
        trainable:bool=True,
    ):
        super().__init__()
        self.use_joint_context=bool(use_joint_context)

        self.A=nn.Parameter(
            torch.empty(
                S59_HIDDEN_DIMENSION,
                S59_REPRESENTATION_DIMENSION,
                dtype=torch.float32,
            )
        )
        self.u=nn.Parameter(torch.empty(S59_HIDDEN_DIMENSION,dtype=torch.float32))
        self.G=nn.Parameter(
            torch.zeros(
                S59_HIDDEN_DIMENSION,
                S68_CONTEXT_DIMENSION,
                dtype=torch.float32,
            )
        )

        g=torch.Generator(device="cpu").manual_seed(S59_HEAD_SEED)
        with torch.no_grad():
            self.A.copy_(
                torch.randn(
                    S59_HIDDEN_DIMENSION,
                    S59_REPRESENTATION_DIMENSION,
                    generator=g,
                )/math.sqrt(S59_REPRESENTATION_DIMENSION)
            )
            self.u.copy_(
                torch.randn(
                    S59_HIDDEN_DIMENSION,
                    generator=g,
                )/math.sqrt(S59_HIDDEN_DIMENSION)
            )
            self.G.zero_()

        tg=torch.Generator(device="cpu").manual_seed(S59_HEAD_SEED+1)
        tie=torch.randn(S59_REPRESENTATION_DIMENSION,generator=tg)
        tie=F.normalize(tie,dim=0)
        self.register_buffer("tie_vector",tie,persistent=True)
        self.register_buffer(
            "context_projection",
            build_fixed_context_projection(),
            persistent=True,
        )

        for p in (self.A,self.u,self.G):
            p.requires_grad_(bool(trainable))

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def context_feature(self,representation:Tensor)->Tensor:
        return pairwise_set_context_feature(
            representation,
            self.context_projection,
            use_joint_context=self.use_joint_context,
        )

    def modulation(self,representation:Tensor)->Tensor:
        z=self.context_feature(representation)
        gamma=1.0+S68_MODULATION_SCALE*torch.tanh(
            F.linear(z,self.G,bias=None)
        )
        if gamma.shape[-1]!=S59_HIDDEN_DIMENSION:
            raise RuntimeError("S68 modulation dimension changed")
        if not bool(torch.isfinite(gamma).all()):
            raise ValueError("S68 modulation non-finite")
        if bool((gamma<1.0-S68_MODULATION_SCALE-1e-7).any()):
            raise RuntimeError("S68 modulation lower bound violated")
        if bool((gamma>1.0+S68_MODULATION_SCALE+1e-7).any()):
            raise RuntimeError("S68 modulation upper bound violated")
        return gamma

    def pairwise_logits(self,representation:Tensor)->Tensor:
        if representation.ndim!=3:
            raise ValueError("S68 representation must be [B,K,512]")
        b,k,d=representation.shape
        if k<2 or d!=S59_REPRESENTATION_DIMENSION:
            raise ValueError("S68 representation shape changed")
        if not bool(torch.isfinite(representation).all()):
            raise ValueError("S68 representation contains non-finite values")

        rep=F.normalize(representation.detach(),dim=-1)
        diff=rep[:,:,None,:]-rep[:,None,:,:]
        hidden=torch.tanh(F.linear(diff,self.A,bias=None))
        gamma=self.modulation(representation).to(
            device=hidden.device,dtype=hidden.dtype
        )
        modulated=hidden*gamma[:,None,None,:]
        raw=torch.einsum("bijk,k->bij",modulated,self.u)

        pair=0.5*(raw-raw.transpose(-1,-2))
        eye=torch.eye(k,dtype=torch.bool,device=pair.device).unsqueeze(0)
        pair=pair.masked_fill(eye,0.0)
        if not bool(torch.isfinite(pair).all()):
            raise ValueError("S68 pairwise logits non-finite")
        return pair

    def aggregate_logits(self,representation:Tensor)->Tensor:
        pair=self.pairwise_logits(representation)
        k=pair.shape[-1]
        score=pair.sum(dim=-1)/float(k-1)
        if not bool(torch.isfinite(score).all()):
            raise ValueError("S68 aggregate logits non-finite")
        return score

    @torch.no_grad()
    def select_with_tiebreak(
        self,
        representation:Tensor,
    )->tuple[Tensor,dict[str,Tensor]]:
        score=self.aggregate_logits(representation)
        rep=F.normalize(representation.detach(),dim=-1)
        top=score.max(dim=-1,keepdim=True).values
        tied=score.eq(top)
        key=torch.einsum(
            "bkd,d->bk",
            rep,
            self.tie_vector.to(device=rep.device,dtype=rep.dtype),
        )
        masked_key=key.masked_fill(~tied,float("-inf"))
        winner=masked_key.argmax(dim=-1)
        tied_count=tied.sum(dim=-1)
        winner_key=masked_key.gather(1,winner[:,None])
        key_ties=(masked_key==winner_key).sum(dim=-1)
        return winner,{
            "top_score_tie_count":tied_count,
            "representation_key_tie_count":key_ties,
            "degenerate_index_fallback":key_ties.gt(1),
        }

    def state_dict_exact(self)->dict[str,Tensor]:
        return {
            key:value.detach().cpu().clone()
            for key,value in self.state_dict().items()
        }

    def load_state_dict_exact(
        self,
        state:dict[str,Tensor],
        *,
        freeze:bool=False,
    )->None:
        self.load_state_dict(state,strict=True)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)


def context_modulated_pairwise_loss(
    head:ContextModulatedPairwiseHead,
    representation:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    pair=head.pairwise_logits(representation)
    loss,diag=gold_pairwise_loss(pair,gold)
    gamma=head.modulation(representation)
    return loss,{
        **diag,
        "antisymmetry_max_abs_error":(
            pair+pair.transpose(-1,-2)
        ).abs().max().detach(),
        "diagonal_max_abs_error":torch.diagonal(
            pair,dim1=-2,dim2=-1
        ).abs().max().detach(),
        "modulation_mean":gamma.mean().detach(),
        "modulation_min":gamma.min().detach(),
        "modulation_max":gamma.max().detach(),
        "modulation_std":gamma.std(unbiased=False).detach(),
    }


__all__=[
    "S68_CONTEXT_DIMENSION",
    "S68_CONTEXT_PROJECTION_SEED",
    "S68_MODULATION_SCALE",
    "S68_MODULATION_PARAMETER_COUNT",
    "S68_PAIRWISE_PARAMETER_COUNT",
    "build_fixed_context_projection",
    "pairwise_set_context_feature",
    "ContextModulatedPairwiseHead",
    "context_modulated_pairwise_loss",
]
