from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


S59_REPRESENTATION_DIMENSION=512
S59_HIDDEN_DIMENSION=64
S59_PAIRWISE_PARAMETER_COUNT=32_832
S59_HEAD_SEED=80_059


def build_pairwise_representation(identity: Tensor, joint_context: Tensor)->Tensor:
    if identity.shape!=joint_context.shape:
        raise ValueError("S59 identity/context shape mismatch")
    if identity.ndim!=3 or identity.shape[-1]*2!=S59_REPRESENTATION_DIMENSION:
        raise ValueError("S59 expects identity/context [B,K,256]")
    if identity.shape[1]<2:
        raise ValueError("S59 requires K>=2")
    if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(joint_context).all()):
        raise ValueError("S59 representation source is non-finite")
    # Ownership boundary: S59 decision head cannot update upstream geometry,
    # correction, native runtime, or cache tensors.
    rep=torch.cat([identity.detach(),joint_context.detach()],dim=-1)
    rep=F.normalize(rep,dim=-1)
    if not bool(torch.isfinite(rep).all()):
        raise ValueError("S59 detached representation is non-finite")
    return rep.detach()


class ExplicitPairwiseDecisionHead(nn.Module):
    """S59 gold-supervised anti-symmetric pairwise decision head."""

    def __init__(
        self,
        *,
        representation_dimension: int=S59_REPRESENTATION_DIMENSION,
        hidden_dimension: int=S59_HIDDEN_DIMENSION,
        seed: int=S59_HEAD_SEED,
        trainable: bool=True,
    ):
        super().__init__()
        if representation_dimension!=S59_REPRESENTATION_DIMENSION:
            raise ValueError("S59 representation dimension frozen at 512")
        if hidden_dimension!=S59_HIDDEN_DIMENSION:
            raise ValueError("S59 hidden dimension frozen at 64")
        if seed!=S59_HEAD_SEED:
            raise ValueError("S59 head seed frozen at 80059")
        self.representation_dimension=int(representation_dimension)
        self.hidden_dimension=int(hidden_dimension)
        self.seed=int(seed)

        self.A=nn.Parameter(torch.empty(hidden_dimension,representation_dimension))
        self.u=nn.Parameter(torch.empty(hidden_dimension))

        g=torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.A.copy_(
                torch.randn(
                    hidden_dimension,representation_dimension,generator=g
                )/math.sqrt(representation_dimension)
            )
            self.u.copy_(
                torch.randn(hidden_dimension,generator=g)
                /math.sqrt(hidden_dimension)
            )

        # Frozen representation-derived tiebreak key. It is not a model
        # parameter and never consumes native/fused/corrected logits.
        tg=torch.Generator().manual_seed(seed+1)
        tie=torch.randn(representation_dimension,generator=tg)
        tie=F.normalize(tie,dim=0)
        self.register_buffer("tie_vector",tie,persistent=True)

        for p in (self.A,self.u):
            p.requires_grad_(bool(trainable))

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def pairwise_logits(self,representation:Tensor)->Tensor:
        if representation.ndim!=3:
            raise ValueError("S59 representation must be [B,K,512]")
        b,k,d=representation.shape
        if k<2 or d!=self.representation_dimension:
            raise ValueError("S59 representation shape changed")
        if not bool(torch.isfinite(representation).all()):
            raise ValueError("S59 representation contains non-finite values")

        rep=F.normalize(representation.detach(),dim=-1)
        diff=rep[:,:,None,:]-rep[:,None,:,:]
        hidden=torch.tanh(F.linear(diff,self.A,bias=None))
        raw=torch.einsum("bijk,k->bij",hidden,self.u)

        # Explicit antisymmetrization means the pair matrix is the actual
        # decision object, not a scalar score later reinterpreted as pairwise.
        pair=0.5*(raw-raw.transpose(-1,-2))
        eye=torch.eye(k,dtype=torch.bool,device=pair.device).unsqueeze(0)
        pair=pair.masked_fill(eye,0.0)
        if not bool(torch.isfinite(pair).all()):
            raise ValueError("S59 pairwise logits are non-finite")
        return pair

    def aggregate_logits(self,representation:Tensor)->Tensor:
        pair=self.pairwise_logits(representation)
        k=pair.shape[-1]
        score=pair.sum(dim=-1)/float(k-1)
        if not bool(torch.isfinite(score).all()):
            raise ValueError("S59 aggregate logits are non-finite")
        return score

    @torch.no_grad()
    def select_with_tiebreak(self,representation:Tensor)->tuple[Tensor,dict[str,Tensor]]:
        score=self.aggregate_logits(representation)
        rep=F.normalize(representation.detach(),dim=-1)
        top=score.max(dim=-1,keepdim=True).values
        tied=score.eq(top)
        key=torch.einsum("bkd,d->bk",rep,self.tie_vector.to(rep.dtype))
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

    def load_state_dict_exact(self,state:dict[str,Tensor],*,freeze:bool=False)->None:
        self.load_state_dict(state,strict=True)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)


def gold_pairwise_supervision_mask(gold:Tensor,k:int)->Tensor:
    if gold.ndim!=1 or gold.dtype not in (torch.int64,torch.long):
        raise ValueError("S59 gold must be int64 [B]")
    if k<2 or bool(((gold<0)|(gold>=k)).any()):
        raise ValueError("S59 gold index out of range")
    b=gold.shape[0]
    rows=torch.arange(b,device=gold.device)[:,None]
    cols=torch.arange(k,device=gold.device)[None,:]
    return cols.ne(gold[:,None])


def gold_pairwise_loss(pair_logits:Tensor,gold:Tensor)->tuple[Tensor,dict[str,Tensor]]:
    if pair_logits.ndim!=3 or pair_logits.shape[1]!=pair_logits.shape[2]:
        raise ValueError("S59 pair logits must be [B,K,K]")
    b,k,_=pair_logits.shape
    if gold.shape!=(b,) or gold.dtype not in (torch.int64,torch.long):
        raise ValueError("S59 gold mismatch")
    if bool(((gold<0)|(gold>=k)).any()):
        raise ValueError("S59 gold index out of range")
    batch=torch.arange(b,device=pair_logits.device)
    gold_row=pair_logits[batch,gold,:]
    mask=gold_pairwise_supervision_mask(gold,k)
    margins=gold_row[mask]
    if margins.numel()!=b*(k-1):
        raise RuntimeError("S59 gold pair supervision count changed")
    loss=F.softplus(-margins).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("S59 pairwise loss non-finite")
    return loss,{
        "supervised_gold_pair_count":torch.tensor(
            margins.numel(),device=pair_logits.device,dtype=torch.long
        ),
        "supervised_distractor_pair_count":torch.tensor(
            0,device=pair_logits.device,dtype=torch.long
        ),
        "mean_gold_pair_margin":margins.detach().mean(),
        "gold_pair_accuracy":margins.detach().gt(0).to(torch.float32).mean(),
    }


def pairwise_head_loss(
    head:ExplicitPairwiseDecisionHead,
    representation:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    pair=head.pairwise_logits(representation)
    loss,diag=gold_pairwise_loss(pair,gold)
    return loss,{
        **diag,
        "antisymmetry_max_abs_error":(
            pair+pair.transpose(-1,-2)
        ).abs().max().detach(),
        "diagonal_max_abs_error":torch.diagonal(
            pair,dim1=-2,dim2=-1
        ).abs().max().detach(),
    }


__all__=[
    "S59_REPRESENTATION_DIMENSION",
    "S59_HIDDEN_DIMENSION",
    "S59_PAIRWISE_PARAMETER_COUNT",
    "S59_HEAD_SEED",
    "build_pairwise_representation",
    "ExplicitPairwiseDecisionHead",
    "gold_pairwise_supervision_mask",
    "gold_pairwise_loss",
    "pairwise_head_loss",
]
