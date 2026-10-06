from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_confidence_adaptive_bounded_hybrid import S61_EPSILON
from .v1_multistat_pairwise_row_composer import pairwise_row_statistics


S74_FUSED_FEATURE_DIM=4
S74_RELATIONAL_FEATURE_DIM=8
S74_INPUT_DIM=S74_FUSED_FEATURE_DIM+S74_RELATIONAL_FEATURE_DIM
S74_HIDDEN_DIM=16
S74_PARAMETER_COUNT=257
S74_INIT_SEED=74_074
S74_JS_WEIGHT=0.10


def _centered_rms(x:Tensor)->Tensor:
    if x.ndim!=2 or x.shape[-1]<2:
        raise ValueError("S74 expects [B,K], K>=2")
    centered=x-x.mean(dim=-1,keepdim=True)
    return centered.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)


def _standardize_options(x:Tensor)->Tensor:
    centered=x-x.mean(dim=-1,keepdim=True)
    return centered/_centered_rms(x)


def _rank_fraction(x:Tensor)->Tensor:
    # Permutation-equivariant deterministic rank in [0,1]. Ties receive the
    # same mid-rank by counting strict-greater and equal elements.
    b,k=x.shape
    greater=(x[:,:,None]<x[:,None,:]).sum(dim=-1).to(x.dtype)
    equal=(x[:,:,None]==x[:,None,:]).sum(dim=-1).to(x.dtype)
    mid=greater+0.5*(equal-1.0)
    denom=max(1,k-1)
    return 1.0-mid/float(denom)


def direct_set_arbitration_features(
    fused_logits:Tensor,
    pair_matrix:Tensor,
    *,
    use_relational:bool,
)->Tensor:
    if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
        raise ValueError("S74 fused logits must be [B,K], K>=2")
    b,k=fused_logits.shape
    if pair_matrix.shape!=(b,k,k):
        raise ValueError("S74 pair matrix shape mismatch")
    if not bool(torch.isfinite(fused_logits).all()):
        raise ValueError("S74 fused logits non-finite")
    if not bool(torch.isfinite(pair_matrix).all()):
        raise ValueError("S74 pair matrix non-finite")

    fused=fused_logits.detach()
    pair=pair_matrix.detach()

    fused_std=_standardize_options(fused)
    fused_prob=torch.softmax(fused,dim=-1)
    fused_logprob=torch.log_softmax(fused,dim=-1)
    fused_logprob_std=_standardize_options(fused_logprob)
    fused_rank=_rank_fraction(fused)

    fused_features=torch.stack(
        [fused_std,fused_prob,fused_logprob_std,fused_rank],
        dim=-1,
    )

    stats=pairwise_row_statistics(pair)
    mean,maxv,minv,rms=stats.unbind(dim=-1)
    mean_std=_standardize_options(mean)
    max_std=_standardize_options(maxv)
    min_std=_standardize_options(minv)
    rms_std=_standardize_options(rms)

    eye=torch.eye(k,device=pair.device,dtype=torch.bool).unsqueeze(0)
    off_mask=(~eye).expand(b,-1,-1)
    off=pair.masked_select(off_mask).reshape(b,k,k-1)
    win_fraction=off.gt(0).to(pair.dtype).mean(dim=-1)
    win_fraction=2.0*win_fraction-1.0

    strongest_loss=(-off.min(dim=-1).values).clamp_min(0.0)
    strongest_loss_std=_standardize_options(strongest_loss)

    abs_mean=off.abs().mean(dim=-1).clamp_min(S61_EPSILON)
    signed_strength=(mean/abs_mean).clamp(-1.0,1.0)

    pair_rank=_rank_fraction(mean)
    rank_disagreement=(pair_rank-fused_rank).clamp(-1.0,1.0)

    relational=torch.stack(
        [
            mean_std,
            max_std,
            min_std,
            rms_std,
            win_fraction,
            strongest_loss_std,
            signed_strength,
            rank_disagreement,
        ],
        dim=-1,
    )
    if not use_relational:
        relational=torch.zeros_like(relational)

    features=torch.cat([fused_features,relational],dim=-1)
    if features.shape!=(b,k,S74_INPUT_DIM):
        raise RuntimeError("S74 feature shape changed")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("S74 features non-finite")
    return features.detach()


class DirectSetArbitrationCore(nn.Module):
    """Permutation-equivariant full-K direct decision core.

    The model emits final arbitration logits directly. It never adds a learned
    residual to fused logits.
    """

    def __init__(self,*,trainable:bool=True):
        super().__init__()
        g=torch.Generator(device="cpu").manual_seed(S74_INIT_SEED)
        self.W_option=nn.Parameter(
            torch.randn(S74_HIDDEN_DIM,S74_INPUT_DIM,generator=g)
            /math.sqrt(S74_INPUT_DIM)
        )
        self.b_option=nn.Parameter(torch.zeros(S74_HIDDEN_DIM))
        self.w_score=nn.Parameter(
            torch.randn(3*S74_HIDDEN_DIM,generator=g)
            /math.sqrt(3*S74_HIDDEN_DIM)
        )
        self.b_score=nn.Parameter(torch.zeros(()))

        for p in self.parameters():
            p.requires_grad_(bool(trainable))
        if self.parameter_count!=S74_PARAMETER_COUNT:
            raise RuntimeError("S74 parameter count changed")

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def logits_from_features(self,features:Tensor)->Tensor:
        if features.ndim!=3 or features.shape[-1]!=S74_INPUT_DIM:
            raise ValueError("S74 features must be [B,K,12]")
        if features.shape[1]<2:
            raise ValueError("S74 requires K>=2")
        x=features.detach()
        hidden=torch.tanh(F.linear(x,self.W_option,self.b_option))
        mean_pool=hidden.mean(dim=1,keepdim=True).expand_as(hidden)
        max_pool=hidden.max(dim=1,keepdim=True).values.expand_as(hidden)
        score_input=torch.cat([hidden,mean_pool,max_pool],dim=-1)
        logits=F.linear(score_input,self.w_score.unsqueeze(0),self.b_score.reshape(1)).squeeze(-1)
        if logits.shape!=features.shape[:2]:
            raise RuntimeError("S74 logit shape changed")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S74 logits non-finite")
        return logits

    def forward(
        self,
        fused_logits:Tensor,
        pair_matrix:Tensor,
        *,
        use_relational:bool,
    )->Tensor:
        features=direct_set_arbitration_features(
            fused_logits,pair_matrix,use_relational=use_relational
        )
        return self.logits_from_features(features)

    def state_dict_exact(self)->dict[str,Tensor]:
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()}

    def load_state_dict_exact(self,state:dict[str,Tensor],*,freeze:bool=False)->None:
        self.load_state_dict(state,strict=True)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)


def direct_set_arbitration_loss(
    core:DirectSetArbitrationCore,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pair_canonical:Tensor,
    pair_paraphrase:Tensor,
    gold:Tensor,
    *,
    use_relational:bool,
)->tuple[Tensor,dict[str,Tensor]]:
    logits_c=core(
        fused_canonical,pair_canonical,use_relational=use_relational
    )
    logits_p=core(
        fused_paraphrase,pair_paraphrase,use_relational=use_relational
    )
    if gold.ndim!=1 or gold.shape[0]!=logits_c.shape[0]:
        raise ValueError("S74 gold mismatch")

    ce_c=F.cross_entropy(logits_c,gold)
    ce_p=F.cross_entropy(logits_p,gold)

    pc=torch.softmax(logits_c,dim=-1)
    pp=torch.softmax(logits_p,dim=-1)
    m=0.5*(pc+pp)
    js=0.5*(
        (pc*(pc.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(dim=-1).mean()
        +(pp*(pp.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(dim=-1).mean()
    )
    loss=0.5*(ce_c+ce_p)+S74_JS_WEIGHT*js
    if not bool(torch.isfinite(loss)):
        raise ValueError("S74 loss non-finite")
    return loss,{
        "canonical_ce":ce_c.detach(),
        "paraphrase_ce":ce_p.detach(),
        "cross_view_js":js.detach(),
    }


__all__=[
    "S74_FUSED_FEATURE_DIM",
    "S74_RELATIONAL_FEATURE_DIM",
    "S74_INPUT_DIM",
    "S74_HIDDEN_DIM",
    "S74_PARAMETER_COUNT",
    "S74_INIT_SEED",
    "S74_JS_WEIGHT",
    "direct_set_arbitration_features",
    "DirectSetArbitrationCore",
    "direct_set_arbitration_loss",
]
