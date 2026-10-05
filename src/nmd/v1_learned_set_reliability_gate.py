from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_confidence_adaptive_bounded_hybrid import (
    S61_ALPHA_INITIAL,
    S61_ALPHA_MAX,
    S61_B_INITIAL,
    S61_EPSILON,
    adaptive_gate_features,
)
from .v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    train_only_reliability_target,
)


S63_OPTION_INPUT_DIM=4
S63_OPTION_HIDDEN_DIM=8
S63_POOLED_DIM=16
S63_REPRESENTATION_DIM=20
S63_GATE_PARAMETER_COUNT=61
S63_INIT_SEED=63_063
S63_ALPHA_MAX=S61_ALPHA_MAX
S63_ALPHA_INITIAL=S61_ALPHA_INITIAL
S63_B_INITIAL=S61_B_INITIAL


def _centered_rms(logits:Tensor)->Tensor:
    if logits.ndim!=2 or logits.shape[-1]<2:
        raise ValueError("S63 logits must be [B,K] with K>=2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S63 logits contain non-finite values")
    centered=logits-logits.mean(dim=-1,keepdim=True)
    return centered.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)


def learned_set_option_features(
    fused_logits:Tensor,
    pairwise_logits:Tensor,
)->Tensor:
    if fused_logits.shape!=pairwise_logits.shape:
        raise ValueError("S63 fused/pairwise shape mismatch")
    if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
        raise ValueError("S63 fused/pairwise logits must be [B,K]")
    if not bool(torch.isfinite(fused_logits).all()) or not bool(torch.isfinite(pairwise_logits).all()):
        raise ValueError("S63 inputs contain non-finite values")

    fused=fused_logits.detach()
    pairwise=pairwise_logits.detach()
    zf=(fused-fused.mean(dim=-1,keepdim=True))/_centered_rms(fused)
    zp=(pairwise-pairwise.mean(dim=-1,keepdim=True))/_centered_rms(pairwise)
    features=torch.stack([zf,zp,zf-zp,zf*zp],dim=-1)
    if features.shape[-1]!=S63_OPTION_INPUT_DIM:
        raise RuntimeError("S63 option feature dimension changed")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("S63 option features non-finite")
    return features.detach()


class LearnedSetReliabilityGate(nn.Module):
    """Tiny permutation-invariant reliability gate over detached decision surfaces."""

    def __init__(self,*,trainable:bool=True):
        super().__init__()
        g=torch.Generator(device="cpu").manual_seed(S63_INIT_SEED)
        w_phi=torch.randn(
            S63_OPTION_HIDDEN_DIM,
            S63_OPTION_INPUT_DIM,
            generator=g,
            dtype=torch.float32,
        )/math.sqrt(S63_OPTION_INPUT_DIM)
        b_phi=torch.randn(
            S63_OPTION_HIDDEN_DIM,
            generator=g,
            dtype=torch.float32,
        )*0.05
        self.W_phi=nn.Parameter(w_phi)
        self.b_phi=nn.Parameter(b_phi)
        self.w_out=nn.Parameter(torch.zeros(S63_REPRESENTATION_DIM,dtype=torch.float32))
        self.b_out=nn.Parameter(torch.tensor(S63_B_INITIAL,dtype=torch.float32))
        for p in (self.W_phi,self.b_phi,self.w_out,self.b_out):
            p.requires_grad_(bool(trainable))

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def representation(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
    )->Tensor:
        option=learned_set_option_features(fused_logits,pairwise_logits)
        hidden=torch.tanh(F.linear(option,self.W_phi,self.b_phi))
        mean_pool=hidden.mean(dim=1)
        max_pool=hidden.max(dim=1).values
        base=adaptive_gate_features(
            fused_logits.detach(),
            pairwise_logits.detach(),
        )
        rep=torch.cat([mean_pool,max_pool,base],dim=-1)
        if rep.shape[-1]!=S63_REPRESENTATION_DIM:
            raise RuntimeError("S63 representation dimension changed")
        if not bool(torch.isfinite(rep).all()):
            raise ValueError("S63 representation non-finite")
        return rep

    def pre_sigmoid_logit(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
    )->Tensor:
        rep=self.representation(fused_logits,pairwise_logits)
        logits=F.linear(rep,self.w_out.unsqueeze(0),self.b_out.reshape(1)).squeeze(-1)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S63 gate logits non-finite")
        return logits

    def alpha(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
    )->Tensor:
        alpha=S63_ALPHA_MAX*torch.sigmoid(
            self.pre_sigmoid_logit(fused_logits,pairwise_logits)
        )
        if not bool(torch.isfinite(alpha).all()):
            raise ValueError("S63 alpha non-finite")
        return alpha.unsqueeze(-1)

    def compose(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
        *,
        alpha_override:float|None=None,
        return_diagnostics:bool=False,
    ):
        if fused_logits.shape!=pairwise_logits.shape:
            raise ValueError("S63 fused/pairwise shape mismatch")
        if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
            raise ValueError("S63 fused/pairwise logits must be [B,K]")
        fused=fused_logits.detach()
        pairwise=pairwise_logits.detach()
        if not bool(torch.isfinite(fused).all()) or not bool(torch.isfinite(pairwise).all()):
            raise ValueError("S63 compose inputs contain non-finite values")

        if alpha_override is not None:
            value=float(alpha_override)
            if value<0.0 or value>S63_ALPHA_MAX:
                raise ValueError("S63 alpha override outside [0,0.35]")
            alpha=torch.full(
                (fused.shape[0],1),
                value,
                device=fused.device,
                dtype=fused.dtype,
            )
            if value==0.0:
                if return_diagnostics:
                    zero=torch.zeros_like(alpha)
                    return fused,{
                        "alpha":zero,
                        "fused_rms":_centered_rms(fused),
                        "residual_max_abs":zero,
                        "residual_bound":zero,
                    }
                return fused
        else:
            alpha=self.alpha(fused,pairwise).to(
                device=fused.device,dtype=fused.dtype
            )

        pair_centered=pairwise-pairwise.mean(dim=-1,keepdim=True)
        pair_z=pair_centered/_centered_rms(pairwise)
        bounded=torch.tanh(pair_z)
        fused_rms=_centered_rms(fused)
        residual=alpha*fused_rms*bounded
        out=fused+residual
        if not bool(torch.isfinite(out).all()):
            raise ValueError("S63 hybrid logits non-finite")
        bound=alpha*fused_rms
        if bool((residual.abs()>bound+1e-7).any()):
            raise RuntimeError("S63 bounded residual contract violated")

        if return_diagnostics:
            return out,{
                "alpha":alpha.detach(),
                "fused_rms":fused_rms.detach(),
                "residual_max_abs":residual.abs().amax(dim=-1,keepdim=True).detach(),
                "residual_bound":bound.detach(),
            }
        return out

    def state_dict_exact(self)->dict[str,Tensor]:
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()}

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


def learned_set_reliability_loss(
    gate:LearnedSetReliabilityGate,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    target,target_diag=train_only_reliability_target(
        fused_canonical,
        fused_paraphrase,
        pairwise_canonical,
        pairwise_paraphrase,
        gold,
    )
    zc=gate.pre_sigmoid_logit(
        fused_canonical.detach(),
        pairwise_canonical.detach(),
    )
    zp=gate.pre_sigmoid_logit(
        fused_paraphrase.detach(),
        pairwise_paraphrase.detach(),
    )
    if zc.shape!=target.shape or zp.shape!=target.shape:
        raise RuntimeError("S63 gate logit/target shape changed")
    loss=0.5*(
        F.binary_cross_entropy_with_logits(zc,target)
        +F.binary_cross_entropy_with_logits(zp,target)
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("S63 reliability BCE non-finite")

    with torch.no_grad():
        ac=gate.alpha(fused_canonical,pairwise_canonical).reshape(-1)
        ap=gate.alpha(fused_paraphrase,pairwise_paraphrase).reshape(-1)
        all_alpha=torch.cat([ac,ap])
    return loss,{
        **target_diag,
        "target_positive_count":target.sum().detach(),
        "target_count":torch.tensor(
            target.numel(),device=target.device,dtype=torch.long
        ),
        "mean_alpha":all_alpha.mean().detach(),
        "min_alpha":all_alpha.min().detach(),
        "max_alpha":all_alpha.max().detach(),
        "std_alpha":all_alpha.std(unbiased=False).detach(),
    }


__all__=[
    "S63_OPTION_INPUT_DIM",
    "S63_OPTION_HIDDEN_DIM",
    "S63_POOLED_DIM",
    "S63_REPRESENTATION_DIM",
    "S63_GATE_PARAMETER_COUNT",
    "S63_INIT_SEED",
    "S63_ALPHA_MAX",
    "S63_ALPHA_INITIAL",
    "learned_set_option_features",
    "LearnedSetReliabilityGate",
    "learned_set_reliability_loss",
]
