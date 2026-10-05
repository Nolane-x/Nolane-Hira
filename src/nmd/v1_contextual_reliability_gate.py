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
from .v1_explicit_pairwise_decision_head import S59_REPRESENTATION_DIMENSION
from .v1_learned_set_reliability_gate import learned_set_option_features
from .v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    train_only_reliability_target,
)


S64_SURFACE_DIM=4
S64_CONTEXT_DIM=4
S64_OPTION_INPUT_DIM=8
S64_OPTION_HIDDEN_DIM=5
S64_POOLED_DIM=10
S64_REPRESENTATION_DIM=14
S64_GATE_PARAMETER_COUNT=60
S64_CONTEXT_PROJECTION_SEED=64_064
S64_ENCODER_INIT_SEED=64_164
S64_ALPHA_MAX=S61_ALPHA_MAX
S64_ALPHA_INITIAL=S61_ALPHA_INITIAL
S64_B_INITIAL=S61_B_INITIAL


def _centered_rms_last_option(x:Tensor)->Tensor:
    if x.ndim!=3:
        raise ValueError("S64 context projection must be [B,K,D]")
    centered=x-x.mean(dim=1,keepdim=True)
    return centered.square().mean(dim=1,keepdim=True).sqrt().clamp_min(S61_EPSILON)


def build_fixed_context_projection()->Tensor:
    g=torch.Generator(device="cpu").manual_seed(S64_CONTEXT_PROJECTION_SEED)
    p=torch.randn(
        S64_CONTEXT_DIM,
        S59_REPRESENTATION_DIMENSION,
        generator=g,
        dtype=torch.float32,
    )
    p=F.normalize(p,dim=-1)
    if p.shape!=(S64_CONTEXT_DIM,S59_REPRESENTATION_DIMENSION):
        raise RuntimeError("S64 context projection shape changed")
    return p


def contextual_option_features(
    context_representation:Tensor,
    projection:Tensor,
)->Tensor:
    if context_representation.ndim!=3:
        raise ValueError("S64 context representation must be [B,K,512]")
    if context_representation.shape[-1]!=S59_REPRESENTATION_DIMENSION:
        raise ValueError("S64 context representation dimension changed")
    if context_representation.shape[1]<2:
        raise ValueError("S64 requires K>=2")
    if projection.shape!=(S64_CONTEXT_DIM,S59_REPRESENTATION_DIMENSION):
        raise ValueError("S64 projection shape changed")
    if not bool(torch.isfinite(context_representation).all()):
        raise ValueError("S64 context representation non-finite")
    if not bool(torch.isfinite(projection).all()):
        raise ValueError("S64 projection non-finite")

    rep=context_representation.detach()
    raw=F.linear(rep,projection.detach())
    centered=raw-raw.mean(dim=1,keepdim=True)
    features=centered/_centered_rms_last_option(raw)
    if features.shape[-1]!=S64_CONTEXT_DIM:
        raise RuntimeError("S64 context feature dimension changed")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("S64 context features non-finite")
    return features.detach()


class ContextInjectedReliabilityGate(nn.Module):
    """Parameter-matched S64 reliability gate.

    Reference and treatment have identical parameters. The only controlled
    difference is whether the four detached contextual channels are exposed.
    """

    def __init__(self,*,use_context:bool,trainable:bool=True):
        super().__init__()
        self.use_context=bool(use_context)

        projection=build_fixed_context_projection()
        self.register_buffer("context_projection",projection,persistent=True)

        g=torch.Generator(device="cpu").manual_seed(S64_ENCODER_INIT_SEED)
        w_phi=torch.randn(
            S64_OPTION_HIDDEN_DIM,
            S64_OPTION_INPUT_DIM,
            generator=g,
            dtype=torch.float32,
        )/math.sqrt(S64_OPTION_INPUT_DIM)
        b_phi=torch.randn(
            S64_OPTION_HIDDEN_DIM,
            generator=g,
            dtype=torch.float32,
        )*0.05

        self.W_phi=nn.Parameter(w_phi)
        self.b_phi=nn.Parameter(b_phi)
        self.w_out=nn.Parameter(torch.zeros(S64_REPRESENTATION_DIM,dtype=torch.float32))
        self.b_out=nn.Parameter(torch.tensor(S64_B_INITIAL,dtype=torch.float32))

        for p in (self.W_phi,self.b_phi,self.w_out,self.b_out):
            p.requires_grad_(bool(trainable))

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def option_input(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
        context_representation:Tensor,
    )->Tensor:
        surface=learned_set_option_features(
            fused_logits.detach(),
            pairwise_logits.detach(),
        )
        if surface.shape[-1]!=S64_SURFACE_DIM:
            raise RuntimeError("S64 surface feature dimension changed")

        if context_representation.shape[:2]!=surface.shape[:2]:
            raise ValueError("S64 context/surface batch-option mismatch")
        context=contextual_option_features(
            context_representation.detach(),
            self.context_projection,
        )
        if not self.use_context:
            context=torch.zeros_like(context)

        option=torch.cat([surface,context],dim=-1)
        if option.shape[-1]!=S64_OPTION_INPUT_DIM:
            raise RuntimeError("S64 option input dimension changed")
        if not bool(torch.isfinite(option).all()):
            raise ValueError("S64 option input non-finite")
        return option

    def representation(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
        context_representation:Tensor,
    )->Tensor:
        option=self.option_input(
            fused_logits,
            pairwise_logits,
            context_representation,
        )
        hidden=torch.tanh(F.linear(option,self.W_phi,self.b_phi))
        mean_pool=hidden.mean(dim=1)
        max_pool=hidden.max(dim=1).values
        base=adaptive_gate_features(
            fused_logits.detach(),
            pairwise_logits.detach(),
        )
        rep=torch.cat([mean_pool,max_pool,base],dim=-1)
        if rep.shape[-1]!=S64_REPRESENTATION_DIM:
            raise RuntimeError("S64 representation dimension changed")
        if not bool(torch.isfinite(rep).all()):
            raise ValueError("S64 representation non-finite")
        return rep

    def pre_sigmoid_logit(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
        context_representation:Tensor,
    )->Tensor:
        rep=self.representation(
            fused_logits,
            pairwise_logits,
            context_representation,
        )
        logits=F.linear(rep,self.w_out.unsqueeze(0),self.b_out.reshape(1)).squeeze(-1)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S64 gate logits non-finite")
        return logits

    def alpha(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
        context_representation:Tensor,
    )->Tensor:
        alpha=S64_ALPHA_MAX*torch.sigmoid(
            self.pre_sigmoid_logit(
                fused_logits,
                pairwise_logits,
                context_representation,
            )
        )
        if not bool(torch.isfinite(alpha).all()):
            raise ValueError("S64 alpha non-finite")
        return alpha.unsqueeze(-1)

    def compose(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
        context_representation:Tensor,
        *,
        alpha_override:float|None=None,
        return_diagnostics:bool=False,
    ):
        if fused_logits.shape!=pairwise_logits.shape:
            raise ValueError("S64 fused/pairwise shape mismatch")
        if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
            raise ValueError("S64 fused/pairwise logits must be [B,K]")
        if context_representation.shape[:2]!=fused_logits.shape:
            raise ValueError("S64 context representation K mismatch")

        fused=fused_logits.detach()
        pairwise=pairwise_logits.detach()
        context=context_representation.detach()
        if not bool(torch.isfinite(fused).all()) or not bool(torch.isfinite(pairwise).all()):
            raise ValueError("S64 compose logits non-finite")

        centered_f=fused-fused.mean(dim=-1,keepdim=True)
        fused_rms=centered_f.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)

        if alpha_override is not None:
            value=float(alpha_override)
            if value<0.0 or value>S64_ALPHA_MAX:
                raise ValueError("S64 alpha override outside [0,0.35]")
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
                        "fused_rms":fused_rms.detach(),
                        "residual_max_abs":zero,
                        "residual_bound":zero,
                    }
                return fused
        else:
            alpha=self.alpha(fused,pairwise,context).to(
                device=fused.device,
                dtype=fused.dtype,
            )

        pair_centered=pairwise-pairwise.mean(dim=-1,keepdim=True)
        pair_rms=pair_centered.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)
        bounded=torch.tanh(pair_centered/pair_rms)
        residual=alpha*fused_rms*bounded
        out=fused+residual
        if not bool(torch.isfinite(out).all()):
            raise ValueError("S64 hybrid logits non-finite")

        bound=alpha*fused_rms
        if bool((residual.abs()>bound+1e-7).any()):
            raise RuntimeError("S64 bounded residual contract violated")

        if return_diagnostics:
            return out,{
                "alpha":alpha.detach(),
                "fused_rms":fused_rms.detach(),
                "residual_max_abs":residual.abs().amax(dim=-1,keepdim=True).detach(),
                "residual_bound":bound.detach(),
            }
        return out

    def parameter_state_dict_exact(self)->dict[str,Tensor]:
        return {
            k:v.detach().cpu().clone()
            for k,v in self.named_parameters()
        }

    def state_dict_exact(self)->dict[str,Tensor]:
        return {
            k:v.detach().cpu().clone()
            for k,v in self.state_dict().items()
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


def contextual_reliability_loss(
    gate:ContextInjectedReliabilityGate,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    context_canonical:Tensor,
    context_paraphrase:Tensor,
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
        context_canonical.detach(),
    )
    zp=gate.pre_sigmoid_logit(
        fused_paraphrase.detach(),
        pairwise_paraphrase.detach(),
        context_paraphrase.detach(),
    )
    if zc.shape!=target.shape or zp.shape!=target.shape:
        raise RuntimeError("S64 gate logit/target shape changed")
    loss=0.5*(
        F.binary_cross_entropy_with_logits(zc,target)
        +F.binary_cross_entropy_with_logits(zp,target)
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("S64 reliability BCE non-finite")

    with torch.no_grad():
        ac=gate.alpha(
            fused_canonical,
            pairwise_canonical,
            context_canonical,
        ).reshape(-1)
        ap=gate.alpha(
            fused_paraphrase,
            pairwise_paraphrase,
            context_paraphrase,
        ).reshape(-1)
        all_alpha=torch.cat([ac,ap])

    return loss,{
        **target_diag,
        "target_positive_count":target.sum().detach(),
        "target_count":torch.tensor(
            target.numel(),
            device=target.device,
            dtype=torch.long,
        ),
        "mean_alpha":all_alpha.mean().detach(),
        "min_alpha":all_alpha.min().detach(),
        "max_alpha":all_alpha.max().detach(),
        "std_alpha":all_alpha.std(unbiased=False).detach(),
    }


__all__=[
    "S64_SURFACE_DIM",
    "S64_CONTEXT_DIM",
    "S64_OPTION_INPUT_DIM",
    "S64_OPTION_HIDDEN_DIM",
    "S64_POOLED_DIM",
    "S64_REPRESENTATION_DIM",
    "S64_GATE_PARAMETER_COUNT",
    "S64_CONTEXT_PROJECTION_SEED",
    "S64_ENCODER_INIT_SEED",
    "S64_ALPHA_MAX",
    "S64_ALPHA_INITIAL",
    "build_fixed_context_projection",
    "contextual_option_features",
    "ContextInjectedReliabilityGate",
    "contextual_reliability_loss",
]
