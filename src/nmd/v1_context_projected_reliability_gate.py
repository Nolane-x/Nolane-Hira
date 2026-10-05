from __future__ import annotations

from hashlib import sha256
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
from .v1_learned_set_reliability_gate import (
    S63_INIT_SEED,
    S63_OPTION_HIDDEN_DIM,
    S63_POOLED_DIM,
    S63_REPRESENTATION_DIM,
)


S64_CONTEXT_DIM=512
S64_CONTEXT_PROJECTION_DIM=4
S64_PROJECTION_SEED=64_064
S64_GATE_PARAMETER_COUNT=61
S64_ALPHA_MAX=S61_ALPHA_MAX
S64_ALPHA_INITIAL=S61_ALPHA_INITIAL
S64_B_INITIAL=S61_B_INITIAL


def fixed_context_projection()->Tensor:
    g=torch.Generator(device="cpu").manual_seed(S64_PROJECTION_SEED)
    bits=torch.randint(
        0,2,(S64_CONTEXT_PROJECTION_DIM,S64_CONTEXT_DIM),
        generator=g,dtype=torch.int64,
    )
    signs=bits.mul(2).sub(1).to(torch.float32)
    return signs/math.sqrt(S64_CONTEXT_DIM)


def projection_sha256(projection:Tensor)->str:
    p=projection.detach().cpu().contiguous()
    d=sha256()
    d.update(str(tuple(p.shape)).encode("ascii"))
    d.update(str(p.dtype).encode("ascii"))
    d.update(p.numpy().tobytes())
    return d.hexdigest()


def _context_rms(x:Tensor)->Tensor:
    return x.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)


def context_projected_option_features(
    context_representation:Tensor,
    projection:Tensor|None=None,
)->Tensor:
    if context_representation.ndim!=3:
        raise ValueError("S64 context representation must be [B,K,512]")
    if context_representation.shape[-1]!=S64_CONTEXT_DIM:
        raise ValueError("S64 context dimension frozen at 512")
    if context_representation.shape[-2]<2:
        raise ValueError("S64 requires K>=2")
    if not bool(torch.isfinite(context_representation).all()):
        raise ValueError("S64 context contains non-finite values")

    source=context_representation.detach()
    centered=source-source.mean(dim=-1,keepdim=True)
    normalized=centered/_context_rms(centered)
    p=fixed_context_projection() if projection is None else projection.detach()
    if p.shape!=(S64_CONTEXT_PROJECTION_DIM,S64_CONTEXT_DIM):
        raise ValueError("S64 projection shape changed")
    p=p.to(device=source.device,dtype=source.dtype)
    features=F.linear(normalized,p)
    if features.shape[-1]!=S64_CONTEXT_PROJECTION_DIM:
        raise RuntimeError("S64 projected feature dimension changed")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("S64 projected context non-finite")
    return features.detach()


class ContextProjectedReliabilityGate(nn.Module):
    """Matched-capacity S64 gate using frozen projected S59 option context."""

    def __init__(self,*,trainable:bool=True):
        super().__init__()
        self.register_buffer("P_context",fixed_context_projection(),persistent=True)

        g=torch.Generator(device="cpu").manual_seed(S63_INIT_SEED)
        w_phi=torch.randn(
            S63_OPTION_HIDDEN_DIM,
            S64_CONTEXT_PROJECTION_DIM,
            generator=g,
            dtype=torch.float32,
        )/math.sqrt(S64_CONTEXT_PROJECTION_DIM)
        b_phi=torch.randn(
            S63_OPTION_HIDDEN_DIM,
            generator=g,
            dtype=torch.float32,
        )*0.05
        self.W_phi=nn.Parameter(w_phi)
        self.b_phi=nn.Parameter(b_phi)
        self.w_out=nn.Parameter(torch.zeros(S63_REPRESENTATION_DIM,dtype=torch.float32))
        self.b_out=nn.Parameter(torch.tensor(S64_B_INITIAL,dtype=torch.float32))
        for p in (self.W_phi,self.b_phi,self.w_out,self.b_out):
            p.requires_grad_(bool(trainable))

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def projection_digest(self)->str:
        return projection_sha256(self.P_context)

    def representation(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
        context_representation:Tensor,
    )->Tensor:
        if fused_logits.shape!=pairwise_logits.shape:
            raise ValueError("S64 fused/pairwise shape mismatch")
        if context_representation.shape[:2]!=fused_logits.shape:
            raise ValueError("S64 context/logit option shape mismatch")
        option=context_projected_option_features(
            context_representation,self.P_context
        )
        hidden=torch.tanh(F.linear(option,self.W_phi,self.b_phi))
        mean_pool=hidden.mean(dim=1)
        max_pool=hidden.max(dim=1).values
        base=adaptive_gate_features(
            fused_logits.detach(),pairwise_logits.detach()
        )
        rep=torch.cat([mean_pool,max_pool,base],dim=-1)
        if rep.shape[-1]!=S63_REPRESENTATION_DIM:
            raise RuntimeError("S64 reliability representation dimension changed")
        if not bool(torch.isfinite(rep).all()):
            raise ValueError("S64 reliability representation non-finite")
        return rep

    def pre_sigmoid_logit(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
        context_representation:Tensor,
    )->Tensor:
        rep=self.representation(
            fused_logits,pairwise_logits,context_representation
        )
        logits=F.linear(
            rep,self.w_out.unsqueeze(0),self.b_out.reshape(1)
        ).squeeze(-1)
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
                fused_logits,pairwise_logits,context_representation
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
        if context_representation.shape[:2]!=fused_logits.shape:
            raise ValueError("S64 context/logit shape mismatch")
        fused=fused_logits.detach()
        pairwise=pairwise_logits.detach()
        context=context_representation.detach()

        if alpha_override is not None:
            value=float(alpha_override)
            if value<0.0 or value>S64_ALPHA_MAX:
                raise ValueError("S64 alpha override outside [0,0.35]")
            alpha=torch.full(
                (fused.shape[0],1),value,
                device=fused.device,dtype=fused.dtype,
            )
            if value==0.0:
                if return_diagnostics:
                    zero=torch.zeros_like(alpha)
                    return fused,{
                        "alpha":zero,
                        "residual_max_abs":zero,
                        "residual_bound":zero,
                    }
                return fused
        else:
            alpha=self.alpha(fused,pairwise,context).to(
                device=fused.device,dtype=fused.dtype
            )

        pair_centered=pairwise-pairwise.mean(dim=-1,keepdim=True)
        pair_rms=pair_centered.square().mean(
            dim=-1,keepdim=True
        ).sqrt().clamp_min(S61_EPSILON)
        bounded=torch.tanh(pair_centered/pair_rms)
        fused_centered=fused-fused.mean(dim=-1,keepdim=True)
        fused_rms=fused_centered.square().mean(
            dim=-1,keepdim=True
        ).sqrt().clamp_min(S61_EPSILON)
        residual=alpha*fused_rms*bounded
        out=fused+residual
        bound=alpha*fused_rms
        if bool((residual.abs()>bound+1e-7).any()):
            raise RuntimeError("S64 residual bound violated")
        if not bool(torch.isfinite(out).all()):
            raise ValueError("S64 hybrid logits non-finite")
        if return_diagnostics:
            return out,{
                "alpha":alpha.detach(),
                "residual_max_abs":residual.abs().amax(
                    dim=-1,keepdim=True
                ).detach(),
                "residual_bound":bound.detach(),
            }
        return out

    def learned_state_dict_exact(self)->dict[str,Tensor]:
        return {
            k:v.detach().cpu().clone()
            for k,v in self.state_dict().items()
            if k!="P_context"
        }

    def state_dict_exact(self)->dict[str,Tensor]:
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()}

    def load_state_dict_exact(
        self,state:dict[str,Tensor],*,freeze:bool=False
    )->None:
        self.load_state_dict(state,strict=True)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)


def context_projected_reliability_loss(
    gate:ContextProjectedReliabilityGate,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    context_canonical:Tensor,
    context_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    target,target_diag=train_only_reliability_target(
        fused_canonical,fused_paraphrase,
        pairwise_canonical,pairwise_paraphrase,gold,
    )
    zc=gate.pre_sigmoid_logit(
        fused_canonical.detach(),pairwise_canonical.detach(),
        context_canonical.detach(),
    )
    zp=gate.pre_sigmoid_logit(
        fused_paraphrase.detach(),pairwise_paraphrase.detach(),
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
            fused_canonical,pairwise_canonical,context_canonical
        ).reshape(-1)
        ap=gate.alpha(
            fused_paraphrase,pairwise_paraphrase,context_paraphrase
        ).reshape(-1)
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
    "S64_CONTEXT_DIM",
    "S64_CONTEXT_PROJECTION_DIM",
    "S64_PROJECTION_SEED",
    "S64_GATE_PARAMETER_COUNT",
    "S64_ALPHA_MAX",
    "S64_ALPHA_INITIAL",
    "fixed_context_projection",
    "projection_sha256",
    "context_projected_option_features",
    "ContextProjectedReliabilityGate",
    "context_projected_reliability_loss",
]
