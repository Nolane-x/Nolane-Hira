from __future__ import annotations

import torch
from torch import Tensor

from .v1_confidence_adaptive_bounded_hybrid import S61_EPSILON
from .v1_multistat_pairwise_row_composer import (
    MultiStatPairwiseRowComposer,
    pairwise_row_mean,
)


S72_PROFILE_EPSILON=1e-8


def _centered_rms_2d(x:Tensor)->Tensor:
    if x.ndim!=2 or x.shape[-1]<2:
        raise ValueError("S72 direction input must be [B,K], K>=2")
    centered=x-x.mean(dim=-1,keepdim=True)
    return centered.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)


def bounded_direction_from_option_scores(scores:Tensor)->Tensor:
    if scores.ndim!=2 or scores.shape[-1]<2:
        raise ValueError("S72 scores must be [B,K], K>=2")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("S72 scores non-finite")
    x=scores.detach()
    centered=x-x.mean(dim=-1,keepdim=True)
    rms=_centered_rms_2d(x)
    direction=torch.tanh(centered/rms)
    if not bool(torch.isfinite(direction).all()):
        raise ValueError("S72 direction non-finite")
    if bool((direction.abs()>1.0+1e-7).any()):
        raise RuntimeError("S72 bounded direction escaped [-1,1]")
    return direction.detach()


def reference_pairwise_mean_direction(pair_matrix:Tensor)->Tensor:
    return bounded_direction_from_option_scores(pairwise_row_mean(pair_matrix))


def opponent_profile_confidence(
    fused_logits:Tensor,
    pair_matrix:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
        raise ValueError("S72 fused logits must be [B,K], K>=2")
    if pair_matrix.ndim!=3:
        raise ValueError("S72 pair matrix must be [B,K,K]")
    b,k=fused_logits.shape
    if pair_matrix.shape!=(b,k,k):
        raise ValueError("S72 pair matrix shape mismatch")
    if not bool(torch.isfinite(fused_logits).all()):
        raise ValueError("S72 fused logits non-finite")
    if not bool(torch.isfinite(pair_matrix).all()):
        raise ValueError("S72 pair matrix non-finite")

    fused=fused_logits.detach()
    pair=pair_matrix.detach()
    out_dtype=fused.dtype

    # Mechanical numerical hardening only: reductions use float64 so option
    # permutations do not change fp32 summation order enough to affect A0.
    work_fused=fused.to(torch.float64) if fused.dtype!=torch.float64 else fused
    work_pair=pair.to(torch.float64) if pair.dtype!=torch.float64 else pair

    prior=torch.softmax(work_fused,dim=-1)
    mask=(~torch.eye(k,device=work_pair.device,dtype=torch.bool)).unsqueeze(0)
    weights=prior.unsqueeze(1).expand(b,k,k).masked_fill(~mask,0.0)
    denom=weights.sum(dim=-1,keepdim=True).clamp_min(S72_PROFILE_EPSILON)
    weights=weights/denom

    signed=(weights*work_pair).sum(dim=-1)
    absolute=(weights*work_pair.abs()).sum(dim=-1)
    confidence=(signed/(absolute+S72_PROFILE_EPSILON)).to(out_dtype)

    if not bool(torch.isfinite(confidence).all()):
        raise ValueError("S72 opponent-profile confidence non-finite")
    if bool((confidence.abs()>1.0+1e-6).any()):
        raise RuntimeError("S72 confidence ratio escaped [-1,1]")
    if bool((weights.diagonal(dim1=-2,dim2=-1).abs()>0).any()):
        raise RuntimeError("S72 pair diagonal contributed")
    opponent_mass=weights.sum(dim=-1)
    if float((opponent_mass-1.0).abs().max())>1e-6:
        raise RuntimeError("S72 opponent weights did not normalize")

    return confidence.detach(),{
        "prior":prior.to(out_dtype).detach(),
        "opponent_weights":weights.to(out_dtype).detach(),
        "signed_evidence":signed.to(out_dtype).detach(),
        "absolute_evidence":absolute.to(out_dtype).detach(),
        "opponent_mass":opponent_mass.to(out_dtype).detach(),
    }


def opponent_profile_vector_direction(
    fused_logits:Tensor,
    pair_matrix:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    confidence,diag=opponent_profile_confidence(fused_logits,pair_matrix)
    direction=bounded_direction_from_option_scores(confidence)
    return direction,{
        **diag,
        "confidence":confidence.detach(),
        "direction":direction.detach(),
    }


def compose_with_residual_direction(
    composer:MultiStatPairwiseRowComposer,
    fused_logits:Tensor,
    pair_matrix:Tensor,
    context_representation:Tensor,
    *,
    use_opponent_profile_direction:bool,
    alpha_override:float|None=None,
    return_diagnostics:bool=False,
):
    if composer.use_multistat:
        raise ValueError("S72 requires exact S71 mean-only composer in both arms")
    if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
        raise ValueError("S72 fused logits must be [B,K], K>=2")
    if pair_matrix.shape[:2]!=fused_logits.shape or pair_matrix.shape[-1]!=fused_logits.shape[-1]:
        raise ValueError("S72 pair matrix shape mismatch")
    if context_representation.shape[:2]!=fused_logits.shape:
        raise ValueError("S72 context shape mismatch")

    fused=fused_logits.detach()
    pair=pair_matrix.detach()
    context=context_representation.detach()

    centered_f=fused-fused.mean(dim=-1,keepdim=True)
    fused_rms=centered_f.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)

    if alpha_override is not None:
        value=float(alpha_override)
        if value<0.0 or value>0.35:
            raise ValueError("S72 alpha override outside [0,0.35]")
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
                    "direction":torch.zeros_like(fused),
                    "residual_max_abs":zero,
                    "residual_bound":zero,
                }
            return fused
    else:
        alpha=composer.alpha(fused,pair,context).to(
            device=fused.device,
            dtype=fused.dtype,
        )

    if use_opponent_profile_direction:
        direction,profile_diag=opponent_profile_vector_direction(fused,pair)
    else:
        direction=reference_pairwise_mean_direction(pair)
        profile_diag={}

    residual=alpha*fused_rms*direction
    bound=alpha*fused_rms
    out=fused+residual

    if bool((residual.abs()>bound+1e-7).any()):
        raise RuntimeError("S72 bounded residual contract violated")
    if not bool(torch.isfinite(out).all()):
        raise ValueError("S72 output non-finite")

    if return_diagnostics:
        return out,{
            "alpha":alpha.detach(),
            "fused_rms":fused_rms.detach(),
            "direction":direction.detach(),
            "residual_max_abs":residual.abs().amax(dim=-1,keepdim=True).detach(),
            "residual_bound":bound.detach(),
            **profile_diag,
        }
    return out


__all__=[
    "S72_PROFILE_EPSILON",
    "bounded_direction_from_option_scores",
    "reference_pairwise_mean_direction",
    "opponent_profile_confidence",
    "opponent_profile_vector_direction",
    "compose_with_residual_direction",
]
