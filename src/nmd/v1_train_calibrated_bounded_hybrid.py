from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


S60_EPSILON=1e-6
S60_ALPHA_MAX=0.35
S60_ALPHA_INITIAL=0.10
S60_A_INITIAL=-0.916290731874155
S60_COMPOSER_PARAMETER_COUNT=1


def _centered_rms(logits:Tensor, *, epsilon:float=S60_EPSILON)->Tensor:
    if logits.ndim!=2 or logits.shape[-1]<2:
        raise ValueError("S60 logits must be [B,K] with K>=2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S60 logits contain non-finite values")
    centered=logits-logits.mean(dim=-1,keepdim=True)
    rms=centered.square().mean(dim=-1,keepdim=True).sqrt()
    return rms.clamp_min(epsilon)


class TrainCalibratedBoundedHybridComposer(nn.Module):
    """S60 one-scalar bounded pairwise residual on exact fused evidence."""

    def __init__(
        self,
        *,
        alpha_max:float=S60_ALPHA_MAX,
        alpha_initial:float=S60_ALPHA_INITIAL,
        epsilon:float=S60_EPSILON,
        trainable:bool=True,
    ):
        super().__init__()
        if alpha_max!=S60_ALPHA_MAX:
            raise ValueError("S60 alpha_max frozen at 0.35")
        if alpha_initial!=S60_ALPHA_INITIAL:
            raise ValueError("S60 alpha_initial frozen at 0.10")
        if epsilon!=S60_EPSILON:
            raise ValueError("S60 epsilon frozen at 1e-6")
        self.alpha_max=float(alpha_max)
        self.alpha_initial=float(alpha_initial)
        self.epsilon=float(epsilon)

        ratio=alpha_initial/alpha_max
        a0=math.log(ratio/(1.0-ratio))
        if abs(a0-S60_A_INITIAL)>1e-12:
            raise RuntimeError("S60 frozen calibration initialization changed")
        self.a=nn.Parameter(torch.tensor(a0,dtype=torch.float32))
        self.a.requires_grad_(bool(trainable))

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def alpha(self)->Tensor:
        return self.alpha_max*torch.sigmoid(self.a)

    def compose(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
        *,
        alpha_override:float|None=None,
        return_diagnostics:bool=False,
    ):
        if fused_logits.shape!=pairwise_logits.shape:
            raise ValueError("S60 fused/pairwise shape mismatch")
        if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
            raise ValueError("S60 fused/pairwise logits must be [B,K]")
        if not bool(torch.isfinite(fused_logits).all()):
            raise ValueError("S60 fused logits are non-finite")
        if not bool(torch.isfinite(pairwise_logits).all()):
            raise ValueError("S60 pairwise logits are non-finite")

        # Strong ownership boundary: composer calibration cannot train either
        # upstream decision surface.
        fused=fused_logits.detach()
        pairwise=pairwise_logits.detach()

        if alpha_override is not None:
            value=float(alpha_override)
            if value<0.0 or value>self.alpha_max:
                raise ValueError("S60 alpha override outside [0,0.35]")
            if value==0.0:
                out=fused
                if return_diagnostics:
                    zero=torch.zeros(
                        fused.shape[0],1,device=fused.device,dtype=fused.dtype
                    )
                    return out,{
                        "alpha":torch.tensor(0.0,device=fused.device,dtype=fused.dtype),
                        "fused_rms":_centered_rms(fused,epsilon=self.epsilon),
                        "pairwise_bounded_direction_max_abs":torch.tensor(
                            0.0,device=fused.device,dtype=fused.dtype
                        ),
                        "residual_max_abs":zero,
                        "residual_bound":zero,
                    }
                return out
            alpha=torch.tensor(value,device=fused.device,dtype=fused.dtype)
        else:
            alpha=self.alpha().to(device=fused.device,dtype=fused.dtype)

        pair_centered=pairwise-pairwise.mean(dim=-1,keepdim=True)
        pair_rms=_centered_rms(pairwise,epsilon=self.epsilon)
        pair_z=pair_centered/pair_rms
        bounded=torch.tanh(pair_z)

        fused_rms=_centered_rms(fused,epsilon=self.epsilon)
        residual=alpha*fused_rms*bounded
        out=fused+residual

        if not bool(torch.isfinite(out).all()):
            raise ValueError("S60 hybrid logits are non-finite")

        bound=alpha*fused_rms
        if bool((residual.abs()>bound+1e-7).any()):
            raise RuntimeError("S60 bounded residual contract violated")

        if return_diagnostics:
            return out,{
                "alpha":alpha.detach(),
                "fused_rms":fused_rms.detach(),
                "pairwise_bounded_direction_max_abs":bounded.abs().max().detach(),
                "residual_max_abs":residual.abs().amax(dim=-1,keepdim=True).detach(),
                "residual_bound":bound.detach(),
            }
        return out

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


def composer_gold_loss(
    composer:TrainCalibratedBoundedHybridComposer,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    if fused_canonical.shape!=fused_paraphrase.shape:
        raise ValueError("S60 fused view shape mismatch")
    if pairwise_canonical.shape!=pairwise_paraphrase.shape:
        raise ValueError("S60 pairwise view shape mismatch")
    if fused_canonical.shape!=pairwise_canonical.shape:
        raise ValueError("S60 decision surface shape mismatch")
    b,k=fused_canonical.shape
    if gold.shape!=(b,) or gold.dtype not in (torch.int64,torch.long):
        raise ValueError("S60 gold must be int64 [B]")
    if bool(((gold<0)|(gold>=k)).any()):
        raise ValueError("S60 gold index out of range")

    hc,dc=composer.compose(
        fused_canonical,pairwise_canonical,return_diagnostics=True
    )
    hp,dp=composer.compose(
        fused_paraphrase,pairwise_paraphrase,return_diagnostics=True
    )
    loss=0.5*(F.cross_entropy(hc,gold)+F.cross_entropy(hp,gold))
    if not bool(torch.isfinite(loss)):
        raise ValueError("S60 composer loss is non-finite")
    return loss,{
        "alpha":composer.alpha().detach(),
        "canonical_mean_fused_rms":dc["fused_rms"].mean(),
        "paraphrase_mean_fused_rms":dp["fused_rms"].mean(),
        "canonical_residual_max_abs":dc["residual_max_abs"].max(),
        "paraphrase_residual_max_abs":dp["residual_max_abs"].max(),
        "canonical_bound_max":dc["residual_bound"].max(),
        "paraphrase_bound_max":dp["residual_bound"].max(),
    }


__all__=[
    "S60_EPSILON",
    "S60_ALPHA_MAX",
    "S60_ALPHA_INITIAL",
    "S60_A_INITIAL",
    "S60_COMPOSER_PARAMETER_COUNT",
    "TrainCalibratedBoundedHybridComposer",
    "composer_gold_loss",
]
