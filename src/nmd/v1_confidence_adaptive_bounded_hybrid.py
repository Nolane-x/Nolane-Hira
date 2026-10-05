from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_train_calibrated_bounded_hybrid import (
    S60_ALPHA_MAX,
    S60_ALPHA_INITIAL,
    S60_A_INITIAL,
    S60_EPSILON,
)


S61_FEATURE_DIMENSION=4
S61_GATE_PARAMETER_COUNT=5
S61_ALPHA_MAX=S60_ALPHA_MAX
S61_ALPHA_INITIAL=S60_ALPHA_INITIAL
S61_B_INITIAL=S60_A_INITIAL
S61_EPSILON=S60_EPSILON


def _centered_rms(logits:Tensor)->Tensor:
    if logits.ndim!=2 or logits.shape[-1]<2:
        raise ValueError("S61 logits must be [B,K] with K>=2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S61 logits contain non-finite values")
    centered=logits-logits.mean(dim=-1,keepdim=True)
    return centered.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)


def adaptive_gate_features(fused_logits:Tensor,pairwise_logits:Tensor)->Tensor:
    if fused_logits.shape!=pairwise_logits.shape:
        raise ValueError("S61 fused/pairwise shape mismatch")
    if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
        raise ValueError("S61 fused/pairwise logits must be [B,K]")
    if not bool(torch.isfinite(fused_logits).all()) or not bool(torch.isfinite(pairwise_logits).all()):
        raise ValueError("S61 feature inputs contain non-finite values")

    fused=fused_logits.detach()
    pairwise=pairwise_logits.detach()

    fused_rms=_centered_rms(fused)
    pair_rms=_centered_rms(pairwise)

    fused_top2=torch.topk(fused,k=2,dim=-1).values
    pair_top2=torch.topk(pairwise,k=2,dim=-1).values
    fused_conf=torch.tanh((fused_top2[:,0:1]-fused_top2[:,1:2])/fused_rms)
    pair_conf=torch.tanh((pair_top2[:,0:1]-pair_top2[:,1:2])/pair_rms)

    agreement=torch.where(
        fused.argmax(dim=-1,keepdim=True).eq(pairwise.argmax(dim=-1,keepdim=True)),
        torch.ones_like(fused_conf),
        -torch.ones_like(fused_conf),
    )

    fused_z=(fused-fused.mean(dim=-1,keepdim=True))/fused_rms
    pair_z=(pairwise-pairwise.mean(dim=-1,keepdim=True))/pair_rms
    alignment=(fused_z*pair_z).mean(dim=-1,keepdim=True).clamp(-1.0,1.0)

    features=torch.cat([fused_conf,pair_conf,agreement,alignment],dim=-1)
    if features.shape[-1]!=S61_FEATURE_DIMENSION:
        raise RuntimeError("S61 feature dimension changed")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("S61 features are non-finite")
    return features.detach()


class ConfidenceAdaptiveBoundedHybridGate(nn.Module):
    """S61 five-parameter per-query confidence-adaptive residual gate."""

    def __init__(self,*,trainable:bool=True):
        super().__init__()
        self.w=nn.Parameter(torch.zeros(S61_FEATURE_DIMENSION,dtype=torch.float32))
        self.b=nn.Parameter(torch.tensor(S61_B_INITIAL,dtype=torch.float32))
        for p in (self.w,self.b):
            p.requires_grad_(bool(trainable))

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def alpha_from_features(self,features:Tensor)->Tensor:
        if features.ndim!=2 or features.shape[-1]!=S61_FEATURE_DIMENSION:
            raise ValueError("S61 features must be [B,4]")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("S61 features are non-finite")
        x=features.detach()
        logits=F.linear(x,self.w.unsqueeze(0),self.b.reshape(1)).squeeze(-1)
        alpha=S61_ALPHA_MAX*torch.sigmoid(logits)
        if not bool(torch.isfinite(alpha).all()):
            raise ValueError("S61 alpha is non-finite")
        return alpha.unsqueeze(-1)

    def alpha(self,fused_logits:Tensor,pairwise_logits:Tensor)->Tensor:
        return self.alpha_from_features(adaptive_gate_features(fused_logits,pairwise_logits))

    def compose(
        self,
        fused_logits:Tensor,
        pairwise_logits:Tensor,
        *,
        alpha_override:float|None=None,
        return_diagnostics:bool=False,
    ):
        if fused_logits.shape!=pairwise_logits.shape:
            raise ValueError("S61 fused/pairwise shape mismatch")
        if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
            raise ValueError("S61 fused/pairwise logits must be [B,K]")
        if not bool(torch.isfinite(fused_logits).all()) or not bool(torch.isfinite(pairwise_logits).all()):
            raise ValueError("S61 compose inputs contain non-finite values")

        fused=fused_logits.detach()
        pairwise=pairwise_logits.detach()

        if alpha_override is not None:
            value=float(alpha_override)
            if value<0.0 or value>S61_ALPHA_MAX:
                raise ValueError("S61 alpha override outside [0,0.35]")
            if value==0.0:
                if return_diagnostics:
                    zero=torch.zeros(fused.shape[0],1,device=fused.device,dtype=fused.dtype)
                    features=adaptive_gate_features(fused,pairwise)
                    return fused,{
                        "features":features,
                        "alpha":zero,
                        "fused_rms":_centered_rms(fused),
                        "pairwise_bounded_direction_max_abs":torch.tensor(0.0,device=fused.device,dtype=fused.dtype),
                        "residual_max_abs":zero,
                        "residual_bound":zero,
                    }
                return fused
            alpha=torch.full(
                (fused.shape[0],1),value,device=fused.device,dtype=fused.dtype
            )
            features=adaptive_gate_features(fused,pairwise)
        else:
            features=adaptive_gate_features(fused,pairwise)
            alpha=self.alpha_from_features(features).to(device=fused.device,dtype=fused.dtype)

        pair_centered=pairwise-pairwise.mean(dim=-1,keepdim=True)
        pair_rms=_centered_rms(pairwise)
        pair_z=pair_centered/pair_rms
        bounded=torch.tanh(pair_z)

        fused_rms=_centered_rms(fused)
        residual=alpha*fused_rms*bounded
        out=fused+residual
        if not bool(torch.isfinite(out).all()):
            raise ValueError("S61 hybrid logits are non-finite")

        bound=alpha*fused_rms
        if bool((residual.abs()>bound+1e-7).any()):
            raise RuntimeError("S61 bounded residual contract violated")

        if return_diagnostics:
            return out,{
                "features":features.detach(),
                "alpha":alpha.detach(),
                "fused_rms":fused_rms.detach(),
                "pairwise_bounded_direction_max_abs":bounded.abs().max().detach(),
                "residual_max_abs":residual.abs().amax(dim=-1,keepdim=True).detach(),
                "residual_bound":bound.detach(),
            }
        return out

    def state_dict_exact(self)->dict[str,Tensor]:
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()}

    def load_state_dict_exact(self,state:dict[str,Tensor],*,freeze:bool=False)->None:
        self.load_state_dict(state,strict=True)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)


def adaptive_gate_gold_loss(
    gate:ConfidenceAdaptiveBoundedHybridGate,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pairwise_canonical:Tensor,
    pairwise_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    if fused_canonical.shape!=fused_paraphrase.shape:
        raise ValueError("S61 fused view shape mismatch")
    if pairwise_canonical.shape!=pairwise_paraphrase.shape:
        raise ValueError("S61 pairwise view shape mismatch")
    if fused_canonical.shape!=pairwise_canonical.shape:
        raise ValueError("S61 surface shape mismatch")
    b,k=fused_canonical.shape
    if gold.shape!=(b,) or gold.dtype not in (torch.int64,torch.long):
        raise ValueError("S61 gold must be int64 [B]")
    if bool(((gold<0)|(gold>=k)).any()):
        raise ValueError("S61 gold index out of range")

    hc,dc=gate.compose(fused_canonical,pairwise_canonical,return_diagnostics=True)
    hp,dp=gate.compose(fused_paraphrase,pairwise_paraphrase,return_diagnostics=True)
    loss=0.5*(F.cross_entropy(hc,gold)+F.cross_entropy(hp,gold))
    if not bool(torch.isfinite(loss)):
        raise ValueError("S61 gate loss is non-finite")

    all_alpha=torch.cat([dc["alpha"],dp["alpha"]],dim=0)
    all_features=torch.cat([dc["features"],dp["features"]],dim=0)
    return loss,{
        "mean_alpha":all_alpha.mean().detach(),
        "min_alpha":all_alpha.min().detach(),
        "max_alpha":all_alpha.max().detach(),
        "std_alpha":all_alpha.std(unbiased=False).detach(),
        "mean_features":all_features.mean(dim=0).detach(),
        "canonical_residual_max_abs":dc["residual_max_abs"].max(),
        "paraphrase_residual_max_abs":dp["residual_max_abs"].max(),
        "canonical_bound_max":dc["residual_bound"].max(),
        "paraphrase_bound_max":dp["residual_bound"].max(),
    }


__all__=[
    "S61_FEATURE_DIMENSION",
    "S61_GATE_PARAMETER_COUNT",
    "S61_ALPHA_MAX",
    "S61_ALPHA_INITIAL",
    "S61_B_INITIAL",
    "S61_EPSILON",
    "adaptive_gate_features",
    "ConfidenceAdaptiveBoundedHybridGate",
    "adaptive_gate_gold_loss",
]
