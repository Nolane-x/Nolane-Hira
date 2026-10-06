from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_confidence_adaptive_bounded_hybrid import (
    S61_ALPHA_MAX,
    S61_ALPHA_INITIAL,
    S61_B_INITIAL,
    S61_EPSILON,
    adaptive_gate_features,
)
from .v1_contextual_reliability_gate import (
    S64_CONTEXT_DIM,
    build_fixed_context_projection,
    contextual_option_features,
)
from .v1_learned_set_reliability_gate import learned_set_option_features
from .v1_per_view_responsibility import per_view_responsibility_targets


S71_ROW_STAT_DIM=4
S71_OPTION_INPUT_DIM=12
S71_HIDDEN_DIM=5
S71_REPRESENTATION_DIM=14
S71_PARAMETER_COUNT=80
S71_ENCODER_INIT_SEED=71_164
S71_ALPHA_MAX=S61_ALPHA_MAX
S71_ALPHA_INITIAL=S61_ALPHA_INITIAL


def _centered_rms_options(x:Tensor)->Tensor:
    if x.ndim!=3:
        raise ValueError("S71 features must be [B,K,D]")
    centered=x-x.mean(dim=1,keepdim=True)
    return centered.square().mean(dim=1,keepdim=True).sqrt().clamp_min(S61_EPSILON)


def pairwise_row_statistics(pair_matrix:Tensor)->Tensor:
    if pair_matrix.ndim!=3:
        raise ValueError("S71 pair matrix must be [B,K,K]")
    b,k,k2=pair_matrix.shape
    if k!=k2 or k<2:
        raise ValueError("S71 pair matrix must be square with K>=2")
    if not bool(torch.isfinite(pair_matrix).all()):
        raise ValueError("S71 pair matrix non-finite")

    p=pair_matrix.detach()
    mask=~torch.eye(k,device=p.device,dtype=torch.bool).unsqueeze(0)
    off=p.masked_select(mask.expand(b,-1,-1)).reshape(b,k,k-1)

    mean=off.mean(dim=-1)
    maxv=off.max(dim=-1).values
    minv=off.min(dim=-1).values
    rms=off.square().mean(dim=-1).sqrt()

    stats=torch.stack([mean,maxv,minv,rms],dim=-1)
    if stats.shape!=(b,k,S71_ROW_STAT_DIM):
        raise RuntimeError("S71 row statistic shape changed")
    if not bool(torch.isfinite(stats).all()):
        raise ValueError("S71 row statistics non-finite")
    return stats.detach()


def normalized_pairwise_row_statistics(pair_matrix:Tensor)->Tensor:
    stats=pairwise_row_statistics(pair_matrix)
    centered=stats-stats.mean(dim=1,keepdim=True)
    norm=centered/_centered_rms_options(stats)
    if not bool(torch.isfinite(norm).all()):
        raise ValueError("S71 normalized row statistics non-finite")
    return norm.detach()


def pairwise_row_mean(pair_matrix:Tensor)->Tensor:
    return pairwise_row_statistics(pair_matrix)[...,0]


class MultiStatPairwiseRowComposer(nn.Module):
    """Matched S71 bounded scalar composer.

    Reference and treatment have identical 80-parameter capacity.
    Reference exposes normalized row mean and zeros the three extra row channels.
    Treatment exposes normalized mean/max/min/RMS.
    """

    def __init__(self,*,use_multistat:bool,trainable:bool=True):
        super().__init__()
        self.use_multistat=bool(use_multistat)

        self.register_buffer(
            "context_projection",
            build_fixed_context_projection(),
            persistent=True,
        )

        g=torch.Generator(device="cpu").manual_seed(S71_ENCODER_INIT_SEED)
        w_phi=torch.randn(
            S71_HIDDEN_DIM,
            S71_OPTION_INPUT_DIM,
            generator=g,
            dtype=torch.float32,
        )/math.sqrt(S71_OPTION_INPUT_DIM)
        b_phi=torch.randn(
            S71_HIDDEN_DIM,
            generator=g,
            dtype=torch.float32,
        )*0.05

        self.W_phi=nn.Parameter(w_phi)
        self.b_phi=nn.Parameter(b_phi)
        self.w_out=nn.Parameter(
            torch.zeros(S71_REPRESENTATION_DIM,dtype=torch.float32)
        )
        self.b_out=nn.Parameter(torch.tensor(S61_B_INITIAL,dtype=torch.float32))

        for p in self.parameters():
            p.requires_grad_(bool(trainable))

        if self.parameter_count!=S71_PARAMETER_COUNT:
            raise RuntimeError("S71 parameter count changed")

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def row_channels(self,pair_matrix:Tensor)->Tensor:
        row=normalized_pairwise_row_statistics(pair_matrix)
        if not self.use_multistat:
            row=torch.cat(
                [row[...,:1],torch.zeros_like(row[...,1:])],
                dim=-1,
            )
        return row.detach()

    def option_input(
        self,
        fused_logits:Tensor,
        pair_matrix:Tensor,
        context_representation:Tensor,
    )->Tensor:
        mean=pairwise_row_mean(pair_matrix)
        surface=learned_set_option_features(
            fused_logits.detach(),
            mean.detach(),
        )
        context=contextual_option_features(
            context_representation.detach(),
            self.context_projection,
        )
        row=self.row_channels(pair_matrix)

        if surface.shape[-1]!=4 or context.shape[-1]!=S64_CONTEXT_DIM:
            raise RuntimeError("S71 inherited feature shape changed")
        if surface.shape[:2]!=context.shape[:2] or surface.shape[:2]!=row.shape[:2]:
            raise ValueError("S71 option feature batch/K mismatch")

        option=torch.cat([surface,context,row],dim=-1)
        if option.shape[-1]!=S71_OPTION_INPUT_DIM:
            raise RuntimeError("S71 option input dimension changed")
        if not bool(torch.isfinite(option).all()):
            raise ValueError("S71 option input non-finite")
        return option

    def representation(
        self,
        fused_logits:Tensor,
        pair_matrix:Tensor,
        context_representation:Tensor,
    )->Tensor:
        mean=pairwise_row_mean(pair_matrix)
        option=self.option_input(
            fused_logits,
            pair_matrix,
            context_representation,
        )
        hidden=torch.tanh(F.linear(option,self.W_phi,self.b_phi))
        mean_pool=hidden.mean(dim=1)
        max_pool=hidden.max(dim=1).values
        base=adaptive_gate_features(
            fused_logits.detach(),
            mean.detach(),
        )
        rep=torch.cat([mean_pool,max_pool,base],dim=-1)
        if rep.shape[-1]!=S71_REPRESENTATION_DIM:
            raise RuntimeError("S71 representation dimension changed")
        if not bool(torch.isfinite(rep).all()):
            raise ValueError("S71 representation non-finite")
        return rep

    def pre_sigmoid_logit(
        self,
        fused_logits:Tensor,
        pair_matrix:Tensor,
        context_representation:Tensor,
    )->Tensor:
        rep=self.representation(
            fused_logits,
            pair_matrix,
            context_representation,
        )
        out=F.linear(
            rep,
            self.w_out.unsqueeze(0),
            self.b_out.reshape(1),
        ).squeeze(-1)
        if not bool(torch.isfinite(out).all()):
            raise ValueError("S71 composer logit non-finite")
        return out

    def alpha(
        self,
        fused_logits:Tensor,
        pair_matrix:Tensor,
        context_representation:Tensor,
    )->Tensor:
        alpha=S71_ALPHA_MAX*torch.sigmoid(
            self.pre_sigmoid_logit(
                fused_logits,
                pair_matrix,
                context_representation,
            )
        )
        if not bool(torch.isfinite(alpha).all()):
            raise ValueError("S71 alpha non-finite")
        return alpha.unsqueeze(-1)

    def compose(
        self,
        fused_logits:Tensor,
        pair_matrix:Tensor,
        context_representation:Tensor,
        *,
        alpha_override:float|None=None,
        return_diagnostics:bool=False,
    ):
        if fused_logits.ndim!=2 or fused_logits.shape[-1]<2:
            raise ValueError("S71 fused logits must be [B,K], K>=2")
        if pair_matrix.shape[:2]!=fused_logits.shape or pair_matrix.shape[-1]!=fused_logits.shape[-1]:
            raise ValueError("S71 pair matrix shape mismatch")
        if context_representation.shape[:2]!=fused_logits.shape:
            raise ValueError("S71 context shape mismatch")

        fused=fused_logits.detach()
        pairwise=pairwise_row_mean(pair_matrix)
        context=context_representation.detach()

        centered_f=fused-fused.mean(dim=-1,keepdim=True)
        fused_rms=centered_f.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)

        if alpha_override is not None:
            value=float(alpha_override)
            if value<0.0 or value>S71_ALPHA_MAX:
                raise ValueError("S71 alpha override outside [0,0.35]")
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
            alpha=self.alpha(fused,pair_matrix,context).to(
                device=fused.device,
                dtype=fused.dtype,
            )

        pair_centered=pairwise-pairwise.mean(dim=-1,keepdim=True)
        pair_rms=pair_centered.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(S61_EPSILON)
        bounded=torch.tanh(pair_centered/pair_rms)
        residual=alpha*fused_rms*bounded
        out=fused+residual
        bound=alpha*fused_rms

        if bool((residual.abs()>bound+1e-7).any()):
            raise RuntimeError("S71 bounded residual contract violated")
        if not bool(torch.isfinite(out).all()):
            raise ValueError("S71 output non-finite")

        if return_diagnostics:
            return out,{
                "alpha":alpha.detach(),
                "fused_rms":fused_rms.detach(),
                "residual_max_abs":residual.abs().amax(dim=-1,keepdim=True).detach(),
                "residual_bound":bound.detach(),
            }
        return out

    def parameter_state_dict_exact(self)->dict[str,Tensor]:
        return {k:v.detach().cpu().clone() for k,v in self.named_parameters()}

    def state_dict_exact(self)->dict[str,Tensor]:
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()}

    def load_state_dict_exact(self,state:dict[str,Tensor],*,freeze:bool=False)->None:
        self.load_state_dict(state,strict=True)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)


def multistat_responsibility_loss(
    composer:MultiStatPairwiseRowComposer,
    fused_canonical:Tensor,
    fused_paraphrase:Tensor,
    pair_matrix_canonical:Tensor,
    pair_matrix_paraphrase:Tensor,
    context_canonical:Tensor,
    context_paraphrase:Tensor,
    gold:Tensor,
)->tuple[Tensor,dict[str,Tensor]]:
    mean_c=pairwise_row_mean(pair_matrix_canonical)
    mean_p=pairwise_row_mean(pair_matrix_paraphrase)
    yc,yp,target_diag=per_view_responsibility_targets(
        fused_canonical,
        fused_paraphrase,
        mean_c,
        mean_p,
        gold,
    )

    zc=composer.pre_sigmoid_logit(
        fused_canonical.detach(),
        pair_matrix_canonical.detach(),
        context_canonical.detach(),
    )
    zp=composer.pre_sigmoid_logit(
        fused_paraphrase.detach(),
        pair_matrix_paraphrase.detach(),
        context_paraphrase.detach(),
    )
    if zc.shape!=yc.shape or zp.shape!=yp.shape:
        raise RuntimeError("S71 logit/target shape changed")

    loss=0.5*(
        F.binary_cross_entropy_with_logits(zc,yc)
        +F.binary_cross_entropy_with_logits(zp,yp)
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("S71 responsibility loss non-finite")

    with torch.no_grad():
        ac=composer.alpha(
            fused_canonical,
            pair_matrix_canonical,
            context_canonical,
        ).reshape(-1)
        ap=composer.alpha(
            fused_paraphrase,
            pair_matrix_paraphrase,
            context_paraphrase,
        ).reshape(-1)
        all_alpha=torch.cat([ac,ap])

    return loss,{
        **target_diag,
        "canonical_target_positive_count":yc.sum().detach(),
        "paraphrase_target_positive_count":yp.sum().detach(),
        "mean_alpha":all_alpha.mean().detach(),
        "min_alpha":all_alpha.min().detach(),
        "max_alpha":all_alpha.max().detach(),
        "std_alpha":all_alpha.std(unbiased=False).detach(),
    }


__all__=[
    "S71_ROW_STAT_DIM",
    "S71_OPTION_INPUT_DIM",
    "S71_HIDDEN_DIM",
    "S71_REPRESENTATION_DIM",
    "S71_PARAMETER_COUNT",
    "S71_ENCODER_INIT_SEED",
    "S71_ALPHA_MAX",
    "S71_ALPHA_INITIAL",
    "pairwise_row_statistics",
    "normalized_pairwise_row_statistics",
    "pairwise_row_mean",
    "MultiStatPairwiseRowComposer",
    "multistat_responsibility_loss",
]
