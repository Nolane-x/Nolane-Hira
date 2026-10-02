from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_relation_canonicalization import CanonicalRelationDiagnostics


@dataclass(frozen=True)
class EntropicTransportDiagnostics:
    base: CanonicalRelationDiagnostics
    max_row_marginal_residual: Tensor
    max_column_marginal_residual: Tensor
    mean_transport_expected_similarity: Tensor

    def to_dict(self) -> dict[str, float]:
        out=self.base.to_dict()
        out.update({
            "max_row_marginal_residual":float(
                self.max_row_marginal_residual.detach().cpu()
            ),
            "max_column_marginal_residual":float(
                self.max_column_marginal_residual.detach().cpu()
            ),
            "mean_transport_expected_similarity":float(
                self.mean_transport_expected_similarity.detach().cpu()
            ),
        })
        return out


class QueryConditionedEntropicRelationTransport(nn.Module):
    """S34 zero-parameter query-conditioned many-to-many relation transport."""

    def __init__(
        self,
        *,
        state_relevance_temperature: float=0.10,
        option_relevance_temperature: float=0.10,
        kernel_temperature: float=0.10,
        logit_temperature: float=0.10,
        sinkhorn_iterations: int=8,
        epsilon: float=1e-12,
    ):
        super().__init__()
        for name,value in (
            ("state relevance",state_relevance_temperature),
            ("option relevance",option_relevance_temperature),
            ("kernel",kernel_temperature),
            ("logit",logit_temperature),
            ("epsilon",epsilon),
        ):
            if value<=0:
                raise ValueError(f"S34 {name} must be positive")
        if sinkhorn_iterations<1:
            raise ValueError("S34 Sinkhorn iterations must be >=1")
        self.state_relevance_temperature=float(state_relevance_temperature)
        self.option_relevance_temperature=float(option_relevance_temperature)
        self.kernel_temperature=float(kernel_temperature)
        self.logit_temperature=float(logit_temperature)
        self.sinkhorn_iterations=int(sinkhorn_iterations)
        self.epsilon=float(epsilon)

    @property
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _project(projection:nn.Linear,x:Tensor)->Tensor:
        if not isinstance(projection,nn.Linear):
            raise TypeError("S34 requires a shared nn.Linear projection")
        if projection.bias is not None:
            raise ValueError("S34 requires bias-free shared projection")
        if x.shape[-1]!=projection.in_features:
            raise ValueError("S34 projection input mismatch")
        y=F.normalize(projection(x),dim=-1)
        if not bool(torch.isfinite(y).all()):
            raise ValueError("S34 projection produced non-finite values")
        return y

    @staticmethod
    def _masked_softmax(scores:Tensor,mask:Tensor,temperature:float)->Tensor:
        masked=scores.masked_fill(~mask,-1e4)
        weights=torch.softmax(masked/temperature,dim=-1)
        weights=weights*mask.to(weights.dtype)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        return weights

    @staticmethod
    def _normalized_entropy(weights:Tensor,mask:Tensor)->Tensor:
        eps=torch.finfo(weights.dtype).eps
        count=mask.sum(-1).to(weights.dtype)
        den=torch.where(count>1,count.log(),torch.ones_like(count))
        entropy=-(weights.clamp_min(eps).log()*weights).sum(-1)
        return entropy/den

    def _validate(
        self,
        *,
        projection:nn.Linear,
        state_tokens:Tensor,
        state_mask:Tensor,
        question_tokens:Tensor,
        question_mask:Tensor,
        option_view_tokens:Tensor,
        option_view_token_mask:Tensor,
        option_view_mask:Tensor,
    )->None:
        if state_tokens.ndim!=3:
            raise ValueError("S34 state_tokens must be [B,S,D]")
        if question_tokens.ndim!=3:
            raise ValueError("S34 question_tokens must be [B,Q,D]")
        if option_view_tokens.ndim!=5:
            raise ValueError("S34 option_view_tokens must be [B,K,V,T,D]")
        if (
            state_tokens.shape[-1]!=projection.in_features
            or question_tokens.shape[-1]!=projection.in_features
            or option_view_tokens.shape[-1]!=projection.in_features
        ):
            raise ValueError("S34 d_model mismatch")
        if state_mask.shape!=state_tokens.shape[:2] or state_mask.dtype!=torch.bool:
            raise ValueError("S34 state_mask mismatch")
        if question_mask.shape!=question_tokens.shape[:2] or question_mask.dtype!=torch.bool:
            raise ValueError("S34 question_mask mismatch")
        if (
            option_view_token_mask.shape!=option_view_tokens.shape[:4]
            or option_view_token_mask.dtype!=torch.bool
        ):
            raise ValueError("S34 option token mask mismatch")
        if option_view_mask.shape!=option_view_tokens.shape[:3] or option_view_mask.dtype!=torch.bool:
            raise ValueError("S34 option view mask mismatch")
        if not (
            state_tokens.shape[0]
            ==question_tokens.shape[0]
            ==option_view_tokens.shape[0]
        ):
            raise ValueError("S34 batch mismatch")
        if option_view_tokens.shape[1]<2:
            raise ValueError("S34 requires K>=2")
        if bool((state_mask.sum(-1)<1).any()):
            raise ValueError("S34 requires state content")
        if bool((question_mask.sum(-1)<1).any()):
            raise ValueError("S34 requires question content")
        if bool((option_view_mask.sum(-1)<1).any()):
            raise ValueError("S34 requires active option views")
        if bool(((option_view_token_mask.sum(-1)<1)&option_view_mask).any()):
            raise ValueError("S34 active option views require content tokens")

    def _sinkhorn(
        self,
        *,
        similarity:Tensor,
        state_marginal:Tensor,
        option_marginal:Tensor,
        state_mask:Tensor,
        option_mask:Tensor,
    )->Tensor:
        # similarity [B,K,V,S,T]
        log_kernel=similarity/self.kernel_temperature

        state_valid=state_mask[:,None,None,:]
        option_valid=option_mask
        neg_inf=torch.tensor(float("-inf"),device=similarity.device,dtype=similarity.dtype)

        log_a=torch.where(
            state_valid,
            state_marginal[:,None,None,:].clamp_min(self.epsilon).log(),
            neg_inf,
        )
        log_b=torch.where(
            option_valid,
            option_marginal.clamp_min(self.epsilon).log(),
            neg_inf,
        )

        # Inactive semantic views receive a one-token numerical mask and are
        # removed after transport. Active views always have >=1 valid token.
        log_v=torch.where(option_valid,torch.zeros_like(log_b),neg_inf)
        log_u=torch.where(state_valid,torch.zeros_like(log_a),neg_inf)

        for _ in range(self.sinkhorn_iterations):
            row_norm=torch.logsumexp(
                log_kernel+log_v[...,None,:],
                dim=-1,
            )
            log_u=torch.where(state_valid,log_a-row_norm,neg_inf)
            col_norm=torch.logsumexp(
                log_kernel+log_u[..., :,None],
                dim=-2,
            )
            log_v=torch.where(option_valid,log_b-col_norm,neg_inf)

        plan=torch.exp(
            log_kernel+log_u[..., :,None]+log_v[...,None,:]
        )
        plan=plan*(
            state_valid[..., :,None]&option_valid[...,None,:]
        ).to(plan.dtype)
        return plan

    def forward_with_components(
        self,
        *,
        projection:nn.Linear,
        state_tokens:Tensor,
        state_mask:Tensor,
        question_tokens:Tensor,
        question_mask:Tensor,
        option_view_tokens:Tensor,
        option_view_token_mask:Tensor,
        option_view_mask:Tensor,
    )->tuple[Tensor,Tensor,EntropicTransportDiagnostics,dict[str,Tensor]]:
        self._validate(
            projection=projection,
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        state=self._project(projection,state_tokens)
        question=self._project(projection,question_tokens)
        options=self._project(projection,option_view_tokens)

        q_state=torch.einsum("bqd,bsd->bqs",question,state)
        q_state=q_state.masked_fill(~question_mask[...,None],-1e4)
        state_relevance=q_state.amax(dim=1)
        state_marginal=self._masked_softmax(
            state_relevance,
            state_mask,
            self.state_relevance_temperature,
        )

        q_option=torch.einsum("bqd,bkvtd->bkvqt",question,options)
        q_option=q_option.masked_fill(
            ~question_mask[:,None,None,:,None],
            -1e4,
        )
        option_relevance=q_option.amax(dim=-2)

        effective_option_mask=(
            option_view_token_mask
            & option_view_mask[...,None]
        )
        safe_option_mask=effective_option_mask.clone()
        inactive=~option_view_mask
        if bool(inactive.any()):
            # Safe numerical token for inactive views. It is zeroed before
            # scoring/signature aggregation and never becomes semantic mass.
            safe_option_mask[...,0]=safe_option_mask[...,0]|inactive

        option_marginal=self._masked_softmax(
            option_relevance,
            safe_option_mask,
            self.option_relevance_temperature,
        )

        similarity=torch.einsum("bsd,bkvtd->bkvst",state,options)
        plan=self._sinkhorn(
            similarity=similarity,
            state_marginal=state_marginal,
            option_marginal=option_marginal,
            state_mask=state_mask,
            option_mask=safe_option_mask,
        )

        active=option_view_mask.to(plan.dtype)
        plan=plan*active[...,None,None]

        expected_similarity=(plan*similarity).sum(dim=(-1,-2))
        view_logits=expected_similarity/self.logit_temperature
        view_logits=view_logits*active
        view_count=option_view_mask.sum(-1).clamp_min(1).to(view_logits.dtype)
        logits=view_logits.sum(-1)/view_count

        pair_delta=F.normalize(
            options[:,:,:,None,:,:]-state[:,None,None,:,None,:],
            dim=-1,
        )
        view_signature=F.normalize(
            (plan[...,None]*pair_delta).sum(dim=(-3,-2)),
            dim=-1,
        )
        view_signature=view_signature*active[...,None]
        signatures=view_signature.sum(-2)/active.sum(-1,keepdim=True).clamp_min(1.0)
        signatures=F.normalize(signatures,dim=-1)

        state_target=state_marginal[:,None,None,:]*active[...,None]
        option_target=option_marginal*active[...,None]
        row_sum=plan.sum(-1)
        col_sum=plan.sum(-2)
        row_residual=(row_sum-state_target).abs()*active[...,None]
        col_residual=(col_sum-option_target).abs()*active[...,None]

        state_entropy=self._normalized_entropy(state_marginal,state_mask)
        option_entropy=self._normalized_entropy(option_marginal,safe_option_mask)
        option_entropy_mean=(option_entropy*active).sum()/active.sum().clamp_min(1.0)
        option_max_mean=(option_marginal.max(-1).values*active).sum()/active.sum().clamp_min(1.0)

        pair_mask=(
            state_mask[:,None,None,:,None]
            & safe_option_mask[...,None,:]
        )
        pair_count=pair_mask.sum(dim=(-1,-2)).to(plan.dtype)
        pair_den=torch.where(
            pair_count>1,
            pair_count.log(),
            torch.ones_like(pair_count),
        )
        eps=torch.finfo(plan.dtype).eps
        pair_entropy=-(plan.clamp_min(eps).log()*plan).sum(dim=(-1,-2))/pair_den
        pair_entropy_mean=(pair_entropy*active).sum()/active.sum().clamp_min(1.0)

        diagnostics=EntropicTransportDiagnostics(
            base=CanonicalRelationDiagnostics(
                state_role_normalized_entropy=state_entropy.mean(),
                state_role_max_weight=state_marginal.max(-1).values.mean(),
                option_role_normalized_entropy=option_entropy_mean,
                option_role_max_weight=option_max_mean,
                mean_pair_entropy=pair_entropy_mean,
                mean_best_pair_score=(
                    expected_similarity*active
                ).sum()/active.sum().clamp_min(1.0),
            ),
            max_row_marginal_residual=row_residual.max(),
            max_column_marginal_residual=col_residual.max(),
            mean_transport_expected_similarity=(
                expected_similarity*active
            ).sum()/active.sum().clamp_min(1.0),
        )

        if not bool(
            torch.isfinite(logits).all()
            and torch.isfinite(signatures).all()
            and torch.isfinite(plan).all()
        ):
            raise ValueError("S34 entropic transport produced non-finite values")

        components={
            "state_marginal":state_marginal,
            "option_marginal":option_marginal,
            "transport_plan":plan,
            "expected_similarity":expected_similarity,
            "row_marginal_residual":row_residual,
            "column_marginal_residual":col_residual,
        }
        return logits,signatures,diagnostics,components

    def forward(
        self,
        *,
        projection:nn.Linear,
        state_tokens:Tensor,
        state_mask:Tensor,
        question_tokens:Tensor,
        question_mask:Tensor,
        option_view_tokens:Tensor,
        option_view_token_mask:Tensor,
        option_view_mask:Tensor,
    )->tuple[Tensor,Tensor,EntropicTransportDiagnostics]:
        logits,signatures,diagnostics,_=self.forward_with_components(
            projection=projection,
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )
        return logits,signatures,diagnostics


__all__=[
    "EntropicTransportDiagnostics",
    "QueryConditionedEntropicRelationTransport",
]
