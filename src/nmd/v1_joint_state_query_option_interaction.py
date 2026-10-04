from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_token_query_option_late_interaction import (
    TokenLateInteractionQueryFreePrivateCorrectionFork,
)


class JointStateQueryOptionLateInteraction:
    """S54 zero-parameter triadic state-query-option context."""

    def __init__(
        self,
        *,
        temperature: float = 0.10,
        native_dimension: int = 256,
        norm_epsilon: float = 1e-12,
    ):
        if temperature != 0.10:
            raise ValueError("S54 temperature is frozen at 0.10")
        if native_dimension != 256:
            raise ValueError("S54 native dimension is frozen at 256")
        if norm_epsilon != 1e-12:
            raise ValueError("S54 norm epsilon is frozen at 1e-12")
        self.temperature=float(temperature)
        self.native_dimension=int(native_dimension)
        self.norm_epsilon=float(norm_epsilon)

    @property
    def parameter_count(self)->int:
        return 0

    def __call__(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        return_diagnostics: bool = False,
    ):
        if state_tokens.ndim!=3 or state_tokens.shape[-1]!=self.native_dimension:
            raise ValueError("S54 state_tokens must be [B,S,256]")
        if state_mask.shape!=state_tokens.shape[:2] or state_mask.dtype!=torch.bool:
            raise ValueError("S54 state_mask mismatch")
        if option_view_tokens.ndim!=5 or option_view_tokens.shape[-1]!=self.native_dimension:
            raise ValueError("S54 option_view_tokens must be [B,K,V,T,256]")
        b,k,v,t,d=option_view_tokens.shape
        if k<2:
            raise ValueError("S54 requires K >= 2")
        if option_view_token_mask.shape!=(b,k,v,t) or option_view_token_mask.dtype!=torch.bool:
            raise ValueError("S54 option token mask mismatch")
        if option_view_mask.shape!=(b,k,v) or option_view_mask.dtype!=torch.bool:
            raise ValueError("S54 option view mask mismatch")
        if question_tokens.ndim!=3 or question_tokens.shape[0]!=b or question_tokens.shape[-1]!=d:
            raise ValueError("S54 question_tokens must be [B,Q,256]")
        if question_mask.shape!=question_tokens.shape[:2] or question_mask.dtype!=torch.bool:
            raise ValueError("S54 question_mask mismatch")
        if state_tokens.shape[0]!=b:
            raise ValueError("S54 state batch mismatch")
        if bool((state_mask.sum(-1)<1).any()):
            raise ValueError("S54 requires at least one active state token")
        if bool((question_mask.sum(-1)<1).any()):
            raise ValueError("S54 requires at least one active query token")

        option_mask=option_view_token_mask & option_view_mask.unsqueeze(-1)
        if bool((option_mask.reshape(b,k,-1).sum(-1)<1).any()):
            raise ValueError("S54 requires at least one active token per option")

        for name,value in (
            ("state_tokens",state_tokens),
            ("option_view_tokens",option_view_tokens),
            ("question_tokens",question_tokens),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"S54 {name} contains non-finite values")

        state=F.normalize(state_tokens.detach(),dim=-1,eps=self.norm_epsilon)
        option=F.normalize(option_view_tokens.detach(),dim=-1,eps=self.norm_epsilon)
        query=F.normalize(question_tokens.detach(),dim=-1,eps=self.norm_epsilon)

        # State support per query token.
        state_score=torch.einsum("bqd,bsd->bqs",query,state)
        state_score=state_score.masked_fill(~state_mask[:,None,:],-1e4)
        state_support=state_score.max(dim=-1).values  # [B,Q]

        # Option support per query token and option. Views/tokens are flattened,
        # but only active view-token pairs participate in the max.
        option_flat=option.reshape(b,k,v*t,d)
        option_mask_flat=option_mask.reshape(b,k,v*t)
        option_score=torch.einsum("bqd,bknd->bkqn",query,option_flat)
        option_score=option_score.masked_fill(
            ~option_mask_flat[:,:,None,:],
            -1e4,
        )
        option_support=option_score.max(dim=-1).values  # [B,K,Q]

        joint=(state_support[:,None,:]+option_support)/self.temperature
        qmask=question_mask[:,None,:]
        joint=joint.masked_fill(~qmask,-1e4)
        weight=torch.softmax(joint,dim=-1)
        weight=weight*qmask.to(weight.dtype)
        weight=weight/weight.sum(dim=-1,keepdim=True).clamp_min(self.norm_epsilon)

        context=torch.einsum("bkq,bqd->bkd",weight,query)
        context=F.normalize(context,dim=-1,eps=self.norm_epsilon)

        if not bool(torch.isfinite(context).all()):
            raise ValueError("S54 joint context is non-finite")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("S54 joint weights are non-finite")
        if not bool(torch.isfinite(state_support).all()):
            raise ValueError("S54 state support is non-finite")
        if not bool(torch.isfinite(option_support).all()):
            raise ValueError("S54 option support is non-finite")

        if return_diagnostics:
            return context,weight,state_support,option_support
        return context


class JointStateQueryOptionPrivateCorrectionFork(
    TokenLateInteractionQueryFreePrivateCorrectionFork
):
    """S54 joint state-query-option context with the exact S53 correction shell."""

    def __init__(
        self,
        *,
        native_dimension: int = 256,
        hidden_dimension: int = 64,
        query_norm_epsilon: float = 1e-12,
        private_norm_epsilon: float = 1e-12,
        residual_scale: float = 1.0,
        adapter_seed: int = 65044,
        train_correction: bool = False,
        pair_temperature: float = 0.10,
        token_temperature: float = 0.10,
    ):
        super().__init__(
            native_dimension=native_dimension,
            hidden_dimension=hidden_dimension,
            query_norm_epsilon=query_norm_epsilon,
            private_norm_epsilon=private_norm_epsilon,
            residual_scale=residual_scale,
            adapter_seed=adapter_seed,
            train_correction=train_correction,
            pair_temperature=pair_temperature,
            token_temperature=token_temperature,
        )
        self.joint_interaction=JointStateQueryOptionLateInteraction(
            temperature=token_temperature,
            native_dimension=native_dimension,
            norm_epsilon=1e-12,
        )

    @property
    def joint_interaction_parameter_count(self)->int:
        return self.joint_interaction.parameter_count

    def joint_query_context(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        return_diagnostics: bool = False,
    ):
        return self.joint_interaction(
            state_tokens=state_tokens,
            state_mask=state_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            return_diagnostics=return_diagnostics,
        )

    def correction_logits_from_state_option(
        self,
        *,
        native_logits: Tensor,
        state_tokens: Tensor,
        state_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        return_context: bool = False,
    ):
        identity=self.identity_signatures(
            state_tokens=state_tokens,
            state_mask=state_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )
        context,weights,state_support,option_support=self.joint_query_context(
            state_tokens=state_tokens,
            state_mask=state_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            return_diagnostics=True,
        )
        logits=self.correction_logits_from_identity_context(
            native_logits=native_logits,
            identity=identity,
            query_context=context,
        )
        if return_context:
            return (
                logits,
                identity,
                context,
                weights,
                state_support,
                option_support,
            )
        return logits,identity


__all__=[
    "JointStateQueryOptionLateInteraction",
    "JointStateQueryOptionPrivateCorrectionFork",
]
