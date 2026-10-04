from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork


class TokenLevelQueryOptionLateInteraction:
    """S53 zero-parameter option-conditioned query-token context."""

    def __init__(
        self,
        *,
        temperature: float = 0.10,
        native_dimension: int = 256,
        norm_epsilon: float = 1e-12,
    ):
        if temperature != 0.10:
            raise ValueError("S53 temperature is frozen at 0.10")
        if native_dimension != 256:
            raise ValueError("S53 native dimension is frozen at 256")
        if norm_epsilon != 1e-12:
            raise ValueError("S53 norm epsilon is frozen at 1e-12")
        self.temperature=float(temperature)
        self.native_dimension=int(native_dimension)
        self.norm_epsilon=float(norm_epsilon)

    @property
    def parameter_count(self)->int:
        return 0

    def __call__(
        self,
        *,
        option_identity: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        return_weights: bool = False,
    ):
        if option_identity.ndim!=3 or option_identity.shape[-1]!=self.native_dimension:
            raise ValueError("S53 option_identity must be [B,K,256]")
        if question_tokens.ndim!=3 or question_tokens.shape[-1]!=self.native_dimension:
            raise ValueError("S53 question_tokens must be [B,Q,256]")
        if question_mask.shape!=question_tokens.shape[:2] or question_mask.dtype!=torch.bool:
            raise ValueError("S53 question_mask mismatch")
        if option_identity.shape[0]!=question_tokens.shape[0]:
            raise ValueError("S53 batch mismatch")
        if option_identity.shape[1]<2:
            raise ValueError("S53 requires K >= 2")
        if bool((question_mask.sum(-1)<1).any()):
            raise ValueError("S53 requires at least one active query token")
        if not bool(torch.isfinite(option_identity).all()):
            raise ValueError("S53 option identity is non-finite")
        if not bool(torch.isfinite(question_tokens).all()):
            raise ValueError("S53 question tokens are non-finite")

        identity=F.normalize(
            option_identity.detach(),
            dim=-1,
            eps=self.norm_epsilon,
        )
        tokens=F.normalize(
            question_tokens.detach(),
            dim=-1,
            eps=self.norm_epsilon,
        )
        score=torch.einsum("bkd,bqd->bkq",identity,tokens)
        score=score/self.temperature
        mask=question_mask[:,None,:]
        score=score.masked_fill(~mask,-1e4)
        weight=torch.softmax(score,dim=-1)
        weight=weight*mask.to(weight.dtype)
        weight=weight/weight.sum(dim=-1,keepdim=True).clamp_min(self.norm_epsilon)
        context=torch.einsum("bkq,bqd->bkd",weight,tokens)
        context=F.normalize(context,dim=-1,eps=self.norm_epsilon)

        if not bool(torch.isfinite(context).all()):
            raise ValueError("S53 token context is non-finite")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("S53 token weights are non-finite")

        if return_weights:
            return context,weight
        return context


class TokenLateInteractionQueryFreePrivateCorrectionFork(
    QueryFreeIdentityPrivateCorrectionFork
):
    """S53 query-free option identity + per-option token-level query context."""

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
        )
        self.token_interaction=TokenLevelQueryOptionLateInteraction(
            temperature=token_temperature,
            native_dimension=native_dimension,
            norm_epsilon=1e-12,
        )

    @property
    def late_interaction_parameter_count(self)->int:
        return self.token_interaction.parameter_count

    def option_conditioned_query_context(
        self,
        *,
        option_identity: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        return_weights: bool = False,
    ):
        return self.token_interaction(
            option_identity=option_identity,
            question_tokens=question_tokens,
            question_mask=question_mask,
            return_weights=return_weights,
        )

    def correction_logits_from_identity_context(
        self,
        *,
        native_logits: Tensor,
        identity: Tensor,
        query_context: Tensor,
    )->Tensor:
        if native_logits.shape!=identity.shape[:2]:
            raise ValueError("S53 native logit/identity shape mismatch")
        if query_context.shape!=identity.shape:
            raise ValueError("S53 query context must match [B,K,256] identity")
        if identity.ndim!=3 or identity.shape[-1]!=self.native_dimension:
            raise ValueError("S53 identity dimension changed")

        base=identity.detach()
        context=query_context.detach()

        x=torch.cat([base,context],dim=-1)
        a=self.adapter_a.to(dtype=base.dtype)
        b=self.adapter_b.to(dtype=base.dtype)
        hidden=F.gelu(F.linear(x,a,bias=None))
        residual=F.linear(hidden,b,bias=None)
        private=F.normalize(
            base+residual,
            dim=-1,
            eps=self.private_norm_epsilon,
        )

        weight=self.bilinear_weight.to(dtype=private.dtype)
        correction=torch.einsum("bkd,de,bke->bk",private,weight,context)
        logits=native_logits.detach()+self.residual_scale*correction

        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S53 corrected logits are non-finite")
        return logits

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
        context,weights=self.option_conditioned_query_context(
            option_identity=identity,
            question_tokens=question_tokens,
            question_mask=question_mask,
            return_weights=True,
        )
        logits=self.correction_logits_from_identity_context(
            native_logits=native_logits,
            identity=identity,
            query_context=context,
        )
        if return_context:
            return logits,identity,context,weights
        return logits,identity


__all__=[
    "TokenLevelQueryOptionLateInteraction",
    "TokenLateInteractionQueryFreePrivateCorrectionFork",
]
