from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_joint_state_query_option_interaction import JointStateQueryOptionLateInteraction
from .v1_token_query_option_late_interaction import (
    TokenLateInteractionQueryFreePrivateCorrectionFork,
)


class LearnedJointRelationTransform(nn.Module):
    """S55 matched learned relation transform over q, joint-state evidence and option identity."""

    def __init__(
        self,
        *,
        native_dimension: int = 256,
        hidden_dimension: int = 64,
        seed: int = 75555,
        norm_epsilon: float = 1e-12,
    ):
        super().__init__()
        if native_dimension != 256:
            raise ValueError("S55 native dimension is frozen at 256")
        if hidden_dimension != 64:
            raise ValueError("S55 hidden dimension is frozen at 64")
        if seed != 75555:
            raise ValueError("S55 transform seed is frozen at 75555")
        if norm_epsilon != 1e-12:
            raise ValueError("S55 norm epsilon is frozen at 1e-12")

        self.native_dimension=int(native_dimension)
        self.hidden_dimension=int(hidden_dimension)
        self.seed=int(seed)
        self.norm_epsilon=float(norm_epsilon)

        g=torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        a=torch.randn(
            self.hidden_dimension,
            3*self.native_dimension,
            generator=g,
            dtype=torch.float32,
        )*0.02
        b=torch.zeros(
            self.native_dimension,
            self.hidden_dimension,
            dtype=torch.float32,
        )
        self.adapter_a=nn.Parameter(a)
        self.adapter_b=nn.Parameter(b)

    @property
    def parameter_count(self)->int:
        return int(self.adapter_a.numel()+self.adapter_b.numel())

    def parameters_list(self)->list[nn.Parameter]:
        return [self.adapter_a,self.adapter_b]

    def forward(
        self,
        *,
        query_context: Tensor,
        state_joint_context: Tensor,
        option_identity: Tensor,
    )->Tensor:
        if query_context.ndim!=3 or query_context.shape[-1]!=self.native_dimension:
            raise ValueError("S55 query_context must be [B,K,256]")
        if state_joint_context.shape!=query_context.shape:
            raise ValueError("S55 state_joint_context shape mismatch")
        if option_identity.shape!=query_context.shape:
            raise ValueError("S55 option_identity shape mismatch")
        if query_context.shape[1]<2:
            raise ValueError("S55 requires K >= 2")

        for name,value in (
            ("query_context",query_context),
            ("state_joint_context",state_joint_context),
            ("option_identity",option_identity),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"S55 {name} contains non-finite values")

        q=F.normalize(query_context,dim=-1,eps=self.norm_epsilon)
        s=F.normalize(state_joint_context,dim=-1,eps=self.norm_epsilon)
        o=F.normalize(option_identity,dim=-1,eps=self.norm_epsilon)

        x=torch.cat([q,s,o],dim=-1)
        a=self.adapter_a.to(dtype=x.dtype)
        b=self.adapter_b.to(dtype=x.dtype)
        hidden=F.gelu(F.linear(x,a,bias=None))
        residual=F.linear(hidden,b,bias=None)
        code=F.normalize(q+residual,dim=-1,eps=self.norm_epsilon)

        if not bool(torch.isfinite(code).all()):
            raise ValueError("S55 learned relation code is non-finite")
        return code


class LearnedJointRelationPrivateCorrectionFork(
    TokenLateInteractionQueryFreePrivateCorrectionFork
):
    """S55 matched-capacity learned relation operator.

    Reference uses use_state_joint_context=False and receives an explicit
    all-zero state channel. Treatment uses the exact same architecture and
    initialization with use_state_joint_context=True.
    """

    def __init__(
        self,
        *,
        use_state_joint_context: bool,
        native_dimension: int = 256,
        hidden_dimension: int = 64,
        query_norm_epsilon: float = 1e-12,
        private_norm_epsilon: float = 1e-12,
        residual_scale: float = 1.0,
        adapter_seed: int = 65044,
        train_correction: bool = False,
        pair_temperature: float = 0.10,
        token_temperature: float = 0.10,
        learned_joint_seed: int = 75555,
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
        self.use_state_joint_context=bool(use_state_joint_context)
        self.state_joint_interaction=JointStateQueryOptionLateInteraction(
            temperature=token_temperature,
            native_dimension=native_dimension,
            norm_epsilon=1e-12,
        )
        self.learned_joint_transform=LearnedJointRelationTransform(
            native_dimension=native_dimension,
            hidden_dimension=64,
            seed=learned_joint_seed,
            norm_epsilon=1e-12,
        )
        if not train_correction:
            for p in self.learned_joint_transform.parameters():
                p.requires_grad_(False)

    @property
    def learned_joint_parameter_count(self)->int:
        return self.learned_joint_transform.parameter_count

    @property
    def private_trainable_parameter_count(self)->int:
        return self.correction_parameter_count+self.learned_joint_parameter_count

    def learned_joint_parameters(self)->list[nn.Parameter]:
        return self.learned_joint_transform.parameters_list()

    def private_parameters(self)->list[nn.Parameter]:
        return self.correction_parameters()+self.learned_joint_parameters()

    def learned_relation_code(
        self,
        *,
        identity: Tensor,
        state_tokens: Tensor,
        state_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        return_contexts: bool = False,
    ):
        query_context,query_weights=self.option_conditioned_query_context(
            option_identity=identity,
            question_tokens=question_tokens,
            question_mask=question_mask,
            return_weights=True,
        )

        if self.use_state_joint_context:
            state_joint_context,joint_weights,state_support,option_support=(
                self.state_joint_interaction(
                    state_tokens=state_tokens,
                    state_mask=state_mask,
                    option_view_tokens=option_view_tokens,
                    option_view_token_mask=option_view_token_mask,
                    option_view_mask=option_view_mask,
                    question_tokens=question_tokens,
                    question_mask=question_mask,
                    return_diagnostics=True,
                )
            )
        else:
            state_joint_context=torch.zeros_like(query_context)
            joint_weights=None
            state_support=None
            option_support=None

        code=self.learned_joint_transform(
            query_context=query_context,
            state_joint_context=state_joint_context,
            option_identity=identity,
        )
        if return_contexts:
            return (
                code,
                query_context,
                state_joint_context,
                query_weights,
                joint_weights,
                state_support,
                option_support,
            )
        return code

    def correction_logits_from_identity_relation_code(
        self,
        *,
        native_logits: Tensor,
        identity: Tensor,
        relation_code: Tensor,
    )->Tensor:
        if native_logits.shape!=identity.shape[:2]:
            raise ValueError("S55 native logit/identity shape mismatch")
        if relation_code.shape!=identity.shape:
            raise ValueError("S55 relation code must match [B,K,256] identity")
        if identity.ndim!=3 or identity.shape[-1]!=self.native_dimension:
            raise ValueError("S55 identity dimension changed")

        # Native/cache-derived identity is frozen, but relation_code intentionally
        # remains attached so the learned joint transform receives gradients.
        base=identity.detach()
        context=relation_code

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
            raise ValueError("S55 corrected logits are non-finite")
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
        (
            relation_code,
            query_context,
            state_joint_context,
            query_weights,
            joint_weights,
            state_support,
            option_support,
        )=self.learned_relation_code(
            identity=identity,
            state_tokens=state_tokens,
            state_mask=state_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            return_contexts=True,
        )
        logits=self.correction_logits_from_identity_relation_code(
            native_logits=native_logits,
            identity=identity,
            relation_code=relation_code,
        )
        if return_context:
            return (
                logits,
                identity,
                relation_code,
                query_context,
                state_joint_context,
                query_weights,
                joint_weights,
                state_support,
                option_support,
            )
        return logits,identity


def learned_joint_initialization_exact(
    reference: LearnedJointRelationPrivateCorrectionFork,
    treatment: LearnedJointRelationPrivateCorrectionFork,
)->bool:
    if reference.use_state_joint_context:
        return False
    if not treatment.use_state_joint_context:
        return False
    r_corr=reference.correction_state_dict()
    t_corr=treatment.correction_state_dict()
    if r_corr.keys()!=t_corr.keys():
        return False
    if not all(torch.equal(r_corr[k],t_corr[k]) for k in r_corr):
        return False
    rp=reference.learned_joint_parameters()
    tp=treatment.learned_joint_parameters()
    return len(rp)==len(tp) and all(torch.equal(a,b) for a,b in zip(rp,tp))


__all__=[
    "LearnedJointRelationTransform",
    "LearnedJointRelationPrivateCorrectionFork",
    "learned_joint_initialization_exact",
]
