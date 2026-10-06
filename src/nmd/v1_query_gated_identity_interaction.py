from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_explicit_pairwise_decision_head import (
    S59_REPRESENTATION_DIMENSION,
    build_pairwise_representation,
)


S69_NATIVE_DIMENSION=256
S69_REPRESENTATION_DIMENSION=S59_REPRESENTATION_DIMENSION
S69_INTERACTION_SCALE=math.sqrt(float(S69_NATIVE_DIMENSION))
S69_REPRESENTATION_PARAMETER_COUNT=0
S69_NORM_EPSILON=1e-12


def _validate_sources(identity:Tensor,joint_context:Tensor)->None:
    if identity.shape!=joint_context.shape:
        raise ValueError("S69 identity/context shape mismatch")
    if identity.ndim!=3 or identity.shape[-1]!=S69_NATIVE_DIMENSION:
        raise ValueError("S69 expects identity/context [B,K,256]")
    if identity.shape[1]<2:
        raise ValueError("S69 requires K>=2")
    if not bool(torch.isfinite(identity).all()):
        raise ValueError("S69 identity contains non-finite values")
    if not bool(torch.isfinite(joint_context).all()):
        raise ValueError("S69 joint context contains non-finite values")


def build_reference_pairwise_representation(
    identity:Tensor,
    joint_context:Tensor,
)->Tensor:
    """Exact S59 representation authority."""
    _validate_sources(identity,joint_context)
    rep=build_pairwise_representation(identity,joint_context)
    if rep.requires_grad:
        raise RuntimeError("S69 reference representation retained gradient")
    return rep


def build_query_gated_context(
    identity:Tensor,
    joint_context:Tensor,
)->Tensor:
    """Zero-parameter state/query-conditioned option interaction.

    The sqrt(256) factor restores a unit-vector Hadamard interaction to the
    same natural scale as the direct joint-context residual.
    """
    _validate_sources(identity,joint_context)
    identity_detached=identity.detach()
    context_detached=joint_context.detach()

    interaction=(
        context_detached
        +S69_INTERACTION_SCALE*(identity_detached*context_detached)
    )
    context=F.normalize(
        interaction,
        dim=-1,
        eps=S69_NORM_EPSILON,
    )
    if not bool(torch.isfinite(context).all()):
        raise ValueError("S69 query-gated context non-finite")
    return context.detach()


def build_query_gated_pairwise_representation(
    identity:Tensor,
    joint_context:Tensor,
)->Tensor:
    _validate_sources(identity,joint_context)
    identity_detached=identity.detach()
    transformed_context=build_query_gated_context(
        identity_detached,
        joint_context.detach(),
    )
    rep=torch.cat([identity_detached,transformed_context],dim=-1)
    rep=F.normalize(rep,dim=-1,eps=S69_NORM_EPSILON)
    if rep.shape[-1]!=S69_REPRESENTATION_DIMENSION:
        raise RuntimeError("S69 representation dimension changed")
    if not bool(torch.isfinite(rep).all()):
        raise ValueError("S69 treatment representation non-finite")
    return rep.detach()


__all__=[
    "S69_NATIVE_DIMENSION",
    "S69_REPRESENTATION_DIMENSION",
    "S69_INTERACTION_SCALE",
    "S69_REPRESENTATION_PARAMETER_COUNT",
    "S69_NORM_EPSILON",
    "build_reference_pairwise_representation",
    "build_query_gated_context",
    "build_query_gated_pairwise_representation",
]
