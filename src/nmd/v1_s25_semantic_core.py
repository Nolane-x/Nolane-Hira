from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .runtime import NolaneHira
from .semantic import HFAutoSemanticEncoder
from .v1_a13_lora import iter_a13_lora_modules
from .v1_role_gated_content_triadic import RoleGatedContentTriadicScorer
from .v1_s21_semantic_core import build_hira_v1_s21_role_content_core

HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT = 16_384
HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT = 32_768
HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT = 32_768
HIRA_V1_S25_TOTAL_PARAMETER_COUNT = 81_920
S25_RELATION_PROJECTION_NAME = "s25_relation_projection"


def get_s25_relation_projection(runtime: NolaneHira) -> nn.Linear:
    projection = getattr(runtime, S25_RELATION_PROJECTION_NAME, None)
    if not isinstance(projection, nn.Linear):
        raise RuntimeError("Hira v1 S25 relation-private projection missing")
    if projection.in_features != 256 or projection.out_features != 128:
        raise RuntimeError("Hira v1 S25 relation-private projection shape changed")
    if projection.bias is not None:
        raise RuntimeError("Hira v1 S25 relation-private projection must be bias-free")
    return projection


def build_hira_v1_s25_decoupled_projection_core(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str,
    train_lora: bool = True,
    train_primary_projection: bool = True,
    train_relation_projection: bool = True,
) -> NolaneHira:
    runtime = build_hira_v1_s21_role_content_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=train_lora,
        train_projection=train_primary_projection,
    )
    scorer = runtime.projection_triadic_scorer
    if not isinstance(scorer, RoleGatedContentTriadicScorer):
        raise RuntimeError("Hira v1 S25 primary role-content scorer missing")

    primary = scorer.projection
    relation = nn.Linear(
        256,
        128,
        bias=False,
        device=primary.weight.device,
        dtype=primary.weight.dtype,
    )
    with torch.no_grad():
        relation.weight.copy_(primary.weight.detach())
    relation.weight.requires_grad_(train_relation_projection)
    runtime.add_module(S25_RELATION_PROJECTION_NAME, relation)

    if primary.weight.data_ptr() == relation.weight.data_ptr():
        raise RuntimeError("Hira v1 S25 private projections unexpectedly share storage")
    if not torch.equal(primary.weight.detach(), relation.weight.detach()):
        raise RuntimeError("Hira v1 S25 private projection initialization diverged")

    lora_trainable = sum(
        p.numel()
        for module in iter_a13_lora_modules(runtime.encoder)
        for p in (module.lora_a, module.lora_b)
        if p.requires_grad
    )
    primary_trainable = (
        primary.weight.numel() if primary.weight.requires_grad else 0
    )
    relation_trainable = (
        relation.weight.numel() if relation.weight.requires_grad else 0
    )
    expected = (
        (HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT if train_lora else 0)
        + (
            HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT
            if train_primary_projection
            else 0
        )
        + (
            HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT
            if train_relation_projection
            else 0
        )
    )
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)

    if lora_trainable != (
        HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT if train_lora else 0
    ):
        raise RuntimeError("Hira v1 S25 shared LoRA surface changed")
    if primary_trainable != (
        HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT
        if train_primary_projection
        else 0
    ):
        raise RuntimeError("Hira v1 S25 primary-private surface changed")
    if relation_trainable != (
        HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT
        if train_relation_projection
        else 0
    ):
        raise RuntimeError("Hira v1 S25 relation-private surface changed")
    if trainable != expected:
        raise RuntimeError(
            f"Hira v1 S25 optimization surface changed: {trainable} != {expected}"
        )

    original_a13_trainable = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original_a13_trainable != 0:
        raise RuntimeError("Hira v1 S25 original A13 weights became trainable")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("Hira v1 S25 HIRACore became trainable")

    runtime.eval()
    relation.eval()
    return runtime


def enforce_s25_eval(runtime: NolaneHira) -> None:
    runtime.encoder.eval()
    runtime.hira.eval()
    scorer = runtime.projection_triadic_scorer
    if not isinstance(scorer, RoleGatedContentTriadicScorer):
        raise RuntimeError("Hira v1 S25 primary scorer missing")
    scorer.eval()
    get_s25_relation_projection(runtime).eval()


__all__ = [
    "HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT",
    "HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT",
    "HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT",
    "HIRA_V1_S25_TOTAL_PARAMETER_COUNT",
    "S25_RELATION_PROJECTION_NAME",
    "build_hira_v1_s25_decoupled_projection_core",
    "enforce_s25_eval",
    "get_s25_relation_projection",
]
