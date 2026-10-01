from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .local_runtime import A13_REVISION
from .semantic import HFAutoSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .typed_competitive_cache import file_sha256
from .v1_a13_lora import load_a13_lora_state_dict
from .v1_s6_semantic_core import HIRA_V1_S6_LORA_RANK
from .v1_s25_semantic_core import get_s25_relation_projection
from .v1_s26_semantic_core import (
    HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S26_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s26_factorized_relation_core,
    build_s26_factorized_relation_operator,
    enforce_s26_eval,
)

S26_CHECKPOINT_SCHEMA = "hira-v1-s26-factorized-role-value-relation-checkpoint-v1"
S26_CHECKPOINT_KIND = "factorized-role-value-relation-a13-w28"
S26_LORA_KEYS = {
    *(f"lora.{i}.a" for i in range(4)),
    *(f"lora.{i}.b" for i in range(4)),
}
S26_PROJECTION_KEYS = {"projection.weight"}


def _clean_projection(value, *, label: str) -> dict[str, Tensor]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Hira v1 S26 checkpoint missing {label} projection state")
    state = {str(k): v for k, v in value.items()}
    if set(state) != S26_PROJECTION_KEYS:
        raise RuntimeError(f"Hira v1 S26 {label} projection keys changed")
    weight = state["projection.weight"]
    if not isinstance(weight, Tensor) or tuple(weight.shape) != (128, 256):
        raise RuntimeError(f"Hira v1 S26 {label} projection shape changed")
    if not bool(torch.isfinite(weight).all()):
        raise RuntimeError(f"Hira v1 S26 {label} projection non-finite")
    return {"projection.weight": weight.detach().float().clone()}


def load_hira_v1_s26_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    initialization_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    semantic_revision: str = A13_REVISION,
) -> tuple[
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, object],
]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = file_sha256(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RuntimeError(
            f"Hira v1 S26 checkpoint SHA mismatch: {actual_sha} != {expected_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != S26_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected Hira v1 S26 checkpoint schema")
    if payload.get("kind") != S26_CHECKPOINT_KIND:
        raise RuntimeError("unexpected Hira v1 S26 checkpoint kind")
    if int(payload.get("lora_parameter_count", -1)) != HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S26 LoRA parameter count changed")
    if int(payload.get("primary_projection_parameter_count", -1)) != HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S26 primary projection parameter count changed")
    if int(payload.get("relation_projection_parameter_count", -1)) != HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S26 relation projection parameter count changed")
    if int(payload.get("total_parameter_count", -1)) != HIRA_V1_S26_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S26 total parameter count changed")
    if int(payload.get("lora_rank", -1)) != HIRA_V1_S6_LORA_RANK:
        raise RuntimeError("Hira v1 S26 LoRA rank changed")
    if payload.get("initialization_t0_sha256") != initialization_t0_sha256:
        raise RuntimeError("Hira v1 S26 initialization T0 identity changed")
    if payload.get("semantic_revision") != semantic_revision:
        raise RuntimeError("Hira v1 S26 semantic revision changed")

    lora_value = payload.get("lora_state_dict")
    if not isinstance(lora_value, Mapping):
        raise RuntimeError("Hira v1 S26 checkpoint missing LoRA state")
    lora = {str(k): v for k, v in lora_value.items()}
    if set(lora) != S26_LORA_KEYS:
        raise RuntimeError("Hira v1 S26 LoRA keys changed")

    clean_lora: dict[str, Tensor] = {}
    for i in range(4):
        a = lora[f"lora.{i}.a"]
        b = lora[f"lora.{i}.b"]
        if not isinstance(a, Tensor) or tuple(a.shape) != (8, 256):
            raise RuntimeError(f"Hira v1 S26 LoRA A shape changed: {i}")
        if not isinstance(b, Tensor) or tuple(b.shape) != (256, 8):
            raise RuntimeError(f"Hira v1 S26 LoRA B shape changed: {i}")
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise RuntimeError(f"Hira v1 S26 LoRA tensor non-finite: {i}")
        clean_lora[f"lora.{i}.a"] = a.detach().float().clone()
        clean_lora[f"lora.{i}.b"] = b.detach().float().clone()

    primary = _clean_projection(
        payload.get("primary_projection_state_dict"),
        label="primary",
    )
    relation = _clean_projection(
        payload.get("relation_projection_state_dict"),
        label="relation",
    )

    epoch = int(payload.get("selected_dev_epoch", -1))
    if not (1 <= epoch <= 24):
        raise RuntimeError("Hira v1 S26 selected DEV epoch is invalid")

    return (
        clean_lora,
        primary,
        relation,
        {
            "sha256": actual_sha,
            "selected_dev_epoch": epoch,
            "total_parameter_count": HIRA_V1_S26_TOTAL_PARAMETER_COUNT,
            "semantic_revision": semantic_revision,
        },
    )


def build_frozen_hira_v1_s26_candidate(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    candidate_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_candidate_sha256: str | None = None,
    semantic_revision: str = A13_REVISION,
):
    if encoder.revision != semantic_revision:
        raise RuntimeError("Hira v1 S26 encoder revision changed")

    runtime = build_hira_v1_s26_factorized_relation_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=True,
        train_primary_projection=True,
        train_relation_projection=True,
    )
    lora, primary, relation, metadata = load_hira_v1_s26_checkpoint(
        candidate_checkpoint_path,
        expected_sha256=expected_candidate_sha256,
        initialization_t0_sha256=expected_t0_sha256,
        semantic_revision=semantic_revision,
    )
    load_a13_lora_state_dict(runtime.encoder, lora, freeze=True)

    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S26 primary replay scorer missing")
    scorer.load_projection_state_dict(primary, freeze=True)

    relation_projection = get_s25_relation_projection(runtime)
    with torch.no_grad():
        relation_projection.weight.copy_(relation["projection.weight"])
    relation_projection.weight.requires_grad_(False)

    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(
            f"Hira v1 S26 frozen replay has trainable parameters: {trainable}"
        )
    if scorer.projection.weight.data_ptr() == relation_projection.weight.data_ptr():
        raise RuntimeError("Hira v1 S26 frozen replay projections share storage")

    enforce_s26_eval(runtime)
    operator = build_s26_factorized_relation_operator()
    operator.eval()
    if operator.factorization_added_parameter_count != 0:
        raise RuntimeError("Hira v1 S26 frozen replay relation operator added params")
    return runtime, operator, metadata


__all__ = [
    "S26_CHECKPOINT_KIND",
    "S26_CHECKPOINT_SCHEMA",
    "S26_LORA_KEYS",
    "S26_PROJECTION_KEYS",
    "build_frozen_hira_v1_s26_candidate",
    "load_hira_v1_s26_checkpoint",
]
