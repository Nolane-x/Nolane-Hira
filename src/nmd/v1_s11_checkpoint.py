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
from .v1_s6_semantic_core import (
    HIRA_V1_S6_LORA_PARAMETER_COUNT,
    HIRA_V1_S6_LORA_RANK,
)
from .v1_s11_semantic_core import (
    HIRA_V1_S11_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S11_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s11_role_binding_core,
)

S11_CHECKPOINT_SCHEMA = "hira-v1-s11-role-binding-checkpoint-v1"
S11_CHECKPOINT_KIND = "state-option-role-preserving-binding-a13-w28"
S11_LORA_KEYS = {
    *(f"lora.{i}.a" for i in range(4)),
    *(f"lora.{i}.b" for i in range(4)),
}
S11_PROJECTION_KEYS = {"projection.weight"}


def load_hira_v1_s11_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    initialization_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    semantic_revision: str = A13_REVISION,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, object]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha = file_sha256(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RuntimeError(
            f"Hira v1 S11 checkpoint SHA mismatch: "
            f"{actual_sha} != {expected_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != S11_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected Hira v1 S11 checkpoint schema")
    if payload.get("kind") != S11_CHECKPOINT_KIND:
        raise RuntimeError("unexpected Hira v1 S11 checkpoint kind")
    if int(payload.get("lora_parameter_count", -1)) != HIRA_V1_S6_LORA_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S11 LoRA parameter count changed")
    if int(payload.get("projection_parameter_count", -1)) != HIRA_V1_S11_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S11 projection parameter count changed")
    if int(payload.get("total_parameter_count", -1)) != HIRA_V1_S11_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S11 total parameter count changed")
    if int(payload.get("lora_rank", -1)) != HIRA_V1_S6_LORA_RANK:
        raise RuntimeError("Hira v1 S11 LoRA rank changed")
    if payload.get("initialization_t0_sha256") != initialization_t0_sha256:
        raise RuntimeError("Hira v1 S11 initialization T0 identity changed")
    if payload.get("semantic_revision") != semantic_revision:
        raise RuntimeError("Hira v1 S11 semantic revision changed")

    lora = payload.get("lora_state_dict")
    projection = payload.get("projection_state_dict")
    if not isinstance(lora, Mapping):
        raise RuntimeError("Hira v1 S11 checkpoint missing LoRA state")
    if not isinstance(projection, Mapping):
        raise RuntimeError("Hira v1 S11 checkpoint missing projection state")

    lora = {str(k): v for k, v in lora.items()}
    projection = {str(k): v for k, v in projection.items()}
    if set(lora) != S11_LORA_KEYS:
        raise RuntimeError("Hira v1 S11 LoRA keys changed")
    if set(projection) != S11_PROJECTION_KEYS:
        raise RuntimeError("Hira v1 S11 projection keys changed")

    clean_lora: dict[str, Tensor] = {}
    for i in range(4):
        a = lora[f"lora.{i}.a"]
        b = lora[f"lora.{i}.b"]
        if not isinstance(a, Tensor) or tuple(a.shape) != (8, 256):
            raise RuntimeError(f"Hira v1 S11 LoRA A shape changed: {i}")
        if not isinstance(b, Tensor) or tuple(b.shape) != (256, 8):
            raise RuntimeError(f"Hira v1 S11 LoRA B shape changed: {i}")
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise RuntimeError(f"Hira v1 S11 LoRA tensor non-finite: {i}")
        clean_lora[f"lora.{i}.a"] = a.detach().float().clone()
        clean_lora[f"lora.{i}.b"] = b.detach().float().clone()

    weight = projection["projection.weight"]
    if not isinstance(weight, Tensor) or tuple(weight.shape) != (128, 256):
        raise RuntimeError("Hira v1 S11 projection shape changed")
    if not bool(torch.isfinite(weight).all()):
        raise RuntimeError("Hira v1 S11 projection tensor non-finite")

    epoch = int(payload.get("selected_dev_epoch", -1))
    if not (1 <= epoch <= 24):
        raise RuntimeError("Hira v1 S11 selected DEV epoch is invalid")

    return (
        clean_lora,
        {"projection.weight": weight.detach().float().clone()},
        {
            "sha256": actual_sha,
            "selected_dev_epoch": epoch,
            "total_parameter_count": HIRA_V1_S11_TOTAL_PARAMETER_COUNT,
            "semantic_revision": semantic_revision,
        },
    )


def build_frozen_hira_v1_s11_candidate(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    binding_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_binding_sha256: str | None = None,
    semantic_revision: str = A13_REVISION,
):
    if encoder.revision != semantic_revision:
        raise RuntimeError("Hira v1 S11 encoder revision changed")

    runtime = build_hira_v1_s11_role_binding_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=True,
        train_projection=True,
    )
    lora, projection, metadata = load_hira_v1_s11_checkpoint(
        binding_checkpoint_path,
        expected_sha256=expected_binding_sha256,
        initialization_t0_sha256=expected_t0_sha256,
        semantic_revision=semantic_revision,
    )
    load_a13_lora_state_dict(runtime.encoder, lora, freeze=True)

    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S11 replay scorer missing")
    scorer.load_projection_state_dict(projection, freeze=True)

    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(
            f"Hira v1 S11 frozen replay has trainable parameters: {trainable}"
        )
    runtime.eval()
    return runtime, metadata


__all__ = [
    "S11_CHECKPOINT_KIND",
    "S11_CHECKPOINT_SCHEMA",
    "S11_LORA_KEYS",
    "S11_PROJECTION_KEYS",
    "build_frozen_hira_v1_s11_candidate",
    "load_hira_v1_s11_checkpoint",
]
