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
    build_hira_v1_s6_a13_lora_core,
)

S6_LORA_CHECKPOINT_SCHEMA = "hira-v1-s6-a13-lora-checkpoint-v1"
S6_LORA_CHECKPOINT_KIND = "a13-last-attention-lora"
S6_LORA_KEYS = {
    *(f"lora.{i}.a" for i in range(4)),
    *(f"lora.{i}.b" for i in range(4)),
}


def load_hira_v1_s6_lora_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    initialization_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    semantic_revision: str = A13_REVISION,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha = file_sha256(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RuntimeError(
            f"Hira v1 S6 LoRA checkpoint SHA mismatch: "
            f"{actual_sha} != {expected_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != S6_LORA_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected Hira v1 S6 LoRA checkpoint schema")
    if payload.get("kind") != S6_LORA_CHECKPOINT_KIND:
        raise RuntimeError("unexpected Hira v1 S6 LoRA checkpoint kind")
    if int(payload.get("lora_parameter_count", -1)) != HIRA_V1_S6_LORA_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S6 LoRA parameter count changed")
    if int(payload.get("lora_rank", -1)) != HIRA_V1_S6_LORA_RANK:
        raise RuntimeError("Hira v1 S6 LoRA rank changed")
    if payload.get("initialization_t0_sha256") != initialization_t0_sha256:
        raise RuntimeError("Hira v1 S6 initialization T0 identity changed")
    if payload.get("semantic_revision") != semantic_revision:
        raise RuntimeError("Hira v1 S6 A13 revision identity changed")

    state = payload.get("lora_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Hira v1 S6 checkpoint missing LoRA state dict")
    state = {str(key): value for key, value in state.items()}
    if set(state) != S6_LORA_KEYS:
        raise RuntimeError("Hira v1 S6 LoRA checkpoint keys changed")

    cleaned = {}
    for i in range(4):
        a = state[f"lora.{i}.a"]
        b = state[f"lora.{i}.b"]
        if not isinstance(a, Tensor) or tuple(a.shape) != (8, 256):
            raise RuntimeError(f"Hira v1 S6 LoRA A tensor shape changed: {i}")
        if not isinstance(b, Tensor) or tuple(b.shape) != (256, 8):
            raise RuntimeError(f"Hira v1 S6 LoRA B tensor shape changed: {i}")
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise RuntimeError(f"Hira v1 S6 LoRA tensor is non-finite: {i}")
        cleaned[f"lora.{i}.a"] = a.detach().float().clone()
        cleaned[f"lora.{i}.b"] = b.detach().float().clone()

    epoch = int(payload.get("selected_dev_epoch", -1))
    if not (1 <= epoch <= 24):
        raise RuntimeError("Hira v1 S6 selected DEV epoch is invalid")

    return cleaned, {
        "sha256": actual_sha,
        "selected_dev_epoch": epoch,
        "lora_parameter_count": HIRA_V1_S6_LORA_PARAMETER_COUNT,
        "semantic_revision": semantic_revision,
    }


def build_frozen_hira_v1_s6_candidate(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    lora_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_lora_sha256: str | None = None,
    semantic_revision: str = A13_REVISION,
):
    if encoder.revision != semantic_revision:
        raise RuntimeError("Hira v1 S6 encoder revision changed")

    runtime = build_hira_v1_s6_a13_lora_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=True,
    )
    state, metadata = load_hira_v1_s6_lora_checkpoint(
        lora_checkpoint_path,
        expected_sha256=expected_lora_sha256,
        initialization_t0_sha256=expected_t0_sha256,
        semantic_revision=semantic_revision,
    )
    load_a13_lora_state_dict(runtime.encoder, state, freeze=True)

    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(
            f"Hira v1 S6 frozen replay has trainable parameters: {trainable}"
        )
    runtime.eval()
    return runtime, metadata


__all__ = [
    "S6_LORA_CHECKPOINT_KIND",
    "S6_LORA_CHECKPOINT_SCHEMA",
    "S6_LORA_KEYS",
    "build_frozen_hira_v1_s6_candidate",
    "load_hira_v1_s6_lora_checkpoint",
]
