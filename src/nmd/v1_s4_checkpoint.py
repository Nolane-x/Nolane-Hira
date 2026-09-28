from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .typed_competitive_cache import file_sha256
from .v1_s4_semantic_core import (
    HIRA_V1_S4_ADAPTER_BOTTLENECK,
    HIRA_V1_S4_ADAPTER_PARAMETER_COUNT,
    build_hira_v1_s4_adapter_core,
)

S4_ADAPTER_CHECKPOINT_SCHEMA = "hira-v1-s4-adapter-checkpoint-v1"
S4_ADAPTER_CHECKPOINT_KIND = "shared-residual-semantic-adapter"
S4_ADAPTER_KEYS = {
    "adapter.down.weight",
    "adapter.up.weight",
}


def load_hira_v1_s4_adapter_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha = file_sha256(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RuntimeError(
            f"Hira v1 S4 adapter checkpoint SHA mismatch: "
            f"{actual_sha} != {expected_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != S4_ADAPTER_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected Hira v1 S4 adapter checkpoint schema")
    if payload.get("kind") != S4_ADAPTER_CHECKPOINT_KIND:
        raise RuntimeError("unexpected Hira v1 S4 adapter checkpoint kind")
    if int(payload.get("bottleneck", -1)) != HIRA_V1_S4_ADAPTER_BOTTLENECK:
        raise RuntimeError("Hira v1 S4 adapter bottleneck changed")
    if (
        int(payload.get("adapter_parameter_count", -1))
        != HIRA_V1_S4_ADAPTER_PARAMETER_COUNT
    ):
        raise RuntimeError("Hira v1 S4 adapter parameter count changed")
    if payload.get("t0_checkpoint_sha256") != expected_t0_sha256:
        raise RuntimeError("Hira v1 S4 T0 identity changed")

    state = payload.get("adapter_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Hira v1 S4 checkpoint missing adapter state dict")
    state = {str(key): value for key, value in state.items()}
    if set(state) != S4_ADAPTER_KEYS:
        raise RuntimeError("Hira v1 S4 checkpoint keys changed")

    shapes = {
        "adapter.down.weight": (HIRA_V1_S4_ADAPTER_BOTTLENECK, 256),
        "adapter.up.weight": (256, HIRA_V1_S4_ADAPTER_BOTTLENECK),
    }
    for key, shape in shapes.items():
        value = state[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(f"Hira v1 S4 tensor shape changed: {key}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Hira v1 S4 tensor is non-finite: {key}")

    epoch = int(payload.get("selected_dev_epoch", -1))
    if not (1 <= epoch <= 24):
        raise RuntimeError("Hira v1 S4 selected DEV epoch is invalid")

    return (
        {key: value.detach().float().clone() for key, value in state.items()},
        {
            "sha256": actual_sha,
            "selected_dev_epoch": epoch,
            "adapter_parameter_count": HIRA_V1_S4_ADAPTER_PARAMETER_COUNT,
        },
    )


def build_frozen_hira_v1_s4_candidate(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    adapter_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_adapter_sha256: str | None = None,
):
    runtime = build_hira_v1_s4_adapter_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_adapter=True,
    )
    state, metadata = load_hira_v1_s4_adapter_checkpoint(
        adapter_checkpoint_path,
        expected_sha256=expected_adapter_sha256,
        expected_t0_sha256=expected_t0_sha256,
    )
    scorer = runtime.adapted_triadic_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S4 scorer missing during frozen replay")
    scorer.load_adapter_state_dict(state, freeze=True)

    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(
            f"Hira v1 S4 frozen replay has trainable parameters: {trainable}"
        )
    runtime.eval()
    return runtime, metadata


__all__ = [
    "S4_ADAPTER_CHECKPOINT_KIND",
    "S4_ADAPTER_CHECKPOINT_SCHEMA",
    "S4_ADAPTER_KEYS",
    "build_frozen_hira_v1_s4_candidate",
    "load_hira_v1_s4_adapter_checkpoint",
]
