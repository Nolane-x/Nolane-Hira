from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .typed_competitive_cache import file_sha256
from .v1_s5_semantic_core import (
    HIRA_V1_S5_PROJECTION_PARAMETER_COUNT,
    build_hira_v1_s5_projection_core,
)

S5_PROJECTION_CHECKPOINT_SCHEMA = "hira-v1-s5-projection-checkpoint-v1"
S5_PROJECTION_CHECKPOINT_KIND = "semantic-projection-relearning"
S5_PROJECTION_KEYS = {"projection.weight"}


def load_hira_v1_s5_projection_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    initialization_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha = file_sha256(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RuntimeError(
            f"Hira v1 S5 projection checkpoint SHA mismatch: "
            f"{actual_sha} != {expected_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != S5_PROJECTION_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected Hira v1 S5 projection checkpoint schema")
    if payload.get("kind") != S5_PROJECTION_CHECKPOINT_KIND:
        raise RuntimeError("unexpected Hira v1 S5 projection checkpoint kind")
    if (
        int(payload.get("projection_parameter_count", -1))
        != HIRA_V1_S5_PROJECTION_PARAMETER_COUNT
    ):
        raise RuntimeError("Hira v1 S5 projection parameter count changed")
    if payload.get("initialization_t0_sha256") != initialization_t0_sha256:
        raise RuntimeError("Hira v1 S5 initialization T0 identity changed")

    state = payload.get("projection_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Hira v1 S5 checkpoint missing projection state dict")
    state = {str(key): value for key, value in state.items()}
    if set(state) != S5_PROJECTION_KEYS:
        raise RuntimeError("Hira v1 S5 checkpoint keys changed")

    weight = state["projection.weight"]
    if not isinstance(weight, Tensor) or tuple(weight.shape) != (128, 256):
        raise RuntimeError("Hira v1 S5 projection tensor shape changed")
    if not bool(torch.isfinite(weight).all()):
        raise RuntimeError("Hira v1 S5 projection tensor is non-finite")

    epoch = int(payload.get("selected_dev_epoch", -1))
    if not (1 <= epoch <= 30):
        raise RuntimeError("Hira v1 S5 selected DEV epoch is invalid")

    return (
        {"projection.weight": weight.detach().float().clone()},
        {
            "sha256": actual_sha,
            "selected_dev_epoch": epoch,
            "projection_parameter_count": HIRA_V1_S5_PROJECTION_PARAMETER_COUNT,
        },
    )


def build_frozen_hira_v1_s5_candidate(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    projection_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_projection_sha256: str | None = None,
):
    runtime = build_hira_v1_s5_projection_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_projection=True,
    )
    state, metadata = load_hira_v1_s5_projection_checkpoint(
        projection_checkpoint_path,
        expected_sha256=expected_projection_sha256,
        initialization_t0_sha256=expected_t0_sha256,
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S5 scorer missing during frozen replay")
    scorer.load_projection_state_dict(state, freeze=True)

    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(
            f"Hira v1 S5 frozen replay has trainable parameters: {trainable}"
        )
    runtime.eval()
    return runtime, metadata


__all__ = [
    "S5_PROJECTION_CHECKPOINT_KIND",
    "S5_PROJECTION_CHECKPOINT_SCHEMA",
    "S5_PROJECTION_KEYS",
    "build_frozen_hira_v1_s5_candidate",
    "load_hira_v1_s5_projection_checkpoint",
]
