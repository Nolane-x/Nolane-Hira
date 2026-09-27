from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .hira import HIRACore
from .runtime import NolaneHira
from .semantic import TextSemanticEncoder
from .semantic_core import (
    W28_T0_CHECKPOINT_SHA256,
    load_rescued_projection_checkpoint,
)
from .semantic_transfer_bridge import BridgedSymmetricSemanticScorer
from .symmetric_semantic import SymmetricSemanticScorer
from .typed_competitive_cache import file_sha256

W30_BRIDGE_CHECKPOINT_SCHEMA = "r8-w30-semantic-transfer-bridge-checkpoint-v1"
W30_BRIDGE_RANK = 8
W30_BRIDGE_PARAMETER_COUNT = 2048


def load_transfer_bridge_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = file_sha256(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RuntimeError(
            f"W30 bridge checkpoint SHA mismatch: {actual_sha} != {expected_sha256}"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != W30_BRIDGE_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected W30 bridge checkpoint schema")
    if checkpoint.get("kind") != "semantic-transfer-bridge":
        raise RuntimeError("unexpected W30 bridge checkpoint kind")
    if int(checkpoint.get("rank", -1)) != W30_BRIDGE_RANK:
        raise RuntimeError("W30 bridge rank changed")
    if int(checkpoint.get("parameter_count", -1)) != W30_BRIDGE_PARAMETER_COUNT:
        raise RuntimeError("W30 bridge parameter count changed")
    if checkpoint.get("t0_checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W30 bridge base T0 identity changed")

    state = checkpoint.get("bridge_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("W30 bridge checkpoint missing state dict")
    state = {str(key): value for key, value in state.items()}
    if set(state) != {"down.weight", "up.weight"}:
        raise RuntimeError("W30 bridge checkpoint keys changed")
    if tuple(state["down.weight"].shape) != (8, 128):
        raise RuntimeError("W30 bridge down shape changed")
    if tuple(state["up.weight"].shape) != (128, 8):
        raise RuntimeError("W30 bridge up shape changed")
    if not all(isinstance(value, Tensor) and bool(torch.isfinite(value).all()) for value in state.values()):
        raise RuntimeError("W30 bridge checkpoint contains invalid tensors")

    metadata = {
        "sha256": actual_sha,
        "selected_dev_epoch": int(checkpoint.get("selected_dev_epoch", -1)),
        "parameter_count": W30_BRIDGE_PARAMETER_COUNT,
    }
    return {key: value.detach().float().clone() for key, value in state.items()}, metadata


def build_hira_v0_transfer_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    bridge_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_bridge_sha256: str | None = None,
    hira: HIRACore | None = None,
    include_unbridged_baseline: bool = True,
) -> NolaneHira:
    if int(encoder.d_model) != 256:
        raise ValueError("HIRA v0 transfer core requires d_model=256")

    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path,
        expected_sha256=expected_t0_sha256,
    )
    bridge_state, _ = load_transfer_bridge_checkpoint(
        bridge_checkpoint_path,
        expected_sha256=expected_bridge_sha256,
    )

    bridged = BridgedSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=W30_BRIDGE_RANK,
    )
    bridged.load_projection_weight(projection, freeze=True)
    bridged.load_bridge_state_dict(bridge_state, freeze=True)
    if bridged.trainable_parameter_count != 0:
        raise RuntimeError("packaged W30 bridge scorer must be fully frozen")

    baseline = None
    if include_unbridged_baseline:
        baseline = SymmetricSemanticScorer(d_model=256, d_rel=128)
        baseline.load_projection_weight(projection, freeze=True)

    return NolaneHira(
        encoder,
        hira or HIRACore(d_model=256),
        symmetric_semantic_scorer=baseline,
        bridged_symmetric_semantic_scorer=bridged,
    )


__all__ = [
    "W30_BRIDGE_CHECKPOINT_SCHEMA",
    "W30_BRIDGE_PARAMETER_COUNT",
    "W30_BRIDGE_RANK",
    "build_hira_v0_transfer_core",
    "load_transfer_bridge_checkpoint",
]
