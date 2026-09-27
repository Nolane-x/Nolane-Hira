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
from .symmetric_semantic import SymmetricSemanticScorer
from .typed_competitive_cache import file_sha256
from .w31_dual_semantic_adapter import DualAdapterSymmetricSemanticScorer

W31_ADAPTER_CHECKPOINT_SCHEMA = "r8-w31-dual-semantic-adapter-checkpoint-v1"
W31_ADAPTER_RANK = 8
W31_ADAPTER_PARAMETER_COUNT = 4096
W31_ADAPTER_KEYS = {
    "state_adapter.down.weight",
    "state_adapter.up.weight",
    "schema_adapter.down.weight",
    "schema_adapter.up.weight",
}


def load_w31_adapter_checkpoint(
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
            f"W31 adapter checkpoint SHA mismatch: {actual_sha} != {expected_sha256}"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != W31_ADAPTER_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected W31 adapter checkpoint schema")
    if checkpoint.get("kind") != "dual-semantic-adapter":
        raise RuntimeError("unexpected W31 adapter checkpoint kind")
    if int(checkpoint.get("rank", -1)) != W31_ADAPTER_RANK:
        raise RuntimeError("W31 adapter rank changed")
    if int(checkpoint.get("parameter_count", -1)) != W31_ADAPTER_PARAMETER_COUNT:
        raise RuntimeError("W31 adapter parameter count changed")
    if checkpoint.get("t0_checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W31 adapter base T0 identity changed")

    state = checkpoint.get("adapter_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("W31 adapter checkpoint missing state dict")
    state = {str(key): value for key, value in state.items()}
    if set(state) != W31_ADAPTER_KEYS:
        raise RuntimeError("W31 adapter checkpoint keys changed")

    shapes = {
        "state_adapter.down.weight": (8, 128),
        "state_adapter.up.weight": (128, 8),
        "schema_adapter.down.weight": (8, 128),
        "schema_adapter.up.weight": (128, 8),
    }
    for key, shape in shapes.items():
        value = state[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(f"W31 adapter shape changed: {key}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"W31 adapter contains invalid tensor: {key}")

    metadata = {
        "sha256": actual_sha,
        "selected_dev_epoch": int(checkpoint.get("selected_dev_epoch", -1)),
        "parameter_count": W31_ADAPTER_PARAMETER_COUNT,
    }
    return {
        key: value.detach().float().clone()
        for key, value in state.items()
    }, metadata


def build_hira_v0_w31_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    adapter_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_adapter_sha256: str | None = None,
    hira: HIRACore | None = None,
    include_unbridged_baseline: bool = True,
) -> NolaneHira:
    if int(encoder.d_model) != 256:
        raise ValueError("HIRA v0 W31 core requires d_model=256")

    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path,
        expected_sha256=expected_t0_sha256,
    )
    adapter_state, _ = load_w31_adapter_checkpoint(
        adapter_checkpoint_path,
        expected_sha256=expected_adapter_sha256,
    )

    dual = DualAdapterSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=W31_ADAPTER_RANK,
    )
    dual.load_projection_weight(projection, freeze=True)
    dual.load_adapter_state_dict(adapter_state, freeze=True)
    if dual.trainable_parameter_count != 0:
        raise RuntimeError("packaged W31 dual scorer must be fully frozen")

    baseline = None
    if include_unbridged_baseline:
        baseline = SymmetricSemanticScorer(d_model=256, d_rel=128)
        baseline.load_projection_weight(projection, freeze=True)

    return NolaneHira(
        encoder,
        hira=hira or HIRACore(d_model=256),
        symmetric_semantic_scorer=baseline,
        dual_symmetric_semantic_scorer=dual,
    )


__all__ = [
    "W31_ADAPTER_CHECKPOINT_SCHEMA",
    "W31_ADAPTER_KEYS",
    "W31_ADAPTER_PARAMETER_COUNT",
    "W31_ADAPTER_RANK",
    "build_hira_v0_w31_core",
    "load_w31_adapter_checkpoint",
]
