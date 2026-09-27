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
from .w34_coevidence_semantic import CoEvidenceSemanticScorer

W34_CANDIDATE_CHECKPOINT_SCHEMA = "r8-w34-coevidence-semantic-checkpoint-v1"
W34_CANDIDATE_RANK = 8
W34_CANDIDATE_PARAMETER_COUNT = 8192
W34_CANDIDATE_KEYS = {
    "state_adapter.down.weight",
    "state_adapter.up.weight",
    "schema_adapter.down.weight",
    "schema_adapter.up.weight",
    "interaction_state.weight",
    "interaction_schema.weight",
    "composition_state.weight",
    "composition_schema.weight",
}


def load_w34_candidate_checkpoint(
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
            f"W34 candidate checkpoint SHA mismatch: {actual_sha} != {expected_sha256}"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != W34_CANDIDATE_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected W34 candidate checkpoint schema")
    if checkpoint.get("kind") != "coevidence-semantic":
        raise RuntimeError("unexpected W34 candidate checkpoint kind")
    if int(checkpoint.get("rank", -1)) != W34_CANDIDATE_RANK:
        raise RuntimeError("W34 candidate rank changed")
    if int(checkpoint.get("parameter_count", -1)) != W34_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("W34 candidate parameter count changed")
    if checkpoint.get("t0_checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W34 candidate base T0 identity changed")

    state = checkpoint.get("candidate_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("W34 candidate checkpoint missing state dict")
    state = {str(key): value for key, value in state.items()}
    if set(state) != W34_CANDIDATE_KEYS:
        raise RuntimeError("W34 candidate checkpoint keys changed")

    shapes = {
        "state_adapter.down.weight": (8, 128),
        "state_adapter.up.weight": (128, 8),
        "schema_adapter.down.weight": (8, 128),
        "schema_adapter.up.weight": (128, 8),
        "interaction_state.weight": (8, 128),
        "interaction_schema.weight": (8, 128),
        "composition_state.weight": (8, 128),
        "composition_schema.weight": (8, 128),
    }
    for key, shape in shapes.items():
        value = state[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(f"W34 candidate shape changed: {key}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"W34 candidate contains invalid tensor: {key}")

    metadata = {
        "sha256": actual_sha,
        "selected_dev_epoch": int(checkpoint.get("selected_dev_epoch", -1)),
        "parameter_count": W34_CANDIDATE_PARAMETER_COUNT,
    }
    return {
        key: value.detach().float().clone()
        for key, value in state.items()
    }, metadata


def build_hira_v0_w34_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    candidate_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_candidate_sha256: str | None = None,
    hira: HIRACore | None = None,
    include_unbridged_baseline: bool = True,
) -> NolaneHira:
    if int(encoder.d_model) != 256:
        raise ValueError("HIRA v0 W34 core requires d_model=256")

    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path,
        expected_sha256=expected_t0_sha256,
    )
    candidate_state, _ = load_w34_candidate_checkpoint(
        candidate_checkpoint_path,
        expected_sha256=expected_candidate_sha256,
    )

    coevidence = CoEvidenceSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=W34_CANDIDATE_RANK,
    )
    coevidence.load_projection_weight(projection, freeze=True)
    coevidence.load_candidate_state_dict(candidate_state, freeze=True)
    if coevidence.trainable_parameter_count != 0:
        raise RuntimeError("packaged W34 co-evidence scorer must be fully frozen")

    baseline = None
    if include_unbridged_baseline:
        baseline = SymmetricSemanticScorer(d_model=256, d_rel=128)
        baseline.load_projection_weight(projection, freeze=True)

    return NolaneHira(
        encoder,
        hira=hira or HIRACore(d_model=256),
        symmetric_semantic_scorer=baseline,
        coevidence_symmetric_semantic_scorer=coevidence,
    )


__all__ = [
    "W34_CANDIDATE_CHECKPOINT_SCHEMA",
    "W34_CANDIDATE_KEYS",
    "W34_CANDIDATE_PARAMETER_COUNT",
    "W34_CANDIDATE_RANK",
    "build_hira_v0_w34_core",
    "load_w34_candidate_checkpoint",
]
