from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .typed_competitive_cache import file_sha256
from .v1_s2_semantic_core import (
    HIRA_V1_S2_CANDIDATE_PARAMETER_COUNT,
    HIRA_V1_S2_FUSION_PARAMETER_COUNT,
    build_hira_v1_s2_fusion_core,
)

S2_QTRF_CHECKPOINT_SCHEMA = "hira-v1-s2-qtrf-checkpoint-v1"
S2_QTRF_CHECKPOINT_KIND = "query-token-residual-fusion"
S2_QTRF_KEYS = {
    "state_query.weight",
    "question_key.weight",
    "question_value.weight",
    "fusion_up.weight",
}


def load_hira_v1_s2_qtrf_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_w34_sha256: str,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = file_sha256(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RuntimeError(
            f"Hira v1 S2 QTRF checkpoint SHA mismatch: {actual_sha} != {expected_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != S2_QTRF_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected Hira v1 S2 QTRF checkpoint schema")
    if payload.get("kind") != S2_QTRF_CHECKPOINT_KIND:
        raise RuntimeError("unexpected Hira v1 S2 QTRF checkpoint kind")
    if int(payload.get("fusion_parameter_count", -1)) != HIRA_V1_S2_FUSION_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S2 fusion parameter count changed")
    if int(payload.get("candidate_parameter_count", -1)) != HIRA_V1_S2_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S2 candidate parameter count changed")
    if payload.get("t0_checkpoint_sha256") != expected_t0_sha256:
        raise RuntimeError("Hira v1 S2 T0 identity changed")
    if payload.get("w34_checkpoint_sha256") != expected_w34_sha256:
        raise RuntimeError("Hira v1 S2 W34 identity changed")

    state = payload.get("fusion_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Hira v1 S2 checkpoint missing fusion state dict")
    state = {str(k): v for k, v in state.items()}
    if set(state) != S2_QTRF_KEYS:
        raise RuntimeError("Hira v1 S2 checkpoint keys changed")

    shapes = {
        "state_query.weight": (16, 128),
        "question_key.weight": (16, 128),
        "question_value.weight": (16, 128),
        "fusion_up.weight": (128, 16),
    }
    for key, shape in shapes.items():
        value = state[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(f"Hira v1 S2 tensor shape changed: {key}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Hira v1 S2 tensor is non-finite: {key}")

    epoch = int(payload.get("selected_dev_epoch", -1))
    if not (1 <= epoch <= 16):
        raise RuntimeError("Hira v1 S2 selected DEV epoch is invalid")

    return (
        {key: value.detach().float().clone() for key, value in state.items()},
        {
            "sha256": actual_sha,
            "selected_dev_epoch": epoch,
            "fusion_parameter_count": HIRA_V1_S2_FUSION_PARAMETER_COUNT,
            "candidate_parameter_count": HIRA_V1_S2_CANDIDATE_PARAMETER_COUNT,
        },
    )


def build_frozen_hira_v1_s2_candidate(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    w34_checkpoint_path: str | Path,
    qtrf_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_w34_sha256: str,
    expected_qtrf_sha256: str | None = None,
):
    runtime = build_hira_v1_s2_fusion_core(
        encoder,
        t0_checkpoint_path,
        w34_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        expected_w34_sha256=expected_w34_sha256,
        train_fusion=True,
    )
    state, metadata = load_hira_v1_s2_qtrf_checkpoint(
        qtrf_checkpoint_path,
        expected_sha256=expected_qtrf_sha256,
        expected_t0_sha256=expected_t0_sha256,
        expected_w34_sha256=expected_w34_sha256,
    )
    scorer = runtime.query_token_residual_fusion_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S2 scorer missing during frozen replay")
    scorer.load_fusion_state_dict(state, freeze=True)
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"Hira v1 S2 frozen replay has trainable parameters: {trainable}")
    runtime.eval()
    return runtime, metadata


__all__ = [
    "S2_QTRF_CHECKPOINT_KIND",
    "S2_QTRF_CHECKPOINT_SCHEMA",
    "S2_QTRF_KEYS",
    "build_frozen_hira_v1_s2_candidate",
    "load_hira_v1_s2_qtrf_checkpoint",
]
