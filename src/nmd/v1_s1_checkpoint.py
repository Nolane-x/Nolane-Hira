from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .typed_competitive_cache import file_sha256
from .v1_s1_semantic_core import (
    HIRA_V1_S1_CANDIDATE_PARAMETER_COUNT,
    HIRA_V1_S1_EXTRACTOR_PARAMETER_COUNT,
    build_hira_v1_s1_query_keyed_core,
)

S1_QKEE_CHECKPOINT_SCHEMA = "hira-v1-s1-qkee-checkpoint-v1"
S1_QKEE_CHECKPOINT_KIND = "query-keyed-evidence"
S1_QKEE_KEYS = {
    "question_heads.weight",
    "state_keys.weight",
}


def load_hira_v1_s1_qkee_checkpoint(
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
            f"Hira v1 S1 QKEE checkpoint SHA mismatch: "
            f"{actual_sha} != {expected_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != S1_QKEE_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected Hira v1 S1 QKEE checkpoint schema")
    if payload.get("kind") != S1_QKEE_CHECKPOINT_KIND:
        raise RuntimeError("unexpected Hira v1 S1 QKEE checkpoint kind")
    if int(payload.get("extractor_parameter_count", -1)) != HIRA_V1_S1_EXTRACTOR_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S1 extractor parameter count changed")
    if int(payload.get("candidate_parameter_count", -1)) != HIRA_V1_S1_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S1 candidate parameter count changed")
    if payload.get("t0_checkpoint_sha256") != expected_t0_sha256:
        raise RuntimeError("Hira v1 S1 T0 identity changed")
    if payload.get("w34_checkpoint_sha256") != expected_w34_sha256:
        raise RuntimeError("Hira v1 S1 W34 identity changed")

    state = payload.get("extractor_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Hira v1 S1 checkpoint missing extractor state dict")
    state = {str(key): value for key, value in state.items()}
    if set(state) != S1_QKEE_KEYS:
        raise RuntimeError("Hira v1 S1 checkpoint keys changed")

    shapes = {
        "question_heads.weight": (32, 128),
        "state_keys.weight": (32, 128),
    }
    for key, shape in shapes.items():
        value = state[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(f"Hira v1 S1 tensor shape changed: {key}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Hira v1 S1 tensor is non-finite: {key}")

    epoch = int(payload.get("selected_dev_epoch", -1))
    if not (1 <= epoch <= 12):
        raise RuntimeError("Hira v1 S1 selected DEV epoch is invalid")

    return (
        {key: value.detach().float().clone() for key, value in state.items()},
        {
            "sha256": actual_sha,
            "selected_dev_epoch": epoch,
            "extractor_parameter_count": HIRA_V1_S1_EXTRACTOR_PARAMETER_COUNT,
            "candidate_parameter_count": HIRA_V1_S1_CANDIDATE_PARAMETER_COUNT,
        },
    )


def build_frozen_hira_v1_s1_candidate(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    w34_checkpoint_path: str | Path,
    qkee_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_w34_sha256: str,
    expected_qkee_sha256: str | None = None,
):
    runtime = build_hira_v1_s1_query_keyed_core(
        encoder,
        t0_checkpoint_path,
        w34_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        expected_w34_sha256=expected_w34_sha256,
        train_extractor=True,
    )
    state, metadata = load_hira_v1_s1_qkee_checkpoint(
        qkee_checkpoint_path,
        expected_sha256=expected_qkee_sha256,
        expected_t0_sha256=expected_t0_sha256,
        expected_w34_sha256=expected_w34_sha256,
    )
    scorer = runtime.query_keyed_evidence_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S1 scorer missing during frozen replay")
    scorer.load_extractor_state_dict(state, freeze=True)

    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(
            f"Hira v1 S1 frozen replay has trainable parameters: {trainable}"
        )
    runtime.eval()
    return runtime, metadata


__all__ = [
    "S1_QKEE_CHECKPOINT_KIND",
    "S1_QKEE_CHECKPOINT_SCHEMA",
    "S1_QKEE_KEYS",
    "build_frozen_hira_v1_s1_candidate",
    "load_hira_v1_s1_qkee_checkpoint",
]
