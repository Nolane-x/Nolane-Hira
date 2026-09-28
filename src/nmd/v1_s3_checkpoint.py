from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .typed_competitive_cache import file_sha256
from .v1_s3_semantic_core import (
    HIRA_V1_S3_FACTOR_PARAMETER_COUNT,
    build_hira_v1_s3_triadic_core,
)

S3_TRIADIC_CHECKPOINT_SCHEMA = "hira-v1-s3-triadic-checkpoint-v1"
S3_TRIADIC_CHECKPOINT_KIND = "triadic-cp-semantic"
S3_TRIADIC_KEYS = {
    "state_factor.weight",
    "question_factor.weight",
    "option_factor.weight",
}


def load_hira_v1_s3_triadic_checkpoint(
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
            f"Hira v1 S3 checkpoint SHA mismatch: {actual_sha} != {expected_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != S3_TRIADIC_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected Hira v1 S3 checkpoint schema")
    if payload.get("kind") != S3_TRIADIC_CHECKPOINT_KIND:
        raise RuntimeError("unexpected Hira v1 S3 checkpoint kind")
    if int(payload.get("factor_parameter_count", -1)) != HIRA_V1_S3_FACTOR_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S3 factor parameter count changed")
    if payload.get("t0_checkpoint_sha256") != expected_t0_sha256:
        raise RuntimeError("Hira v1 S3 T0 identity changed")

    state = payload.get("factor_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Hira v1 S3 checkpoint missing factor state dict")
    state = {str(key): value for key, value in state.items()}
    if set(state) != S3_TRIADIC_KEYS:
        raise RuntimeError("Hira v1 S3 checkpoint keys changed")

    for key in S3_TRIADIC_KEYS:
        value = state[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != (32, 128):
            raise RuntimeError(f"Hira v1 S3 factor shape changed: {key}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Hira v1 S3 factor tensor non-finite: {key}")

    epoch = int(payload.get("selected_dev_epoch", -1))
    if not (1 <= epoch <= 20):
        raise RuntimeError("Hira v1 S3 selected DEV epoch is invalid")

    return (
        {key: value.detach().float().clone() for key, value in state.items()},
        {
            "sha256": actual_sha,
            "selected_dev_epoch": epoch,
            "factor_parameter_count": HIRA_V1_S3_FACTOR_PARAMETER_COUNT,
        },
    )


def build_frozen_hira_v1_s3_candidate(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    triadic_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_triadic_sha256: str | None = None,
):
    runtime = build_hira_v1_s3_triadic_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_factors=True,
    )
    state, metadata = load_hira_v1_s3_triadic_checkpoint(
        triadic_checkpoint_path,
        expected_sha256=expected_triadic_sha256,
        expected_t0_sha256=expected_t0_sha256,
    )
    scorer = runtime.triadic_cp_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S3 scorer missing during frozen replay")
    scorer.load_factor_state_dict(state, freeze=True)

    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(
            f"Hira v1 S3 frozen replay has trainable parameters: {trainable}"
        )
    runtime.eval()
    return runtime, metadata


__all__ = [
    "S3_TRIADIC_CHECKPOINT_KIND",
    "S3_TRIADIC_CHECKPOINT_SCHEMA",
    "S3_TRIADIC_KEYS",
    "build_frozen_hira_v1_s3_candidate",
    "load_hira_v1_s3_triadic_checkpoint",
]
