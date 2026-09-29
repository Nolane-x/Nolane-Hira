from __future__ import annotations

from pathlib import Path

from .semantic import HFAutoSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .v1_s12_semantic_core import (
    HIRA_V1_S12_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S12_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s12_relation_binding_core,
    enforce_s12_eval,
)

HIRA_V1_S13_PROJECTION_PARAMETER_COUNT = HIRA_V1_S12_PROJECTION_PARAMETER_COUNT
HIRA_V1_S13_TOTAL_PARAMETER_COUNT = HIRA_V1_S12_TOTAL_PARAMETER_COUNT


def build_hira_v1_s13_canonicalization_core(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    train_lora: bool = True,
    train_projection: bool = True,
):
    runtime = build_hira_v1_s12_relation_binding_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=train_lora,
        train_projection=train_projection,
    )
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    expected = (
        (16_384 if train_lora else 0)
        + (32_768 if train_projection else 0)
    )
    if trainable != expected:
        raise RuntimeError(
            f"Hira v1 S13 optimization surface changed: {trainable} != {expected}"
        )
    return runtime


def enforce_s13_eval(runtime) -> None:
    enforce_s12_eval(runtime)


__all__ = [
    "HIRA_V1_S13_PROJECTION_PARAMETER_COUNT",
    "HIRA_V1_S13_TOTAL_PARAMETER_COUNT",
    "build_hira_v1_s13_canonicalization_core",
    "enforce_s13_eval",
]
