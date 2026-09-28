from __future__ import annotations

from pathlib import Path

from .semantic import HFAutoSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .v1_s7_semantic_core import (
    HIRA_V1_S7_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S7_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s7_coadapt_core,
    enforce_s7_eval,
)

HIRA_V1_S8_PROJECTION_PARAMETER_COUNT = HIRA_V1_S7_PROJECTION_PARAMETER_COUNT
HIRA_V1_S8_TOTAL_PARAMETER_COUNT = HIRA_V1_S7_TOTAL_PARAMETER_COUNT


def build_hira_v1_s8_invariant_core(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    train_lora: bool = True,
    train_projection: bool = True,
):
    runtime = build_hira_v1_s7_coadapt_core(
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
            f"Hira v1 S8 inherited optimization surface changed: "
            f"{trainable} != {expected}"
        )
    return runtime


def enforce_s8_eval(runtime) -> None:
    enforce_s7_eval(runtime)


__all__ = [
    "HIRA_V1_S8_PROJECTION_PARAMETER_COUNT",
    "HIRA_V1_S8_TOTAL_PARAMETER_COUNT",
    "build_hira_v1_s8_invariant_core",
    "enforce_s8_eval",
]
