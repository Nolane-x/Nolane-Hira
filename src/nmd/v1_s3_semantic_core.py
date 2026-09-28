from __future__ import annotations

from pathlib import Path

from .hira import HIRACore
from .runtime import NolaneHira
from .semantic import TextSemanticEncoder
from .semantic_core import (
    W28_T0_CHECKPOINT_SHA256,
    load_rescued_projection_checkpoint,
)
from .v1_triadic_semantic import (
    ParameterFreeTriadicScorer,
    TriadicCPSemanticScorer,
)

HIRA_V1_S3_FACTOR_PARAMETER_COUNT = 12_288


def _freeze_encoder_hira(encoder: TextSemanticEncoder) -> HIRACore:
    if int(encoder.d_model) != 256:
        raise ValueError("Hira v1 S3 requires d_model=256")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()

    hira = HIRACore(d_model=256, dropout=0.0)
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    hira.eval()
    return hira


def build_hira_v1_s3_parameter_free_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
) -> NolaneHira:
    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path,
        expected_sha256=expected_t0_sha256,
    )
    hira = _freeze_encoder_hira(encoder)
    scorer = ParameterFreeTriadicScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(projection, freeze=True)

    if scorer.added_parameter_count != 0:
        raise RuntimeError("S3 A0 unexpectedly added parameters")
    if scorer.trainable_parameter_count != 0:
        raise RuntimeError("S3 A0 scorer must be frozen")

    runtime = NolaneHira(
        encoder,
        hira=hira,
        parameter_free_triadic_scorer=scorer,
    )
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad) != 0:
        raise RuntimeError("S3 A0 runtime unexpectedly trainable")
    return runtime


def build_hira_v1_s3_triadic_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    train_factors: bool = True,
) -> NolaneHira:
    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path,
        expected_sha256=expected_t0_sha256,
    )
    hira = _freeze_encoder_hira(encoder)
    scorer = TriadicCPSemanticScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(projection, freeze=True)

    if scorer.factor_parameter_count != HIRA_V1_S3_FACTOR_PARAMETER_COUNT:
        raise RuntimeError("S3 factor parameter count changed")
    if not train_factors:
        scorer.freeze_factors()

    runtime = NolaneHira(
        encoder,
        hira=hira,
        triadic_cp_scorer=scorer,
    )
    trainable = sum(
        p.numel()
        for p in runtime.parameters()
        if p.requires_grad
    )
    expected = HIRA_V1_S3_FACTOR_PARAMETER_COUNT if train_factors else 0
    if trainable != expected:
        raise RuntimeError(
            f"S3 optimization surface changed: {trainable} != {expected}"
        )
    return runtime


__all__ = [
    "HIRA_V1_S3_FACTOR_PARAMETER_COUNT",
    "build_hira_v1_s3_parameter_free_core",
    "build_hira_v1_s3_triadic_core",
]
