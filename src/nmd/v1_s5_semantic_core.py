from __future__ import annotations

from pathlib import Path

from .hira import HIRACore
from .runtime import NolaneHira
from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256, load_rescued_projection_checkpoint
from .v1_projection_relearning import TrainableProjectionTriadicScorer

HIRA_V1_S5_PROJECTION_PARAMETER_COUNT = 32_768


def build_hira_v1_s5_projection_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    train_projection: bool = True,
) -> NolaneHira:
    if int(encoder.d_model) != 256:
        raise ValueError("Hira v1 S5 requires d_model=256")

    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path,
        expected_sha256=expected_t0_sha256,
    )

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()

    hira = HIRACore(d_model=256, dropout=0.0)
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    hira.eval()

    scorer = TrainableProjectionTriadicScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(projection, freeze=not train_projection)

    if scorer.projection_parameter_count != HIRA_V1_S5_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S5 projection parameter count changed")

    runtime = NolaneHira(
        encoder,
        hira=hira,
        projection_triadic_scorer=scorer,
    )

    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    expected = HIRA_V1_S5_PROJECTION_PARAMETER_COUNT if train_projection else 0
    if trainable != expected:
        raise RuntimeError(
            f"Hira v1 S5 optimization surface changed: {trainable} != {expected}"
        )
    return runtime


__all__ = [
    "HIRA_V1_S5_PROJECTION_PARAMETER_COUNT",
    "build_hira_v1_s5_projection_core",
]
