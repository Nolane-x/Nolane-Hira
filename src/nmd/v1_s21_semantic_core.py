from __future__ import annotations

from pathlib import Path

from .hira import HIRACore
from .runtime import NolaneHira
from .semantic import HFAutoSemanticEncoder
from .semantic_core import (
    W28_T0_CHECKPOINT_SHA256,
    load_rescued_projection_checkpoint,
)
from .v1_a13_lora import (
    inject_a13_last_attention_lora,
    iter_a13_lora_modules,
)
from .v1_role_gated_content_triadic import RoleGatedContentTriadicScorer
from .v1_s6_semantic_core import (
    HIRA_V1_S6_LORA_ALPHA,
    HIRA_V1_S6_LORA_DROPOUT,
    HIRA_V1_S6_LORA_PARAMETER_COUNT,
    HIRA_V1_S6_LORA_RANK,
)

HIRA_V1_S21_PROJECTION_PARAMETER_COUNT = 32_768
HIRA_V1_S21_TOTAL_PARAMETER_COUNT = (
    HIRA_V1_S6_LORA_PARAMETER_COUNT
    + HIRA_V1_S21_PROJECTION_PARAMETER_COUNT
)


def build_hira_v1_s21_role_content_core(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    train_lora: bool = True,
    train_projection: bool = True,
) -> NolaneHira:
    if not isinstance(encoder, HFAutoSemanticEncoder):
        raise TypeError("Hira v1 S21 requires HFAutoSemanticEncoder")
    if int(encoder.d_model) != 256:
        raise ValueError("Hira v1 S21 requires d_model=256")

    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path,
        expected_sha256=expected_t0_sha256,
    )

    receipt = inject_a13_last_attention_lora(
        encoder,
        rank=HIRA_V1_S6_LORA_RANK,
        alpha=HIRA_V1_S6_LORA_ALPHA,
        dropout=HIRA_V1_S6_LORA_DROPOUT,
    )
    if receipt.trainable_parameters != HIRA_V1_S6_LORA_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S21 LoRA parameter count changed")
    if receipt.original_trainable_parameters != 0:
        raise RuntimeError("Hira v1 S21 original A13 weights became trainable")

    if not train_lora:
        for module in iter_a13_lora_modules(encoder):
            module.lora_a.requires_grad_(False)
            module.lora_b.requires_grad_(False)

    encoder.eval()

    hira = HIRACore(d_model=256, dropout=0.0)
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    hira.eval()

    scorer = RoleGatedContentTriadicScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(
        projection,
        freeze=not train_projection,
    )
    if scorer.projection_parameter_count != HIRA_V1_S21_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S21 projection parameter count changed")
    if scorer.factorization_added_parameter_count != 0:
        raise RuntimeError("Hira v1 S21 factorization unexpectedly added parameters")

    runtime = NolaneHira(
        encoder,
        hira=hira,
        projection_triadic_scorer=scorer,
    )
    runtime.eval()

    expected = (
        (HIRA_V1_S6_LORA_PARAMETER_COUNT if train_lora else 0)
        + (
            HIRA_V1_S21_PROJECTION_PARAMETER_COUNT
            if train_projection
            else 0
        )
    )
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable != expected:
        raise RuntimeError(
            f"Hira v1 S21 optimization surface changed: {trainable} != {expected}"
        )

    original_a13_trainable = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original_a13_trainable != 0:
        raise RuntimeError("Hira v1 S21 original A13 weights became trainable")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("Hira v1 S21 HIRACore became trainable")

    return runtime


def enforce_s21_eval(runtime: NolaneHira) -> None:
    runtime.encoder.eval()
    runtime.hira.eval()
    scorer = runtime.projection_triadic_scorer
    if not isinstance(scorer, RoleGatedContentTriadicScorer):
        raise RuntimeError("Hira v1 S21 role-content scorer missing")
    scorer.eval()


__all__ = [
    "HIRA_V1_S21_PROJECTION_PARAMETER_COUNT",
    "HIRA_V1_S21_TOTAL_PARAMETER_COUNT",
    "build_hira_v1_s21_role_content_core",
    "enforce_s21_eval",
]
