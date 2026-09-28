from __future__ import annotations

from pathlib import Path

from .hira import HIRACore
from .runtime import NolaneHira
from .semantic import HFAutoSemanticEncoder
from .semantic_core import (
    W28_T0_CHECKPOINT_SHA256,
    load_rescued_projection_checkpoint,
)
from .v1_a13_lora import inject_a13_last_attention_lora, iter_a13_lora_modules
from .v1_triadic_semantic import ParameterFreeTriadicScorer

HIRA_V1_S6_LORA_RANK = 8
HIRA_V1_S6_LORA_ALPHA = 8.0
HIRA_V1_S6_LORA_DROPOUT = 0.0
HIRA_V1_S6_LORA_PARAMETER_COUNT = 16_384


def build_hira_v1_s6_a13_lora_core(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    train_lora: bool = True,
) -> NolaneHira:
    if not isinstance(encoder, HFAutoSemanticEncoder):
        raise TypeError("Hira v1 S6 requires HFAutoSemanticEncoder")
    if int(encoder.d_model) != 256:
        raise ValueError("Hira v1 S6 requires d_model=256")

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
        raise RuntimeError("Hira v1 S6 LoRA parameter count changed")
    if receipt.original_trainable_parameters != 0:
        raise RuntimeError("Hira v1 S6 original A13 weights became trainable")

    if not train_lora:
        for module in iter_a13_lora_modules(encoder):
            module.lora_a.requires_grad_(False)
            module.lora_b.requires_grad_(False)

    # Keep the frozen A13 base deterministic. Gradients still flow through the
    # eval-mode graph into LoRA tensors because requires_grad remains enabled.
    encoder.eval()

    hira = HIRACore(d_model=256, dropout=0.0)
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    hira.eval()

    scorer = ParameterFreeTriadicScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(projection, freeze=True)
    scorer.eval()

    runtime = NolaneHira(
        encoder,
        hira=hira,
        parameter_free_triadic_scorer=scorer,
    )
    runtime.eval()

    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    expected = HIRA_V1_S6_LORA_PARAMETER_COUNT if train_lora else 0
    if trainable != expected:
        raise RuntimeError(
            f"Hira v1 S6 optimization surface changed: {trainable} != {expected}"
        )

    if scorer.trainable_parameter_count != 0:
        raise RuntimeError("Hira v1 S6 W28 scorer must remain frozen")
    if any(p.requires_grad for p in hira.parameters()):
        raise RuntimeError("Hira v1 S6 HIRACore must remain frozen")

    return runtime


def enforce_s6_encoder_eval(runtime: NolaneHira) -> None:
    """Restore deterministic frozen-base eval mode during LoRA optimization."""
    runtime.encoder.eval()
    runtime.hira.eval()
    scorer = runtime.parameter_free_triadic_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S6 parameter-free triadic scorer missing")
    scorer.eval()


__all__ = [
    "HIRA_V1_S6_LORA_ALPHA",
    "HIRA_V1_S6_LORA_DROPOUT",
    "HIRA_V1_S6_LORA_PARAMETER_COUNT",
    "HIRA_V1_S6_LORA_RANK",
    "build_hira_v1_s6_a13_lora_core",
    "enforce_s6_encoder_eval",
]
