from __future__ import annotations

from pathlib import Path

from .hira import HIRACore
from .runtime import NolaneHira
from .semantic import HFAutoSemanticEncoder
from .semantic_core import (
    W28_T0_CHECKPOINT_SHA256,
    load_rescued_projection_checkpoint,
)
from .v1_a13_ffn_lora import (
    FFN_ONLY_LORA_PARAMETER_COUNT,
    inject_a13_last_ffn_only_lora,
    iter_a13_ffn_only_lora_modules,
)
from .v1_projection_relearning import TrainableProjectionTriadicScorer
from .v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)

S30_ATTENTION_LORA_PARAMETER_COUNT = 16_384
S30_ATTENTION_TOTAL_PARAMETER_COUNT = HIRA_V1_S17_TOTAL_PARAMETER_COUNT
S30_FFN_LORA_PARAMETER_COUNT = FFN_ONLY_LORA_PARAMETER_COUNT
S30_PROJECTION_PARAMETER_COUNT = HIRA_V1_S17_PROJECTION_PARAMETER_COUNT
S30_FFN_TOTAL_PARAMETER_COUNT = (
    S30_FFN_LORA_PARAMETER_COUNT + S30_PROJECTION_PARAMETER_COUNT
)


def build_s30_attention_arm(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    train_lora: bool = True,
    train_projection: bool = True,
):
    runtime = build_hira_v1_s17_norm_balanced_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=train_lora,
        train_projection=train_projection,
    )
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    expected = (
        (S30_ATTENTION_LORA_PARAMETER_COUNT if train_lora else 0)
        + (S30_PROJECTION_PARAMETER_COUNT if train_projection else 0)
    )
    if trainable != expected:
        raise RuntimeError(
            f"Hira v1 S30 attention arm surface changed: {trainable} != {expected}"
        )
    return runtime


def build_s30_ffn_arm(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    train_lora: bool = True,
    train_projection: bool = True,
) -> NolaneHira:
    if not isinstance(encoder, HFAutoSemanticEncoder):
        raise TypeError("Hira v1 S30 requires HFAutoSemanticEncoder")
    if int(encoder.d_model) != 256:
        raise ValueError("Hira v1 S30 requires d_model=256")

    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path,
        expected_sha256=expected_t0_sha256,
    )
    receipt = inject_a13_last_ffn_only_lora(
        encoder,
        rank=8,
        alpha=8.0,
        dropout=0.0,
    )
    if receipt.trainable_parameters != S30_FFN_LORA_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S30 FFN-only LoRA parameter count changed")
    if receipt.original_trainable_parameters != 0:
        raise RuntimeError("Hira v1 S30 original A13 weights became trainable")

    if not train_lora:
        for module in iter_a13_ffn_only_lora_modules(encoder):
            module.lora_a.requires_grad_(False)
            module.lora_b.requires_grad_(False)
    encoder.eval()

    hira = HIRACore(d_model=256, dropout=0.0)
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    hira.eval()

    scorer = TrainableProjectionTriadicScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(
        projection,
        freeze=not train_projection,
    )
    if scorer.projection_parameter_count != S30_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S30 projection parameter count changed")

    runtime = NolaneHira(
        encoder,
        hira=hira,
        projection_triadic_scorer=scorer,
    )
    runtime.eval()

    expected = (
        (S30_FFN_LORA_PARAMETER_COUNT if train_lora else 0)
        + (S30_PROJECTION_PARAMETER_COUNT if train_projection else 0)
    )
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable != expected:
        raise RuntimeError(
            f"Hira v1 S30 FFN arm surface changed: {trainable} != {expected}"
        )

    original_a13 = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original_a13 != 0:
        raise RuntimeError("Hira v1 S30 original A13 weights became trainable")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("Hira v1 S30 HIRACore became trainable")
    return runtime


def enforce_s30_attention_eval(runtime) -> None:
    enforce_s17_eval(runtime)


def enforce_s30_ffn_eval(runtime) -> None:
    runtime.encoder.eval()
    runtime.hira.eval()
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S30 FFN arm projection scorer missing")
    scorer.eval()


__all__ = [
    "S30_ATTENTION_LORA_PARAMETER_COUNT",
    "S30_ATTENTION_TOTAL_PARAMETER_COUNT",
    "S30_FFN_LORA_PARAMETER_COUNT",
    "S30_PROJECTION_PARAMETER_COUNT",
    "S30_FFN_TOTAL_PARAMETER_COUNT",
    "build_s30_attention_arm",
    "build_s30_ffn_arm",
    "enforce_s30_attention_eval",
    "enforce_s30_ffn_eval",
]
