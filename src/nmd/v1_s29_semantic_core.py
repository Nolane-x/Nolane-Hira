from __future__ import annotations

from pathlib import Path

from .semantic import HFAutoSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .v1_a13_fullblock_lora import (
    FULL_BLOCK_LORA_PARAMETER_COUNT,
    inject_a13_last_ffn_lora,
    iter_a13_full_block_lora_modules,
)
from .v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)

HIRA_V1_S29_LORA_PARAMETER_COUNT = FULL_BLOCK_LORA_PARAMETER_COUNT
HIRA_V1_S29_PROJECTION_PARAMETER_COUNT = HIRA_V1_S17_PROJECTION_PARAMETER_COUNT
HIRA_V1_S29_TOTAL_PARAMETER_COUNT = (
    HIRA_V1_S29_LORA_PARAMETER_COUNT
    + HIRA_V1_S29_PROJECTION_PARAMETER_COUNT
)


def build_hira_v1_s29_full_block_lora_core(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    train_lora: bool = True,
    train_projection: bool = True,
):
    # Build the exact S17 shell first. This installs the four attention LoRA
    # modules and the shared trainable/frozen projection according to flags.
    runtime = build_hira_v1_s17_norm_balanced_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=train_lora,
        train_projection=train_projection,
    )

    # S29 changes only encoder adaptation coverage: add the two zero-init FFN
    # LoRA adapters to the already-existing final attention LoRA surface.
    ffn = inject_a13_last_ffn_lora(
        runtime.encoder,
        rank=8,
        alpha=8.0,
        dropout=0.0,
    )
    if not train_lora:
        for module in ffn:
            module.lora_a.requires_grad_(False)
            module.lora_b.requires_grad_(False)

    modules = iter_a13_full_block_lora_modules(runtime.encoder)
    if len(modules) != 6:
        raise RuntimeError("Hira v1 S29 expected exactly six LoRA modules")

    lora_trainable = sum(
        p.numel()
        for module in modules
        for p in (module.lora_a, module.lora_b)
        if p.requires_grad
    )
    expected_lora = HIRA_V1_S29_LORA_PARAMETER_COUNT if train_lora else 0
    if lora_trainable != expected_lora:
        raise RuntimeError(
            f"Hira v1 S29 LoRA optimization surface changed: "
            f"{lora_trainable} != {expected_lora}"
        )

    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    expected = (
        (HIRA_V1_S29_LORA_PARAMETER_COUNT if train_lora else 0)
        + (HIRA_V1_S29_PROJECTION_PARAMETER_COUNT if train_projection else 0)
    )
    if trainable != expected:
        raise RuntimeError(
            f"Hira v1 S29 optimization surface changed: {trainable} != {expected}"
        )

    original_a13 = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original_a13 != 0:
        raise RuntimeError("Hira v1 S29 original A13 parameters became trainable")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("Hira v1 S29 HIRACore became trainable")

    enforce_s17_eval(runtime)
    return runtime


def enforce_s29_eval(runtime) -> None:
    enforce_s17_eval(runtime)


__all__ = [
    "HIRA_V1_S29_LORA_PARAMETER_COUNT",
    "HIRA_V1_S29_PROJECTION_PARAMETER_COUNT",
    "HIRA_V1_S29_TOTAL_PARAMETER_COUNT",
    "build_hira_v1_s29_full_block_lora_core",
    "enforce_s29_eval",
]
