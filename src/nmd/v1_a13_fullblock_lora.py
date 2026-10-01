from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .semantic import HFAutoSemanticEncoder
from .v1_a13_lora import (
    LoRALinear,
    inject_a13_last_attention_lora,
    iter_a13_lora_modules,
)


ATTENTION_LORA_PARAMETER_COUNT = 16_384
FFN_INTERMEDIATE_LORA_PARAMETER_COUNT = 10_240
FFN_OUTPUT_LORA_PARAMETER_COUNT = 10_240
FFN_LORA_PARAMETER_COUNT = (
    FFN_INTERMEDIATE_LORA_PARAMETER_COUNT
    + FFN_OUTPUT_LORA_PARAMETER_COUNT
)
FULL_BLOCK_LORA_PARAMETER_COUNT = (
    ATTENTION_LORA_PARAMETER_COUNT
    + FFN_LORA_PARAMETER_COUNT
)

_FFN_TARGETS = (
    ("intermediate.dense", ("intermediate", "dense"), 256, 1024),
    ("output.dense", ("output", "dense"), 1024, 256),
)


@dataclass(frozen=True)
class A13FullBlockLoRAReceipt:
    rank: int
    alpha: float
    dropout: float
    layer_index: int
    wrapped_modules: tuple[str, ...]
    attention_trainable_parameters: int
    ffn_trainable_parameters: int
    trainable_parameters: int
    original_trainable_parameters: int

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "layer_index": self.layer_index,
            "wrapped_modules": list(self.wrapped_modules),
            "attention_trainable_parameters":
                self.attention_trainable_parameters,
            "ffn_trainable_parameters": self.ffn_trainable_parameters,
            "trainable_parameters": self.trainable_parameters,
            "original_trainable_parameters":
                self.original_trainable_parameters,
        }


def _resolve_attr(root: object, path: tuple[str, ...]) -> object:
    current = root
    for name in path:
        if not hasattr(current, name):
            raise RuntimeError(
                "Hira v1 S29 pinned A13 module path missing: "
                + ".".join(path)
            )
        current = getattr(current, name)
    return current


def _replace_attr(
    root: object,
    path: tuple[str, ...],
    value: object,
) -> None:
    parent = _resolve_attr(root, path[:-1]) if len(path) > 1 else root
    setattr(parent, path[-1], value)


def inject_a13_last_full_block_lora(
    encoder: HFAutoSemanticEncoder,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> A13FullBlockLoRAReceipt:
    """Inject zero-init LoRA into final A13 attention + FFN linears.

    Existing S6 attention-only semantics are reused unchanged. S29 then wraps
    exactly the two final-block feed-forward dense layers.
    """

    attention = inject_a13_last_attention_lora(
        encoder,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    if attention.trainable_parameters != ATTENTION_LORA_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S29 inherited attention LoRA count changed")

    final_layer = encoder.model.encoder.layer[-1]
    wrapped = list(attention.wrapped_modules)
    ffn_count = 0

    for label, path, in_features, out_features in _FFN_TARGETS:
        base = _resolve_attr(final_layer, path)
        if isinstance(base, LoRALinear):
            raise RuntimeError(f"Hira v1 S29 FFN LoRA already injected: {label}")
        if not isinstance(base, nn.Linear):
            raise RuntimeError(f"Hira v1 S29 FFN target is not nn.Linear: {label}")
        if (
            base.in_features != in_features
            or base.out_features != out_features
        ):
            raise RuntimeError(
                "Hira v1 S29 FFN target shape changed: "
                f"{label}={base.in_features}->{base.out_features}"
            )
        adapter = LoRALinear(
            base,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        expected = rank * (in_features + out_features)
        if adapter.lora_parameter_count != expected:
            raise RuntimeError(
                f"Hira v1 S29 FFN LoRA parameter count changed: {label}"
            )
        _replace_attr(final_layer, path, adapter)
        ffn_count += adapter.lora_parameter_count
        wrapped.append(label)

    if ffn_count != FFN_LORA_PARAMETER_COUNT:
        raise RuntimeError(
            f"Hira v1 S29 FFN LoRA surface changed: "
            f"{ffn_count} != {FFN_LORA_PARAMETER_COUNT}"
        )

    trainable = sum(
        p.numel()
        for p in encoder.model.parameters()
        if p.requires_grad
    )
    if trainable != FULL_BLOCK_LORA_PARAMETER_COUNT:
        raise RuntimeError(
            f"Hira v1 S29 full-block LoRA surface changed: "
            f"{trainable} != {FULL_BLOCK_LORA_PARAMETER_COUNT}"
        )

    original_trainable = sum(
        p.numel()
        for name, p in encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original_trainable != 0:
        raise RuntimeError(
            "Hira v1 S29 original A13 parameters became trainable"
        )

    return A13FullBlockLoRAReceipt(
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        layer_index=5,
        wrapped_modules=tuple(wrapped),
        attention_trainable_parameters=
            ATTENTION_LORA_PARAMETER_COUNT,
        ffn_trainable_parameters=ffn_count,
        trainable_parameters=trainable,
        original_trainable_parameters=original_trainable,
    )


def iter_a13_full_block_lora_modules(
    encoder: HFAutoSemanticEncoder,
) -> tuple[LoRALinear, ...]:
    attention = iter_a13_lora_modules(encoder)
    final_layer = encoder.model.encoder.layer[-1]
    ffn = []
    for label, path, _in_features, _out_features in _FFN_TARGETS:
        module = _resolve_attr(final_layer, path)
        if not isinstance(module, LoRALinear):
            raise RuntimeError(
                f"Hira v1 S29 expected FFN LoRA module missing: {label}"
            )
        ffn.append(module)
    modules = (*attention, *ffn)
    if len(modules) != 6:
        raise RuntimeError("Hira v1 S29 expected exactly six LoRA modules")
    return tuple(modules)


def a13_full_block_lora_state_dict(
    encoder: HFAutoSemanticEncoder,
) -> dict[str, Tensor]:
    state = {}
    for index, module in enumerate(
        iter_a13_full_block_lora_modules(encoder)
    ):
        state[f"lora.{index}.a"] = (
            module.lora_a.detach().cpu().clone()
        )
        state[f"lora.{index}.b"] = (
            module.lora_b.detach().cpu().clone()
        )
    return state


def load_a13_full_block_lora_state_dict(
    encoder: HFAutoSemanticEncoder,
    state_dict: dict[str, Tensor],
    *,
    freeze: bool = True,
) -> None:
    modules = iter_a13_full_block_lora_modules(encoder)
    expected = (
        {f"lora.{i}.a" for i in range(len(modules))}
        | {f"lora.{i}.b" for i in range(len(modules))}
    )
    if set(state_dict) != expected:
        raise ValueError("Hira v1 S29 full-block LoRA checkpoint keys changed")

    import torch

    with torch.no_grad():
        for i, module in enumerate(modules):
            a = state_dict[f"lora.{i}.a"]
            b = state_dict[f"lora.{i}.b"]
            if tuple(a.shape) != tuple(module.lora_a.shape):
                raise ValueError(
                    f"Hira v1 S29 LoRA A shape mismatch: {i}"
                )
            if tuple(b.shape) != tuple(module.lora_b.shape):
                raise ValueError(
                    f"Hira v1 S29 LoRA B shape mismatch: {i}"
                )
            if not bool(
                torch.isfinite(a).all()
                and torch.isfinite(b).all()
            ):
                raise ValueError(
                    f"Hira v1 S29 LoRA tensor non-finite: {i}"
                )
            module.lora_a.copy_(
                a.to(
                    device=module.lora_a.device,
                    dtype=module.lora_a.dtype,
                )
            )
            module.lora_b.copy_(
                b.to(
                    device=module.lora_b.device,
                    dtype=module.lora_b.dtype,
                )
            )
            module.lora_a.requires_grad_(not freeze)
            module.lora_b.requires_grad_(not freeze)


__all__ = [
    "ATTENTION_LORA_PARAMETER_COUNT",
    "FFN_INTERMEDIATE_LORA_PARAMETER_COUNT",
    "FFN_OUTPUT_LORA_PARAMETER_COUNT",
    "FFN_LORA_PARAMETER_COUNT",
    "FULL_BLOCK_LORA_PARAMETER_COUNT",
    "A13FullBlockLoRAReceipt",
    "inject_a13_last_full_block_lora",
    "iter_a13_full_block_lora_modules",
    "a13_full_block_lora_state_dict",
    "load_a13_full_block_lora_state_dict",
]
