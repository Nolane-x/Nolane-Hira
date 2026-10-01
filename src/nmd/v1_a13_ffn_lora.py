from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .semantic import HFAutoSemanticEncoder
from .v1_a13_lora import LoRALinear

FFN_INTERMEDIATE_LORA_PARAMETER_COUNT = 10_240
FFN_OUTPUT_LORA_PARAMETER_COUNT = 10_240
FFN_ONLY_LORA_PARAMETER_COUNT = (
    FFN_INTERMEDIATE_LORA_PARAMETER_COUNT
    + FFN_OUTPUT_LORA_PARAMETER_COUNT
)

_TARGETS = (
    ("intermediate.dense", ("intermediate", "dense"), 256, 1024),
    ("output.dense", ("output", "dense"), 1024, 256),
)


@dataclass(frozen=True)
class A13FFNLoRAReceipt:
    rank: int
    alpha: float
    dropout: float
    layer_index: int
    wrapped_modules: tuple[str, ...]
    trainable_parameters: int
    original_trainable_parameters: int

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "layer_index": self.layer_index,
            "wrapped_modules": list(self.wrapped_modules),
            "trainable_parameters": self.trainable_parameters,
            "original_trainable_parameters": self.original_trainable_parameters,
        }


def _resolve_attr(root: object, path: tuple[str, ...]) -> object:
    current = root
    for name in path:
        if not hasattr(current, name):
            raise RuntimeError(
                "Hira v1 S30 pinned A13 module path missing: "
                + ".".join(path)
            )
        current = getattr(current, name)
    return current


def _replace_attr(root: object, path: tuple[str, ...], value: object) -> None:
    parent = _resolve_attr(root, path[:-1]) if len(path) > 1 else root
    setattr(parent, path[-1], value)


def inject_a13_last_ffn_only_lora(
    encoder: HFAutoSemanticEncoder,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> A13FFNLoRAReceipt:
    if not isinstance(encoder, HFAutoSemanticEncoder):
        raise TypeError("Hira v1 S30 requires HFAutoSemanticEncoder")
    model = encoder.model
    config = model.config
    if getattr(config, "model_type", None) != "bert":
        raise RuntimeError("Hira v1 S30 requires pinned BERT A13")
    if int(getattr(config, "hidden_size", -1)) != 256:
        raise RuntimeError("Hira v1 S30 requires A13 hidden_size=256")
    if int(getattr(config, "num_hidden_layers", -1)) != 6:
        raise RuntimeError("Hira v1 S30 requires A13 num_hidden_layers=6")
    if int(getattr(config, "num_attention_heads", -1)) != 8:
        raise RuntimeError("Hira v1 S30 requires A13 num_attention_heads=8")

    bert_encoder = getattr(model, "encoder", None)
    layers = None if bert_encoder is None else getattr(bert_encoder, "layer", None)
    if layers is None or len(layers) != 6:
        raise RuntimeError("Hira v1 S30 pinned A13 encoder.layer structure changed")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    final_layer = layers[-1]
    wrapped = []
    count = 0
    for label, path, in_features, out_features in _TARGETS:
        module = _resolve_attr(final_layer, path)
        if isinstance(module, LoRALinear):
            raise RuntimeError(f"Hira v1 S30 FFN LoRA already injected: {label}")
        if not isinstance(module, nn.Linear):
            raise RuntimeError(f"Hira v1 S30 FFN target is not nn.Linear: {label}")
        if module.in_features != in_features or module.out_features != out_features:
            raise RuntimeError(
                f"Hira v1 S30 FFN target shape changed: "
                f"{label}={module.in_features}->{module.out_features}"
            )
        adapter = LoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        expected = rank * (in_features + out_features)
        if adapter.lora_parameter_count != expected:
            raise RuntimeError(
                f"Hira v1 S30 FFN LoRA parameter count changed: {label}"
            )
        _replace_attr(final_layer, path, adapter)
        wrapped.append(label)
        count += adapter.lora_parameter_count

    if count != FFN_ONLY_LORA_PARAMETER_COUNT:
        raise RuntimeError(
            f"Hira v1 S30 FFN-only surface changed: "
            f"{count} != {FFN_ONLY_LORA_PARAMETER_COUNT}"
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != FFN_ONLY_LORA_PARAMETER_COUNT:
        raise RuntimeError(
            f"Hira v1 S30 trainable surface changed: "
            f"{trainable} != {FFN_ONLY_LORA_PARAMETER_COUNT}"
        )
    original_trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original_trainable != 0:
        raise RuntimeError("Hira v1 S30 original A13 weights became trainable")

    return A13FFNLoRAReceipt(
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        layer_index=5,
        wrapped_modules=tuple(wrapped),
        trainable_parameters=trainable,
        original_trainable_parameters=original_trainable,
    )


def iter_a13_ffn_only_lora_modules(
    encoder: HFAutoSemanticEncoder,
) -> tuple[LoRALinear, LoRALinear]:
    final_layer = encoder.model.encoder.layer[-1]
    modules = []
    for label, path, _in_features, _out_features in _TARGETS:
        module = _resolve_attr(final_layer, path)
        if not isinstance(module, LoRALinear):
            raise RuntimeError(
                f"Hira v1 S30 expected FFN LoRA module missing: {label}"
            )
        modules.append(module)
    if len(modules) != 2:
        raise RuntimeError("Hira v1 S30 expected exactly two FFN LoRA modules")
    return tuple(modules)  # type: ignore[return-value]


def a13_ffn_only_lora_state_dict(
    encoder: HFAutoSemanticEncoder,
) -> dict[str, Tensor]:
    state = {}
    for index, module in enumerate(iter_a13_ffn_only_lora_modules(encoder)):
        state[f"lora.{index}.a"] = module.lora_a.detach().cpu().clone()
        state[f"lora.{index}.b"] = module.lora_b.detach().cpu().clone()
    return state


def load_a13_ffn_only_lora_state_dict(
    encoder: HFAutoSemanticEncoder,
    state_dict: dict[str, Tensor],
    *,
    freeze: bool = True,
) -> None:
    modules = iter_a13_ffn_only_lora_modules(encoder)
    expected = (
        {f"lora.{i}.a" for i in range(2)}
        | {f"lora.{i}.b" for i in range(2)}
    )
    if set(state_dict) != expected:
        raise ValueError("Hira v1 S30 FFN-only checkpoint keys changed")

    with torch.no_grad():
        for i, module in enumerate(modules):
            a = state_dict[f"lora.{i}.a"]
            b = state_dict[f"lora.{i}.b"]
            if tuple(a.shape) != tuple(module.lora_a.shape):
                raise ValueError(f"Hira v1 S30 LoRA A shape mismatch: {i}")
            if tuple(b.shape) != tuple(module.lora_b.shape):
                raise ValueError(f"Hira v1 S30 LoRA B shape mismatch: {i}")
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise ValueError(f"Hira v1 S30 LoRA tensor non-finite: {i}")
            module.lora_a.copy_(
                a.to(device=module.lora_a.device, dtype=module.lora_a.dtype)
            )
            module.lora_b.copy_(
                b.to(device=module.lora_b.device, dtype=module.lora_b.dtype)
            )
            module.lora_a.requires_grad_(not freeze)
            module.lora_b.requires_grad_(not freeze)


__all__ = [
    "A13FFNLoRAReceipt",
    "FFN_INTERMEDIATE_LORA_PARAMETER_COUNT",
    "FFN_OUTPUT_LORA_PARAMETER_COUNT",
    "FFN_ONLY_LORA_PARAMETER_COUNT",
    "inject_a13_last_ffn_only_lora",
    "iter_a13_ffn_only_lora_modules",
    "a13_ffn_only_lora_state_dict",
    "load_a13_ffn_only_lora_state_dict",
]
