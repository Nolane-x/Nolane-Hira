from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .semantic import HFAutoSemanticEncoder


class LoRALinear(nn.Module):
    """Frozen linear layer with a zero-initialized low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear base must be nn.Linear")
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0,1)")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.dropout_p = float(dropout)

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        self.lora_a = nn.Parameter(
            torch.empty(self.rank, self.base.in_features)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(self.base.out_features, self.rank)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    @property
    def lora_parameter_count(self) -> int:
        return self.lora_a.numel() + self.lora_b.numel()

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        adapted_input = F.dropout(
            x,
            p=self.dropout_p,
            training=self.training,
        )
        low_rank = F.linear(adapted_input, self.lora_a)
        residual = F.linear(low_rank, self.lora_b)
        return base_out + residual * self.scale


@dataclass(frozen=True)
class A13LoRAReceipt:
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


_TARGETS = (
    ("attention.self.query", ("attention", "self", "query")),
    ("attention.self.key", ("attention", "self", "key")),
    ("attention.self.value", ("attention", "self", "value")),
    ("attention.output.dense", ("attention", "output", "dense")),
)


def _resolve_attr(root: object, path: tuple[str, ...]) -> object:
    current = root
    for name in path:
        if not hasattr(current, name):
            raise RuntimeError(
                "Hira v1 S6 pinned A13 module path missing: "
                + ".".join(path)
            )
        current = getattr(current, name)
    return current


def _replace_attr(root: object, path: tuple[str, ...], value: object) -> None:
    parent = _resolve_attr(root, path[:-1]) if len(path) > 1 else root
    setattr(parent, path[-1], value)


def inject_a13_last_attention_lora(
    encoder: HFAutoSemanticEncoder,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> A13LoRAReceipt:
    """Inject S6 LoRA into the final attention block of pinned A13.

    This is deliberately strict: the pinned A13 must be a 6-layer BERT with
    hidden size 256 and the four target attention linears must all be 256x256.
    """

    if not isinstance(encoder, HFAutoSemanticEncoder):
        raise TypeError("Hira v1 S6 requires HFAutoSemanticEncoder")
    model = encoder.model
    config = model.config

    if getattr(config, "model_type", None) != "bert":
        raise RuntimeError("Hira v1 S6 requires pinned BERT A13")
    if int(getattr(config, "hidden_size", -1)) != 256:
        raise RuntimeError("Hira v1 S6 requires A13 hidden_size=256")
    if int(getattr(config, "num_hidden_layers", -1)) != 6:
        raise RuntimeError("Hira v1 S6 requires A13 num_hidden_layers=6")
    if int(getattr(config, "num_attention_heads", -1)) != 8:
        raise RuntimeError("Hira v1 S6 requires A13 num_attention_heads=8")

    bert_encoder = getattr(model, "encoder", None)
    layers = None if bert_encoder is None else getattr(bert_encoder, "layer", None)
    if layers is None or len(layers) != 6:
        raise RuntimeError("Hira v1 S6 pinned A13 encoder.layer structure changed")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    final_layer = layers[-1]
    wrapped = []
    expected_per_module = 2 * 256 * int(rank)

    for label, path in _TARGETS:
        module = _resolve_attr(final_layer, path)
        if isinstance(module, LoRALinear):
            raise RuntimeError(f"Hira v1 S6 LoRA already injected: {label}")
        if not isinstance(module, nn.Linear):
            raise RuntimeError(f"Hira v1 S6 target is not nn.Linear: {label}")
        if module.in_features != 256 or module.out_features != 256:
            raise RuntimeError(f"Hira v1 S6 target shape changed: {label}")

        adapter = LoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if adapter.lora_parameter_count != expected_per_module:
            raise RuntimeError(f"Hira v1 S6 LoRA parameter count changed: {label}")
        _replace_attr(final_layer, path, adapter)
        wrapped.append(label)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_total = len(_TARGETS) * expected_per_module
    original_trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if trainable != expected_total:
        raise RuntimeError(
            f"Hira v1 S6 trainable surface changed: {trainable} != {expected_total}"
        )
    if original_trainable != 0:
        raise RuntimeError("Hira v1 S6 original A13 parameters became trainable")

    return A13LoRAReceipt(
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        layer_index=5,
        wrapped_modules=tuple(wrapped),
        trainable_parameters=trainable,
        original_trainable_parameters=original_trainable,
    )


def iter_a13_lora_modules(
    encoder: HFAutoSemanticEncoder,
) -> tuple[LoRALinear, ...]:
    model = encoder.model
    final_layer = model.encoder.layer[-1]
    modules = []
    for _, path in _TARGETS:
        module = _resolve_attr(final_layer, path)
        if not isinstance(module, LoRALinear):
            raise RuntimeError("Hira v1 S6 expected injected LoRA module missing")
        modules.append(module)
    return tuple(modules)


def a13_lora_state_dict(
    encoder: HFAutoSemanticEncoder,
) -> dict[str, Tensor]:
    state = {}
    for index, module in enumerate(iter_a13_lora_modules(encoder)):
        state[f"lora.{index}.a"] = module.lora_a.detach().cpu().clone()
        state[f"lora.{index}.b"] = module.lora_b.detach().cpu().clone()
    return state


def load_a13_lora_state_dict(
    encoder: HFAutoSemanticEncoder,
    state_dict: dict[str, Tensor],
    *,
    freeze: bool = True,
) -> None:
    modules = iter_a13_lora_modules(encoder)
    expected = {
        **{f"lora.{i}.a" for i in range(len(modules))},
        **{f"lora.{i}.b" for i in range(len(modules))},
    }
    if set(state_dict) != expected:
        raise ValueError("Hira v1 S6 LoRA checkpoint keys changed")

    with torch.no_grad():
        for i, module in enumerate(modules):
            a = state_dict[f"lora.{i}.a"]
            b = state_dict[f"lora.{i}.b"]
            if tuple(a.shape) != tuple(module.lora_a.shape):
                raise ValueError(f"Hira v1 S6 LoRA A shape mismatch: {i}")
            if tuple(b.shape) != tuple(module.lora_b.shape):
                raise ValueError(f"Hira v1 S6 LoRA B shape mismatch: {i}")
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise ValueError(f"Hira v1 S6 LoRA tensor non-finite: {i}")
            module.lora_a.copy_(
                a.to(device=module.lora_a.device, dtype=module.lora_a.dtype)
            )
            module.lora_b.copy_(
                b.to(device=module.lora_b.device, dtype=module.lora_b.dtype)
            )
            module.lora_a.requires_grad_(not freeze)
            module.lora_b.requires_grad_(not freeze)


__all__ = [
    "A13LoRAReceipt",
    "LoRALinear",
    "a13_lora_state_dict",
    "inject_a13_last_attention_lora",
    "iter_a13_lora_modules",
    "load_a13_lora_state_dict",
]
