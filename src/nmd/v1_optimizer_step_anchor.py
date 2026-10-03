from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class AdamWParameterState:
    step: int
    exp_avg: Tensor
    exp_avg_sq: Tensor


@dataclass(frozen=True)
class AdamWStepProjectionDiagnostics:
    candidate_delta_norm: float
    anchor_gradient_norm: float
    pre_dot: float
    post_dot: float
    projected: bool
    projection_coefficient: float

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "candidate_delta_norm": self.candidate_delta_norm,
            "anchor_gradient_norm": self.anchor_gradient_norm,
            "pre_dot": self.pre_dot,
            "post_dot": self.post_dot,
            "projected": self.projected,
            "projection_coefficient": self.projection_coefficient,
        }


def _validate_pair(a: Sequence[Tensor], b: Sequence[Tensor], *, label: str) -> None:
    if not a or len(a) != len(b):
        raise ValueError(f"S41 {label} lists must be nonempty and equal length")
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape or x.device != y.device or x.dtype != y.dtype:
            raise ValueError(f"S41 {label} mismatch at {i}")
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError(f"S41 non-finite {label} at {i}")


def _dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    total = a[0].new_zeros(())
    for x, y in zip(a, b):
        total = total + torch.sum(x * y)
    return total


def _sq(a: Sequence[Tensor]) -> Tensor:
    return _dot(a, a)


def initialize_adamw_state(params: Sequence[Tensor]) -> list[AdamWParameterState]:
    if not params:
        raise ValueError("S41 AdamW requires at least one parameter")
    states = []
    for i, p in enumerate(params):
        if not bool(torch.is_floating_point(p)):
            raise ValueError(f"S41 AdamW parameter {i} must be floating point")
        if not bool(torch.isfinite(p).all()):
            raise ValueError(f"S41 non-finite parameter {i}")
        states.append(
            AdamWParameterState(
                step=0,
                exp_avg=torch.zeros_like(p, memory_format=torch.preserve_format),
                exp_avg_sq=torch.zeros_like(p, memory_format=torch.preserve_format),
            )
        )
    return states


def adamw_candidate_deltas(
    params: Sequence[Tensor],
    grads: Sequence[Tensor],
    states: Sequence[AdamWParameterState],
    *,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[list[Tensor], list[AdamWParameterState]]:
    """Return the exact candidate delta/state of standard PyTorch AdamW.

    S41 constrains the *actual optimizer step*, so this helper deliberately
    delegates the stateful AdamW transition to torch.optim.AdamW itself on
    lightweight shadow copies of the trainable tensors. The real treatment
    parameters are never mutated here.
    """
    if lr <= 0:
        raise ValueError("S41 AdamW lr must be positive")
    if eps <= 0:
        raise ValueError("S41 AdamW eps must be positive")
    if weight_decay < 0:
        raise ValueError("S41 AdamW weight_decay must be nonnegative")
    beta1, beta2 = betas
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("S41 AdamW betas must be in [0,1)")
    if len(params) != len(states):
        raise ValueError("S41 AdamW state length mismatch")
    _validate_pair(params, grads, label="parameter/gradient")

    shadow = [
        torch.nn.Parameter(p.detach().clone(memory_format=torch.preserve_format))
        for p in params
    ]
    optimizer = torch.optim.AdamW(
        shadow,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
        amsgrad=False,
        maximize=False,
        capturable=False,
        differentiable=False,
    )

    for i, (probe, grad, state, param) in enumerate(zip(shadow, grads, states, params)):
        if state.step < 0:
            raise ValueError(f"S41 AdamW negative step at {i}")
        if (
            state.exp_avg.shape != param.shape
            or state.exp_avg_sq.shape != param.shape
            or state.exp_avg.device != param.device
            or state.exp_avg_sq.device != param.device
            or state.exp_avg.dtype != param.dtype
            or state.exp_avg_sq.dtype != param.dtype
        ):
            raise ValueError(f"S41 AdamW state mismatch at {i}")
        if not bool(torch.isfinite(state.exp_avg).all() and torch.isfinite(state.exp_avg_sq).all()):
            raise ValueError(f"S41 non-finite AdamW state at {i}")

        probe.grad = grad.detach().clone(memory_format=torch.preserve_format)
        if state.step > 0:
            optimizer.state[probe] = {
                "step": torch.tensor(float(state.step), dtype=torch.float32),
                "exp_avg": state.exp_avg.detach().clone(memory_format=torch.preserve_format),
                "exp_avg_sq": state.exp_avg_sq.detach().clone(memory_format=torch.preserve_format),
            }

    optimizer.step()

    deltas: list[Tensor] = []
    next_states: list[AdamWParameterState] = []
    for i, (param, probe) in enumerate(zip(params, shadow)):
        delta = probe.detach() - param.detach()
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError(f"S41 AdamW candidate delta non-finite at {i}")
        state = optimizer.state[probe]
        deltas.append(delta.detach().clone(memory_format=torch.preserve_format))
        next_states.append(
            AdamWParameterState(
                step=int(state["step"].item()),
                exp_avg=state["exp_avg"].detach().clone(memory_format=torch.preserve_format),
                exp_avg_sq=state["exp_avg_sq"].detach().clone(memory_format=torch.preserve_format),
            )
        )

    return deltas, next_states

def project_adamw_runtime_delta_against_anchor(
    candidate_delta: Sequence[Tensor],
    anchor_gradient: Sequence[Tensor],
    *,
    epsilon: float = 1e-12,
) -> tuple[list[Tensor], AdamWStepProjectionDiagnostics]:
    if epsilon <= 0:
        raise ValueError("S41 projection epsilon must be positive")
    _validate_pair(candidate_delta, anchor_gradient, label="delta/anchor")

    delta = [x.detach().clone() for x in candidate_delta]
    anchor = [x.detach().clone() for x in anchor_gradient]

    delta_norm_t = torch.sqrt(_sq(delta))
    anchor_sq = _sq(anchor)
    anchor_norm_t = torch.sqrt(anchor_sq)
    pre = _dot(anchor, delta)

    projected = bool(pre.item() > 0.0 and float(anchor_norm_t.item()) > 0.0)
    coeff = pre.new_zeros(())
    out = delta
    if projected:
        coeff = pre / (anchor_sq + epsilon)
        out = [x - coeff * y for x, y in zip(delta, anchor)]

    post = _dot(anchor, out)
    return out, AdamWStepProjectionDiagnostics(
        candidate_delta_norm=float(delta_norm_t.item()),
        anchor_gradient_norm=float(anchor_norm_t.item()),
        pre_dot=float(pre.item()),
        post_dot=float(post.item()),
        projected=projected,
        projection_coefficient=float(coeff.item()),
    )


def apply_parameter_deltas(params: Sequence[Tensor], deltas: Sequence[Tensor]) -> None:
    _validate_pair(params, deltas, label="parameter/delta")
    with torch.no_grad():
        for param, delta in zip(params, deltas):
            # Define the applied projected step by its quantized parameter
            # target. This makes finite-precision movement semantics explicit.
            target = param.detach() + delta
            param.copy_(target)


__all__ = [
    "AdamWParameterState",
    "AdamWStepProjectionDiagnostics",
    "adamw_candidate_deltas",
    "apply_parameter_deltas",
    "initialize_adamw_state",
    "project_adamw_runtime_delta_against_anchor",
]
