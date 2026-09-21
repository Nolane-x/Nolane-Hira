from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import LogicalOption, Primitive
from .losses import LossWeights, typed_decision_loss
from .runtime import NolaneHira


@dataclass(frozen=True)
class DecisionExample:
    state_text: str
    primitive: Primitive
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int
    teacher_probs: tuple[float, ...] | None = None
    gold_score: float | None = None


def forward_example(
    model: NolaneHira, example: DecisionExample, *, forced_budget: int | None = None
) -> tuple[Tensor, Tensor]:
    memory = model.compile_state(example.state_text)
    schema, _ = model.compile_schema(
        primitive=example.primitive,
        question_text=example.question_text,
        options=example.options,
        use_cache=False,
    )
    out = model.forward_compiled(memory, schema, forced_budget=forced_budget)
    return out.logits.unsqueeze(0), out.probabilities.unsqueeze(0)


def loss_example(
    model: NolaneHira,
    example: DecisionExample,
    *,
    weights: LossWeights = LossWeights(),
    forced_budget: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    logits, _ = forward_example(model, example, forced_budget=forced_budget)
    gold = torch.tensor([example.gold_index], device=logits.device, dtype=torch.long)
    teacher = None
    if example.teacher_probs is not None:
        teacher = torch.tensor([example.teacher_probs], device=logits.device, dtype=logits.dtype)
    gold_score = None
    if example.gold_score is not None:
        gold_score = torch.tensor([example.gold_score], device=logits.device, dtype=logits.dtype)
    return typed_decision_loss(
        logits,
        gold,
        teacher_probs=teacher,
        gold_score=gold_score,
        weights=weights,
    )


def optimizer_step(
    model: NolaneHira,
    example: DecisionExample,
    optimizer: torch.optim.Optimizer,
    *,
    weights: LossWeights = LossWeights(),
    forced_budget: int | None = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = loss_example(model, example, weights=weights, forced_budget=forced_budget)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
