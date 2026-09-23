from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import LogicalOption, Primitive, StateMemory
from .losses import LossWeights, typed_decision_loss
from .runtime import CoarseMode, NolaneHira
from .typed_decisions import TypedDecision, TypedDecisionCase


@dataclass(frozen=True)
class DecisionExample:
    state_text: str
    primitive: Primitive
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int
    teacher_probs: tuple[float, ...] | None = None
    gold_score: float | None = None


@dataclass(frozen=True)
class CaseTrainingReceipt:
    state_encode_calls: int
    decision_count: int


def _score_support(
    primitive: Primitive,
    options: tuple[LogicalOption, ...],
    *,
    device,
    dtype,
) -> Tensor | None:
    if primitive != "score":
        return None
    values = [option.value for option in options]
    if any(value is None for value in values):
        raise ValueError(
            "score training requires explicit numeric option values"
        )
    return torch.tensor(
        values,
        device=device,
        dtype=dtype,
    )


def _loss_from_logits(
    logits: Tensor,
    *,
    primitive: Primitive,
    options: tuple[LogicalOption, ...],
    gold_index: int,
    teacher_probs: tuple[float, ...] | None,
    gold_score_value: float | None,
    weights: LossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    gold = torch.tensor(
        [gold_index],
        device=logits.device,
        dtype=torch.long,
    )
    teacher = None
    if teacher_probs is not None:
        teacher = torch.tensor(
            [teacher_probs],
            device=logits.device,
            dtype=logits.dtype,
        )
    gold_score = None
    if gold_score_value is not None:
        gold_score = torch.tensor(
            [gold_score_value],
            device=logits.device,
            dtype=logits.dtype,
        )
    support = _score_support(
        primitive,
        options,
        device=logits.device,
        dtype=logits.dtype,
    )
    return typed_decision_loss(
        logits,
        gold,
        teacher_probs=teacher,
        gold_score=gold_score,
        score_support=support,
        weights=weights,
    )


def forward_example(
    model: NolaneHira,
    example: DecisionExample,
    *,
    forced_budget: int | None = None,
    coarse_mode: CoarseMode = "legacy",
) -> tuple[Tensor, Tensor]:
    memory = model.compile_state(example.state_text)
    schema, _ = model.compile_schema(
        primitive=example.primitive,
        question_text=example.question_text,
        options=example.options,
        use_cache=False,
        include_token_artifacts=(coarse_mode == "competitive"),
    )
    out = model.forward_compiled(
        memory,
        schema,
        forced_budget=forced_budget,
        coarse_mode=coarse_mode,
    )
    return out.logits.unsqueeze(0), out.probabilities.unsqueeze(0)


def loss_example(
    model: NolaneHira,
    example: DecisionExample,
    *,
    weights: LossWeights = LossWeights(),
    forced_budget: int | None = None,
    coarse_mode: CoarseMode = "legacy",
) -> tuple[Tensor, dict[str, Tensor]]:
    logits, _ = forward_example(
        model,
        example,
        forced_budget=forced_budget,
        coarse_mode=coarse_mode,
    )
    return _loss_from_logits(
        logits,
        primitive=example.primitive,
        options=example.options,
        gold_index=example.gold_index,
        teacher_probs=example.teacher_probs,
        gold_score_value=example.gold_score,
        weights=weights,
    )


def forward_typed_decision_with_memory(
    model: NolaneHira,
    memory: StateMemory,
    decision: TypedDecision,
    *,
    forced_budget: int | None = None,
    coarse_mode: CoarseMode = "legacy",
) -> tuple[Tensor, Tensor]:
    schema, _ = model.compile_schema(
        primitive=decision.primitive,
        question_text=decision.question_text,
        options=decision.options,
        use_cache=False,
        include_token_artifacts=(coarse_mode == "competitive"),
    )
    out = model.forward_compiled(
        memory,
        schema,
        forced_budget=forced_budget,
        coarse_mode=coarse_mode,
    )
    return out.logits.unsqueeze(0), out.probabilities.unsqueeze(0)


def loss_typed_case(
    model: NolaneHira,
    case: TypedDecisionCase,
    *,
    weights: LossWeights = LossWeights(),
    forced_budget: int | None = None,
    coarse_mode: CoarseMode = "legacy",
) -> tuple[Tensor, dict[str, Tensor], CaseTrainingReceipt]:
    if not case.decisions:
        raise ValueError("typed case must contain at least one decision")

    before = model.state_encode_calls
    memory = model.compile_state(case.state_text)
    losses: list[Tensor] = []
    part_values: dict[str, list[Tensor]] = {}

    for decision in case.decisions:
        logits, _ = forward_typed_decision_with_memory(
            model,
            memory,
            decision,
            forced_budget=forced_budget,
            coarse_mode=coarse_mode,
        )
        loss, parts = _loss_from_logits(
            logits,
            primitive=decision.primitive,
            options=decision.options,
            gold_index=decision.gold_index,
            teacher_probs=decision.gold_probabilities,
            gold_score_value=decision.gold_score,
            weights=weights,
        )
        losses.append(loss)
        for name, value in parts.items():
            part_values.setdefault(name, []).append(value)

    state_encode_calls = model.state_encode_calls - before
    if state_encode_calls != 1:
        raise RuntimeError(
            "typed case training violated state-once semantics"
        )

    mean_loss = torch.stack(losses).mean()
    mean_parts = {
        name: torch.stack(values).mean()
        for name, values in part_values.items()
    }
    return (
        mean_loss,
        mean_parts,
        CaseTrainingReceipt(
            state_encode_calls=state_encode_calls,
            decision_count=len(case.decisions),
        ),
    )


def optimizer_step(
    model: NolaneHira,
    example: DecisionExample,
    optimizer: torch.optim.Optimizer,
    *,
    weights: LossWeights = LossWeights(),
    forced_budget: int | None = None,
    coarse_mode: CoarseMode = "legacy",
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = loss_example(
        model,
        example,
        weights=weights,
        forced_budget=forced_budget,
        coarse_mode=coarse_mode,
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def optimizer_step_typed_case(
    model: NolaneHira,
    case: TypedDecisionCase,
    optimizer: torch.optim.Optimizer,
    *,
    weights: LossWeights = LossWeights(),
    forced_budget: int | None = None,
    coarse_mode: CoarseMode = "legacy",
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = loss_typed_case(
        model,
        case,
        weights=weights,
        forced_budget=forced_budget,
        coarse_mode=coarse_mode,
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
