from __future__ import annotations

import random
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .competitive import CompetitiveCoarseScorer
from .hira import HIRACore
from .semantic_alignment_bridge import (
    ASYMMETRIC_BRIDGE_PARAMETER_COUNT,
    SHARED_BRIDGE_PARAMETER_COUNT,
    SemanticAlignmentBridgeScorer,
    semantic_alignment_scores,
)
from .semantic_alignment_cache import validate_w9_cache
from .semantic_alignment_eval import (
    ALIGNMENT_TEMPERATURE,
    evaluate_w9_checkpoint,
)


CANDIDATES = (
    "frozen-w6e-control",
    "projection-semantic-control",
    "shared-bridge-semantic-control",
    "asymmetric-bridge-primary",
    "asymmetric-bridge-replica",
)

EPOCHS = 8
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PROJECTION_PARAMETER_COUNT = 32768

SEEDS = {
    "frozen-w6e-control": 0,
    "projection-semantic-control": 2101,
    "shared-bridge-semantic-control": 2111,
    "asymmetric-bridge-primary": 2129,
    "asymmetric-bridge-replica": 2137,
}


def candidate_seed(candidate: str) -> int:
    if candidate not in SEEDS:
        raise ValueError(f"unknown W9 candidate: {candidate}")
    return int(SEEDS[candidate])


def expected_trainable_parameters(candidate: str) -> int:
    if candidate == "frozen-w6e-control":
        return 0
    if candidate == "projection-semantic-control":
        return PROJECTION_PARAMETER_COUNT
    if candidate == "shared-bridge-semantic-control":
        return SHARED_BRIDGE_PARAMETER_COUNT
    if candidate in {
        "asymmetric-bridge-primary",
        "asymmetric-bridge-replica",
    }:
        return ASYMMETRIC_BRIDGE_PARAMETER_COUNT
    raise ValueError(f"unknown W9 candidate: {candidate}")


def configure_trainability(
    candidate: str,
    hira: HIRACore,
    scorer: nn.Module,
) -> list[Tensor]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown W9 candidate: {candidate}")

    for parameter in hira.parameters():
        parameter.requires_grad_(False)

    if candidate == "frozen-w6e-control":
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
        return []

    if candidate == "projection-semantic-control":
        if not isinstance(scorer, CompetitiveCoarseScorer):
            raise ValueError("W9 projection control requires base scorer")
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
        scorer.projection.weight.requires_grad_(True)
        parameters = [scorer.projection.weight]

    elif candidate == "shared-bridge-semantic-control":
        if (
            not isinstance(scorer, SemanticAlignmentBridgeScorer)
            or scorer.mode != "shared"
        ):
            raise ValueError("W9 shared control requires shared bridge scorer")
        scorer.freeze_base()
        parameters = list(scorer.bridge_parameters())

    else:
        if (
            not isinstance(scorer, SemanticAlignmentBridgeScorer)
            or scorer.mode != "asymmetric"
        ):
            raise ValueError(
                "W9 asymmetric candidates require asymmetric bridge scorer"
            )
        scorer.freeze_base()
        parameters = list(scorer.bridge_parameters())

    count = sum(parameter.numel() for parameter in parameters)
    expected = expected_trainable_parameters(candidate)
    if count != expected:
        raise RuntimeError(
            f"W9 parameter contract changed for {candidate}: "
            f"{count} != {expected}"
        )
    return parameters


def _definition_k16_view(base: dict) -> dict:
    matches = [
        view
        for view in base["views"]
        if view["view_id"] == "definition"
        and int(view["diagnosis_k"]) == 16
    ]
    if len(matches) != 1:
        raise ValueError("W9 base requires one definition K16 view")
    return matches[0]


def semantic_alignment_loss(
    scorer: nn.Module,
    base: dict,
) -> Tensor:
    view = _definition_k16_view(base)
    state = base["state_content_tokens"].float().unsqueeze(0)
    state_mask = torch.ones(
        1,
        state.shape[1],
        dtype=torch.bool,
    )
    definitions = view["option_tokens"].float().unsqueeze(0)
    definition_mask = view["option_content_mask"].bool().unsqueeze(0)
    scores = semantic_alignment_scores(
        scorer,
        state_tokens=state,
        state_mask=state_mask,
        definition_tokens=definitions,
        definition_mask=definition_mask,
    )
    target = torch.tensor(
        [int(view["gold_index"])],
        dtype=torch.long,
    )
    return F.cross_entropy(
        scores / ALIGNMENT_TEMPERATURE,
        target,
    )


def dev_selection_key(
    metrics: dict[str, object],
    epoch: int,
) -> tuple:
    pooled = metrics["pooled"]
    views = pooled["views"]
    alignment = metrics["alignment"]["pooled"]
    return (
        -float(views["definition"]["4"]["final_top1"]),
        -float(views["definition"]["16"]["final_top1"]),
        -float(views["definition"]["4"]["coarse_top1"]),
        -float(views["definition"]["16"]["final_mrr"]),
        -float(views["label"]["4"]["final_top1"]),
        float(alignment["ce"]),
        int(epoch),
    )


def train_w9_candidate(
    candidate: str,
    hira: HIRACore,
    scorer: nn.Module,
    train_cache: dict,
    dev_cache: dict,
    *,
    epochs: int = EPOCHS,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object]]:
    if candidate == "frozen-w6e-control":
        raise ValueError("frozen W9 control is not trainable")
    validate_w9_cache(train_cache)
    validate_w9_cache(dev_cache)

    parameters = configure_trainability(candidate, hira, scorer)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    seed = candidate_seed(candidate)

    train_bases: Sequence[dict] = train_cache["bases"]
    if not train_bases:
        raise ValueError("W9 training cache is empty")

    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_metrics = None

    for epoch in range(1, int(epochs) + 1):
        hira.eval()
        scorer.train()
        order = list(range(len(train_bases)))
        random.Random(seed + epoch).shuffle(order)

        loss_sum = 0.0
        grad_norm_sum = 0.0

        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = semantic_alignment_loss(
                scorer,
                train_bases[index],
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                GRAD_CLIP,
            )
            optimizer.step()
            loss_sum += float(loss.detach())
            grad_norm_sum += float(grad_norm)

        metrics = evaluate_w9_checkpoint(
            hira,
            scorer,
            dev_cache,
        )
        metrics["epoch"] = epoch
        metrics["mean_train_alignment_loss"] = (
            loss_sum / len(train_bases)
        )
        metrics["mean_preclip_grad_norm"] = (
            grad_norm_sum / len(train_bases)
        )
        metrics["optimizer_steps"] = len(train_bases)
        history.append(metrics)

        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in scorer.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("W9 candidate produced no DEV checkpoint")

    scorer.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics
