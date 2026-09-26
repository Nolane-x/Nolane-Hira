from __future__ import annotations

from copy import deepcopy
import random
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .atomic_geometry_eval import _prediction_row, _summarize_factor_predictions
from .semantic_transfer_bridge import BridgedSymmetricSemanticScorer
from .symmetric_semantic import SymmetricSemanticScorer
from .w30_transfer_authority import FACTOR_IDS
from .w30_transfer_cache import validate_w30_cache

BRIDGE_SEED = 3011
BRIDGE_RANK = 8
BRIDGE_EPOCHS = 12
BRIDGE_BATCH_SIZE = 32
BRIDGE_LR = 3e-4
BRIDGE_WEIGHT_DECAY = 0.01
BRIDGE_GRAD_CLIP = 1.0
BRIDGE_TEMPERATURE = 0.07


def _factor_logits(
    state_tokens: Tensor,
    payload: Mapping[str, object],
    scorer: SymmetricSemanticScorer,
) -> Tensor:
    state = state_tokens.float()
    option_tokens = payload["option_view_tokens"].float()
    option_token_mask = payload["option_view_token_mask"].bool()
    option_view_mask = payload["option_view_mask"].bool()
    return scorer(
        state_tokens=state.unsqueeze(0),
        state_mask=torch.ones(
            1,
            state.shape[0],
            dtype=torch.bool,
            device=state.device,
        ),
        option_view_tokens=option_tokens.unsqueeze(0),
        option_view_token_mask=option_token_mask.unsqueeze(0),
        option_view_mask=option_view_mask.unsqueeze(0),
    )[0]


@torch.inference_mode()
def evaluate_w30_cache(
    cache: dict[str, object],
    scorer: SymmetricSemanticScorer,
) -> dict[str, object]:
    validate_w30_cache(cache)
    scorer.eval()
    raw = []
    for case in cache["cases"]:
        pack = cache["schemas"][case["domain_id"]]
        logits = {
            factor: _factor_logits(case["state_tokens"], pack[factor], scorer)
            for factor in FACTOR_IDS
        }
        raw.append(_prediction_row(case, logits))

    per_domain = {}
    for domain in cache["metadata"]["domains"]:
        rows = [row for row in raw if row["domain_id"] == domain]
        per_domain[domain] = _summarize_factor_predictions(rows)
    return {
        "per_domain": per_domain,
        "pooled": _summarize_factor_predictions(raw),
        "predictions": {row["case_id"]: row for row in raw},
    }


def _loss_for_case(
    case: Mapping[str, object],
    pack: Mapping[str, object],
    scorer: BridgedSymmetricSemanticScorer,
) -> Tensor:
    losses = []
    for index, factor in enumerate(FACTOR_IDS):
        logits = _factor_logits(case["state_tokens"], pack[factor], scorer)
        target = torch.tensor(
            [int(case["factor_vector"][index])],
            dtype=torch.long,
            device=logits.device,
        )
        losses.append(
            F.cross_entropy(
                (logits / BRIDGE_TEMPERATURE).unsqueeze(0),
                target,
            )
        )
    return torch.stack(losses).sum()


def _selection_key(summary: Mapping[str, object], epoch: int) -> tuple[float, ...]:
    factors = summary["factors"]
    factor_top1 = [float(factors[factor]["top1"]) for factor in FACTOR_IDS]
    factor_ba = [float(factors[factor]["balanced_accuracy"]) for factor in FACTOR_IDS]
    return (
        float(summary["composed_severity_top1"]),
        float(summary["factor_vector_top1"]),
        min(factor_top1),
        min(factor_ba),
        sum(factor_top1) / len(factor_top1),
        -float(summary["invalid_factor_vector_rate"]),
        -float(epoch),
    )


def train_transfer_bridge(
    train_cache: dict[str, object],
    dev_cache: dict[str, object],
    projection_weight: Tensor,
) -> tuple[dict[str, Tensor], dict[str, object], list[dict[str, object]]]:
    validate_w30_cache(train_cache, expected_partition="train")
    validate_w30_cache(dev_cache, expected_partition="dev")
    if tuple(projection_weight.shape) != (128, 256):
        raise ValueError("W30 T0 projection must be [128,256]")

    torch.manual_seed(BRIDGE_SEED)
    random.seed(BRIDGE_SEED)
    scorer = BridgedSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=BRIDGE_RANK,
    )
    scorer.load_projection_weight(projection_weight.detach().float(), freeze=True)
    if scorer.bridge_parameter_count != 2048:
        raise RuntimeError("W30 bridge parameter count changed")
    if scorer.trainable_parameter_count != 2048:
        raise RuntimeError("W30 must train exactly 2,048 parameters")
    if scorer.projection.weight.requires_grad:
        raise RuntimeError("W30 T0 projection must remain frozen")

    optimizer = torch.optim.AdamW(
        list(scorer.bridge.parameters()),
        lr=BRIDGE_LR,
        weight_decay=BRIDGE_WEIGHT_DECAY,
    )

    cases = train_cache["cases"]
    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_receipt = None

    for epoch in range(1, BRIDGE_EPOCHS + 1):
        scorer.train()
        indices = list(range(len(cases)))
        random.Random(BRIDGE_SEED + epoch).shuffle(indices)
        running = 0.0

        for start in range(0, len(indices), BRIDGE_BATCH_SIZE):
            batch_indices = indices[start : start + BRIDGE_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for index in batch_indices:
                case = cases[index]
                pack = train_cache["schemas"][case["domain_id"]]
                batch_losses.append(_loss_for_case(case, pack, scorer))
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            if scorer.projection.weight.grad is not None:
                raise RuntimeError("W30 frozen T0 projection received a gradient")
            torch.nn.utils.clip_grad_norm_(
                list(scorer.bridge.parameters()),
                BRIDGE_GRAD_CLIP,
            )
            optimizer.step()
            running += float(loss.detach()) * len(batch_indices)

        dev_eval = evaluate_w30_cache(dev_cache, scorer)
        summary = dev_eval["pooled"]
        key = _selection_key(summary, epoch)
        row = {
            "epoch": epoch,
            "train_factor_ce": running / len(cases),
            "dev": summary,
            "selection_key": list(key),
        }
        history.append(row)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in scorer.bridge.state_dict().items()
            }
            best_receipt = deepcopy(row)

    if best_state is None or best_receipt is None:
        raise RuntimeError("W30 bridge selection produced no checkpoint")
    return best_state, best_receipt, history


def build_baseline_scorer(projection_weight: Tensor) -> SymmetricSemanticScorer:
    scorer = SymmetricSemanticScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(projection_weight.detach().float(), freeze=True)
    scorer.eval()
    return scorer


def build_bridged_scorer(
    projection_weight: Tensor,
    bridge_state: Mapping[str, Tensor],
    *,
    freeze: bool = True,
) -> BridgedSymmetricSemanticScorer:
    scorer = BridgedSymmetricSemanticScorer(d_model=256, d_rel=128, rank=BRIDGE_RANK)
    scorer.load_projection_weight(projection_weight.detach().float(), freeze=True)
    scorer.load_bridge_state_dict(dict(bridge_state), freeze=freeze)
    scorer.eval()
    return scorer


__all__ = [
    "BRIDGE_BATCH_SIZE",
    "BRIDGE_EPOCHS",
    "BRIDGE_GRAD_CLIP",
    "BRIDGE_LR",
    "BRIDGE_RANK",
    "BRIDGE_SEED",
    "BRIDGE_TEMPERATURE",
    "BRIDGE_WEIGHT_DECAY",
    "build_baseline_scorer",
    "build_bridged_scorer",
    "evaluate_w30_cache",
    "train_transfer_bridge",
]
