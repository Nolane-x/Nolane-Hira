from __future__ import annotations

from copy import deepcopy
import random
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .atomic_geometry_eval import _prediction_row, _summarize_factor_predictions
from .symmetric_semantic import SymmetricSemanticScorer
from .w31_dual_semantic_adapter import DualAdapterSymmetricSemanticScorer
from .w31_transfer_authority import FACTOR_IDS
from .w31_transfer_cache import validate_w31_cache

ADAPTER_SEED = 3117
ADAPTER_RANK = 8
ADAPTER_EPOCHS = 16
ADAPTER_BATCH_SIZE = 32
ADAPTER_LR = 2e-4
ADAPTER_WEIGHT_DECAY = 0.01
ADAPTER_GRAD_CLIP = 1.0
ADAPTER_TEMPERATURE = 0.07
ANCHOR_COEFFICIENT = 0.35
ANCHOR_MARGIN_THRESHOLD = 0.08


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
def evaluate_w31_cache(
    cache: dict[str, object],
    scorer: SymmetricSemanticScorer,
) -> dict[str, object]:
    validate_w31_cache(cache)
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


def _signed_margin(logits: Tensor, target: int) -> Tensor:
    if logits.numel() != 2 or target not in (0, 1):
        raise ValueError("W31 anchor requires binary factor logits")
    other = 1 - int(target)
    return logits[int(target)] - logits[other]


@torch.inference_mode()
def build_anchor_map(
    train_cache: dict[str, object],
    baseline: SymmetricSemanticScorer,
) -> dict[tuple[str, str], bool]:
    validate_w31_cache(train_cache, expected_partition="train")
    baseline.eval()
    anchors: dict[tuple[str, str], bool] = {}
    for case in train_cache["cases"]:
        pack = train_cache["schemas"][case["domain_id"]]
        for index, factor in enumerate(FACTOR_IDS):
            target = int(case["factor_vector"][index])
            logits = _factor_logits(case["state_tokens"], pack[factor], baseline)
            pred = int(logits.argmax())
            margin = float(_signed_margin(logits, target))
            anchors[(case["case_id"], factor)] = (
                pred == target and margin >= ANCHOR_MARGIN_THRESHOLD
            )
    return anchors


def _batch_loss(
    cases: list[Mapping[str, object]],
    schemas: Mapping[str, object],
    scorer: DualAdapterSymmetricSemanticScorer,
    anchor_map: Mapping[tuple[str, str], bool],
) -> tuple[Tensor, Tensor, Tensor, int]:
    factor_losses: dict[str, list[Tensor]] = {factor: [] for factor in FACTOR_IDS}
    anchor_losses: list[Tensor] = []
    anchor_count = 0

    for case in cases:
        pack = schemas[case["domain_id"]]
        for index, factor in enumerate(FACTOR_IDS):
            target = int(case["factor_vector"][index])
            logits = _factor_logits(case["state_tokens"], pack[factor], scorer)
            factor_losses[factor].append(
                F.cross_entropy(
                    (logits / ADAPTER_TEMPERATURE).unsqueeze(0),
                    torch.tensor([target], dtype=torch.long, device=logits.device),
                )
            )
            if anchor_map[(case["case_id"], factor)]:
                anchor_count += 1
                adapted_margin = _signed_margin(logits, target)
                anchor_losses.append(
                    F.relu(
                        logits.new_tensor(ANCHOR_MARGIN_THRESHOLD)
                        - adapted_margin
                    )
                )

    per_factor = torch.stack([
        torch.stack(factor_losses[factor]).mean()
        for factor in FACTOR_IDS
    ])
    primary = per_factor.mean()
    if anchor_losses:
        anchor = torch.stack(anchor_losses).mean()
    else:
        anchor = primary.new_zeros(())
    total = primary + ANCHOR_COEFFICIENT * anchor
    return total, primary, anchor, anchor_count


def _selection_key(summary: Mapping[str, object], epoch: int) -> tuple[float, ...]:
    factors = summary["factors"]
    worst_ba = min(float(factors[factor]["balanced_accuracy"]) for factor in FACTOR_IDS)
    worst_top1 = min(float(factors[factor]["top1"]) for factor in FACTOR_IDS)
    return (
        worst_ba,
        worst_top1,
        float(summary["composed_severity_top1"]),
        -float(summary["invalid_factor_vector_rate"]),
        -float(epoch),
    )


def build_baseline_scorer(projection_weight: Tensor) -> SymmetricSemanticScorer:
    scorer = SymmetricSemanticScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(projection_weight.detach().float(), freeze=True)
    scorer.eval()
    return scorer


def build_dual_scorer(
    projection_weight: Tensor,
    adapter_state: Mapping[str, Tensor] | None = None,
    *,
    freeze: bool = False,
) -> DualAdapterSymmetricSemanticScorer:
    scorer = DualAdapterSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=ADAPTER_RANK,
    )
    scorer.load_projection_weight(projection_weight.detach().float(), freeze=True)
    if adapter_state is not None:
        scorer.load_adapter_state_dict(dict(adapter_state), freeze=freeze)
    elif freeze:
        scorer.freeze_adapters()
    return scorer


def train_dual_adapter(
    train_cache: dict[str, object],
    dev_cache: dict[str, object],
    projection_weight: Tensor,
) -> tuple[dict[str, Tensor], dict[str, object], list[dict[str, object]], dict[str, object]]:
    validate_w31_cache(train_cache, expected_partition="train")
    validate_w31_cache(dev_cache, expected_partition="dev")
    if tuple(projection_weight.shape) != (128, 256):
        raise ValueError("W31 T0 projection must be [128,256]")

    torch.manual_seed(ADAPTER_SEED)
    random.seed(ADAPTER_SEED)

    baseline = build_baseline_scorer(projection_weight)
    scorer = build_dual_scorer(projection_weight)
    if scorer.adapter_parameter_count != 4096:
        raise RuntimeError("W31 adapter parameter count changed")
    if scorer.adapter_trainable_parameter_count != 4096:
        raise RuntimeError("W31 must train exactly 4,096 adapter parameters")
    if scorer.trainable_parameter_count != 4096:
        raise RuntimeError("W31 unexpected trainable parameter surface")
    if scorer.projection.weight.requires_grad:
        raise RuntimeError("W31 T0 projection must remain frozen")

    anchor_map = build_anchor_map(train_cache, baseline)
    anchor_total = sum(anchor_map.values())
    baseline_dev = evaluate_w31_cache(dev_cache, baseline)

    optimizer = torch.optim.AdamW(
        list(scorer.state_adapter.parameters())
        + list(scorer.schema_adapter.parameters()),
        lr=ADAPTER_LR,
        weight_decay=ADAPTER_WEIGHT_DECAY,
    )

    cases = train_cache["cases"]
    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_receipt = None

    for epoch in range(1, ADAPTER_EPOCHS + 1):
        scorer.train()
        indices = list(range(len(cases)))
        random.Random(ADAPTER_SEED + epoch).shuffle(indices)
        running_total = 0.0
        running_primary = 0.0
        running_anchor = 0.0
        seen = 0

        for start in range(0, len(indices), ADAPTER_BATCH_SIZE):
            batch_indices = indices[start : start + ADAPTER_BATCH_SIZE]
            batch = [cases[index] for index in batch_indices]
            optimizer.zero_grad(set_to_none=True)
            total, primary, anchor, _ = _batch_loss(
                batch,
                train_cache["schemas"],
                scorer,
                anchor_map,
            )
            total.backward()
            if scorer.projection.weight.grad is not None:
                raise RuntimeError("W31 frozen T0 projection received a gradient")
            torch.nn.utils.clip_grad_norm_(
                list(scorer.state_adapter.parameters())
                + list(scorer.schema_adapter.parameters()),
                ADAPTER_GRAD_CLIP,
            )
            optimizer.step()

            n = len(batch)
            seen += n
            running_total += float(total.detach()) * n
            running_primary += float(primary.detach()) * n
            running_anchor += float(anchor.detach()) * n

        dev_eval = evaluate_w31_cache(dev_cache, scorer)
        summary = dev_eval["pooled"]
        key = _selection_key(summary, epoch)
        row = {
            "epoch": epoch,
            "train_total_loss": running_total / max(1, seen),
            "train_factor_balanced_ce": running_primary / max(1, seen),
            "train_anchor_loss": running_anchor / max(1, seen),
            "dev": summary,
            "selection_key": list(key),
        }
        history.append(row)

        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in scorer.state_dict().items()
                if key.startswith("state_adapter.") or key.startswith("schema_adapter.")
            }
            best_receipt = deepcopy(row)

    if best_state is None or best_receipt is None:
        raise RuntimeError("W31 adapter selection produced no checkpoint")

    metadata = {
        "anchor_count": int(anchor_total),
        "anchor_candidate_count": len(cases) * len(FACTOR_IDS),
        "anchor_rate": anchor_total / max(1, len(cases) * len(FACTOR_IDS)),
        "baseline_dev": baseline_dev["pooled"],
    }
    return best_state, best_receipt, history, metadata


__all__ = [
    "ADAPTER_BATCH_SIZE",
    "ADAPTER_EPOCHS",
    "ADAPTER_GRAD_CLIP",
    "ADAPTER_LR",
    "ADAPTER_RANK",
    "ADAPTER_SEED",
    "ADAPTER_TEMPERATURE",
    "ADAPTER_WEIGHT_DECAY",
    "ANCHOR_COEFFICIENT",
    "ANCHOR_MARGIN_THRESHOLD",
    "build_anchor_map",
    "build_baseline_scorer",
    "build_dual_scorer",
    "evaluate_w31_cache",
    "train_dual_adapter",
]
