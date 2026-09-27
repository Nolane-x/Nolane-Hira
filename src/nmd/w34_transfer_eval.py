from __future__ import annotations

from copy import deepcopy
import random
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .atomic_geometry_eval import _prediction_row, _summarize_factor_predictions
from .symmetric_semantic import SymmetricSemanticScorer
from .w34_coevidence_semantic import CoEvidenceSemanticScorer
from .w34_transfer_authority import FACTOR_IDS
from .w34_transfer_cache import validate_w34_cache

CANDIDATE_SEED = 3417
CANDIDATE_RANK = 8
CANDIDATE_EPOCHS = 20
CANDIDATE_BATCH_SIZE = 32
CANDIDATE_LR = 2e-4
CANDIDATE_WEIGHT_DECAY = 0.01
CANDIDATE_GRAD_CLIP = 1.0
CANDIDATE_TEMPERATURE = 0.07
ANCHOR_COEFFICIENT = 0.35
ANCHOR_MARGIN_THRESHOLD = 0.08
VECTOR_VALIDITY_COEFFICIENT = 0.15


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
def evaluate_w34_cache(
    cache: dict[str, object],
    scorer: SymmetricSemanticScorer,
) -> dict[str, object]:
    validate_w34_cache(cache)
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
        raise ValueError("W34 anchor requires binary factor logits")
    return logits[int(target)] - logits[1 - int(target)]


def invalid_vector_mass(
    logits_by_factor: Mapping[str, Tensor],
) -> Tensor:
    """Independent factor joint mass assigned to invalid monotone vectors.

    Valid vectors: 000, 100, 110, 111.
    Invalid vectors: 001, 010, 011, 101.
    """
    if set(logits_by_factor) != set(FACTOR_IDS):
        raise ValueError("W34 vector validity requires exactly F0/F1/F2")
    p = {}
    for factor in FACTOR_IDS:
        logits = logits_by_factor[factor]
        if logits.numel() != 2:
            raise ValueError("W34 factor logits must be binary")
        p[factor] = torch.softmax(
            logits / CANDIDATE_TEMPERATURE,
            dim=-1,
        )[1]

    p0, p1, p2 = p["F0"], p["F1"], p["F2"]
    q0, q1, q2 = 1.0 - p0, 1.0 - p1, 1.0 - p2
    invalid = (
        q0 * q1 * p2  # 001
        + q0 * p1 * q2  # 010
        + q0 * p1 * p2  # 011
        + p0 * q1 * p2  # 101
    )
    return invalid.clamp(min=0.0, max=1.0)


@torch.inference_mode()
def build_anchor_map(
    train_cache: dict[str, object],
    baseline: SymmetricSemanticScorer,
) -> dict[tuple[str, str], bool]:
    validate_w34_cache(train_cache, expected_partition="train")
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
    scorer: CoEvidenceSemanticScorer,
    anchor_map: Mapping[tuple[str, str], bool],
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    factor_losses: dict[str, list[Tensor]] = {factor: [] for factor in FACTOR_IDS}
    anchor_losses: list[Tensor] = []
    validity_losses: list[Tensor] = []
    anchor_count = 0

    for case in cases:
        pack = schemas[case["domain_id"]]
        logits_by_factor: dict[str, Tensor] = {}
        for index, factor in enumerate(FACTOR_IDS):
            target = int(case["factor_vector"][index])
            logits = _factor_logits(case["state_tokens"], pack[factor], scorer)
            logits_by_factor[factor] = logits
            factor_losses[factor].append(
                F.cross_entropy(
                    (logits / CANDIDATE_TEMPERATURE).unsqueeze(0),
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
        validity_losses.append(invalid_vector_mass(logits_by_factor))

    per_factor = torch.stack([
        torch.stack(factor_losses[factor]).mean()
        for factor in FACTOR_IDS
    ])
    primary = per_factor.mean()
    anchor = (
        torch.stack(anchor_losses).mean()
        if anchor_losses
        else primary.new_zeros(())
    )
    validity = (
        torch.stack(validity_losses).mean()
        if validity_losses
        else primary.new_zeros(())
    )
    total = (
        primary
        + ANCHOR_COEFFICIENT * anchor
        + VECTOR_VALIDITY_COEFFICIENT * validity
    )
    return total, primary, anchor, validity, anchor_count


def quality_gate_pass(summary: Mapping[str, object]) -> bool:
    factors = summary["factors"]
    return (
        all(float(factors[f]["top1"]) >= 0.90 for f in FACTOR_IDS)
        and all(float(factors[f]["balanced_accuracy"]) >= 0.88 for f in FACTOR_IDS)
        and float(summary["factor_vector_top1"]) >= 0.82
        and float(summary["composed_severity_top1"]) >= 0.82
        and float(summary["invalid_factor_vector_rate"]) <= 0.05
        and float(summary["factor_probability_mass_max_error"]) <= 1e-6
    )


def transfer_gate_summary(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    b_factors = baseline["factors"]
    c_factors = candidate["factors"]
    baseline_worst = min(float(b_factors[f]["top1"]) for f in FACTOR_IDS)
    candidate_worst = min(float(c_factors[f]["top1"]) for f in FACTOR_IDS)
    regressions = {
        factor: float(b_factors[factor]["top1"]) - float(c_factors[factor]["top1"])
        for factor in FACTOR_IDS
    }
    severity_delta = (
        float(candidate["composed_severity_top1"])
        - float(baseline["composed_severity_top1"])
    )
    worst_factor_delta = candidate_worst - baseline_worst
    max_factor_regression = max(regressions.values())
    passed = (
        severity_delta >= 0.15
        and worst_factor_delta >= 0.08
        and max_factor_regression <= 0.02
    )
    return {
        "severity_delta": severity_delta,
        "worst_factor_top1_delta": worst_factor_delta,
        "factor_top1_regressions": regressions,
        "max_factor_top1_regression": max_factor_regression,
        "pass": passed,
    }


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


def build_coevidence_scorer(
    projection_weight: Tensor,
    candidate_state: Mapping[str, Tensor] | None = None,
    *,
    freeze: bool = False,
) -> CoEvidenceSemanticScorer:
    scorer = CoEvidenceSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=CANDIDATE_RANK,
    )
    scorer.load_projection_weight(projection_weight.detach().float(), freeze=True)
    if candidate_state is not None:
        scorer.load_candidate_state_dict(dict(candidate_state), freeze=freeze)
    elif freeze:
        scorer.freeze_candidate()
    return scorer


def train_coevidence_candidate(
    train_cache: dict[str, object],
    dev_cache: dict[str, object],
    projection_weight: Tensor,
) -> tuple[dict[str, Tensor], dict[str, object], list[dict[str, object]], dict[str, object]]:
    validate_w34_cache(train_cache, expected_partition="train")
    validate_w34_cache(dev_cache, expected_partition="dev")
    if tuple(projection_weight.shape) != (128, 256):
        raise ValueError("W34 T0 projection must be [128,256]")

    torch.manual_seed(CANDIDATE_SEED)
    random.seed(CANDIDATE_SEED)

    baseline = build_baseline_scorer(projection_weight)
    scorer = build_coevidence_scorer(projection_weight)
    if scorer.candidate_parameter_count != 8192:
        raise RuntimeError("W34 candidate parameter count changed")
    if scorer.candidate_trainable_parameter_count != 8192:
        raise RuntimeError("W34 must train exactly 8,192 candidate parameters")
    if scorer.trainable_parameter_count != 8192:
        raise RuntimeError("W34 unexpected trainable parameter surface")
    if scorer.projection.weight.requires_grad:
        raise RuntimeError("W34 T0 projection must remain frozen")

    anchor_map = build_anchor_map(train_cache, baseline)
    anchor_total = sum(anchor_map.values())
    baseline_dev = evaluate_w34_cache(dev_cache, baseline)

    candidate_parameters = (
        list(scorer.state_adapter.parameters())
        + list(scorer.schema_adapter.parameters())
        + list(scorer.interaction_state.parameters())
        + list(scorer.interaction_schema.parameters())
        + list(scorer.composition_state.parameters())
        + list(scorer.composition_schema.parameters())
    )
    optimizer = torch.optim.AdamW(
        candidate_parameters,
        lr=CANDIDATE_LR,
        weight_decay=CANDIDATE_WEIGHT_DECAY,
    )

    cases = train_cache["cases"]
    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_receipt = None

    for epoch in range(1, CANDIDATE_EPOCHS + 1):
        scorer.train()
        indices = list(range(len(cases)))
        random.Random(CANDIDATE_SEED + epoch).shuffle(indices)
        running_total = 0.0
        running_primary = 0.0
        running_anchor = 0.0
        running_validity = 0.0
        seen = 0

        for start in range(0, len(indices), CANDIDATE_BATCH_SIZE):
            batch_indices = indices[start : start + CANDIDATE_BATCH_SIZE]
            batch = [cases[index] for index in batch_indices]
            optimizer.zero_grad(set_to_none=True)
            total, primary, anchor, validity, _ = _batch_loss(
                batch,
                train_cache["schemas"],
                scorer,
                anchor_map,
            )
            total.backward()
            if scorer.projection.weight.grad is not None:
                raise RuntimeError("W34 frozen T0 projection received a gradient")
            torch.nn.utils.clip_grad_norm_(
                candidate_parameters,
                CANDIDATE_GRAD_CLIP,
            )
            optimizer.step()

            n = len(batch)
            seen += n
            running_total += float(total.detach()) * n
            running_primary += float(primary.detach()) * n
            running_anchor += float(anchor.detach()) * n
            running_validity += float(validity.detach()) * n

        dev_eval = evaluate_w34_cache(dev_cache, scorer)
        summary = dev_eval["pooled"]
        key = _selection_key(summary, epoch)
        row = {
            "epoch": epoch,
            "train_total_loss": running_total / max(1, seen),
            "train_factor_balanced_ce": running_primary / max(1, seen),
            "train_anchor_loss": running_anchor / max(1, seen),
            "train_invalid_vector_mass": running_validity / max(1, seen),
            "dev": summary,
            "selection_key": list(key),
        }
        history.append(row)

        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in scorer.state_dict().items()
                if (
                    key.startswith("state_adapter.")
                    or key.startswith("schema_adapter.")
                    or key.startswith("interaction_state.")
                    or key.startswith("interaction_schema.")
                    or key.startswith("composition_state.")
                    or key.startswith("composition_schema.")
                )
            }
            best_receipt = deepcopy(row)

    if best_state is None or best_receipt is None:
        raise RuntimeError("W34 candidate selection produced no checkpoint")

    metadata = {
        "anchor_count": int(anchor_total),
        "anchor_candidate_count": len(cases) * len(FACTOR_IDS),
        "anchor_rate": anchor_total / max(1, len(cases) * len(FACTOR_IDS)),
        "baseline_dev": baseline_dev["pooled"],
    }
    return best_state, best_receipt, history, metadata


__all__ = [
    "ANCHOR_COEFFICIENT",
    "ANCHOR_MARGIN_THRESHOLD",
    "CANDIDATE_BATCH_SIZE",
    "CANDIDATE_EPOCHS",
    "CANDIDATE_GRAD_CLIP",
    "CANDIDATE_LR",
    "CANDIDATE_RANK",
    "CANDIDATE_SEED",
    "CANDIDATE_TEMPERATURE",
    "CANDIDATE_WEIGHT_DECAY",
    "VECTOR_VALIDITY_COEFFICIENT",
    "build_anchor_map",
    "build_baseline_scorer",
    "build_coevidence_scorer",
    "evaluate_w34_cache",
    "invalid_vector_mass",
    "quality_gate_pass",
    "transfer_gate_summary",
    "train_coevidence_candidate",
]
