from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .competitive import CompetitiveCoarseScorer, MIN_COVERAGE_WEIGHT
from .hira import HIRACore
from .runtime import NolaneHira
from .typed_competitive_cache import PRIMITIVE_TO_ID, compile_w6b_cache
from .high_k_localization_authority import (
    DOMAINS,
    HighKDiagnosticView,
)


def compile_w6f_cache(
    model: NolaneHira,
    views: Sequence[HighKDiagnosticView],
) -> dict:
    """Compile W6f through the already-tested W6b state/schema cache path."""
    cache = compile_w6b_cache(
        model,
        views,
        expected_split="diagnostic",
    )
    if len(cache["cases"]) != len(views):
        raise RuntimeError("W6f cache/view count mismatch")

    base_ids: set[str] = set()
    for row, view in zip(cache["cases"], views):
        if len(row["decisions"]) != 1:
            raise RuntimeError("W6f view must contain diagnosis only")
        row["base_id"] = view.base_id
        row["domain_id"] = view.domain_id
        row["gold_signature"] = tuple(view.gold_signature)
        row["option_signatures"] = tuple(view.option_signatures)
        row["option_distances"] = tuple(int(x) for x in view.option_distances)
        row["roles"] = tuple(DOMAINS[view.domain_id].roles)
        base_ids.add(view.base_id)

    cache["metadata"]["w6f_schema_version"] = "r8-w6f-high-k-cache-v1"
    cache["metadata"]["base_count"] = len(base_ids)
    cache["metadata"]["view_count"] = len(views)
    cache["metadata"]["domain_counts"] = dict(
        Counter(view.domain_id for view in views)
    )
    cache["metadata"]["k_counts"] = {
        str(k): sum(view.diagnosis_k == k for view in views)
        for k in (8, 16, 32, 64)
    }
    return cache


def uniform_salience_logits(
    scorer: CompetitiveCoarseScorer,
    *,
    state_tokens: Tensor,
    state_mask: Tensor,
    question_tokens: Tensor,
    question_mask: Tensor,
    option_tokens: Tensor,
    option_token_ids: Tensor,
    option_mask: Tensor,
) -> Tensor:
    """Diagnostic-only competitive logits with unit valid-token salience.

    Everything except candidate-relative IDF is kept identical to production.
    """
    if state_tokens.ndim != 3 or state_tokens.shape[-1] != scorer.d_model:
        raise ValueError("state_tokens must be [B,S,D]")
    if question_tokens.ndim != 3 or question_tokens.shape[-1] != scorer.d_model:
        raise ValueError("question_tokens must be [B,Q,D]")
    if option_tokens.ndim != 4 or option_tokens.shape[-1] != scorer.d_model:
        raise ValueError("option_tokens must be [B,K,T,D]")
    if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
        raise ValueError("state_mask mismatch")
    if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
        raise ValueError("question_mask mismatch")
    if option_mask.shape != option_tokens.shape[:3] or option_mask.dtype != torch.bool:
        raise ValueError("option_mask mismatch")
    if option_token_ids.shape != option_mask.shape:
        raise ValueError("option_token_ids mismatch")
    if option_token_ids.dtype != torch.long:
        raise ValueError("option_token_ids must be torch.long")
    if (option_mask.sum(-1) < 1).any():
        raise ValueError("every option requires content tokens")

    context = torch.cat([state_tokens, question_tokens], dim=1)
    context_mask = torch.cat([state_mask, question_mask], dim=1)
    projected_context = scorer._project(context)
    projected_options = scorer._project(option_tokens)

    similarity = torch.einsum(
        "bktd,bcd->bktc",
        projected_options,
        projected_context,
    )
    similarity = similarity.masked_fill(
        ~context_mask[:, None, None, :],
        -1e4,
    )

    weights = option_mask.to(similarity.dtype)
    denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
    common = (
        similarity * weights[..., None]
    ).sum(dim=2) / denom
    adjusted = similarity - common[:, :, None, :]
    adjusted = adjusted.masked_fill(
        ~context_mask[:, None, None, :],
        -1e4,
    )
    coverage = adjusted.max(dim=-1).values

    weighted_mean = (
        (coverage * weights).sum(-1)
        / weights.sum(-1).clamp_min(1e-8)
    )
    min_coverage = coverage.masked_fill(
        ~option_mask,
        1e4,
    ).min(dim=-1).values
    raw = weighted_mean + MIN_COVERAGE_WEIGHT * min_coverage
    logits = raw * scorer.scale().to(raw.device, raw.dtype)
    if not torch.isfinite(logits).all():
        raise ValueError("uniform-salience diagnostic produced non-finite logits")
    return logits


def _cached_tensors(case: dict, decision: dict) -> dict[str, Tensor]:
    state_tokens = case["state_content_tokens"].float().unsqueeze(0)
    return {
        "state_segments": case["state_segments"].float().unsqueeze(0),
        "state_tokens": state_tokens,
        "state_mask": torch.ones(
            1,
            state_tokens.shape[1],
            dtype=torch.bool,
        ),
        "question": decision["question_embedding"].float().unsqueeze(0),
        "question_tokens": decision["question_tokens"].float().unsqueeze(0),
        "question_mask": decision[
            "question_content_mask"
        ].bool().unsqueeze(0),
        "options": decision["option_embeddings"].float().unsqueeze(0),
        "option_tokens": decision["option_tokens"].float().unsqueeze(0),
        "option_token_ids": decision["option_token_ids"].long().unsqueeze(0),
        "option_mask": decision["option_content_mask"].bool().unsqueeze(0),
        "qtype": torch.tensor(
            [PRIMITIVE_TO_ID[decision["primitive"]]],
            dtype=torch.long,
        ),
    }


def _rank_metrics(logits: Tensor, gold_index: int) -> dict[str, float | int | bool]:
    row = logits.detach().cpu().to(torch.float64).flatten()
    if not 0 <= gold_index < row.numel():
        raise ValueError("gold index outside logits")
    gold_logit = float(row[gold_index])
    rank = 1 + int((row > row[gold_index]).sum().item())
    predicted = int(row.argmax().item())

    negative = row.clone()
    negative[gold_index] = -math.inf
    best_negative_index = int(negative.argmax().item())
    margin = gold_logit - float(negative[best_negative_index])
    return {
        "rank": rank,
        "top1": rank == 1,
        "top5": rank <= min(5, row.numel()),
        "reciprocal_rank": 1.0 / rank,
        "gold_logit": gold_logit,
        "margin": margin,
        "predicted_index": predicted,
        "best_negative_index": best_negative_index,
    }


@torch.inference_mode()
def diagnose_cached_view(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    case: dict,
) -> dict[str, object]:
    if len(case["decisions"]) != 1:
        raise ValueError("W6f cached case must contain exactly one decision")
    decision = case["decisions"][0]
    if decision["question_id"] != "diagnosis":
        raise ValueError("W6f cached decision must be diagnosis")

    tensors = _cached_tensors(case, decision)
    native_coarse = scorer(
        state_tokens=tensors["state_tokens"],
        state_mask=tensors["state_mask"],
        question_tokens=tensors["question_tokens"],
        question_mask=tensors["question_mask"],
        option_tokens=tensors["option_tokens"],
        option_token_ids=tensors["option_token_ids"],
        option_mask=tensors["option_mask"],
    )
    uniform_coarse = uniform_salience_logits(
        scorer,
        state_tokens=tensors["state_tokens"],
        state_mask=tensors["state_mask"],
        question_tokens=tensors["question_tokens"],
        question_mask=tensors["question_mask"],
        option_tokens=tensors["option_tokens"],
        option_token_ids=tensors["option_token_ids"],
        option_mask=tensors["option_mask"],
    )

    out = hira(
        tensors["question"],
        tensors["state_segments"],
        tensors["options"],
        tensors["qtype"],
        coarse_override=native_coarse,
        forced_budget=tensors["options"].shape[1],
        adaptive_budget=False,
    )

    gold = int(decision["gold_index"])
    native = _rank_metrics(out.coarse_logits[0], gold)
    final = _rank_metrics(out.logits[0], gold)
    uniform = _rank_metrics(uniform_coarse[0], gold)

    relation = out.relation_delta[0].detach().cpu().to(torch.float64)
    gold_delta = float(relation[gold])
    final_best_negative = int(final["best_negative_index"])
    coarse_best_negative = int(native["best_negative_index"])

    predicted_index = int(final["predicted_index"])
    winner_fields: tuple[str, ...] = ()
    winner_distance = 0
    if predicted_index != gold:
        gold_signature = tuple(case["gold_signature"])
        predicted_signature = tuple(case["option_signatures"][predicted_index])
        roles = tuple(case["roles"])
        winner_fields = tuple(
            role
            for role, left, right in zip(
                roles,
                gold_signature,
                predicted_signature,
            )
            if left != right
        )
        winner_distance = sum(
            left != right
            for left, right in zip(gold_signature, predicted_signature)
        )

    return {
        "base_id": case["base_id"],
        "case_id": case["case_id"],
        "domain_id": case["domain_id"],
        "diagnosis_k": int(case["diagnosis_k"]),
        "gold_index": gold,
        "native_coarse": native,
        "uniform_coarse": uniform,
        "final": final,
        "relation": {
            "gold_delta": gold_delta,
            "coarse_best_negative_delta": float(
                relation[coarse_best_negative]
            ),
            "final_best_negative_delta": float(
                relation[final_best_negative]
            ),
            "rescued": (not bool(native["top1"])) and bool(final["top1"]),
            "damaged": bool(native["top1"]) and (not bool(final["top1"])),
            "gold_rank_change": int(final["rank"]) - int(native["rank"]),
        },
        "winner_mismatch_fields": winner_fields,
        "winner_distance": int(winner_distance),
    }


def _mean(records: Iterable[dict], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for record in records:
        value: object = record
        for key in path:
            value = value[key]  # type: ignore[index]
        values.append(float(value))
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _aggregate_k(records: list[dict]) -> dict[str, object]:
    native_top1 = _mean(records, ("native_coarse", "top1"))
    native_top5 = _mean(records, ("native_coarse", "top5"))
    uniform_top1 = _mean(records, ("uniform_coarse", "top1"))
    uniform_top5 = _mean(records, ("uniform_coarse", "top5"))
    final_top1 = _mean(records, ("final", "top1"))
    final_top5 = _mean(records, ("final", "top5"))
    rescue = _mean(records, ("relation", "rescued"))
    damage = _mean(records, ("relation", "damaged"))

    final_errors = [row for row in records if not row["final"]["top1"]]
    coarse_top5_in_final_errors = (
        _mean(final_errors, ("native_coarse", "top5"))
        if final_errors
        else 0.0
    )
    return {
        "n": len(records),
        "native_coarse_top1": native_top1,
        "native_coarse_top5": native_top5,
        "native_coarse_mrr": _mean(
            records,
            ("native_coarse", "reciprocal_rank"),
        ),
        "native_coarse_mean_margin": _mean(
            records,
            ("native_coarse", "margin"),
        ),
        "uniform_coarse_top1": uniform_top1,
        "uniform_coarse_top5": uniform_top5,
        "uniform_coarse_mrr": _mean(
            records,
            ("uniform_coarse", "reciprocal_rank"),
        ),
        "uniform_coarse_mean_margin": _mean(
            records,
            ("uniform_coarse", "margin"),
        ),
        "final_top1": final_top1,
        "final_top5": final_top5,
        "final_mrr": _mean(records, ("final", "reciprocal_rank")),
        "final_mean_margin": _mean(records, ("final", "margin")),
        "relation_rescue_rate": rescue,
        "relation_damage_rate": damage,
        "mean_gold_relation_delta": _mean(
            records,
            ("relation", "gold_delta"),
        ),
        "mean_final_best_negative_relation_delta": _mean(
            records,
            ("relation", "final_best_negative_delta"),
        ),
        "mean_gold_rank_change": _mean(
            records,
            ("relation", "gold_rank_change"),
        ),
        "final_error_gold_already_coarse_top5": (
            coarse_top5_in_final_errors
        ),
    }


def _trajectory(records: list[dict]) -> dict[str, object]:
    by_base: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in records:
        by_base[str(row["base_id"])][int(row["diagnosis_k"])] = row

    complete = {
        base_id: views
        for base_id, views in by_base.items()
        if set(views) == {8, 16, 32, 64}
    }
    coarse_rank_drift: list[float] = []
    final_rank_drift: list[float] = []
    coarse_logit_abs_drift: list[float] = []
    margin_collapse: list[float] = []
    first_loss = Counter()

    for views in complete.values():
        k8, k64 = views[8], views[64]
        coarse_rank_drift.append(
            float(k64["native_coarse"]["rank"])
            - float(k8["native_coarse"]["rank"])
        )
        final_rank_drift.append(
            float(k64["final"]["rank"])
            - float(k8["final"]["rank"])
        )
        coarse_logit_abs_drift.append(
            abs(
                float(k64["native_coarse"]["gold_logit"])
                - float(k8["native_coarse"]["gold_logit"])
            )
        )
        margin_collapse.append(
            float(k8["native_coarse"]["margin"])
            - float(k64["native_coarse"]["margin"])
        )

        lost = "never"
        for k in (8, 16, 32, 64):
            if not bool(views[k]["final"]["top1"]):
                lost = str(k)
                break
        first_loss[lost] += 1

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    return {
        "complete_base_count": len(complete),
        "mean_native_coarse_rank_drift_k8_to_k64": avg(
            coarse_rank_drift
        ),
        "mean_final_rank_drift_k8_to_k64": avg(final_rank_drift),
        "mean_abs_gold_coarse_logit_drift_k8_to_k64": avg(
            coarse_logit_abs_drift
        ),
        "mean_native_margin_collapse_k8_to_k64": avg(margin_collapse),
        "first_final_top1_loss_k": dict(first_loss),
    }


def localization_classification(
    k8: dict[str, object],
    k64: dict[str, object],
) -> dict[str, object]:
    salience = (
        (
            float(k64["uniform_coarse_top1"])
            - float(k64["native_coarse_top1"])
            >= 0.08
        )
        or (
            float(k64["uniform_coarse_top5"])
            - float(k64["native_coarse_top5"])
            >= 0.08
        )
    ) and (
        float(k8["uniform_coarse_top1"])
        - float(k8["native_coarse_top1"])
        >= -0.03
    )

    reranking = (
        float(k64["native_coarse_top1"])
        - float(k64["final_top1"])
        >= 0.05
    ) or (
        float(k64["relation_damage_rate"])
        - float(k64["relation_rescue_rate"])
        >= 0.05
    )

    coarse = (
        float(k64["native_coarse_top5"]) < 0.75
        and not salience
        and not reranking
    )

    if salience and reranking:
        label = "MIXED_HIGH_K_FAILURE"
    elif salience:
        label = "CANDIDATE_RELATIVE_SALIENCE_IMPLICATED"
    elif reranking:
        label = "RELATION_RERANKING_IMPLICATED"
    elif coarse:
        label = "COARSE_BINDING_LIMIT"
    else:
        label = "UNRESOLVED_HIGH_K_FAILURE"

    return {
        "classification": label,
        "candidate_relative_salience_implicated": salience,
        "relation_reranking_implicated": reranking,
        "coarse_binding_limit": coarse,
        "k64_uniform_top1_gain": (
            float(k64["uniform_coarse_top1"])
            - float(k64["native_coarse_top1"])
        ),
        "k64_uniform_top5_gain": (
            float(k64["uniform_coarse_top5"])
            - float(k64["native_coarse_top5"])
        ),
        "k8_uniform_top1_gain": (
            float(k8["uniform_coarse_top1"])
            - float(k8["native_coarse_top1"])
        ),
        "k64_final_minus_coarse_top1": (
            float(k64["final_top1"])
            - float(k64["native_coarse_top1"])
        ),
        "k64_damage_minus_rescue": (
            float(k64["relation_damage_rate"])
            - float(k64["relation_rescue_rate"])
        ),
    }


def aggregate_w6f_records(records: Sequence[dict]) -> dict[str, object]:
    if not records:
        raise ValueError("W6f aggregation requires records")

    by_k = {
        k: [row for row in records if int(row["diagnosis_k"]) == k]
        for k in (8, 16, 32, 64)
    }
    if any(not rows for rows in by_k.values()):
        raise ValueError("W6f aggregation requires every K")

    metrics = {
        str(k): _aggregate_k(rows)
        for k, rows in by_k.items()
    }
    classification = localization_classification(
        metrics["8"],
        metrics["64"],
    )

    wrong_fields = Counter(
        field
        for row in records
        for field in row["winner_mismatch_fields"]
    )
    winner_distances = Counter(
        int(row["winner_distance"])
        for row in records
        if int(row["winner_distance"]) > 0
    )

    return {
        "view_count": len(records),
        "base_trajectory": _trajectory(list(records)),
        "per_k": metrics,
        "classification": classification,
        "winning_wrong_field_counts": dict(wrong_fields),
        "winning_wrong_distance_counts": dict(winner_distances),
    }
