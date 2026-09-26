from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

import torch
from torch import Tensor

from .anchor_preserving_residual_training import _margin, _rank_metrics
from .atomic_severity_authority import FACTOR_IDS, compose_severity
from .atomic_severity_cache import validate_w24_cache
from .competitive import CompetitiveCoarseScorer
from .interface_decomposition_eval import _q0_symmetric

CONTROL_ORDER = ("deberta_nli", "roberta_nli")
VIEW_IDS = ("D0", "D1", "D2")


@torch.inference_mode()
def _multiview_logits(
    scorer: CompetitiveCoarseScorer,
    state_tokens: Tensor,
    views: Mapping[str, Mapping[str, object]],
) -> Tensor:
    projection = scorer.projection.weight.detach().float()
    rows = []
    for view_id in VIEW_IDS:
        view = views[view_id]
        rows.append(
            _q0_symmetric(
                state_tokens.float(),
                view["option_tokens"].float(),
                view["option_content_mask"].bool(),
                projection,
            )
        )
    return torch.stack(rows, dim=0).mean(dim=0) * scorer.scale().detach().float()


def _record(logits: Tensor, gold: int) -> dict[str, object]:
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("W24 requires finite logits")
    rank, mrr, _ = _rank_metrics(logits, int(gold))
    pred = int(logits.argmax())
    probs = torch.softmax(logits.float(), dim=0)
    return {
        "gold": int(gold),
        "pred": pred,
        "correct": pred == int(gold),
        "rank": int(rank),
        "mrr": float(mrr),
        "margin": float(_margin(logits, int(gold))),
        "probability_mass_error": abs(float(probs.sum()) - 1.0),
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "top1": 0.0,
            "mrr": 0.0,
            "mean_margin": 0.0,
            "mae": 0.0,
        }
    return {
        "n": n,
        "top1": sum(bool(row["correct"]) for row in rows) / n,
        "mrr": sum(float(row["mrr"]) for row in rows) / n,
        "mean_margin": sum(float(row["margin"]) for row in rows) / n,
        "mae": sum(abs(int(row["pred"]) - int(row["gold"])) for row in rows) / n,
    }


def _balanced_accuracy_binary(rows: list[dict[str, object]]) -> float:
    recalls = []
    for gold in (0, 1):
        subset = [row for row in rows if int(row["gold"]) == gold]
        if not subset:
            return 0.0
        recalls.append(sum(bool(row["correct"]) for row in subset) / len(subset))
    return sum(recalls) / 2.0


def _factor_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    value = _summary(rows)
    value["balanced_accuracy"] = _balanced_accuracy_binary(rows)
    return value


def _case_from_hira_logits(
    case: dict,
    direct_severity_logits: Tensor,
    confidence_logits: Tensor,
    factor_logits: Mapping[str, Tensor],
) -> dict[str, object]:
    severity_gold = int(case["severity"])
    confidence_gold = int(case["confidence_index"])
    factor_gold = tuple(int(x) for x in case["factor_vector"])

    direct = _record(direct_severity_logits, severity_gold)
    confidence = _record(confidence_logits, confidence_gold)
    factors = {
        factor_id: _record(factor_logits[factor_id], factor_gold[index])
        for index, factor_id in enumerate(FACTOR_IDS)
    }
    pred_vector = tuple(int(factors[factor_id]["pred"]) for factor_id in FACTOR_IDS)
    composed = compose_severity(pred_vector)
    invalid = composed is None
    composed_correct = composed == severity_gold if composed is not None else False
    composed_mae = (
        abs(int(composed) - severity_gold) if composed is not None else 3
    )

    return {
        "case_id": str(case["case_id"]),
        "domain_id": str(case["domain_id"]),
        "severity_gold": severity_gold,
        "confidence_gold": confidence_gold,
        "factor_gold": factor_gold,
        "direct_severity": direct,
        "confidence": confidence,
        "factors": factors,
        "factor_pred_vector": pred_vector,
        "factor_vector_correct": pred_vector == factor_gold,
        "composed_severity": composed,
        "composed_correct": composed_correct,
        "composed_mae": composed_mae,
        "invalid_factor_vector": invalid,
        "factor_probability_mass_error": max(
            [float(row["probability_mass_error"]) for row in factors.values()] or [0.0]
        ),
    }


def _summarize_hira(raw: list[dict[str, object]]) -> dict[str, object]:
    n = len(raw)
    direct_rows = [row["direct_severity"] for row in raw]
    confidence_rows = [row["confidence"] for row in raw]
    factor_rows = {
        factor_id: [row["factors"][factor_id] for row in raw]
        for factor_id in FACTOR_IDS
    }
    vector_accuracy = sum(bool(row["factor_vector_correct"]) for row in raw) / max(1, n)
    invalid_rate = sum(bool(row["invalid_factor_vector"]) for row in raw) / max(1, n)
    composed_top1 = sum(bool(row["composed_correct"]) for row in raw) / max(1, n)
    composed_mae = sum(float(row["composed_mae"]) for row in raw) / max(1, n)
    direct_top1 = _summary(direct_rows)["top1"]

    wrong_to_right = sum(
        (not bool(row["direct_severity"]["correct"])) and bool(row["composed_correct"])
        for row in raw
    )
    right_to_wrong = sum(
        bool(row["direct_severity"]["correct"]) and (not bool(row["composed_correct"]))
        for row in raw
    )

    return {
        "case_count": n,
        "direct_severity": _summary(direct_rows),
        "confidence": _summary(confidence_rows),
        "factors": {
            factor_id: _factor_summary(factor_rows[factor_id])
            for factor_id in FACTOR_IDS
        },
        "factor_vector_top1": vector_accuracy,
        "invalid_factor_vector_rate": invalid_rate,
        "composed_severity_top1": composed_top1,
        "composed_severity_mae": composed_mae,
        "factor_probability_mass_max_error": max(
            [float(row["factor_probability_mass_error"]) for row in raw] or [0.0]
        ),
        "factorization_gain": composed_top1 - float(direct_top1),
        "direct_to_composed_wrong_to_right": wrong_to_right,
        "direct_to_composed_right_to_wrong": right_to_wrong,
        "representation_accounting": {
            "logical_state_compiles_per_case": 1,
            "a13_query_invocations_per_case": 1,
            "encoded_query_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
            "factor_ids": ["F0", "F1", "F2"],
            "factor_decoder": {"000": 0, "100": 1, "110": 2, "111": 3},
        },
    }


@torch.inference_mode()
def evaluate_w24(
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w24_cache(cache)
    scorer.eval()
    raw = []
    for case in cache["cases"]:
        domain = str(case["domain_id"])
        pack = cache["schemas"][domain]
        severity_tokens = case["representations"]["severity_tokens"].float()
        confidence_tokens = case["representations"]["confidence_tokens"].float()
        factor_logits = {
            factor_id: _multiview_logits(
                scorer,
                severity_tokens,
                pack[f"factor_{factor_id.lower()}"],
            )
            for factor_id in FACTOR_IDS
        }
        raw.append(
            _case_from_hira_logits(
                case,
                _multiview_logits(scorer, severity_tokens, pack["direct_severity"]),
                _multiview_logits(scorer, confidence_tokens, pack["confidence"]),
                factor_logits,
            )
        )

    by_domain = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)
    return {
        "per_domain": {
            domain: _summarize_hira(rows)
            for domain, rows in sorted(by_domain.items())
        },
        "pooled": _summarize_hira(raw),
    }


def _reference_case(
    case: dict,
    score_row: Mapping[str, object],
) -> dict[str, object]:
    gold_vector = tuple(int(x) for x in case["factor_vector"])
    factors = {}
    for index, factor_id in enumerate(FACTOR_IDS):
        logits = torch.tensor(score_row[factor_id], dtype=torch.float32)
        factors[factor_id] = _record(logits, gold_vector[index])
    pred_vector = tuple(int(factors[factor_id]["pred"]) for factor_id in FACTOR_IDS)
    composed = compose_severity(pred_vector)
    severity_gold = int(case["severity"])
    return {
        "case_id": str(case["case_id"]),
        "domain_id": str(case["domain_id"]),
        "severity_gold": severity_gold,
        "factor_gold": gold_vector,
        "factors": factors,
        "factor_pred_vector": pred_vector,
        "factor_vector_correct": pred_vector == gold_vector,
        "composed_severity": composed,
        "composed_correct": composed == severity_gold if composed is not None else False,
        "composed_mae": abs(int(composed) - severity_gold) if composed is not None else 3,
        "invalid_factor_vector": composed is None,
        "factor_probability_mass_error": max(
            [float(row["probability_mass_error"]) for row in factors.values()] or [0.0]
        ),
    }


def _summarize_reference(raw: list[dict[str, object]]) -> dict[str, object]:
    n = len(raw)
    factor_rows = {
        factor_id: [row["factors"][factor_id] for row in raw]
        for factor_id in FACTOR_IDS
    }
    return {
        "case_count": n,
        "factors": {
            factor_id: _factor_summary(factor_rows[factor_id])
            for factor_id in FACTOR_IDS
        },
        "factor_vector_top1": sum(bool(row["factor_vector_correct"]) for row in raw) / max(1, n),
        "invalid_factor_vector_rate": sum(bool(row["invalid_factor_vector"]) for row in raw) / max(1, n),
        "composed_severity_top1": sum(bool(row["composed_correct"]) for row in raw) / max(1, n),
        "composed_severity_mae": sum(float(row["composed_mae"]) for row in raw) / max(1, n),
        "factor_probability_mass_max_error": max(
            [float(row["factor_probability_mass_error"]) for row in raw] or [0.0]
        ),
    }


def atomic_reference_evaluation_from_scores(
    cases: list[dict],
    score_lookup: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    raw = []
    predictions = {}
    for case in cases:
        case_id = str(case["case_id"])
        score_row = score_lookup.get(case_id)
        if score_row is None:
            raise ValueError(f"missing W24 atomic reference scores for {case_id}")
        row = _reference_case(case, score_row)
        raw.append(row)
        predictions[case_id] = {
            "domain_id": str(row["domain_id"]),
            "severity_gold": int(row["severity_gold"]),
            "factor_gold": tuple(int(x) for x in row["factor_gold"]),
            "factor_pred": tuple(int(x) for x in row["factor_pred_vector"]),
            "composed_severity": row["composed_severity"],
        }

    by_domain = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)
    return {
        "summary": {
            "per_domain": {
                domain: _summarize_reference(rows)
                for domain, rows in sorted(by_domain.items())
            },
            "pooled": _summarize_reference(raw),
        },
        "predictions": predictions,
    }


def _confidence_control_domain(
    domain: str,
    references: Mapping[str, Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    for name in CONTROL_ORDER:
        if name not in references or name not in predictions:
            raise ValueError(f"W24 missing frozen confidence control {name}")
    a_summary = references["deberta_nli"]["per_domain"][domain]["prototype"]["confidence"]
    b_summary = references["roberta_nli"]["per_domain"][domain]["prototype"]["confidence"]

    case_ids = sorted(
        case_id
        for case_id, row in predictions["deberta_nli"].items()
        if str(row["domain_id"]) == domain
    )
    if not case_ids:
        raise ValueError(f"W24 no confidence control cases for {domain}")
    agreement = sum(
        int(predictions["deberta_nli"][case_id]["confidence_pred"])
        == int(predictions["roberta_nli"][case_id]["confidence_pred"])
        for case_id in case_ids
    ) / len(case_ids)
    stable = (
        float(a_summary["top1"]) >= .95
        and float(b_summary["top1"]) >= .90
        and agreement >= .90
    )
    return {
        "deberta_confidence_top1": float(a_summary["top1"]),
        "roberta_confidence_top1": float(b_summary["top1"]),
        "confidence_prediction_agreement": agreement,
        "reference_environment_stable": stable,
    }


def _atomic_reference_adequate(summary: Mapping[str, object]) -> bool:
    return (
        all(float(summary["factors"][factor_id]["top1"]) >= .92 for factor_id in FACTOR_IDS)
        and all(
            float(summary["factors"][factor_id]["balanced_accuracy"]) >= .90
            for factor_id in FACTOR_IDS
        )
        and float(summary["factor_vector_top1"]) >= .85
        and float(summary["composed_severity_top1"]) >= .85
        and float(summary["invalid_factor_vector_rate"]) <= .05
        and float(summary["factor_probability_mass_max_error"]) <= 1e-6
    )


def _hira_atomic_adequate(summary: Mapping[str, object]) -> bool:
    return (
        all(float(summary["factors"][factor_id]["top1"]) >= .80 for factor_id in FACTOR_IDS)
        and float(summary["factor_vector_top1"]) >= .62
        and float(summary["composed_severity_top1"]) >= .65
        and float(summary["invalid_factor_vector_rate"]) <= .15
        and float(summary["factor_probability_mass_max_error"]) <= 1e-6
    )


def classify_w24(
    metrics: dict[str, object],
    atomic_reference: Mapping[str, object],
    control_references: Mapping[str, Mapping[str, object]],
    control_predictions: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    domains = sorted(metrics["per_domain"])
    per_domain = {}

    for domain in domains:
        control = _confidence_control_domain(
            domain,
            control_references,
            control_predictions,
        )
        ref_summary = atomic_reference["per_domain"][domain]
        atomic_ref_ok = _atomic_reference_adequate(ref_summary)
        hira_summary = metrics["per_domain"][domain]
        hira_ok = _hira_atomic_adequate(hira_summary)
        direct_top1 = float(hira_summary["direct_severity"]["top1"])
        gain = float(hira_summary["factorization_gain"])

        integrity = (
            float(hira_summary["factor_probability_mass_max_error"]) <= 1e-6
            and hira_summary["representation_accounting"] == {
                "logical_state_compiles_per_case": 1,
                "a13_query_invocations_per_case": 1,
                "encoded_query_sequences_per_case": 2,
                "isolated_fields": ["severity", "confidence"],
                "factor_ids": ["F0", "F1", "F2"],
                "factor_decoder": {"000": 0, "100": 1, "110": 2, "111": 3},
            }
        )

        if not bool(control["reference_environment_stable"]):
            classification = "W24_REFERENCE_ENVIRONMENT_UNSTABLE"
        elif not atomic_ref_ok:
            classification = "W24_ATOMIC_REFERENCE_INADEQUATE"
        elif hira_ok and gain >= .12:
            classification = "ATOMIC_SEVERITY_INTERFACE_LIMIT"
        elif not hira_ok:
            classification = "HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT"
        elif direct_top1 >= .75 and gain < .05:
            classification = "DIRECT_SEVERITY_ALREADY_ADEQUATE"
        else:
            classification = "ATOMIC_SEVERITY_UNRESOLVED"

        per_domain[domain] = {
            **control,
            "atomic_reference_adequate": atomic_ref_ok,
            "hira_atomic_adequate": hira_ok,
            "direct_severity_top1": direct_top1,
            "factorization_gain": gain,
            "integrity": integrity,
            "classification": classification,
        }

    counts = Counter(row["classification"] for row in per_domain.values())
    interpretable = (
        "ATOMIC_SEVERITY_INTERFACE_LIMIT",
        "HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT",
        "DIRECT_SEVERITY_ALREADY_ADEQUATE",
    )
    stable = None
    for name in interpretable:
        if counts[name] >= 3:
            stable = name
            break

    env_unstable_count = counts["W24_REFERENCE_ENVIRONMENT_UNSTABLE"]
    atomic_ref_bad_count = counts["W24_ATOMIC_REFERENCE_INADEQUATE"]
    mixed_interpretable = sum(1 for name in interpretable if counts[name] >= 2) >= 2

    if stable is not None:
        outcome = "STABLE_ATOMIC_SEVERITY_LOCALIZATION"
    elif env_unstable_count >= 3:
        outcome = "REFERENCE_ENVIRONMENT_UNSTABLE"
    elif atomic_ref_bad_count >= 3:
        outcome = "ATOMIC_AUTHORITY_UNRESOLVED"
    elif mixed_interpretable:
        outcome = "MIXED_ATOMIC_SEVERITY_LOCALIZATION"
    else:
        outcome = "ATOMIC_SEVERITY_UNRESOLVED"

    return {
        "outcome": outcome,
        "stable_classification": stable,
        "classification_counts": dict(sorted(counts.items())),
        "reference_environment_stable_domain_count": sum(
            bool(row["reference_environment_stable"]) for row in per_domain.values()
        ),
        "atomic_reference_adequate_domain_count": sum(
            bool(row["atomic_reference_adequate"]) for row in per_domain.values()
        ),
        "per_domain": per_domain,
    }


__all__ = [
    "CONTROL_ORDER",
    "atomic_reference_evaluation_from_scores",
    "classify_w24",
    "evaluate_w24",
]
