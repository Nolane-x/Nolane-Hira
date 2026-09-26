from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .anchor_preserving_residual_training import _margin, _rank_metrics
from .projection_replication_authority import FACTOR_IDS, compose_severity

REFERENCE_NAMES = ("deberta_nli", "roberta_nli")


def _record(logits: Tensor, gold: int) -> dict[str, object]:
    logits = logits.float()
    if logits.ndim != 1 or logits.numel() != 2 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("W26 reference requires two finite option scores")
    rank, mrr, _ = _rank_metrics(logits, int(gold))
    pred = int(logits.argmax())
    probs = torch.softmax(logits, dim=0)
    return {
        "gold": int(gold),
        "pred": pred,
        "correct": pred == int(gold),
        "rank": int(rank),
        "mrr": float(mrr),
        "margin": float(_margin(logits, int(gold))),
        "probability_mass_error": abs(float(probs.sum()) - 1.0),
    }


def _balanced_accuracy(rows: Sequence[Mapping[str, object]]) -> float:
    recalls = []
    for gold in (0, 1):
        subset = [row for row in rows if int(row["gold"]) == gold]
        if not subset:
            return 0.0
        recalls.append(sum(bool(row["correct"]) for row in subset) / len(subset))
    return sum(recalls) / 2.0


def _factor_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "top1": 0.0,
            "balanced_accuracy": 0.0,
            "mrr": 0.0,
            "mean_margin": 0.0,
        }
    return {
        "n": n,
        "top1": sum(bool(row["correct"]) for row in rows) / n,
        "balanced_accuracy": _balanced_accuracy(rows),
        "mrr": sum(float(row["mrr"]) for row in rows) / n,
        "mean_margin": sum(float(row["margin"]) for row in rows) / n,
    }


def _prediction_row(
    case: Mapping[str, object],
    score_row: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    gold_vector = tuple(int(x) for x in case["factor_vector"])
    factors = {
        factor_id: _record(
            torch.tensor(score_row[factor_id], dtype=torch.float32),
            gold_vector[index],
        )
        for index, factor_id in enumerate(FACTOR_IDS)
    }
    pred_vector = tuple(int(factors[f]["pred"]) for f in FACTOR_IDS)
    composed = compose_severity(pred_vector)
    severity = int(case["severity"])
    return {
        "case_id": str(case["case_id"]),
        "domain_id": str(case["domain_id"]),
        "severity_gold": severity,
        "factor_gold": gold_vector,
        "factors": factors,
        "factor_pred_vector": pred_vector,
        "vector_correct": pred_vector == gold_vector,
        "composed_severity": composed,
        "composed_correct": composed == severity if composed is not None else False,
        "composed_mae": abs(int(composed) - severity) if composed is not None else 3,
        "invalid": composed is None,
    }


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    n = len(rows)
    factor_rows = {
        factor_id: [row["factors"][factor_id] for row in rows]
        for factor_id in FACTOR_IDS
    }
    return {
        "case_count": n,
        "factors": {
            factor_id: _factor_summary(factor_rows[factor_id])
            for factor_id in FACTOR_IDS
        },
        "factor_vector_top1": sum(bool(row["vector_correct"]) for row in rows) / max(1, n),
        "invalid_factor_vector_rate": sum(bool(row["invalid"]) for row in rows) / max(1, n),
        "composed_severity_top1": sum(bool(row["composed_correct"]) for row in rows) / max(1, n),
        "composed_severity_mae": sum(float(row["composed_mae"]) for row in rows) / max(1, n),
        "factor_probability_mass_max_error": max(
            [
                max(float(v["probability_mass_error"]) for v in row["factors"].values())
                for row in rows
            ]
            or [0.0]
        ),
    }


def evaluate_score_lookup(
    cases: Sequence[Mapping[str, object]],
    score_lookup: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, object]:
    raw = []
    for case in cases:
        case_id = str(case["case_id"])
        if case_id not in score_lookup:
            raise RuntimeError(f"W26 missing reference scores for {case_id}")
        raw.append(_prediction_row(case, score_lookup[case_id]))

    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)

    return {
        "per_domain": {
            domain: _summary(rows)
            for domain, rows in sorted(by_domain.items())
        },
        "pooled": _summary(raw),
        "predictions": {str(row["case_id"]): row for row in raw},
    }


def _consensus_lookup(
    model_scores: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
) -> dict[str, dict[str, list[float]]]:
    if set(model_scores) != set(REFERENCE_NAMES):
        raise ValueError("W26 reference panel model set changed")
    case_ids = set(model_scores[REFERENCE_NAMES[0]])
    if set(model_scores[REFERENCE_NAMES[1]]) != case_ids:
        raise ValueError("W26 reference panel case identity changed")

    out: dict[str, dict[str, list[float]]] = {}
    for case_id in sorted(case_ids):
        out[case_id] = {}
        for factor_id in FACTOR_IDS:
            rows = [
                model_scores[name][case_id][factor_id]
                for name in REFERENCE_NAMES
            ]
            if any(len(row) != 2 for row in rows):
                raise ValueError("W26 reference factor score width changed")
            out[case_id][factor_id] = [
                sum(float(row[value]) for row in rows) / len(rows)
                for value in (0, 1)
            ]
    return out


def _agreement(
    cases: Sequence[Mapping[str, object]],
    individual: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    out = {}
    for domain in sorted({str(case["domain_id"]) for case in cases}):
        ids = [
            str(case["case_id"])
            for case in cases
            if str(case["domain_id"]) == domain
        ]
        domain_row = {}
        for factor_id in FACTOR_IDS:
            matches = 0
            for case_id in ids:
                pred0 = int(
                    individual[REFERENCE_NAMES[0]]["predictions"][case_id]["factors"][factor_id]["pred"]
                )
                pred1 = int(
                    individual[REFERENCE_NAMES[1]]["predictions"][case_id]["factors"][factor_id]["pred"]
                )
                matches += pred0 == pred1
            domain_row[factor_id] = matches / max(1, len(ids))
        out[domain] = domain_row
    return out


def evaluate_reference_panel(
    cases: Sequence[Mapping[str, object]],
    model_scores: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
) -> dict[str, object]:
    individual = {
        name: evaluate_score_lookup(cases, model_scores[name])
        for name in REFERENCE_NAMES
    }
    consensus = evaluate_score_lookup(cases, _consensus_lookup(model_scores))
    agreement = _agreement(cases, individual)
    return {
        "individual": individual,
        "consensus": consensus,
        "agreement": agreement,
    }


def _individual_floor(row: Mapping[str, object], *, qualification: bool) -> bool:
    factor_floor = 0.85 if qualification else 0.82
    ba_floor = 0.85 if qualification else 0.0
    vector_floor = 0.75 if qualification else 0.0
    composed_floor = 0.75 if qualification else 0.72
    invalid_ceiling = 0.10 if qualification else 1.0

    return bool(
        all(float(row["factors"][f]["top1"]) >= factor_floor for f in FACTOR_IDS)
        and all(float(row["factors"][f]["balanced_accuracy"]) >= ba_floor for f in FACTOR_IDS)
        and float(row["factor_vector_top1"]) >= vector_floor
        and float(row["composed_severity_top1"]) >= composed_floor
        and float(row["invalid_factor_vector_rate"]) <= invalid_ceiling
    )


def qualification_domain_pass(
    panel: Mapping[str, object],
    domain: str,
) -> bool:
    consensus = panel["consensus"]["per_domain"][domain]
    agreement = panel["agreement"][domain]
    return bool(
        all(float(consensus["factors"][f]["top1"]) >= 0.94 for f in FACTOR_IDS)
        and all(float(consensus["factors"][f]["balanced_accuracy"]) >= 0.92 for f in FACTOR_IDS)
        and float(consensus["factor_vector_top1"]) >= 0.90
        and float(consensus["composed_severity_top1"]) >= 0.90
        and float(consensus["invalid_factor_vector_rate"]) <= 0.03
        and float(consensus["factor_probability_mass_max_error"]) <= 1e-6
        and all(
            _individual_floor(panel["individual"][name]["per_domain"][domain], qualification=True)
            for name in REFERENCE_NAMES
        )
        and all(float(agreement[f]) >= 0.88 for f in FACTOR_IDS)
    )


def confirm_domain_pass(
    panel: Mapping[str, object],
    domain: str,
) -> bool:
    consensus = panel["consensus"]["per_domain"][domain]
    agreement = panel["agreement"][domain]
    return bool(
        all(float(consensus["factors"][f]["top1"]) >= 0.92 for f in FACTOR_IDS)
        and all(float(consensus["factors"][f]["balanced_accuracy"]) >= 0.90 for f in FACTOR_IDS)
        and float(consensus["factor_vector_top1"]) >= 0.85
        and float(consensus["composed_severity_top1"]) >= 0.85
        and float(consensus["invalid_factor_vector_rate"]) <= 0.05
        and float(consensus["factor_probability_mass_max_error"]) <= 1e-6
        and all(
            _individual_floor(panel["individual"][name]["per_domain"][domain], qualification=False)
            for name in REFERENCE_NAMES
        )
        and all(float(agreement[f]) >= 0.85 for f in FACTOR_IDS)
    )


def qualification_result(panel: Mapping[str, object]) -> dict[str, object]:
    domains = ("EA", "EB")
    per_domain = {
        domain: qualification_domain_pass(panel, domain)
        for domain in domains
    }
    return {
        "status": "PASS" if all(per_domain.values()) else "FAIL",
        "outcome": (
            "REFERENCE_QUALIFIED"
            if all(per_domain.values())
            else "W26_REFERENCE_QUALIFICATION_FAIL"
        ),
        "per_domain": per_domain,
    }


__all__ = [
    "REFERENCE_NAMES",
    "confirm_domain_pass",
    "evaluate_reference_panel",
    "evaluate_score_lookup",
    "qualification_domain_pass",
    "qualification_result",
]
