from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import torch

from .f2_compositional_authority import DOMAINS, TASK_IDS, compose_f2

REFERENCE_NAMES = ("deberta_nli", "roberta_nli")


def _binary_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    n = len(rows)
    if n == 0:
        raise ValueError("W27 binary metric cell cannot be empty")
    recalls = {}
    for gold in (0, 1):
        subset = [row for row in rows if int(row["gold"]) == gold]
        if not subset:
            raise ValueError("W27 binary metric requires both labels")
        recalls[gold] = sum(bool(row["correct"]) for row in subset) / len(subset)
    return {
        "n": n,
        "top1": sum(bool(row["correct"]) for row in rows) / n,
        "balanced_accuracy": (recalls[0] + recalls[1]) / 2.0,
        "negative_recall": recalls[0],
        "positive_recall": recalls[1],
        "mrr": sum(float(row["mrr"]) for row in rows) / n,
        "mean_margin": sum(float(row["margin"]) for row in rows) / n,
        "probability_mass_max_error": max(
            float(row["probability_mass_error"]) for row in rows
        ),
    }


def _score_row(scores: Sequence[float], gold: int) -> dict[str, object]:
    logits = torch.tensor(list(scores), dtype=torch.float32)
    if tuple(logits.shape) != (2,) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("W27 binary reference scores must be finite length-2")
    probs = torch.softmax(logits, dim=0)
    pred = int(logits.argmax())
    order = torch.argsort(logits, descending=True)
    rank = int((order == int(gold)).nonzero(as_tuple=False)[0].item()) + 1
    other = 1 - int(gold)
    return {
        "gold": int(gold),
        "pred": pred,
        "correct": pred == int(gold),
        "mrr": 1.0 / rank,
        "margin": float(logits[int(gold)] - logits[other]),
        "probability_mass_error": abs(float(probs.sum()) - 1.0),
    }


def evaluate_score_lookup(
    cases: Sequence[Mapping[str, object]],
    score_lookup: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, object]:
    raw = []
    for case in cases:
        case_id = str(case["case_id"])
        domain = str(case["domain_id"])
        golds = {
            "U": int(case["timing_atom"]),
            "C": int(case["consequence_atom"]),
            "F2": int(case["f2"]),
        }
        task_rows = {
            task: _score_row(score_lookup[case_id][task], golds[task])
            for task in TASK_IDS
        }
        composed_pred = compose_f2(
            int(task_rows["U"]["pred"]),
            int(task_rows["C"]["pred"]),
        )
        raw.append(
            {
                "case_id": case_id,
                "domain_id": domain,
                "tasks": task_rows,
                "composed_f2_gold": int(case["f2"]),
                "composed_f2_pred": composed_pred,
                "composed_correct": composed_pred == int(case["f2"]),
            }
        )

    def summarize(rows):
        tasks = {
            task: _binary_metrics([row["tasks"][task] for row in rows])
            for task in TASK_IDS
        }
        composed_rows = [
            {
                "gold": int(row["composed_f2_gold"]),
                "pred": int(row["composed_f2_pred"]),
                "correct": bool(row["composed_correct"]),
                "mrr": 1.0 if row["composed_correct"] else 0.5,
                "margin": 1.0 if row["composed_correct"] else -1.0,
                "probability_mass_error": max(
                    float(row["tasks"]["U"]["probability_mass_error"]),
                    float(row["tasks"]["C"]["probability_mass_error"]),
                ),
            }
            for row in rows
        ]
        return {
            "tasks": tasks,
            "composed_f2": _binary_metrics(composed_rows),
        }

    by_domain = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)

    return {
        "per_domain": {
            domain: summarize(rows)
            for domain, rows in sorted(by_domain.items())
        },
        "pooled": summarize(raw),
        "predictions": {str(row["case_id"]): row for row in raw},
    }


def _mean_lookup(
    model_scores: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
) -> dict[str, dict[str, list[float]]]:
    names = tuple(model_scores)
    first = model_scores[names[0]]
    out = {}
    for case_id in first:
        out[case_id] = {}
        for task in TASK_IDS:
            values = []
            for option in (0, 1):
                values.append(
                    sum(
                        float(model_scores[name][case_id][task][option])
                        for name in names
                    )
                    / len(names)
                )
            out[case_id][task] = values
    return out


def _agreement(
    cases: Sequence[Mapping[str, object]],
    individual: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    a = individual[REFERENCE_NAMES[0]]["predictions"]
    b = individual[REFERENCE_NAMES[1]]["predictions"]
    out = {}
    for domain in DOMAINS:
        ids = [str(row["case_id"]) for row in cases if str(row["domain_id"]) == domain]
        out[domain] = {}
        for task in TASK_IDS:
            out[domain][task] = sum(
                int(a[case_id]["tasks"][task]["pred"])
                == int(b[case_id]["tasks"][task]["pred"])
                for case_id in ids
            ) / len(ids)
    return out


def evaluate_reference_panel(
    cases: Sequence[Mapping[str, object]],
    model_scores: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
) -> dict[str, object]:
    if set(model_scores) != set(REFERENCE_NAMES):
        raise ValueError("W27 requires exactly frozen DeBERTa/RoBERTa references")
    individual = {
        name: evaluate_score_lookup(cases, model_scores[name])
        for name in REFERENCE_NAMES
    }
    consensus = evaluate_score_lookup(cases, _mean_lookup(model_scores))
    return {
        "individual": individual,
        "consensus": consensus,
        "agreement": _agreement(cases, individual),
    }


def _individual_atomic_floor(row: Mapping[str, object]) -> bool:
    return bool(
        float(row["tasks"]["U"]["balanced_accuracy"]) >= 0.86
        and float(row["tasks"]["C"]["balanced_accuracy"]) >= 0.86
        and float(row["composed_f2"]["balanced_accuracy"]) >= 0.86
    )


def atomic_domain_pass(panel: Mapping[str, object], domain: str) -> bool:
    row = panel["consensus"]["per_domain"][domain]
    agreement = panel["agreement"][domain]
    return bool(
        float(row["tasks"]["U"]["top1"]) >= 0.94
        and float(row["tasks"]["U"]["balanced_accuracy"]) >= 0.94
        and float(row["tasks"]["C"]["top1"]) >= 0.94
        and float(row["tasks"]["C"]["balanced_accuracy"]) >= 0.94
        and float(row["composed_f2"]["balanced_accuracy"]) >= 0.94
        and float(row["composed_f2"]["positive_recall"]) >= 0.90
        and float(row["composed_f2"]["negative_recall"]) >= 0.95
        and float(row["composed_f2"]["probability_mass_max_error"]) <= 1e-6
        and all(
            _individual_atomic_floor(
                panel["individual"][name]["per_domain"][domain]
            )
            for name in REFERENCE_NAMES
        )
        and float(agreement["U"]) >= 0.90
        and float(agreement["C"]) >= 0.90
    )


def direct_domain_pass(panel: Mapping[str, object], domain: str) -> bool:
    row = panel["consensus"]["per_domain"][domain]["tasks"]["F2"]
    agreement = panel["agreement"][domain]["F2"]
    return bool(
        float(row["balanced_accuracy"]) >= 0.88
        and float(row["positive_recall"]) >= 0.80
        and float(row["negative_recall"]) >= 0.90
        and float(agreement) >= 0.85
    )


def qualification_result(panel: Mapping[str, object]) -> dict[str, object]:
    atomic = {domain: atomic_domain_pass(panel, domain) for domain in DOMAINS}
    direct = {domain: direct_domain_pass(panel, domain) for domain in DOMAINS}
    if not all(atomic.values()):
        outcome = "W27_ATOMIC_REFERENCE_INADEQUATE"
    elif not all(direct.values()):
        outcome = "F2_DIRECT_PACKAGING_LIMIT"
    else:
        outcome = "F2_AUTHORITY_FULLY_QUALIFIED"
    return {
        "outcome": outcome,
        "atomic_per_domain": atomic,
        "direct_per_domain": direct,
        "atomic_qualified_domain_count": sum(atomic.values()),
        "direct_qualified_domain_count": sum(direct.values()),
    }


__all__ = [
    "REFERENCE_NAMES",
    "atomic_domain_pass",
    "direct_domain_pass",
    "evaluate_reference_panel",
    "evaluate_score_lookup",
    "qualification_result",
]
