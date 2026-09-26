from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import torch

from .compositional_projection_authority import PARTITION_DOMAINS, compose_f2

REFERENCE_NAMES = ("deberta_nli", "roberta_nli")
REFERENCE_TASKS = ("F0", "F1", "U", "C", "F2")


def _binary_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("W28 binary metric rows cannot be empty")
    recalls = {}
    for gold in (0, 1):
        subset = [row for row in rows if int(row["gold"]) == gold]
        if not subset:
            raise ValueError("W28 binary metric requires both labels")
        recalls[gold] = sum(bool(row["correct"]) for row in subset) / len(subset)
    return {
        "n": len(rows),
        "top1": sum(bool(row["correct"]) for row in rows) / len(rows),
        "balanced_accuracy": (recalls[0] + recalls[1]) / 2.0,
        "negative_recall": recalls[0],
        "positive_recall": recalls[1],
        "mrr": sum(float(row["mrr"]) for row in rows) / len(rows),
        "mean_margin": sum(float(row["margin"]) for row in rows) / len(rows),
        "probability_mass_max_error": max(
            float(row["probability_mass_error"]) for row in rows
        ),
    }


def _score_row(scores: Sequence[float], gold: int) -> dict[str, object]:
    logits = torch.tensor(list(scores), dtype=torch.float32)
    if tuple(logits.shape) != (2,) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("W28 binary scores must be finite length-2")
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


def _case_golds(case: Mapping[str, object]) -> dict[str, int]:
    f0, f1, u, c = (int(x) for x in case["evidence_vector"])
    f2 = compose_f2(u, c)
    if tuple(case["factor_vector"]) != (f0, f1, f2):
        raise RuntimeError("W28 factor/evidence integrity changed")
    return {"F0": f0, "F1": f1, "U": u, "C": c, "F2": f2}


def evaluate_score_lookup(
    cases: Sequence[Mapping[str, object]],
    score_lookup: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, object]:
    raw = []
    for case in cases:
        case_id = str(case["case_id"])
        golds = _case_golds(case)
        tasks = {
            task: _score_row(score_lookup[case_id][task], golds[task])
            for task in REFERENCE_TASKS
        }
        composed_pred = compose_f2(
            int(tasks["U"]["pred"]),
            int(tasks["C"]["pred"]),
        )
        raw.append(
            {
                "case_id": case_id,
                "domain_id": str(case["domain_id"]),
                "tasks": tasks,
                "composed_f2_gold": golds["F2"],
                "composed_f2_pred": composed_pred,
                "composed_correct": composed_pred == golds["F2"],
            }
        )

    def summarize(rows):
        tasks = {
            task: _binary_metrics([row["tasks"][task] for row in rows])
            for task in REFERENCE_TASKS
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
        for task in REFERENCE_TASKS:
            out[case_id][task] = [
                sum(float(model_scores[name][case_id][task][value]) for name in names)
                / len(names)
                for value in (0, 1)
            ]
    return out


def _agreement(
    cases: Sequence[Mapping[str, object]],
    individual: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    a = individual[REFERENCE_NAMES[0]]["predictions"]
    b = individual[REFERENCE_NAMES[1]]["predictions"]
    domains = sorted({str(case["domain_id"]) for case in cases})
    out = {}
    for domain in domains:
        ids = [str(case["case_id"]) for case in cases if str(case["domain_id"]) == domain]
        out[domain] = {
            task: sum(
                int(a[case_id]["tasks"][task]["pred"])
                == int(b[case_id]["tasks"][task]["pred"])
                for case_id in ids
            ) / len(ids)
            for task in REFERENCE_TASKS
        }
    return out


def evaluate_reference_panel(
    cases: Sequence[Mapping[str, object]],
    model_scores: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
) -> dict[str, object]:
    if set(model_scores) != set(REFERENCE_NAMES):
        raise ValueError("W28 requires frozen DeBERTa/RoBERTa panel")
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


def _individual_floor(row: Mapping[str, object], *, confirm: bool) -> bool:
    atom_floor = 0.82 if confirm else 0.85
    uc_floor = 0.84 if confirm else 0.86
    composed_floor = 0.84 if confirm else 0.86
    tasks = row["tasks"]
    return bool(
        float(tasks["F0"]["balanced_accuracy"]) >= atom_floor
        and float(tasks["F1"]["balanced_accuracy"]) >= atom_floor
        and float(tasks["U"]["balanced_accuracy"]) >= uc_floor
        and float(tasks["C"]["balanced_accuracy"]) >= uc_floor
        and float(row["composed_f2"]["balanced_accuracy"]) >= composed_floor
    )


def qualification_domain_pass(panel: Mapping[str, object], domain: str) -> bool:
    row = panel["consensus"]["per_domain"][domain]
    tasks = row["tasks"]
    agreement = panel["agreement"][domain]
    return bool(
        float(tasks["F0"]["top1"]) >= 0.94
        and float(tasks["F0"]["balanced_accuracy"]) >= 0.92
        and float(tasks["F1"]["top1"]) >= 0.94
        and float(tasks["F1"]["balanced_accuracy"]) >= 0.92
        and float(tasks["U"]["top1"]) >= 0.94
        and float(tasks["U"]["balanced_accuracy"]) >= 0.94
        and float(tasks["C"]["top1"]) >= 0.94
        and float(tasks["C"]["balanced_accuracy"]) >= 0.94
        and float(row["composed_f2"]["balanced_accuracy"]) >= 0.94
        and float(row["composed_f2"]["positive_recall"]) >= 0.90
        and float(row["composed_f2"]["negative_recall"]) >= 0.95
        and float(row["composed_f2"]["probability_mass_max_error"]) <= 1e-6
        and all(
            _individual_floor(
                panel["individual"][name]["per_domain"][domain],
                confirm=False,
            )
            for name in REFERENCE_NAMES
        )
        and float(agreement["F0"]) >= 0.88
        and float(agreement["F1"]) >= 0.88
        and float(agreement["U"]) >= 0.90
        and float(agreement["C"]) >= 0.90
    )


def confirm_domain_pass(panel: Mapping[str, object], domain: str) -> bool:
    row = panel["consensus"]["per_domain"][domain]
    tasks = row["tasks"]
    agreement = panel["agreement"][domain]
    return bool(
        float(tasks["F0"]["top1"]) >= 0.92
        and float(tasks["F0"]["balanced_accuracy"]) >= 0.90
        and float(tasks["F1"]["top1"]) >= 0.92
        and float(tasks["F1"]["balanced_accuracy"]) >= 0.90
        and float(tasks["U"]["top1"]) >= 0.92
        and float(tasks["U"]["balanced_accuracy"]) >= 0.92
        and float(tasks["C"]["top1"]) >= 0.92
        and float(tasks["C"]["balanced_accuracy"]) >= 0.92
        and float(row["composed_f2"]["balanced_accuracy"]) >= 0.92
        and float(row["composed_f2"]["positive_recall"]) >= 0.88
        and float(row["composed_f2"]["negative_recall"]) >= 0.94
        and float(row["composed_f2"]["probability_mass_max_error"]) <= 1e-6
        and all(
            _individual_floor(
                panel["individual"][name]["per_domain"][domain],
                confirm=True,
            )
            for name in REFERENCE_NAMES
        )
        and float(agreement["F0"]) >= 0.85
        and float(agreement["F1"]) >= 0.85
        and float(agreement["U"]) >= 0.88
        and float(agreement["C"]) >= 0.88
    )


def qualification_result(panel: Mapping[str, object]) -> dict[str, object]:
    domains = PARTITION_DOMAINS["qualification"]
    per_domain = {
        domain: qualification_domain_pass(panel, domain)
        for domain in domains
    }
    passed = all(per_domain.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "outcome": (
            "W28_REFERENCE_QUALIFIED"
            if passed
            else "W28_REFERENCE_QUALIFICATION_FAIL"
        ),
        "per_domain": per_domain,
    }


__all__ = [
    "REFERENCE_NAMES",
    "REFERENCE_TASKS",
    "confirm_domain_pass",
    "evaluate_reference_panel",
    "evaluate_score_lookup",
    "qualification_domain_pass",
    "qualification_result",
]
