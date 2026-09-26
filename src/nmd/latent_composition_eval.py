from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping

import torch
from torch import Tensor

from .anchor_preserving_residual_training import _margin, _rank_metrics
from .competitive import CompetitiveCoarseScorer
from .interface_decomposition_eval import _q0_symmetric
from .latent_composition_cache import validate_w18_cache

PATHS = (
    "FIELD_DIRECT",
    "LATENT_COMPOSED_HARD",
    "LATENT_COMPOSED_SOFT",
    "ORACLE_LATENT_COMPOSED",
)
VIEW_IDS = ("D0", "D1", "D2")
DIRECT_FIELDS = {
    "diagnosis": ("intent",),
    "response": ("severity",),
    "needs_review": ("severity", "confidence"),
    "risk": ("severity",),
    "urgency": ("severity", "confidence"),
}


def _empty_diag() -> dict[int, dict[str, float]]:
    return {
        4: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0, "margin": 0.0},
        8: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0, "margin": 0.0},
        16: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0, "margin": 0.0},
    }


def _summarize_diag(table: dict[int, dict[str, float]]) -> dict[str, dict[str, float | int]]:
    out = {}
    for k, slot in table.items():
        n = int(slot["n"])
        out[str(k)] = {
            "n": n,
            "top1": float(slot["correct"]) / max(1, n),
            "top5": float(slot["top5"]) / max(1, n),
            "mrr": float(slot["mrr"]) / max(1, n),
            "mean_margin": float(slot["margin"]) / max(1, n),
        }
    return out


def _field_tokens(case: dict, names: tuple[str, ...]) -> Tensor:
    reps = case["representations"]
    return torch.cat([reps[f"{name}_tokens"].float() for name in names], dim=0)


@torch.inference_mode()
def _multiview_logits(
    scorer: CompetitiveCoarseScorer,
    state_tokens: Tensor,
    views: Mapping[str, Mapping[str, object]],
) -> Tensor:
    projection = scorer.projection.weight.detach().float()
    logits = []
    for view_id in VIEW_IDS:
        view = views[view_id]
        raw = _q0_symmetric(
            state_tokens,
            view["option_tokens"].float(),
            view["option_content_mask"].bool(),
            projection,
        )
        logits.append(raw)
    return torch.stack(logits, dim=0).mean(dim=0) * scorer.scale().detach().float()


def _one_hot(index: int, width: int) -> Tensor:
    out = torch.zeros(width, dtype=torch.float64)
    out[int(index)] = 1.0
    return out


def _hard_composed(
    p_intent: Tensor,
    p_severity: Tensor,
    p_confidence: Tensor,
) -> dict[str, Tensor]:
    i = int(p_intent.argmax())
    s = int(p_severity.argmax())
    c = int(p_confidence.argmax())
    review = int(s == 3 or c == 2)
    urgency = min(3, s + int(c == 2))
    return {
        "diagnosis": _one_hot(i, int(p_intent.numel())),
        "response": _one_hot(s, 4),
        "needs_review": _one_hot(review, 2),
        "risk": _one_hot(s, 4),
        "urgency": _one_hot(urgency, 4),
    }


def _soft_composed(
    p_intent: Tensor,
    p_severity: Tensor,
    p_confidence: Tensor,
) -> dict[str, Tensor]:
    p_i = p_intent.double()
    p_s = p_severity.double()
    p_c = p_confidence.double()
    p_true = 1.0 - (1.0 - p_s[3]) * (1.0 - p_c[2])
    review = torch.stack((1.0 - p_true, p_true))
    urgency = torch.zeros(4, dtype=torch.float64)
    for s in range(4):
        for c in range(3):
            u = min(3, s + int(c == 2))
            urgency[u] += p_s[s] * p_c[c]
    return {
        "diagnosis": p_i,
        "response": p_s,
        "needs_review": review,
        "risk": p_s,
        "urgency": urgency,
    }


def _oracle_composed(case: dict, diagnosis_gold: int, diagnosis_width: int) -> dict[str, Tensor]:
    severity = int(case["severity"])
    confidence = int(case["confidence_index"])
    review = int(severity == 3 or confidence == 2)
    urgency = min(3, severity + int(confidence == 2))
    return {
        "diagnosis": _one_hot(diagnosis_gold, diagnosis_width),
        "response": _one_hot(severity, 4),
        "needs_review": _one_hot(review, 2),
        "risk": _one_hot(severity, 4),
        "urgency": _one_hot(urgency, 4),
    }


def _safe_log_probs(p: Tensor) -> Tensor:
    return torch.log(p.double().clamp_min(1e-12)).float()


def _atomic_record(logits: Tensor, gold: int) -> dict[str, float | int | bool]:
    rank, mrr, top5 = _rank_metrics(logits, gold)
    return {
        "rank": rank,
        "mrr": float(mrr),
        "top5": bool(top5),
        "correct": rank == 1,
        "margin": _margin(logits, gold),
    }


def _summarize_atomic(
    intent_rows: list[dict[str, object]],
    severity_rows: list[dict[str, object]],
    confidence_rows: list[dict[str, object]],
) -> dict[str, object]:
    diag = _empty_diag()
    for row in intent_rows:
        k = int(row["k"])
        slot = diag[k]
        slot["n"] += 1
        slot["correct"] += int(bool(row["correct"]))
        slot["top5"] += int(bool(row["top5"]))
        slot["mrr"] += float(row["mrr"])
        slot["margin"] += float(row["margin"])

    def one(rows):
        n = len(rows)
        return {
            "n": n,
            "top1": sum(bool(row["correct"]) for row in rows) / max(1, n),
            "mrr": sum(float(row["mrr"]) for row in rows) / max(1, n),
            "mean_margin": sum(float(row["margin"]) for row in rows) / max(1, n),
        }

    joint = sum(
        bool(s["correct"]) and bool(c["correct"])
        for s, c in zip(severity_rows, confidence_rows)
    ) / max(1, len(severity_rows))
    return {
        "intent_per_k": _summarize_diag(diag),
        "severity": one(severity_rows),
        "confidence": one(confidence_rows),
        "joint_severity_confidence_top1": joint,
    }


def _path_accumulator():
    return {
        "decision_count": 0,
        "correct": 0,
        "primitive_n": defaultdict(int),
        "primitive_correct": defaultdict(int),
        "question_n": defaultdict(int),
        "question_correct": defaultdict(int),
        "non_diag_n": 0,
        "non_diag_correct": 0,
        "hard_brier": 0.0,
        "soft_brier": 0.0,
        "score_error": [],
        "diagnosis": _empty_diag(),
        "max_mass_error": 0.0,
    }


def _finalize_path(slot: dict[str, object], case_count: int) -> dict[str, object]:
    n = int(slot["decision_count"])
    primitive_accuracy = {
        name: slot["primitive_correct"][name] / max(1, slot["primitive_n"][name])
        for name in ("choice", "noul", "score")
    }
    question_accuracy = {
        name: slot["question_correct"][name] / max(1, slot["question_n"][name])
        for name in ("diagnosis", "response", "needs_review", "risk", "urgency")
    }
    return {
        "case_count": case_count,
        "decision_count": n,
        "accuracy": int(slot["correct"]) / max(1, n),
        "primitive_accuracy": primitive_accuracy,
        "question_accuracy": question_accuracy,
        "non_diagnosis_accuracy": int(slot["non_diag_correct"]) / max(1, int(slot["non_diag_n"])),
        "hard_brier": float(slot["hard_brier"]) / max(1, n),
        "soft_brier": float(slot["soft_brier"]) / max(1, n),
        "score_mae": sum(slot["score_error"]) / max(1, len(slot["score_error"])),
        "probability_mass_max_error": float(slot["max_mass_error"]),
        "diagnosis_per_k": _summarize_diag(slot["diagnosis"]),
    }


@torch.inference_mode()
def _evaluate_cases(
    scorer: CompetitiveCoarseScorer,
    cases: list[dict],
) -> dict[str, object]:
    scorer.eval()
    accum = {path: _path_accumulator() for path in PATHS}
    intent_rows = []
    severity_rows = []
    confidence_rows = []

    transitions = {
        q: {"direct_wrong_hard_right": 0, "direct_right_hard_wrong": 0, "n": 0}
        for q in ("diagnosis", "response", "needs_review", "risk", "urgency")
    }
    conditioned = {
        "severity_correct": {
            "response_n": 0,
            "response_correct": 0,
            "risk_n": 0,
            "risk_correct": 0,
        },
        "severity_confidence_correct": {
            "needs_review_n": 0,
            "needs_review_correct": 0,
            "urgency_n": 0,
            "urgency_correct": 0,
        },
    }
    hard_soft_same = 0
    hard_soft_total = 0

    for case in cases:
        decisions = {str(row["question_id"]): row for row in case["decisions"]}
        diagnosis = decisions["diagnosis"]

        intent_logits = _multiview_logits(
            scorer,
            _field_tokens(case, ("intent",)),
            diagnosis["views"],
        )
        severity_logits = _multiview_logits(
            scorer,
            _field_tokens(case, ("severity",)),
            case["latent_views"]["severity"],
        )
        confidence_logits = _multiview_logits(
            scorer,
            _field_tokens(case, ("confidence",)),
            case["latent_views"]["confidence"],
        )
        p_i = torch.softmax(intent_logits.double(), dim=-1)
        p_s = torch.softmax(severity_logits.double(), dim=-1)
        p_c = torch.softmax(confidence_logits.double(), dim=-1)

        irec = _atomic_record(intent_logits, int(diagnosis["gold_index"]))
        irec["k"] = int(case["diagnosis_k"])
        intent_rows.append(irec)
        severity_rows.append(_atomic_record(severity_logits, int(case["severity"])))
        confidence_rows.append(
            _atomic_record(confidence_logits, int(case["confidence_index"]))
        )

        direct_prob = {}
        for q, decision in decisions.items():
            names = DIRECT_FIELDS[q]
            logits = _multiview_logits(
                scorer,
                _field_tokens(case, names),
                decision["views"],
            )
            direct_prob[q] = torch.softmax(logits.double(), dim=-1)

        hard_prob = _hard_composed(p_i, p_s, p_c)
        soft_prob = _soft_composed(p_i, p_s, p_c)
        oracle_prob = _oracle_composed(
            case,
            int(diagnosis["gold_index"]),
            len(diagnosis["views"]["D0"]["option_ids"]),
        )
        path_probs = {
            "FIELD_DIRECT": direct_prob,
            "LATENT_COMPOSED_HARD": hard_prob,
            "LATENT_COMPOSED_SOFT": soft_prob,
            "ORACLE_LATENT_COMPOSED": oracle_prob,
        }

        severity_correct = int(p_s.argmax()) == int(case["severity"])
        confidence_correct = int(p_c.argmax()) == int(case["confidence_index"])

        for q in transitions:
            gold = int(decisions[q]["gold_index"])
            dcorrect = int(direct_prob[q].argmax()) == gold
            hcorrect = int(hard_prob[q].argmax()) == gold
            transitions[q]["n"] += 1
            transitions[q]["direct_wrong_hard_right"] += int((not dcorrect) and hcorrect)
            transitions[q]["direct_right_hard_wrong"] += int(dcorrect and (not hcorrect))

        if severity_correct:
            for q in ("response", "risk"):
                conditioned["severity_correct"][f"{q}_n"] += 1
                conditioned["severity_correct"][f"{q}_correct"] += int(
                    int(hard_prob[q].argmax()) == int(decisions[q]["gold_index"])
                )
        if severity_correct and confidence_correct:
            for q in ("needs_review", "urgency"):
                conditioned["severity_confidence_correct"][f"{q}_n"] += 1
                conditioned["severity_confidence_correct"][f"{q}_correct"] += int(
                    int(hard_prob[q].argmax()) == int(decisions[q]["gold_index"])
                )

        for q in decisions:
            hard_soft_total += 1
            hard_soft_same += int(
                int(hard_prob[q].argmax()) == int(soft_prob[q].argmax())
            )

        for path, probs_by_q in path_probs.items():
            slot = accum[path]
            for q, decision in decisions.items():
                p = probs_by_q[q].double()
                mass_error = abs(float(p.sum()) - 1.0)
                slot["max_mass_error"] = max(float(slot["max_mass_error"]), mass_error)
                gold = int(decision["gold_index"])
                pred = int(p.argmax())
                correct = pred == gold
                slot["decision_count"] = int(slot["decision_count"]) + 1
                slot["correct"] = int(slot["correct"]) + int(correct)

                primitive = str(decision["primitive"])
                slot["primitive_n"][primitive] += 1
                slot["primitive_correct"][primitive] += int(correct)
                slot["question_n"][q] += 1
                slot["question_correct"][q] += int(correct)
                if q != "diagnosis":
                    slot["non_diag_n"] = int(slot["non_diag_n"]) + 1
                    slot["non_diag_correct"] = int(slot["non_diag_correct"]) + int(correct)

                hard_target = torch.zeros_like(p)
                hard_target[gold] = 1.0
                teacher = decision["gold_probabilities"].double()
                slot["hard_brier"] = float(slot["hard_brier"]) + float(
                    ((p - hard_target) ** 2).sum()
                )
                slot["soft_brier"] = float(slot["soft_brier"]) + float(
                    ((p - teacher) ** 2).sum()
                )

                if primitive == "score":
                    support = decision["score_support"].double()
                    expected = float((p * support).sum())
                    slot["score_error"].append(
                        abs(expected - float(decision["gold_score"]))
                    )

                if q == "diagnosis":
                    logits = (
                        intent_logits
                        if path in ("FIELD_DIRECT", "LATENT_COMPOSED_SOFT")
                        else _safe_log_probs(p)
                    )
                    if path == "LATENT_COMPOSED_HARD":
                        logits = _safe_log_probs(p)
                    if path == "ORACLE_LATENT_COMPOSED":
                        logits = _safe_log_probs(p)
                    rank, mrr, top5 = _rank_metrics(logits, gold)
                    d = slot["diagnosis"][int(case["diagnosis_k"])]
                    d["n"] += 1
                    d["correct"] += int(rank == 1)
                    d["top5"] += int(top5)
                    d["mrr"] += float(mrr)
                    d["margin"] += _margin(logits, gold)

    path_result = {
        path: _finalize_path(slot, len(cases))
        for path, slot in accum.items()
    }
    transition_result = {}
    for q, row in transitions.items():
        n = int(row["n"])
        transition_result[q] = {
            **row,
            "direct_wrong_hard_right_rate": int(row["direct_wrong_hard_right"]) / max(1, n),
            "direct_right_hard_wrong_rate": int(row["direct_right_hard_wrong"]) / max(1, n),
        }

    cond_result = {
        "severity_correct": {
            "response_n": conditioned["severity_correct"]["response_n"],
            "response_accuracy": conditioned["severity_correct"]["response_correct"]
            / max(1, conditioned["severity_correct"]["response_n"]),
            "risk_n": conditioned["severity_correct"]["risk_n"],
            "risk_accuracy": conditioned["severity_correct"]["risk_correct"]
            / max(1, conditioned["severity_correct"]["risk_n"]),
        },
        "severity_confidence_correct": {
            "needs_review_n": conditioned["severity_confidence_correct"]["needs_review_n"],
            "needs_review_accuracy": conditioned["severity_confidence_correct"]["needs_review_correct"]
            / max(1, conditioned["severity_confidence_correct"]["needs_review_n"]),
            "urgency_n": conditioned["severity_confidence_correct"]["urgency_n"],
            "urgency_accuracy": conditioned["severity_confidence_correct"]["urgency_correct"]
            / max(1, conditioned["severity_confidence_correct"]["urgency_n"]),
        },
    }

    return {
        "atomic": _summarize_atomic(intent_rows, severity_rows, confidence_rows),
        "paths": path_result,
        "composition_transitions": transition_result,
        "conditioned_hard_accuracy": cond_result,
        "hard_soft_prediction_identity_rate": hard_soft_same / max(1, hard_soft_total),
        "representation_accounting": {
            "logical_state_compiles_per_case": 1,
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 3,
            "isolated_fields": ["intent", "severity", "confidence"],
        },
    }


@torch.inference_mode()
def evaluate_w18(
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w18_cache(cache)
    scorer.eval()
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for case in cache["cases"]:
        by_domain[str(case["domain_id"])].append(case)
    per_domain = {
        domain: _evaluate_cases(scorer, rows)
        for domain, rows in sorted(by_domain.items())
    }
    pooled = _evaluate_cases(scorer, list(cache["cases"]))
    return {
        "per_domain": per_domain,
        "pooled": pooled,
        "metadata": cache["metadata"],
    }


def _reference_one(logits: Tensor, gold: int) -> dict[str, float | bool]:
    rank, mrr, _ = _rank_metrics(logits, gold)
    return {
        "correct": rank == 1,
        "mrr": float(mrr),
        "margin": _margin(logits, gold),
    }


def reference_summary_from_scores(
    cases: Iterable[dict],
    score_lookup: Mapping[str, Mapping[str, list[float]]],
) -> dict[str, object]:
    rows = list(cases)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for case in rows:
        by_domain[str(case["domain_id"])].append(case)

    def summarize(domain_cases: list[dict]) -> dict[str, object]:
        diag = _empty_diag()
        severity = []
        confidence = []
        for case in domain_cases:
            key = str(case["case_id"])
            lookup = score_lookup.get(key)
            if lookup is None:
                raise ValueError(f"missing W18 reference scores for {key}")
            decision = next(
                row for row in case["decisions"] if row["question_id"] == "diagnosis"
            )
            ilogits = torch.tensor(lookup["intent"], dtype=torch.float32)
            gold = int(decision["gold_index"])
            k = int(case["diagnosis_k"])
            rank, mrr, top5 = _rank_metrics(ilogits, gold)
            d = diag[k]
            d["n"] += 1
            d["correct"] += int(rank == 1)
            d["top5"] += int(top5)
            d["mrr"] += float(mrr)
            d["margin"] += _margin(ilogits, gold)
            severity.append(
                _reference_one(
                    torch.tensor(lookup["severity"], dtype=torch.float32),
                    int(case["severity"]),
                )
            )
            confidence.append(
                _reference_one(
                    torch.tensor(lookup["confidence"], dtype=torch.float32),
                    int(case["confidence_index"]),
                )
            )

        def one(values):
            n = len(values)
            return {
                "n": n,
                "top1": sum(bool(x["correct"]) for x in values) / max(1, n),
                "mrr": sum(float(x["mrr"]) for x in values) / max(1, n),
                "mean_margin": sum(float(x["margin"]) for x in values) / max(1, n),
            }

        return {
            "intent_per_k": _summarize_diag(diag),
            "severity": one(severity),
            "confidence": one(confidence),
        }

    return {
        "per_domain": {
            domain: summarize(domain_cases)
            for domain, domain_cases in sorted(by_domain.items())
        },
        "pooled": summarize(rows),
    }


def _domain_gate(metrics: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    atomic = metrics["atomic"]
    direct = metrics["paths"]["FIELD_DIRECT"]
    hard = metrics["paths"]["LATENT_COMPOSED_HARD"]
    soft = metrics["paths"]["LATENT_COMPOSED_SOFT"]
    oracle = metrics["paths"]["ORACLE_LATENT_COMPOSED"]

    ref_intent = reference["intent_per_k"]
    reference_adequacy = (
        float(oracle["accuracy"]) == 1.0
        and float(ref_intent["4"]["top1"]) >= .80
        and float(ref_intent["16"]["top1"]) >= .60
        and float(reference["severity"]["top1"]) >= .90
        and float(reference["confidence"]["top1"]) >= .90
    )

    intent = atomic["intent_per_k"]
    atomic_adequacy = (
        float(intent["4"]["top1"]) >= .82
        and float(intent["16"]["top1"]) >= .50
        and float(atomic["severity"]["top1"]) >= .80
        and float(atomic["confidence"]["top1"]) >= .80
        and float(atomic["joint_severity_confidence_top1"]) >= .68
    )

    hq = hard["question_accuracy"]
    dq = direct["question_accuracy"]
    conditioned = metrics["conditioned_hard_accuracy"]

    one_field = (
        atomic_adequacy
        and float(hq["response"]) >= .80
        and float(hq["risk"]) >= .80
        and float(hq["response"]) >= float(dq["response"]) + .20
        and float(hq["risk"]) >= float(dq["risk"]) + .10
    )

    multi_field = (
        atomic_adequacy
        and float(hq["needs_review"]) >= .80
        and float(hq["urgency"]) >= .75
        and float(hq["needs_review"]) >= float(dq["needs_review"]) + .15
        and float(hq["urgency"]) >= float(dq["urgency"]) + .30
        and float(conditioned["severity_confidence_correct"]["needs_review_accuracy"]) == 1.0
        and float(conditioned["severity_confidence_correct"]["urgency_accuracy"]) == 1.0
    )

    soft_viability = (
        float(soft["accuracy"]) >= float(hard["accuracy"]) - .03
        and float(soft["non_diagnosis_accuracy"]) >= float(hard["non_diagnosis_accuracy"]) - .03
        and float(soft["probability_mass_max_error"]) <= 1e-6
    )

    integrity = (
        float(oracle["accuracy"]) == 1.0
        and float(oracle["probability_mass_max_error"]) <= 1e-6
        and float(hard["probability_mass_max_error"]) <= 1e-6
        and metrics["representation_accounting"] == {
            "logical_state_compiles_per_case": 1,
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 3,
            "isolated_fields": ["intent", "severity", "confidence"],
        }
    )

    if not reference_adequacy:
        classification = "W18_REFERENCE_INADEQUATE"
    elif not atomic_adequacy:
        classification = "LATENT_FIELD_EXTRACTION_LIMIT"
    elif one_field and multi_field:
        classification = "TYPED_COMPOSITION_LIMIT"
    elif one_field and not multi_field:
        classification = "ONE_FIELD_MAPPING_LIMIT"
    elif multi_field and not one_field:
        classification = "MULTI_FIELD_COMPOSITION_LIMIT"
    else:
        classification = "TYPED_COMPOSITION_UNRESOLVED"

    return {
        "integrity": integrity,
        "reference_adequacy": reference_adequacy,
        "atomic_adequacy": atomic_adequacy,
        "one_field_composition_rescue": one_field,
        "multi_field_composition_rescue": multi_field,
        "soft_viability": soft_viability,
        "classification": classification,
    }


def classify_w18(
    metrics: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    domains = sorted(metrics["per_domain"])
    per_domain = {
        domain: _domain_gate(
            metrics["per_domain"][domain],
            reference["per_domain"][domain],
        )
        for domain in domains
    }
    counts = Counter(row["classification"] for row in per_domain.values())
    stable = None
    excluded = {"TYPED_COMPOSITION_UNRESOLVED", "W18_REFERENCE_INADEQUATE"}
    for name, count in counts.items():
        if name not in excluded and count >= 3:
            stable = name
            break

    if stable is not None:
        outcome = "STABLE_TYPED_COMPOSITION_LOCALIZATION"
    else:
        nontrivial = [
            name for name, count in counts.items()
            if name not in excluded and count >= 2
        ]
        outcome = (
            "MIXED_TYPED_COMPOSITION_LOCALIZATION"
            if len(nontrivial) >= 2
            else "TYPED_COMPOSITION_UNRESOLVED"
        )

    return {
        "outcome": outcome,
        "stable_classification": stable,
        "classification_counts": dict(sorted(counts.items())),
        "per_domain": per_domain,
    }


__all__ = [
    "DIRECT_FIELDS",
    "PATHS",
    "classify_w18",
    "evaluate_w18",
    "reference_summary_from_scores",
]
