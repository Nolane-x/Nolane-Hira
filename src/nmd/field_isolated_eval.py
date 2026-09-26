from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

from .anchor_preserving_residual_training import _ece15, _margin, _rank_metrics
from .competitive import CompetitiveCoarseScorer
from .field_isolated_cache import validate_w17_cache
from .hira import HIRACore
from .interface_decomposition_eval import _q0_symmetric


PATHS = (
    "FULL_SINGLE",
    "FULL_TRIPLICATE",
    "FIELD_ISOLATED",
    "PRODUCTION_D0_FULL",
)
SEMANTIC_PATHS = PATHS[:3]
PRIMITIVE_FIELDS = {
    "diagnosis": ("intent",),
    "response": ("severity",),
    "needs_review": ("severity", "confidence"),
    "risk": ("severity",),
    "urgency": ("severity", "confidence"),
}


def _semantic_state_tokens(case: dict, question_id: str, path: str) -> Tensor:
    reps = case["representations"]
    if path == "FULL_SINGLE":
        return reps["FULL_SINGLE"]["tokens"].float()
    if path == "FULL_TRIPLICATE":
        return reps["FULL_TRIPLICATE"]["tokens"].float()
    if path != "FIELD_ISOLATED":
        raise ValueError(f"unknown W17 semantic path: {path}")
    names = PRIMITIVE_FIELDS[str(question_id)]
    rows = [reps["FIELD_ISOLATED"][f"{name}_tokens"].float() for name in names]
    return torch.cat(rows, dim=0)


@torch.inference_mode()
def _semantic_logits(
    scorer: CompetitiveCoarseScorer,
    case: dict,
    decision: dict,
    path: str,
) -> Tensor:
    projection = scorer.projection.weight.detach().float()
    state_tokens = _semantic_state_tokens(case, str(decision["question_id"]), path)
    view_logits = []
    for view_id in ("D0", "D1", "D2"):
        view = decision["views"][view_id]
        raw = _q0_symmetric(
            state_tokens,
            view["option_tokens"].float(),
            view["option_content_mask"].bool(),
            projection,
        )
        view_logits.append(raw)
    return torch.stack(view_logits, dim=0).mean(dim=0) * scorer.scale().detach().float()


@torch.inference_mode()
def _production_logits(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    case: dict,
    decision: dict,
) -> tuple[Tensor, Tensor]:
    state = case["representations"]["FULL_SINGLE"]["tokens"].float().unsqueeze(0)
    d0 = decision["views"]["D0"]
    coarse = scorer(
        state_tokens=state,
        state_mask=torch.ones(1, state.shape[1], dtype=torch.bool),
        question_tokens=d0["question_tokens"].float().unsqueeze(0),
        question_mask=d0["question_content_mask"].bool().unsqueeze(0),
        option_tokens=d0["option_tokens"].float().unsqueeze(0),
        option_token_ids=d0["option_token_ids"].long().unsqueeze(0),
        option_mask=d0["option_content_mask"].bool().unsqueeze(0),
    )[0]
    qtype = torch.tensor([int(decision["qtype"])], dtype=torch.long)
    segments = case["representations"]["FULL_SINGLE"]["segments"].float().unsqueeze(0)
    question = d0["question_embedding"].float().unsqueeze(0)
    options = d0["option_embeddings"].float().unsqueeze(0)
    out = hira(
        question,
        segments,
        options,
        qtype,
        coarse_override=coarse.unsqueeze(0),
        forced_budget=options.shape[1],
        adaptive_budget=False,
    )
    return out.logits[0].detach().float(), out.probabilities[0].detach().double()


def _empty_diag() -> dict[int, dict[str, float]]:
    return {
        4: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0, "margin": 0.0},
        8: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0, "margin": 0.0},
        16: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0, "margin": 0.0},
    }


def _summarize_diag(table: dict[int, dict[str, float]]) -> dict[str, dict[str, float | int]]:
    result = {}
    for k, slot in table.items():
        n = int(slot["n"])
        result[str(k)] = {
            "n": n,
            "top1": float(slot["correct"]) / max(1, n),
            "top5": float(slot["top5"]) / max(1, n),
            "mrr": float(slot["mrr"]) / max(1, n),
            "mean_margin": float(slot["margin"]) / max(1, n),
        }
    return result


@torch.inference_mode()
def evaluate_w17(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w17_cache(cache)
    hira.eval()
    scorer.eval()

    accum: dict[str, dict[str, object]] = {}
    for path in PATHS:
        accum[path] = {
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
            "confidence": [],
            "hard_target": [],
            "soft_target": [],
            "score_error": [],
            "diagnosis": _empty_diag(),
            "max_mass_error": 0.0,
            "active_tokens": defaultdict(list),
        }

    triplicate_total = 0
    triplicate_top1_equal = 0
    triplicate_max_logit_diff = 0.0

    for case in cache["cases"]:
        per_decision_semantic: dict[str, dict[str, Tensor]] = {}
        for decision in case["decisions"]:
            question_id = str(decision["question_id"])
            per_decision_semantic[question_id] = {
                path: _semantic_logits(scorer, case, decision, path)
                for path in SEMANTIC_PATHS
            }
            a = per_decision_semantic[question_id]["FULL_SINGLE"]
            b = per_decision_semantic[question_id]["FULL_TRIPLICATE"]
            triplicate_total += 1
            triplicate_top1_equal += int(int(a.argmax()) == int(b.argmax()))
            triplicate_max_logit_diff = max(
                triplicate_max_logit_diff,
                float((a - b).abs().max()),
            )

            production_logits, production_prob = _production_logits(
                hira, scorer, case, decision
            )
            all_logits = {
                **per_decision_semantic[question_id],
                "PRODUCTION_D0_FULL": production_logits,
            }

            for path, logits in all_logits.items():
                slot = accum[path]
                if path == "PRODUCTION_D0_FULL":
                    p = production_prob
                else:
                    p = torch.softmax(logits.double(), dim=-1)
                gold = int(decision["gold_index"])
                pred = int(p.argmax())
                correct = pred == gold

                slot["decision_count"] = int(slot["decision_count"]) + 1
                slot["correct"] = int(slot["correct"]) + int(correct)
                primitive = str(decision["primitive"])
                slot["primitive_n"][primitive] += 1
                slot["primitive_correct"][primitive] += int(correct)
                slot["question_n"][question_id] += 1
                slot["question_correct"][question_id] += int(correct)
                if question_id != "diagnosis":
                    slot["non_diag_n"] = int(slot["non_diag_n"]) + 1
                    slot["non_diag_correct"] = int(slot["non_diag_correct"]) + int(correct)

                hard = torch.zeros_like(p)
                hard[gold] = 1.0
                teacher = decision["gold_probabilities"].double()
                slot["hard_brier"] = float(slot["hard_brier"]) + float(((p - hard) ** 2).sum())
                slot["soft_brier"] = float(slot["soft_brier"]) + float(((p - teacher) ** 2).sum())
                slot["confidence"].append(float(p[pred]))
                slot["hard_target"].append(float(correct))
                slot["soft_target"].append(float(teacher[pred]))
                slot["max_mass_error"] = max(
                    float(slot["max_mass_error"]),
                    abs(float(p.sum()) - 1.0),
                )

                if primitive == "score":
                    support = decision["score_support"].double()
                    expected = float((p * support).sum())
                    slot["score_error"].append(
                        abs(expected - float(decision["gold_score"]))
                    )

                if question_id == "diagnosis":
                    k = int(case["diagnosis_k"])
                    rank, mrr, top5 = _rank_metrics(logits, gold)
                    d = slot["diagnosis"][k]
                    d["n"] += 1
                    d["correct"] += int(rank == 1)
                    d["top5"] += int(top5)
                    d["mrr"] += float(mrr)
                    d["margin"] += _margin(logits, gold)

                if path in SEMANTIC_PATHS:
                    slot["active_tokens"][question_id].append(
                        int(_semantic_state_tokens(case, question_id, path).shape[0])
                    )

    result: dict[str, object] = {}
    for path in PATHS:
        slot = accum[path]
        n = int(slot["decision_count"])
        primitive_accuracy = {
            name: slot["primitive_correct"][name] / max(1, slot["primitive_n"][name])
            for name in ("choice", "noul", "score")
        }
        question_accuracy = {
            name: slot["question_correct"][name] / max(1, slot["question_n"][name])
            for name in ("diagnosis", "response", "needs_review", "risk", "urgency")
        }
        active = {}
        if path in SEMANTIC_PATHS:
            active = {
                name: sum(values) / max(1, len(values))
                for name, values in slot["active_tokens"].items()
            }
        result[path] = {
            "case_count": len(cache["cases"]),
            "decision_count": n,
            "accuracy": int(slot["correct"]) / max(1, n),
            "primitive_accuracy": primitive_accuracy,
            "question_accuracy": question_accuracy,
            "non_diagnosis_accuracy": int(slot["non_diag_correct"]) / max(1, int(slot["non_diag_n"])),
            "hard_brier": float(slot["hard_brier"]) / max(1, n),
            "soft_brier": float(slot["soft_brier"]) / max(1, n),
            "raw_ece": _ece15(slot["confidence"], slot["hard_target"]),
            "soft_ece": _ece15(slot["confidence"], slot["soft_target"]),
            "score_mae": sum(slot["score_error"]) / max(1, len(slot["score_error"])),
            "probability_mass_max_error": float(slot["max_mass_error"]),
            "diagnosis_per_k": _summarize_diag(slot["diagnosis"]),
            "mean_active_state_tokens_per_question": active,
        }

    return {
        "paths": result,
        "control": {
            "triplicate_prediction_identity_rate": (
                triplicate_top1_equal / max(1, triplicate_total)
            ),
            "triplicate_max_semantic_logit_diff": triplicate_max_logit_diff,
            "triplicate_decision_count": triplicate_total,
        },
        "representation_accounting": {
            **cache["metadata"],
            "primitive_fields": {
                key: list(value) for key, value in PRIMITIVE_FIELDS.items()
            },
        },
    }


def _domain_gate(
    metrics: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    full = metrics["paths"]["FULL_SINGLE"]
    iso = metrics["paths"]["FIELD_ISOLATED"]
    control = metrics["control"]
    accounting = metrics["representation_accounting"]
    ik = iso["diagnosis_per_k"]
    fk = full["diagnosis_per_k"]

    integrity = (
        accounting["logical_state_compiles_per_candidate_case"] == {
            "FULL_SINGLE": 1,
            "FULL_TRIPLICATE": 1,
            "FIELD_ISOLATED": 1,
        }
        and accounting["a13_invocations_per_candidate_case"] == {
            "FULL_SINGLE": 1,
            "FULL_TRIPLICATE": 1,
            "FIELD_ISOLATED": 1,
        }
        and accounting["encoded_sequences_per_candidate_case"] == {
            "FULL_SINGLE": 1,
            "FULL_TRIPLICATE": 3,
            "FIELD_ISOLATED": 3,
        }
        and accounting["primitive_fields"] == {
            "diagnosis": ["intent"],
            "response": ["severity"],
            "needs_review": ["severity", "confidence"],
            "risk": ["severity"],
            "urgency": ["severity", "confidence"],
        }
        and float(control["triplicate_prediction_identity_rate"]) == 1.0
        and float(control["triplicate_max_semantic_logit_diff"]) <= 1e-5
        and max(
            float(metrics["paths"][name]["probability_mass_max_error"])
            for name in PATHS
        ) <= 1e-6
    )

    pa = iso["primitive_accuracy"]
    qa = iso["question_accuracy"]
    competence = (
        float(iso["accuracy"]) >= .82
        and float(pa["choice"]) >= .82
        and float(pa["noul"]) >= .80
        and float(pa["score"]) >= .80
        and float(qa["response"]) >= .80
        and float(qa["needs_review"]) >= .80
        and float(qa["risk"]) >= .80
        and float(qa["urgency"]) >= .75
        and float(ik["4"]["top1"]) >= .85
        and float(ik["8"]["top1"]) >= .78
        and float(ik["16"]["top1"]) >= .65
    )
    gain = (
        float(iso["accuracy"]) >= float(full["accuracy"]) + .18
        and float(iso["non_diagnosis_accuracy"]) >= float(full["non_diagnosis_accuracy"]) + .12
        and float(ik["4"]["top1"]) >= float(fk["4"]["top1"]) + .20
        and float(ik["8"]["top1"]) >= float(fk["8"]["top1"]) + .25
        and float(ik["16"]["top1"]) >= float(fk["16"]["top1"]) + .25
    )
    ref = reference["diagnosis_per_k"]
    reference_adequacy = (
        float(ref["4"]["top1"]) >= .80
        and float(ref["16"]["top1"]) >= .70
    )
    partial = (
        integrity
        and float(iso["accuracy"]) >= float(full["accuracy"]) + .12
        and float(iso["non_diagnosis_accuracy"]) >= float(full["non_diagnosis_accuracy"]) + .08
        and float(ik["16"]["top1"]) >= float(fk["16"]["top1"]) + .20
        and float(ik["16"]["top1"]) >= .55
    )
    return {
        "integrity": integrity,
        "competence": competence,
        "causal_gain": gain,
        "reference_adequacy": reference_adequacy,
        "partial": partial,
    }


def w17_verdict(
    cp_metrics: dict[str, object],
    cq_metrics: dict[str, object],
    cp_reference: dict[str, object],
    cq_reference: dict[str, object],
) -> tuple[str, dict[str, object]]:
    cp = _domain_gate(cp_metrics, cp_reference)
    cq = _domain_gate(cq_metrics, cq_reference)
    details = {"CP": cp, "CQ": cq}
    if not cp["integrity"] or not cq["integrity"]:
        return "FIELD_ISOLATION_CONTROL_INVALID", details
    if all(
        gate["reference_adequacy"] and gate["competence"] and gate["causal_gain"]
        for gate in (cp, cq)
    ):
        return "FIELD_ISOLATED_TYPED_RESCUE", details
    if cp["partial"] and cq["partial"]:
        return "FIELD_ISOLATED_TYPED_PARTIAL", details
    return "FIELD_ISOLATED_TYPED_FAIL", details


def reference_summary_from_scores(
    cases: Iterable[dict],
    score_lookup: dict[str, list[float]],
) -> dict[str, object]:
    diag = _empty_diag()
    for case in cases:
        decision = next(
            row for row in case["decisions"] if row["question_id"] == "diagnosis"
        )
        key = str(case["case_id"])
        if key not in score_lookup:
            raise ValueError(f"missing W17 reference scores for {key}")
        logits = torch.tensor(score_lookup[key], dtype=torch.float32)
        gold = int(decision["gold_index"])
        k = int(case["diagnosis_k"])
        rank, mrr, top5 = _rank_metrics(logits, gold)
        diag[k]["n"] += 1
        diag[k]["correct"] += int(rank == 1)
        diag[k]["top5"] += int(top5)
        diag[k]["mrr"] += float(mrr)
        diag[k]["margin"] += _margin(logits, gold)
    return {"diagnosis_per_k": _summarize_diag(diag)}


__all__ = [
    "PATHS",
    "PRIMITIVE_FIELDS",
    "evaluate_w17",
    "reference_summary_from_scores",
    "w17_verdict",
]
