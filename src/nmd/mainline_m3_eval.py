from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import torch

from .mainline import HiraV0Mainline
from .mainline_m3_authority import M3PairedCase

M3_EN_TOP1_MIN = 0.65
M3_VI_TOP1_MIN = 0.65
M3_VI_EN_TOP1_RATIO_MIN = 0.90
M3_VI_EN_MRR_RATIO_MIN = 0.90
M3_PAIRED_AGREEMENT_MIN = 0.85


def _gold_rank(probabilities: torch.Tensor, gold_index: int) -> int:
    if probabilities.ndim != 1:
        raise ValueError("M3 probabilities must be rank-1")
    order = torch.argsort(probabilities, descending=True)
    match = (order == int(gold_index)).nonzero(as_tuple=False)
    if match.numel() != 1:
        raise RuntimeError("M3 gold index missing from probability vector")
    return int(match.item()) + 1


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 1.0 if numerator <= 0.0 else float("inf")
    return numerator / denominator


def _language_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("M3 language summary requires rows")

    correct = sum(bool(row["correct"]) for row in rows)
    mrr = sum(float(row["reciprocal_rank"]) for row in rows) / len(rows)
    by_primitive: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_primitive[str(row["primitive"])].append(bool(row["correct"]))

    return {
        "case_count": len(rows),
        "top1": correct / len(rows),
        "mrr": mrr,
        "per_primitive_top1": {
            primitive: sum(values) / len(values)
            for primitive, values in sorted(by_primitive.items())
        },
        "mean_gold_rank": sum(float(row["gold_rank"]) for row in rows) / len(rows),
        "mean_confidence": sum(float(row["confidence"]) for row in rows) / len(rows),
    }


@torch.inference_mode()
def evaluate_m3_paired_cases(
    model: HiraV0Mainline,
    rows: Sequence[M3PairedCase],
) -> dict[str, object]:
    if not rows:
        raise ValueError("M3 paired evaluation requires cases")
    if model.manifest.multilingual != "provisional":
        raise RuntimeError("M3 baseline requires multilingual=provisional")
    if model.manifest.high_k_mechanics != "available":
        raise RuntimeError("M3 baseline requires M2 high-K mechanics authority")
    if model.manifest.production_ready:
        raise RuntimeError("M3 baseline cannot claim production readiness")

    before_encodes = model.runtime.state_encode_calls
    language_rows: dict[str, list[dict[str, object]]] = {"en": [], "vi": []}
    pair_reports: list[dict[str, object]] = []

    probability_mass_max_error = 0.0
    relation_delta_max = 0.0
    finite = True
    full_k = True

    for row in rows:
        language_outputs = {}
        for language in ("en", "vi"):
            state_text = getattr(row, f"{language}_state_text")
            question_text = getattr(row, f"{language}_question_text")
            options = getattr(row, f"{language}_options")

            session = model.open_session(state_text)
            output = session.decide(
                primitive=row.primitive,
                question_text=question_text,
                options=options,
            )
            if session.query_count != 1:
                raise RuntimeError("M3 paired case must issue exactly one query")

            probabilities = output.probabilities.detach().float().cpu()
            selected_index = int(probabilities.argmax())
            rank = _gold_rank(probabilities, row.gold_index)
            candidate_budget = int(output.hira.candidate_budget.item())
            selected_mask_all = bool(output.hira.selected_mask.all())
            relation_delta = float(output.hira.relation_delta.abs().max())
            mass_error = abs(float(probabilities.sum()) - 1.0)
            finite_case = bool(
                torch.isfinite(output.logits).all()
                and torch.isfinite(output.probabilities).all()
            )

            probability_mass_max_error = max(
                probability_mass_max_error,
                mass_error,
            )
            relation_delta_max = max(relation_delta_max, relation_delta)
            finite = finite and finite_case
            full_k = (
                full_k
                and candidate_budget == len(options)
                and selected_mask_all
            )

            result = {
                "pair_id": row.pair_id,
                "domain_id": row.domain_id,
                "primitive": row.primitive,
                "language": language,
                "k": len(options),
                "gold_index": row.gold_index,
                "selected_index": selected_index,
                "selected_option_id": options[selected_index].option_id,
                "gold_option_id": options[row.gold_index].option_id,
                "correct": selected_index == row.gold_index,
                "gold_rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "confidence": float(probabilities.max()),
                "probability_mass_error": mass_error,
                "relation_delta_max": relation_delta,
                "finite": finite_case,
                "full_k": candidate_budget == len(options) and selected_mask_all,
            }
            language_rows[language].append(result)
            language_outputs[language] = result

        en = language_outputs["en"]
        vi = language_outputs["vi"]
        if en["gold_option_id"] != vi["gold_option_id"]:
            raise RuntimeError("M3 paired EN/VI gold option identity changed")

        pair_reports.append(
            {
                "pair_id": row.pair_id,
                "domain_id": row.domain_id,
                "primitive": row.primitive,
                "prediction_agreement": (
                    en["selected_option_id"] == vi["selected_option_id"]
                ),
                "both_correct": bool(en["correct"] and vi["correct"]),
                "either_correct": bool(en["correct"] or vi["correct"]),
                "en_selected_option_id": en["selected_option_id"],
                "vi_selected_option_id": vi["selected_option_id"],
                "gold_option_id": en["gold_option_id"],
            }
        )

    encode_delta = model.runtime.state_encode_calls - before_encodes
    expected_encodes = 2 * len(rows)
    if encode_delta != expected_encodes:
        raise RuntimeError("M3 paired evaluation violated state-once language cases")

    en_summary = _language_summary(language_rows["en"])
    vi_summary = _language_summary(language_rows["vi"])
    paired_agreement = (
        sum(bool(row["prediction_agreement"]) for row in pair_reports)
        / len(pair_reports)
    )
    both_correct = (
        sum(bool(row["both_correct"]) for row in pair_reports)
        / len(pair_reports)
    )

    report = {
        "pair_count": len(rows),
        "language_case_count": 2 * len(rows),
        "state_encode_count": encode_delta,
        "state_encodes_per_language_case": encode_delta / (2 * len(rows)),
        "en": en_summary,
        "vi": vi_summary,
        "vi_en_top1_ratio": _safe_ratio(
            float(vi_summary["top1"]),
            float(en_summary["top1"]),
        ),
        "vi_en_mrr_ratio": _safe_ratio(
            float(vi_summary["mrr"]),
            float(en_summary["mrr"]),
        ),
        "paired_prediction_agreement": paired_agreement,
        "paired_both_correct_rate": both_correct,
        "probability_mass_max_error": probability_mass_max_error,
        "relation_delta_max": relation_delta_max,
        "finite": float(finite),
        "full_k": float(full_k),
        "gradient_updates_used": False,
        "candidate_pruning_used": False,
        "relation_refinement_used": False,
        "adaptive_budget_used": False,
        "pairs": pair_reports,
    }
    return report


def m3_multilingual_qualification(
    report: Mapping[str, object],
) -> dict[str, object]:
    en = report["en"]
    vi = report["vi"]
    if not isinstance(en, Mapping) or not isinstance(vi, Mapping):
        raise ValueError("M3 report missing language summaries")

    gates = {
        "en_top1": float(en["top1"]) >= M3_EN_TOP1_MIN,
        "vi_top1": float(vi["top1"]) >= M3_VI_TOP1_MIN,
        "vi_en_top1_ratio": (
            float(report["vi_en_top1_ratio"]) >= M3_VI_EN_TOP1_RATIO_MIN
        ),
        "vi_en_mrr_ratio": (
            float(report["vi_en_mrr_ratio"]) >= M3_VI_EN_MRR_RATIO_MIN
        ),
        "paired_prediction_agreement": (
            float(report["paired_prediction_agreement"])
            >= M3_PAIRED_AGREEMENT_MIN
        ),
        "state_once": float(report["state_encodes_per_language_case"]) == 1.0,
        "full_k": float(report["full_k"]) == 1.0,
        "finite": float(report["finite"]) == 1.0,
        "probability_integrity": (
            float(report["probability_mass_max_error"]) <= 1e-6
        ),
        "relation_delta_zero": float(report["relation_delta_max"]) == 0.0,
        "no_training": report.get("gradient_updates_used") is False,
        "no_pruning": report.get("candidate_pruning_used") is False,
        "no_relation_refinement": (
            report.get("relation_refinement_used") is False
        ),
        "no_adaptive_budget": report.get("adaptive_budget_used") is False,
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "thresholds": {
            "en_top1_min": M3_EN_TOP1_MIN,
            "vi_top1_min": M3_VI_TOP1_MIN,
            "vi_en_top1_ratio_min": M3_VI_EN_TOP1_RATIO_MIN,
            "vi_en_mrr_ratio_min": M3_VI_EN_MRR_RATIO_MIN,
            "paired_prediction_agreement_min": M3_PAIRED_AGREEMENT_MIN,
        },
    }


__all__ = [
    "M3_EN_TOP1_MIN",
    "M3_PAIRED_AGREEMENT_MIN",
    "M3_VI_EN_MRR_RATIO_MIN",
    "M3_VI_EN_TOP1_RATIO_MIN",
    "M3_VI_TOP1_MIN",
    "evaluate_m3_paired_cases",
    "m3_multilingual_qualification",
]
