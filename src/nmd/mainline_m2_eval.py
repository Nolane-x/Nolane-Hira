from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Mapping, Sequence

import torch

from .contracts import LogicalOption
from .mainline import HiraV0Mainline
from .mainline_m2_authority import M2HighKSemanticCase

M2_SEMANTIC_MASS_TOL = 1e-6
M2_SEMANTIC_PERM_TOL = 2e-6

M2_DEV_GATES = {
    64: {
        "top1": 0.70,
        "top5": 0.90,
        "mrr": 0.75,
    },
    128: {
        "top1": 0.60,
        "top5": 0.85,
        "mrr": 0.65,
    },
}

M2_SEALED_GATES = {
    255: {
        "top1": 0.50,
        "top5": 0.80,
        "mrr": 0.58,
    },
}


@dataclass(frozen=True)
class M2SemanticMetrics:
    k: int
    case_count: int
    top1: float
    top5: float
    mrr: float
    mean_gold_rank: float
    mean_gold_probability: float
    mean_confidence: float
    probability_mass_max_error: float
    permutation_max_error: float
    selected_option_invariant_rate: float
    full_k_rate: float
    state_once_rate: float
    relation_delta_max: float
    finite_rate: float
    mean_case_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "k": self.k,
            "case_count": self.case_count,
            "top1": self.top1,
            "top5": self.top5,
            "mrr": self.mrr,
            "mean_gold_rank": self.mean_gold_rank,
            "mean_gold_probability": self.mean_gold_probability,
            "mean_confidence": self.mean_confidence,
            "probability_mass_max_error": self.probability_mass_max_error,
            "permutation_max_error": self.permutation_max_error,
            "selected_option_invariant_rate": self.selected_option_invariant_rate,
            "full_k_rate": self.full_k_rate,
            "state_once_rate": self.state_once_rate,
            "relation_delta_max": self.relation_delta_max,
            "finite_rate": self.finite_rate,
            "mean_case_ms": self.mean_case_ms,
        }


def _rotate(
    options: tuple[LogicalOption, ...],
) -> tuple[LogicalOption, ...]:
    shift = max(1, len(options) // 3)
    return options[shift:] + options[:shift]


def _probability_by_id(
    probabilities: torch.Tensor,
    options: tuple[LogicalOption, ...],
) -> dict[str, float]:
    return {
        option.option_id: float(probabilities[index])
        for index, option in enumerate(options)
    }


@torch.inference_mode()
def evaluate_m2_semantic_cases(
    model: HiraV0Mainline,
    rows: Sequence[M2HighKSemanticCase],
) -> dict[str, object]:
    if not rows:
        raise ValueError("M2 semantic evaluation requires cases")

    grouped: dict[int, list[M2HighKSemanticCase]] = {}
    for row in rows:
        grouped.setdefault(row.k, []).append(row)

    before_total_state = model.runtime.state_encode_calls
    per_k: dict[str, dict[str, float | int]] = {}

    for k in sorted(grouped):
        cases = grouped[k]
        correct = 0
        top5 = 0
        reciprocal_ranks: list[float] = []
        gold_ranks: list[float] = []
        gold_probabilities: list[float] = []
        confidences: list[float] = []
        mass_errors: list[float] = []
        permutation_errors: list[float] = []
        invariant = 0
        full_k = 0
        state_once = 0
        relation_delta_max = 0.0
        finite_count = 0
        elapsed_ms: list[float] = []

        for row in cases:
            if row.k != k or len(row.options) != k:
                raise ValueError("M2 semantic case K mismatch")

            before_case_state = model.runtime.state_encode_calls
            start = time.perf_counter()
            session = model.open_session(row.state_text)

            canonical = session.decide(
                primitive="choice",
                question_text=row.question_text,
                options=row.options,
                use_schema_cache=False,
            )
            permuted_options = _rotate(row.options)
            permuted = session.decide(
                primitive="choice",
                question_text=row.question_text,
                options=permuted_options,
                use_schema_cache=False,
            )
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

            state_delta = model.runtime.state_encode_calls - before_case_state
            state_once += int(state_delta == 1 and session.query_count == 2)

            p = canonical.probabilities.detach().cpu()
            order = torch.argsort(p, descending=True)
            predicted = int(order[0])
            correct += int(predicted == row.gold_index)
            top5 += int(row.gold_index in set(order[: min(5, k)].tolist()))
            rank = int((order == row.gold_index).nonzero(as_tuple=False)[0]) + 1
            reciprocal_ranks.append(1.0 / rank)
            gold_ranks.append(float(rank))
            gold_probabilities.append(float(p[row.gold_index]))
            confidences.append(float(p.max()))
            mass_errors.append(abs(float(p.sum()) - 1.0))

            finite = bool(
                torch.isfinite(canonical.logits).all()
                and torch.isfinite(canonical.probabilities).all()
                and torch.isfinite(permuted.logits).all()
                and torch.isfinite(permuted.probabilities).all()
            )
            finite_count += int(finite)

            canonical_full = bool(
                int(canonical.hira.candidate_budget.item()) == k
                and bool(canonical.hira.selected_mask.all())
                and canonical.probabilities.numel() == k
            )
            permuted_full = bool(
                int(permuted.hira.candidate_budget.item()) == k
                and bool(permuted.hira.selected_mask.all())
                and permuted.probabilities.numel() == k
            )
            full_k += int(canonical_full and permuted_full)

            relation_delta_max = max(
                relation_delta_max,
                float(canonical.hira.relation_delta.abs().max()),
                float(permuted.hira.relation_delta.abs().max()),
            )

            canonical_by_id = _probability_by_id(
                canonical.probabilities.detach().cpu(),
                row.options,
            )
            permuted_by_id = _probability_by_id(
                permuted.probabilities.detach().cpu(),
                permuted_options,
            )
            permutation_errors.append(
                max(
                    abs(canonical_by_id[option_id] - permuted_by_id[option_id])
                    for option_id in canonical_by_id
                )
            )
            invariant += int(
                canonical.selected_option_id == permuted.selected_option_id
            )

        count = len(cases)
        metrics = M2SemanticMetrics(
            k=k,
            case_count=count,
            top1=correct / count,
            top5=top5 / count,
            mrr=statistics.fmean(reciprocal_ranks),
            mean_gold_rank=statistics.fmean(gold_ranks),
            mean_gold_probability=statistics.fmean(gold_probabilities),
            mean_confidence=statistics.fmean(confidences),
            probability_mass_max_error=max(mass_errors),
            permutation_max_error=max(permutation_errors),
            selected_option_invariant_rate=invariant / count,
            full_k_rate=full_k / count,
            state_once_rate=state_once / count,
            relation_delta_max=relation_delta_max,
            finite_rate=finite_count / count,
            mean_case_ms=statistics.fmean(elapsed_ms),
        )
        per_k[str(k)] = metrics.to_dict()

    state_encode_count = model.runtime.state_encode_calls - before_total_state
    return {
        "case_count": len(rows),
        "state_encode_count": state_encode_count,
        "state_encodes_per_case": state_encode_count / len(rows),
        "per_k": per_k,
        "candidate_pruning_used": False,
        "relation_refinement_used": False,
        "adaptive_budget_used": False,
    }


def _mechanics_gate(metrics: Mapping[str, object]) -> dict[str, bool]:
    return {
        "probability_mass": (
            float(metrics["probability_mass_max_error"])
            <= M2_SEMANTIC_MASS_TOL
        ),
        "permutation": (
            float(metrics["permutation_max_error"])
            <= M2_SEMANTIC_PERM_TOL
            and float(metrics["selected_option_invariant_rate"]) == 1.0
        ),
        "full_k": float(metrics["full_k_rate"]) == 1.0,
        "state_once": float(metrics["state_once_rate"]) == 1.0,
        "relation_off": float(metrics["relation_delta_max"]) == 0.0,
        "finite": float(metrics["finite_rate"]) == 1.0,
    }


def m2_dev_qualification(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    per_k = evaluation["per_k"]
    result: dict[str, object] = {}
    passed = True

    for k, thresholds in M2_DEV_GATES.items():
        metrics = per_k[str(k)]
        mechanics = _mechanics_gate(metrics)
        semantic = {
            "top1": float(metrics["top1"]) >= thresholds["top1"],
            "top5": float(metrics["top5"]) >= thresholds["top5"],
            "mrr": float(metrics["mrr"]) >= thresholds["mrr"],
        }
        k_pass = all(mechanics.values()) and all(semantic.values())
        result[str(k)] = {
            "pass": k_pass,
            "mechanics": mechanics,
            "semantic": semantic,
            "thresholds": thresholds,
        }
        passed = passed and k_pass

    return {
        "pass": passed,
        "per_k": result,
    }


def m2_sealed_qualification(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    per_k = evaluation["per_k"]
    metrics = per_k["255"]
    thresholds = M2_SEALED_GATES[255]
    mechanics = _mechanics_gate(metrics)
    semantic = {
        "top1": float(metrics["top1"]) >= thresholds["top1"],
        "top5": float(metrics["top5"]) >= thresholds["top5"],
        "mrr": float(metrics["mrr"]) >= thresholds["mrr"],
    }
    passed = all(mechanics.values()) and all(semantic.values())
    return {
        "pass": passed,
        "mechanics": mechanics,
        "semantic": semantic,
        "thresholds": thresholds,
    }


__all__ = [
    "M2SemanticMetrics",
    "M2_DEV_GATES",
    "M2_SEALED_GATES",
    "M2_SEMANTIC_MASS_TOL",
    "M2_SEMANTIC_PERM_TOL",
    "evaluate_m2_semantic_cases",
    "m2_dev_qualification",
    "m2_sealed_qualification",
]
