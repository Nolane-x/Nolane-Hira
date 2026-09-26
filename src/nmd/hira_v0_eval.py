from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Mapping, Sequence

from .hira_v0_authority import DOMAINS, FACTOR_IDS, PRIMITIVES, compose_severity

PRIMITIVE_PAIRS = tuple(combinations(PRIMITIVES, 2))


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / max(1, len(values))


def summarize_hira_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_domain: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain_id"])].append(row)

    def summarize(domain_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        factor_primitive = {}
        agreement = {}
        for factor in FACTOR_IDS:
            factor_primitive[factor] = {}
            for primitive in PRIMITIVES:
                factor_primitive[factor][primitive] = sum(
                    int(row["predictions"][factor][primitive])
                    == int(row["golds"][factor])
                    for row in domain_rows
                ) / len(domain_rows)
            agreement[factor] = {}
            for a, b in PRIMITIVE_PAIRS:
                agreement[factor][f"{a}_vs_{b}"] = sum(
                    int(row["predictions"][factor][a])
                    == int(row["predictions"][factor][b])
                    for row in domain_rows
                ) / len(domain_rows)

        return {
            "case_count": len(domain_rows),
            "factor_primitive_top1": factor_primitive,
            "cross_primitive_agreement": agreement,
            "factor_vector_top1": sum(
                bool(row["factor_vector_correct"]) for row in domain_rows
            ) / len(domain_rows),
            "composed_severity_top1": sum(
                bool(row["severity_correct"]) for row in domain_rows
            ) / len(domain_rows),
            "invalid_factor_vector_rate": sum(
                bool(row["invalid_factor_vector"]) for row in domain_rows
            ) / len(domain_rows),
            "option_order_invariance": sum(
                bool(row["option_order_invariant"]) for row in domain_rows
            ) / len(domain_rows),
            "state_once_rate": sum(
                int(row["state_encode_delta"]) == 1 for row in domain_rows
            ) / len(domain_rows),
            "relation_delta_max_abs": max(
                float(row["relation_delta_max_abs"]) for row in domain_rows
            ),
            "probability_mass_max_error": max(
                float(row["probability_mass_max_error"]) for row in domain_rows
            ),
            "full_k_rate": sum(
                bool(row["full_k"]) for row in domain_rows
            ) / len(domain_rows),
        }

    per_domain = {
        domain: summarize(by_domain[domain])
        for domain in DOMAINS
    }
    return {
        "per_domain": per_domain,
        "pooled": summarize(rows),
    }


def reference_domain_pass_w29(panel: Mapping[str, object], domain: str) -> bool:
    consensus = panel["consensus"]["per_domain"][domain]
    tasks = consensus["tasks"]
    individual = panel["individual"]

    panel_pass = bool(
        float(tasks["F0"]["balanced_accuracy"]) >= 0.90
        and float(tasks["F1"]["balanced_accuracy"]) >= 0.90
        and float(tasks["U"]["balanced_accuracy"]) >= 0.92
        and float(tasks["C"]["balanced_accuracy"]) >= 0.92
        and float(consensus["composed_f2"]["balanced_accuracy"]) >= 0.92
        and float(consensus["composed_f2"]["positive_recall"]) >= 0.88
        and float(consensus["composed_f2"]["negative_recall"]) >= 0.94
        and float(consensus["composed_f2"]["probability_mass_max_error"]) <= 1e-6
    )
    model_floor = all(
        float(individual[name]["per_domain"][domain]["tasks"]["F0"]["balanced_accuracy"]) >= 0.80
        and float(individual[name]["per_domain"][domain]["tasks"]["F1"]["balanced_accuracy"]) >= 0.80
        and float(individual[name]["per_domain"][domain]["tasks"]["U"]["balanced_accuracy"]) >= 0.82
        and float(individual[name]["per_domain"][domain]["tasks"]["C"]["balanced_accuracy"]) >= 0.82
        and float(individual[name]["per_domain"][domain]["composed_f2"]["balanced_accuracy"]) >= 0.82
        for name in ("deberta_nli", "roberta_nli")
    )
    return panel_pass and model_floor


def hira_domain_pass(row: Mapping[str, object]) -> bool:
    for factor in FACTOR_IDS:
        for primitive in PRIMITIVES:
            if float(row["factor_primitive_top1"][factor][primitive]) < 0.90:
                return False
        for agreement in row["cross_primitive_agreement"][factor].values():
            if float(agreement) < 0.98:
                return False
    return bool(
        float(row["factor_vector_top1"]) >= 0.85
        and float(row["composed_severity_top1"]) >= 0.85
        and float(row["invalid_factor_vector_rate"]) <= 0.05
        and float(row["probability_mass_max_error"]) <= 1e-6
    )


def runtime_domain_pass(row: Mapping[str, object]) -> bool:
    return bool(
        float(row["state_once_rate"]) == 1.0
        and float(row["option_order_invariance"]) == 1.0
        and float(row["relation_delta_max_abs"]) == 0.0
        and float(row["full_k_rate"]) == 1.0
    )


def classify_w29(
    *,
    panel: Mapping[str, object],
    hira_summary: Mapping[str, object],
    runtime_integrity: Mapping[str, object],
) -> dict[str, object]:
    reference = {
        domain: reference_domain_pass_w29(panel, domain)
        for domain in DOMAINS
    }
    hira = {
        domain: hira_domain_pass(hira_summary["per_domain"][domain])
        for domain in DOMAINS
    }
    runtime = {
        domain: runtime_domain_pass(hira_summary["per_domain"][domain])
        for domain in DOMAINS
    }

    global_runtime = bool(
        runtime_integrity.get("t0_checkpoint_exact", False)
        and runtime_integrity.get("projection_trainable_parameter_count") == 0
        and runtime_integrity.get("schema_cache_hit_after_first_compile", False)
        and runtime_integrity.get("relation_refinement_disabled", False)
        and runtime_integrity.get("candidate_truncation_used") is False
        and runtime_integrity.get("reference_outputs_used_as_model_inputs") is False
    )

    if not all(reference.values()):
        outcome = "W29_REFERENCE_INADEQUATE"
    elif not (all(hira.values()) and all(runtime.values()) and global_runtime):
        outcome = "W29_RUNTIME_INTEGRATION_FAIL"
    else:
        outcome = "HIRA_V0_SEMANTIC_CORE_READY"

    return {
        "outcome": outcome,
        "reference_per_domain": reference,
        "hira_per_domain": hira,
        "runtime_per_domain": runtime,
        "global_runtime_integrity": global_runtime,
    }


__all__ = [
    "PRIMITIVE_PAIRS",
    "classify_w29",
    "hira_domain_pass",
    "reference_domain_pass_w29",
    "runtime_domain_pass",
    "summarize_hira_rows",
]
