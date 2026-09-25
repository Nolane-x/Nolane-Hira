from __future__ import annotations

from collections import Counter

DOMAINS = ("BJ", "BK", "BL", "BM")

SEMANTIC_BASELINE_INADEQUATE = "SEMANTIC_BASELINE_INADEQUATE"
DIRECTIONALITY_LOSS = "DIRECTIONALITY_LOSS"
QUESTION_CONTEXT_LOSS = "QUESTION_CONTEXT_LOSS"
IDF_WEIGHTING_LOSS = "IDF_WEIGHTING_LOSS"
COMMON_MODE_SUBTRACTION_LOSS = "COMMON_MODE_SUBTRACTION_LOSS"
SALIENT_MIN_COVERAGE_LOSS = "SALIENT_MIN_COVERAGE_LOSS"
RELATION_RERANKING_LOSS = "RELATION_RERANKING_LOSS"
MULTI_STAGE_INTERFACE_LOSS = "MULTI_STAGE_INTERFACE_LOSS"
DISTRIBUTED_INTERFACE_LOSS = "DISTRIBUTED_INTERFACE_LOSS"
INTERFACE_DECOMPOSITION_UNRESOLVED = "INTERFACE_DECOMPOSITION_UNRESOLVED"

OUTCOME_STABLE = "STABLE_INTERFACE_LOCALIZATION"
OUTCOME_MIXED = "MIXED_INTERFACE_LOCALIZATION"
OUTCOME_UNRESOLVED = "INTERFACE_DECOMPOSITION_UNRESOLVED"

TRANSITIONS = (
    ("Q0", "Q1", DIRECTIONALITY_LOSS),
    ("Q1", "Q2", QUESTION_CONTEXT_LOSS),
    ("Q2", "Q3", IDF_WEIGHTING_LOSS),
    ("Q3", "Q4", COMMON_MODE_SUBTRACTION_LOSS),
    ("Q4", "Q5", SALIENT_MIN_COVERAGE_LOSS),
    ("Q6", "Q7", RELATION_RERANKING_LOSS),
)


def _damaging(
    metrics: dict[str, object],
    source: str,
    target: str,
) -> bool:
    stages = metrics["stages"]
    transitions = metrics["transitions"]
    source_k4 = float(stages[source]["4"]["top1"])
    target_k4 = float(stages[target]["4"]["top1"])
    source_k16 = float(stages[source]["16"]["top1"])
    target_k16 = float(stages[target]["16"]["top1"])
    trans = transitions[f"{source}->{target}"]
    return (
        target_k4 <= source_k4 - .10
        and target_k16 <= source_k16 - .08
        and float(trans["gate_correct_to_wrong_rate"]) >= .10
        and float(trans["gate_wrong_to_correct_rate"])
        <= float(trans["gate_correct_to_wrong_rate"]) - .04
    )


def classify_domain(metrics: dict[str, object]) -> dict[str, object]:
    stages = metrics["stages"]
    reference = metrics["reference"]

    r0_k4 = float(reference["4"]["top1"])
    q0_k4 = float(stages["Q0"]["4"]["top1"])
    q0_k16 = float(stages["Q0"]["16"]["top1"])
    adequate = r0_k4 >= .75 and q0_k4 >= .70 and q0_k16 >= .45

    if not adequate:
        return {
            "classification": SEMANTIC_BASELINE_INADEQUATE,
            "adequate": False,
            "damaging_stages": [],
            "r0_k4_top1": r0_k4,
            "q0_k4_top1": q0_k4,
            "q0_k16_top1": q0_k16,
        }

    damaging = [
        label
        for source, target, label in TRANSITIONS
        if _damaging(metrics, source, target)
    ]

    if len(damaging) == 1:
        classification = damaging[0]
    elif len(damaging) > 1:
        classification = MULTI_STAGE_INTERFACE_LOSS
    else:
        q7_k4 = float(stages["Q7"]["4"]["top1"])
        q7_k16 = float(stages["Q7"]["16"]["top1"])
        if q7_k4 <= q0_k4 - .15 and q7_k16 <= q0_k16 - .10:
            classification = DISTRIBUTED_INTERFACE_LOSS
        else:
            classification = INTERFACE_DECOMPOSITION_UNRESOLVED

    return {
        "classification": classification,
        "adequate": True,
        "damaging_stages": damaging,
        "r0_k4_top1": r0_k4,
        "q0_k4_top1": q0_k4,
        "q0_k16_top1": q0_k16,
    }


def cross_domain_outcome(
    per_domain: dict[str, dict[str, object]],
) -> dict[str, object]:
    if set(per_domain) != set(DOMAINS):
        raise ValueError("W11 requires BJ/BK/BL/BM classifications")

    excluded = {
        SEMANTIC_BASELINE_INADEQUATE,
        INTERFACE_DECOMPOSITION_UNRESOLVED,
    }
    counts = Counter(
        str(per_domain[domain]["classification"])
        for domain in DOMAINS
        if str(per_domain[domain]["classification"]) not in excluded
    )

    stable = [name for name, count in counts.items() if count >= 3]
    if len(stable) == 1:
        return {
            "outcome": OUTCOME_STABLE,
            "stable_classification": stable[0],
            "counts": dict(counts),
        }

    mixed = [name for name, count in counts.items() if count >= 2]
    if len(mixed) >= 2:
        return {
            "outcome": OUTCOME_MIXED,
            "stable_classification": None,
            "counts": dict(counts),
        }

    return {
        "outcome": OUTCOME_UNRESOLVED,
        "stable_classification": None,
        "counts": dict(counts),
    }
