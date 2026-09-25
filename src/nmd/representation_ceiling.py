from __future__ import annotations

from collections import Counter

DOMAINS = ("BF", "BG", "BH", "BI")

A13_REPRESENTATION_CEILING = "A13_REPRESENTATION_CEILING"
PROJECTION_DEGRADATION_LIMIT = "PROJECTION_DEGRADATION_LIMIT"
PROJECTION_RECOVERABLE_GEOMETRY = "PROJECTION_RECOVERABLE_GEOMETRY"
SCORING_INTERFACE_LIMIT = "SCORING_INTERFACE_LIMIT"
REFERENCE_TASK_AMBIGUOUS = "REFERENCE_TASK_AMBIGUOUS"
REPRESENTATION_CEILING_UNRESOLVED = "REPRESENTATION_CEILING_UNRESOLVED"

OUTCOME_STABLE = "STABLE_REPRESENTATION_CEILING_LOCALIZATION"
OUTCOME_MIXED = "MIXED_REPRESENTATION_CEILING_LOCALIZATION"
OUTCOME_UNRESOLVED = "REPRESENTATION_CEILING_UNRESOLVED"


def classify_domain(metrics: dict[str, float]) -> dict[str, object]:
    a0 = float(metrics["a0_k4_top1"])
    a1 = float(metrics["a1_k4_top1"])
    a1_k16 = float(metrics["a1_k16_top1"])
    p0 = float(metrics["p0_k4_top1"])
    p1 = float(metrics["p1_k4_top1"])
    p1_k16 = float(metrics["p1_k16_top1"])
    s1 = float(metrics["s1_k4_top1"])
    s1_k16 = float(metrics["s1_k16_top1"])
    r0 = float(metrics["r0_k4_top1"])

    active: list[str] = []
    raw_best = max(a0, a1)

    if (
        a1 < .60
        and a0 < .60
        and r0 >= .75
        and r0 - raw_best >= .20
        and p1 < raw_best + .15
    ):
        active.append(A13_REPRESENTATION_CEILING)

    if (
        a1 >= .70
        and p0 <= a1 - .15
        and r0 >= .75
        and a1_k16 >= .45
    ):
        active.append(PROJECTION_DEGRADATION_LIMIT)

    if (
        p1 >= .70
        and p1 >= p0 + .15
        and p1_k16 >= .45
        and r0 >= .75
    ):
        active.append(PROJECTION_RECOVERABLE_GEOMETRY)

    if (
        p1 >= .70
        and s1 <= p1 - .15
        and p1_k16 >= .45
        and s1_k16 <= p1_k16 - .10
    ):
        active.append(SCORING_INTERFACE_LIMIT)

    if r0 < .70:
        active.append(REFERENCE_TASK_AMBIGUOUS)

    primary = [
        x for x in active
        if x not in {REFERENCE_TASK_AMBIGUOUS}
    ]
    classification = (
        primary[0]
        if len(primary) == 1
        else REPRESENTATION_CEILING_UNRESOLVED
    )
    return {
        "classification": classification,
        "active_rules": active,
        "raw_best_k4_top1": raw_best,
    }


def cross_domain_outcome(per_domain: dict[str, dict[str, object]]) -> dict[str, object]:
    if set(per_domain) != set(DOMAINS):
        raise ValueError("W10 requires BF/BG/BH/BI classifications")

    primary_counts: Counter[str] = Counter()
    ambiguous_count = 0
    for domain in DOMAINS:
        row = per_domain[domain]
        if REFERENCE_TASK_AMBIGUOUS in row.get("active_rules", []):
            ambiguous_count += 1
        cls = str(row["classification"])
        if cls != REPRESENTATION_CEILING_UNRESOLVED:
            primary_counts[cls] += 1

    stable = [name for name, count in primary_counts.items() if count >= 3]
    if ambiguous_count >= 3:
        return {
            "outcome": OUTCOME_UNRESOLVED,
            "stable_classification": REFERENCE_TASK_AMBIGUOUS,
            "counts": dict(primary_counts),
            "reference_ambiguous_domains": ambiguous_count,
        }

    if len(stable) == 1:
        return {
            "outcome": OUTCOME_STABLE,
            "stable_classification": stable[0],
            "counts": dict(primary_counts),
            "reference_ambiguous_domains": ambiguous_count,
        }
    if len(stable) > 1 or sum(count >= 2 for count in primary_counts.values()) >= 2:
        return {
            "outcome": OUTCOME_MIXED,
            "stable_classification": None,
            "counts": dict(primary_counts),
            "reference_ambiguous_domains": ambiguous_count,
        }
    return {
        "outcome": OUTCOME_UNRESOLVED,
        "stable_classification": None,
        "counts": dict(primary_counts),
        "reference_ambiguous_domains": ambiguous_count,
    }
