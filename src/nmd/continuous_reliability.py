from __future__ import annotations

from collections import Counter
from typing import Sequence

from .anchor_residual import normalized_anchor_margin, spearman
from .semantic_consistency import rank_order_spearman

DOMAINS = ("BV", "BW", "BX", "BY")
K_VALUES = (4, 8, 16)

CONTINUOUS_CONSISTENCY_RELIABLE = "CONTINUOUS_CONSISTENCY_RELIABLE"
CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE = "CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE"
PRODUCTION_VALUE_LOW_RELIABILITY = "PRODUCTION_VALUE_LOW_RELIABILITY"
CONTINUOUS_CONSISTENCY_UNINFORMATIVE = "CONTINUOUS_CONSISTENCY_UNINFORMATIVE"
CONTINUOUS_RELIABILITY_UNRESOLVED = "CONTINUOUS_RELIABILITY_UNRESOLVED"

OUTCOME_STABLE = "STABLE_CONTINUOUS_RELIABILITY_LOCALIZATION"
OUTCOME_MIXED = "MIXED_CONTINUOUS_RELIABILITY_LOCALIZATION"
OUTCOME_UNRESOLVED = "CONTINUOUS_RELIABILITY_UNRESOLVED"


def vote_strength(top1_ids: Sequence[str]) -> float:
    if len(top1_ids) != 3:
        raise ValueError("W14 requires exactly three paraphrase top1 IDs")
    vmax = max(Counter(str(x) for x in top1_ids).values())
    return (float(vmax) - 1.0) / 2.0


def mean_rank_stability(orders: Sequence[Sequence[int]]) -> float:
    if len(orders) != 3:
        raise ValueError("W14 requires exactly three paraphrase rank orders")
    values = (
        rank_order_spearman(orders[0], orders[1]),
        rank_order_spearman(orders[0], orders[2]),
        rank_order_spearman(orders[1], orders[2]),
    )
    return max(0.0, min(1.0, (sum(values) / 3.0 + 1.0) / 2.0))


def mean_top3_overlap(orders: Sequence[Sequence[int]]) -> float:
    if len(orders) != 3:
        raise ValueError("W14 requires exactly three paraphrase rank orders")

    def top(order: Sequence[int]) -> set[int]:
        n = min(3, len(order))
        return {int(x) for x in order[:n]}

    sets = [top(order) for order in orders]
    values: list[float] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        union = sets[left] | sets[right]
        if not union:
            raise ValueError("W14 top-k set cannot be empty")
        values.append(len(sets[left] & sets[right]) / len(union))
    return sum(values) / 3.0


def continuous_reliability(
    top1_ids: Sequence[str],
    orders: Sequence[Sequence[int]],
) -> dict[str, float]:
    v = vote_strength(top1_ids)
    s = mean_rank_stability(orders)
    o = mean_top3_overlap(orders)
    r = (v + s + o) / 3.0
    return {"V": v, "S": s, "O": o, "R": r}


def fixed_tertiles(rows: Sequence[tuple[str, float]]) -> dict[str, set[str]]:
    if len(rows) != 64:
        raise ValueError("W14 tertiles require exactly 64 cases per domain/K")
    normalized = [(str(base_id), float(score)) for base_id, score in rows]
    if len({base_id for base_id, _ in normalized}) != 64:
        raise ValueError("duplicate W14 base ID in reliability tertiles")
    ordered = sorted(normalized, key=lambda item: (-item[1], item[0]))
    return {
        "HIGH": {base_id for base_id, _ in ordered[:21]},
        "MIDDLE": {base_id for base_id, _ in ordered[21:43]},
        "LOW": {base_id for base_id, _ in ordered[43:]},
    }


def top_margin_ids(rows: Sequence[tuple[str, float]]) -> set[str]:
    if len(rows) != 64:
        raise ValueError("W14 margin control requires exactly 64 cases")
    normalized = [(str(base_id), float(score)) for base_id, score in rows]
    if len({base_id for base_id, _ in normalized}) != 64:
        raise ValueError("duplicate W14 base ID in margin control")
    ordered = sorted(normalized, key=lambda item: (-item[1], item[0]))
    return {base_id for base_id, _ in ordered[:21]}


def classify_domain(metrics: dict[str, object]) -> dict[str, object]:
    reference = metrics["reference"]
    ensemble = metrics["ensemble"]
    final = metrics["final"]
    tertiles = metrics["tertiles"]
    guards = metrics["guards"]
    correlations = metrics["correlations"]

    r0_k4 = float(reference["4"]["top1"])
    e4 = float(ensemble["4"]["top1"])
    e16 = float(ensemble["16"]["top1"])
    f4 = float(final["4"]["top1"])
    f16 = float(final["16"]["top1"])

    high4 = tertiles["HIGH"]["4"]
    high16 = tertiles["HIGH"]["16"]
    low4 = tertiles["LOW"]["4"]
    low16 = tertiles["LOW"]["16"]
    low_transition = tertiles["LOW"]["pooled_primary_transition"]

    adequate = r0_k4 >= .75 and e4 >= .70 and e16 >= .45
    if not adequate:
        return {
            "classification": CONTINUOUS_RELIABILITY_UNRESOLVED,
            "adequate": False,
            "r0_k4_top1": r0_k4,
            "ensemble_k4_top1": e4,
            "ensemble_k16_top1": e16,
        }

    high_gap4 = float(high4["ensemble_top1"]) - float(low4["ensemble_top1"])
    high_gap16 = float(high16["ensemble_top1"]) - float(low16["ensemble_top1"])
    rho4 = float(correlations["4"]["spearman_R_ensemble_correct"])
    rho16 = float(correlations["16"]["spearman_R_ensemble_correct"])

    reliable = (
        float(high4["ensemble_top1"]) >= .85
        and float(high16["ensemble_top1"]) >= .70
        and high_gap4 >= .15
        and high_gap16 >= .15
        and rho4 >= .25
        and rho16 >= .20
        and float(high4["final_top1"]) <= float(high4["ensemble_top1"]) - .10
        and float(high16["final_top1"]) <= float(high16["ensemble_top1"]) - .08
        and float(low4["final_top1"]) >= float(low4["ensemble_top1"]) - .03
        and float(low16["final_top1"]) >= float(low16["ensemble_top1"]) - .03
        and float(guards["G_consistency_high"]["4"]["top1"]) >= f4 + .05
        and float(guards["G_consistency_high"]["16"]["top1"]) >= f16 + .04
        and (
            float(guards["G_consistency_high"]["4"]["top1"])
            >= float(guards["G_margin_high"]["4"]["top1"]) + .03
            or float(guards["G_consistency_high"]["16"]["top1"])
            >= float(guards["G_margin_high"]["16"]["top1"]) + .03
        )
    )

    if reliable:
        classification = CONTINUOUS_CONSISTENCY_RELIABLE
    else:
        anchor_dominance = (
            e4 >= f4 + .08
            and e16 >= f16 + .08
            and (
                (
                    float(high4["final_top1"])
                    <= float(high4["ensemble_top1"]) - .05
                    and float(low4["final_top1"])
                    <= float(low4["ensemble_top1"]) - .05
                )
                or (
                    float(high16["final_top1"])
                    <= float(high16["ensemble_top1"]) - .05
                    and float(low16["final_top1"])
                    <= float(low16["ensemble_top1"]) - .05
                )
            )
        )
        production_value = (
            float(low_transition["wrong_to_correct_rate"])
            - float(low_transition["correct_to_wrong_rate"]) >= .08
            and float(high4["final_top1"]) >= float(high4["ensemble_top1"]) - .05
            and float(high16["final_top1"]) >= float(high16["ensemble_top1"]) - .05
            and (
                f4 >= e4 + .05
                or f16 >= e16 + .05
                or (
                    float(guards["G_consistency_high"]["4"]["top1"]) >= e4 - .03
                    and float(guards["G_consistency_high"]["16"]["top1"]) >= e16 - .03
                )
            )
        )
        uninformative = (
            high_gap4 < .10
            and high_gap16 < .10
            and abs(rho4) < .15
            and abs(rho16) < .15
        )

        if anchor_dominance:
            classification = CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE
        elif production_value:
            classification = PRODUCTION_VALUE_LOW_RELIABILITY
        elif uninformative:
            classification = CONTINUOUS_CONSISTENCY_UNINFORMATIVE
        else:
            classification = CONTINUOUS_RELIABILITY_UNRESOLVED

    return {
        "classification": classification,
        "adequate": True,
        "r0_k4_top1": r0_k4,
        "ensemble_k4_top1": e4,
        "ensemble_k16_top1": e16,
        "final_k4_top1": f4,
        "final_k16_top1": f16,
        "high_low_gap_k4": high_gap4,
        "high_low_gap_k16": high_gap16,
        "spearman_R_ensemble_correct_k4": rho4,
        "spearman_R_ensemble_correct_k16": rho16,
    }


def cross_domain_outcome(
    per_domain: dict[str, dict[str, object]],
) -> dict[str, object]:
    if set(per_domain) != set(DOMAINS):
        raise ValueError("W14 requires BV/BW/BX/BY classifications")

    ignored = {CONTINUOUS_RELIABILITY_UNRESOLVED}
    counts = Counter(
        str(per_domain[domain]["classification"])
        for domain in DOMAINS
        if str(per_domain[domain]["classification"]) not in ignored
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


__all__ = [
    "DOMAINS",
    "K_VALUES",
    "CONTINUOUS_CONSISTENCY_RELIABLE",
    "CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE",
    "PRODUCTION_VALUE_LOW_RELIABILITY",
    "CONTINUOUS_CONSISTENCY_UNINFORMATIVE",
    "CONTINUOUS_RELIABILITY_UNRESOLVED",
    "OUTCOME_STABLE",
    "OUTCOME_MIXED",
    "OUTCOME_UNRESOLVED",
    "normalized_anchor_margin",
    "spearman",
    "vote_strength",
    "mean_rank_stability",
    "mean_top3_overlap",
    "continuous_reliability",
    "fixed_tertiles",
    "top_margin_ids",
    "classify_domain",
    "cross_domain_outcome",
]
