from __future__ import annotations

from collections import Counter
from typing import Sequence

from .anchor_residual import normalized_anchor_margin, spearman

DOMAINS = ("BR", "BS", "BT", "BU")
K_VALUES = (4, 8, 16)

PARAPHRASE_CONSISTENCY_RELIABLE = "PARAPHRASE_CONSISTENCY_RELIABLE"
MULTIVIEW_ANCHOR_DOMINANCE = "MULTIVIEW_ANCHOR_DOMINANCE"
PRODUCTION_VALUE_ON_INCONSISTENCY = "PRODUCTION_VALUE_ON_INCONSISTENCY"
PARAPHRASE_CONSISTENCY_UNINFORMATIVE = "PARAPHRASE_CONSISTENCY_UNINFORMATIVE"
CONSISTENCY_BASELINE_INADEQUATE = "CONSISTENCY_BASELINE_INADEQUATE"
SEMANTIC_CONSISTENCY_UNRESOLVED = "SEMANTIC_CONSISTENCY_UNRESOLVED"

OUTCOME_STABLE = "STABLE_SEMANTIC_CONSISTENCY_LOCALIZATION"
OUTCOME_MIXED = "MIXED_SEMANTIC_CONSISTENCY_LOCALIZATION"
OUTCOME_UNRESOLVED = "SEMANTIC_CONSISTENCY_UNRESOLVED"


def top1_consistency(top1_ids: Sequence[str]) -> str:
    if len(top1_ids) != 3:
        raise ValueError("W13 requires exactly three paraphrase top1 IDs")
    unique = len(set(str(x) for x in top1_ids))
    if unique == 1:
        return "STRICT"
    if unique == 2:
        return "MAJORITY"
    return "SPLIT"


def rank_order_spearman(order_a: Sequence[int], order_b: Sequence[int]) -> float:
    if len(order_a) != len(order_b) or set(order_a) != set(order_b):
        raise ValueError("W13 rank orders must contain the same candidate indices")
    if not order_a:
        raise ValueError("W13 rank orders cannot be empty")
    rank_a = [0.0] * len(order_a)
    rank_b = [0.0] * len(order_b)
    for rank, candidate in enumerate(order_a):
        rank_a[int(candidate)] = float(rank + 1)
    for rank, candidate in enumerate(order_b):
        rank_b[int(candidate)] = float(rank + 1)
    return spearman(rank_a, rank_b)


def mean_pairwise_rank_stability(
    orders: Sequence[Sequence[int]],
) -> float:
    if len(orders) != 3:
        raise ValueError("W13 requires exactly three paraphrase rank orders")
    values = (
        rank_order_spearman(orders[0], orders[1]),
        rank_order_spearman(orders[0], orders[2]),
        rank_order_spearman(orders[1], orders[2]),
    )
    return sum(values) / 3.0


def countmatched_margin_ids(
    rows: Sequence[tuple[str, float]],
    n_selected: int,
) -> set[str]:
    if not 0 <= n_selected <= len(rows):
        raise ValueError("invalid W13 count-matched selection size")
    if len({str(base_id) for base_id, _ in rows}) != len(rows):
        raise ValueError("duplicate W13 base ID in count-matched margin control")
    ordered = sorted(
        ((str(base_id), float(margin)) for base_id, margin in rows),
        key=lambda x: (-x[1], x[0]),
    )
    return {base_id for base_id, _ in ordered[:n_selected]}


def classify_domain(metrics: dict[str, object]) -> dict[str, object]:
    reference = metrics["reference"]
    ensemble = metrics["ensemble"]
    final = metrics["final"]
    strata = metrics["strata"]
    guards = metrics["guards"]

    r0_k4 = float(reference["4"]["top1"])
    e4 = float(ensemble["4"]["top1"])
    e16 = float(ensemble["16"]["top1"])
    f4 = float(final["4"]["top1"])
    f16 = float(final["16"]["top1"])

    strict4 = strata["STRICT"]["4"]
    strict16 = strata["STRICT"]["16"]
    non4 = strata["NON_STRICT"]["4"]
    non16 = strata["NON_STRICT"]["16"]
    strict_transition = strata["STRICT"]["pooled_primary_transition"]
    non_transition = strata["NON_STRICT"]["pooled_primary_transition"]

    strict_cov4 = float(strict4["coverage"])
    strict_cov16 = float(strict16["coverage"])
    adequate = (
        r0_k4 >= .75
        and e4 >= .70
        and e16 >= .45
        and strict_cov4 >= .25
        and strict_cov16 >= .25
    )
    if not adequate:
        return {
            "classification": CONSISTENCY_BASELINE_INADEQUATE,
            "adequate": False,
            "r0_k4_top1": r0_k4,
            "ensemble_k4_top1": e4,
            "ensemble_k16_top1": e16,
            "strict_coverage_k4": strict_cov4,
            "strict_coverage_k16": strict_cov16,
        }

    reliable = (
        float(strict4["ensemble_top1"]) >= .85
        and float(strict16["ensemble_top1"]) >= .70
        and float(strict4["ensemble_top1"]) - float(non4["ensemble_top1"]) >= .15
        and float(strict16["ensemble_top1"]) - float(non16["ensemble_top1"]) >= .15
        and float(strict4["final_top1"]) <= float(strict4["ensemble_top1"]) - .10
        and float(strict16["final_top1"]) <= float(strict16["ensemble_top1"]) - .08
        and float(strict_transition["correct_to_wrong_rate"])
        - float(strict_transition["wrong_to_correct_rate"]) >= .08
        and float(non4["final_top1"]) >= float(non4["ensemble_top1"]) - .03
        and float(non16["final_top1"]) >= float(non16["ensemble_top1"]) - .03
        and float(guards["G_strict_ensemble"]["4"]["top1"]) >= f4 + .05
        and float(guards["G_strict_ensemble"]["16"]["top1"]) >= f16 + .04
        and (
            float(guards["G_strict_ensemble"]["4"]["top1"])
            >= float(guards["G_margin_countmatched"]["4"]["top1"]) + .03
            or float(guards["G_strict_ensemble"]["16"]["top1"])
            >= float(guards["G_margin_countmatched"]["16"]["top1"]) + .03
        )
    )
    if reliable:
        classification = PARAPHRASE_CONSISTENCY_RELIABLE
    else:
        strict_damage_k4 = (
            float(strict4["final_top1"])
            <= float(strict4["ensemble_top1"]) - .05
        )
        strict_damage_k16 = (
            float(strict16["final_top1"])
            <= float(strict16["ensemble_top1"]) - .05
        )
        non_damage_k4 = (
            float(non4["final_top1"])
            <= float(non4["ensemble_top1"]) - .05
        )
        non_damage_k16 = (
            float(non16["final_top1"])
            <= float(non16["ensemble_top1"]) - .05
        )
        global_anchor = (
            e4 >= f4 + .08
            and e16 >= f16 + .08
            and (
                (strict_damage_k4 and non_damage_k4)
                or (strict_damage_k16 and non_damage_k16)
            )
        )

        production_value = (
            (f4 >= e4 + .05 or f16 >= e16 + .05)
            and float(non_transition["wrong_to_correct_rate"])
            - float(non_transition["correct_to_wrong_rate"]) >= .08
            and float(strict4["final_top1"])
            >= float(strict4["ensemble_top1"]) - .05
            and float(strict16["final_top1"])
            >= float(strict16["ensemble_top1"]) - .05
        )

        uninformative = (
            float(strict4["ensemble_top1"]) - float(non4["ensemble_top1"]) < .15
            and float(strict16["ensemble_top1"]) - float(non16["ensemble_top1"]) < .15
        )

        if global_anchor:
            classification = MULTIVIEW_ANCHOR_DOMINANCE
        elif production_value:
            classification = PRODUCTION_VALUE_ON_INCONSISTENCY
        elif uninformative:
            classification = PARAPHRASE_CONSISTENCY_UNINFORMATIVE
        else:
            classification = SEMANTIC_CONSISTENCY_UNRESOLVED

    return {
        "classification": classification,
        "adequate": True,
        "r0_k4_top1": r0_k4,
        "ensemble_k4_top1": e4,
        "ensemble_k16_top1": e16,
        "final_k4_top1": f4,
        "final_k16_top1": f16,
        "strict_coverage_k4": strict_cov4,
        "strict_coverage_k16": strict_cov16,
    }


def cross_domain_outcome(
    per_domain: dict[str, dict[str, object]],
) -> dict[str, object]:
    if set(per_domain) != set(DOMAINS):
        raise ValueError("W13 requires BR/BS/BT/BU classifications")

    ignored = {
        SEMANTIC_CONSISTENCY_UNRESOLVED,
        CONSISTENCY_BASELINE_INADEQUATE,
    }
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
    "PARAPHRASE_CONSISTENCY_RELIABLE",
    "MULTIVIEW_ANCHOR_DOMINANCE",
    "PRODUCTION_VALUE_ON_INCONSISTENCY",
    "PARAPHRASE_CONSISTENCY_UNINFORMATIVE",
    "CONSISTENCY_BASELINE_INADEQUATE",
    "SEMANTIC_CONSISTENCY_UNRESOLVED",
    "OUTCOME_STABLE",
    "OUTCOME_MIXED",
    "OUTCOME_UNRESOLVED",
    "normalized_anchor_margin",
    "spearman",
    "top1_consistency",
    "rank_order_spearman",
    "mean_pairwise_rank_stability",
    "countmatched_margin_ids",
    "classify_domain",
    "cross_domain_outcome",
]
