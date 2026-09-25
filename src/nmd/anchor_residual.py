from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

DOMAINS = ("BN", "BO", "BP", "BQ")

CONFIDENCE_GATED_RESIDUAL_SIGNAL = "CONFIDENCE_GATED_RESIDUAL_SIGNAL"
GLOBAL_ANCHOR_DOMINANCE = "GLOBAL_ANCHOR_DOMINANCE"
COMPETITIVE_RESIDUAL_VALUE = "COMPETITIVE_RESIDUAL_VALUE"
ANCHOR_MARGIN_UNRELIABLE = "ANCHOR_MARGIN_UNRELIABLE"
ANCHOR_RESIDUAL_UNRESOLVED = "ANCHOR_RESIDUAL_UNRESOLVED"

OUTCOME_STABLE = "STABLE_ANCHOR_RESIDUAL_LOCALIZATION"
OUTCOME_MIXED = "MIXED_ANCHOR_RESIDUAL_LOCALIZATION"
OUTCOME_UNRESOLVED = "ANCHOR_RESIDUAL_UNRESOLVED"


@dataclass(frozen=True)
class ConfidenceRow:
    base_id: str
    margin: float


def normalized_anchor_margin(scores: Sequence[float]) -> float:
    if len(scores) < 2:
        raise ValueError("W12 anchor confidence requires at least two candidates")
    values = sorted(float(x) for x in scores)
    top1 = values[-1]
    top2 = values[-2]
    median = _median(values)
    deviations = sorted(abs(x - median) for x in values)
    mad = _median(deviations)
    return (top1 - top2) / max(mad, 1e-6)


def deterministic_quartiles(rows: Iterable[ConfidenceRow]) -> dict[str, str]:
    rows = list(rows)
    if len(rows) != 64:
        raise ValueError("W12 requires exactly 64 cases per domain/K for quartiles")
    ordered = sorted(rows, key=lambda x: (x.margin, x.base_id))
    out: dict[str, str] = {}
    for index, row in enumerate(ordered):
        if index < 16:
            bucket = "LOW"
        elif index >= 48:
            bucket = "HIGH"
        else:
            bucket = "MIDDLE"
        out[row.base_id] = bucket
    if Counter(out.values()) != Counter({"LOW": 16, "MIDDLE": 32, "HIGH": 16}):
        raise RuntimeError("W12 confidence quartiles must be 16/32/16")
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or not xs:
        raise ValueError("Spearman inputs must be equal non-zero length")
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    dx = [x - mx for x in rx]
    dy = [y - my for y in ry]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / denom


def classify_domain(metrics: dict[str, object]) -> dict[str, object]:
    reference = metrics["reference"]
    anchor = metrics["anchor"]
    final = metrics["final"]
    strata = metrics["strata"]
    guards = metrics["guards"]

    r0_k4 = float(reference["4"]["top1"])
    a4 = float(anchor["4"]["top1"])
    a16 = float(anchor["16"]["top1"])
    f4 = float(final["4"]["top1"])
    f16 = float(final["16"]["top1"])
    adequate = r0_k4 >= .75 and a4 >= .70 and a16 >= .45

    if not adequate:
        return {
            "classification": ANCHOR_RESIDUAL_UNRESOLVED,
            "adequate": False,
            "r0_k4_top1": r0_k4,
            "anchor_k4_top1": a4,
            "anchor_k16_top1": a16,
        }

    high4 = strata["HIGH"]["4"]
    high16 = strata["HIGH"]["16"]
    low4 = strata["LOW"]["4"]
    low16 = strata["LOW"]["16"]
    high_transition = strata["HIGH"]["pooled_primary_transition"]
    low_transition = strata["LOW"]["pooled_primary_transition"]

    gated = (
        float(high4["anchor_top1"]) >= .80
        and float(high16["anchor_top1"]) >= .65
        and float(high4["final_top1"]) <= float(high4["anchor_top1"]) - .10
        and float(high16["final_top1"]) <= float(high16["anchor_top1"]) - .08
        and float(high_transition["correct_to_wrong_rate"])
        - float(high_transition["wrong_to_correct_rate"]) >= .08
        and float(low4["final_top1"]) >= float(low4["anchor_top1"]) - .03
        and float(low16["final_top1"]) >= float(low16["anchor_top1"]) - .03
        and float(guards["G_high_anchor"]["4"]["top1"]) >= f4 + .05
        and float(guards["G_high_anchor"]["16"]["top1"]) >= f16 + .04
        and (
            float(guards["G_high_anchor"]["4"]["top1"])
            >= float(guards["G_low_anchor_control"]["4"]["top1"]) + .03
            or float(guards["G_high_anchor"]["16"]["top1"])
            >= float(guards["G_low_anchor_control"]["16"]["top1"]) + .03
        )
    )
    if gated:
        classification = CONFIDENCE_GATED_RESIDUAL_SIGNAL
    else:
        low_damage_k4 = (
            float(low4["final_top1"]) <= float(low4["anchor_top1"]) - .05
        )
        low_damage_k16 = (
            float(low16["final_top1"]) <= float(low16["anchor_top1"]) - .05
        )
        high_damage_k4 = (
            float(high4["final_top1"]) <= float(high4["anchor_top1"]) - .05
        )
        high_damage_k16 = (
            float(high16["final_top1"]) <= float(high16["anchor_top1"]) - .05
        )
        global_anchor = (
            a4 >= f4 + .08
            and a16 >= f16 + .08
            and (
                (high_damage_k4 and low_damage_k4)
                or (high_damage_k16 and low_damage_k16)
            )
        )
        residual_value = (
            (f4 >= a4 + .05 or f16 >= a16 + .05)
            and float(low_transition["wrong_to_correct_rate"])
            - float(low_transition["correct_to_wrong_rate"]) >= .08
            and float(high4["final_top1"]) >= float(high4["anchor_top1"]) - .05
            and float(high16["final_top1"]) >= float(high16["anchor_top1"]) - .05
        )
        margin_unreliable = (
            float(high4["anchor_top1"]) - float(low4["anchor_top1"]) < .15
            and float(high16["anchor_top1"]) - float(low16["anchor_top1"]) < .15
        )

        if global_anchor:
            classification = GLOBAL_ANCHOR_DOMINANCE
        elif residual_value:
            classification = COMPETITIVE_RESIDUAL_VALUE
        elif margin_unreliable:
            classification = ANCHOR_MARGIN_UNRELIABLE
        else:
            classification = ANCHOR_RESIDUAL_UNRESOLVED

    return {
        "classification": classification,
        "adequate": True,
        "r0_k4_top1": r0_k4,
        "anchor_k4_top1": a4,
        "anchor_k16_top1": a16,
        "final_k4_top1": f4,
        "final_k16_top1": f16,
    }


def cross_domain_outcome(
    per_domain: dict[str, dict[str, object]],
) -> dict[str, object]:
    if set(per_domain) != set(DOMAINS):
        raise ValueError("W12 requires BN/BO/BP/BQ classifications")

    counts = Counter(
        str(per_domain[domain]["classification"])
        for domain in DOMAINS
        if str(per_domain[domain]["classification"]) != ANCHOR_RESIDUAL_UNRESOLVED
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


def _median(values: Sequence[float]) -> float:
    values = sorted(float(x) for x in values)
    n = len(values)
    if n == 0:
        raise ValueError("median requires data")
    mid = n // 2
    if n % 2:
        return values[mid]
    return .5 * (values[mid - 1] + values[mid])


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda x: (float(x[1]), x[0]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and float(ordered[j][1]) == float(ordered[i][1]):
            j += 1
        avg = .5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[ordered[k][0]] = avg
        i = j
    return ranks
