from __future__ import annotations

from collections import Counter
from typing import Mapping


LABELS = (
    "PAIRWISE_MULTIPLICITY_LIMIT",
    "SET_CONTEXT_RANK_REVERSAL",
    "COARSE_CONJUNCTION_LIMIT",
    "RELATION_DECOMPOSITION_DAMAGE",
)
UNRESOLVED = "HIGH_CARDINALITY_DECOMPOSITION_UNRESOLVED"
MIXED = "MIXED_HIGH_CARDINALITY_ARCHITECTURE"


def _f(metrics: Mapping[str, float], key: str) -> float:
    try:
        return float(metrics[key])
    except KeyError as exc:
        raise ValueError(f"missing W6j metric: {key}") from exc


def architecture_classification(
    metrics: Mapping[str, float],
) -> dict[str, object]:
    """Apply preregistered W6j architectural-localization gates.

    The function deliberately does not impose a post-hoc precedence order.
    If more than one mechanism independently satisfies its frozen gate, the
    diagnostic is mixed rather than selecting the most convenient label.
    """

    k64 = _f(metrics, "k64_final_top1")
    pair_loss_given_global_error = _f(
        metrics,
        "global_error_has_pair_loss_rate",
    )
    all_pair_win_k64_fail = _f(
        metrics,
        "all_pair_win_but_k64_fail_rate",
    )
    reversal = _f(metrics, "winner_reversal_rate")
    sign_reversal = _f(
        metrics,
        "winner_k2_final_to_k64_final_sign_reversal_rate",
    )
    relation_damage_given_correct = _f(
        metrics,
        "relation_damage_given_coarse_correct_rate",
    )
    winner_beats_k2_coarse = _f(
        metrics,
        "global_error_winner_beats_gold_k2_coarse_rate",
    )
    relation_rescue_gain = _f(metrics, "relation_rescue_top1_gain")
    oracle_final = _f(metrics, "oracle_final_top1")
    coarse_top1 = _f(metrics, "k64_coarse_top1")
    relation_damage = _f(metrics, "relation_damage_rate")
    relation_rescue = _f(metrics, "relation_rescue_rate")
    oracle_error = 1.0 - oracle_final

    active: list[str] = []

    if (
        k64 < 0.65
        and pair_loss_given_global_error >= 0.80
        and all_pair_win_k64_fail < 0.10
        and reversal < 0.15
        and relation_damage_given_correct < 0.10
    ):
        active.append("PAIRWISE_MULTIPLICITY_LIMIT")

    if (
        k64 < 0.65
        and reversal >= 0.25
        and all_pair_win_k64_fail >= 0.15
        and sign_reversal >= 0.20
    ):
        active.append("SET_CONTEXT_RANK_REVERSAL")

    if (
        k64 < 0.65
        and winner_beats_k2_coarse >= 0.70
        and relation_rescue_gain < 0.10
        and oracle_final >= 0.95
    ):
        active.append("COARSE_CONJUNCTION_LIMIT")

    # A stable coarse-correct measurement is represented by either already
    # adequate coarse top1 or a substantial coarse-correct subset.
    if (
        (coarse_top1 >= 0.65 or _f(metrics, "coarse_correct_case_rate") >= 0.50)
        and relation_damage_given_correct >= 0.15
        and oracle_error >= 0.05
        and (relation_damage - relation_rescue) >= 0.05
    ):
        active.append("RELATION_DECOMPOSITION_DAMAGE")

    if len(active) == 1:
        label = active[0]
    elif len(active) > 1:
        label = MIXED
    else:
        label = UNRESOLVED

    return {
        "classification": label,
        "active_rules": active,
    }


def checkpoint_stability(
    per_domain: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Require the same non-mixed mechanism on at least two fresh domains."""

    labels = [
        str(row["classification"])
        for row in per_domain.values()
    ]
    counts = Counter(
        label
        for label in labels
        if label not in {UNRESOLVED, MIXED}
    )
    if not counts:
        return {
            "stable": False,
            "classification": UNRESOLVED,
            "domains": [],
        }

    label, count = counts.most_common(1)[0]
    tied = [
        name
        for name, n in counts.items()
        if n == count
    ]
    if count < 2 or len(tied) != 1:
        return {
            "stable": False,
            "classification": MIXED if len(tied) > 1 else UNRESOLVED,
            "domains": [],
        }

    domains = [
        domain
        for domain, row in per_domain.items()
        if str(row["classification"]) == label
    ]
    return {
        "stable": True,
        "classification": label,
        "domains": sorted(domains),
    }


def diagnostic_stability(
    checkpoint_domains: Mapping[
        str,
        Mapping[str, Mapping[str, object]],
    ],
) -> dict[str, object]:
    """Apply the frozen W6j cross-checkpoint stability rule.

    Required checkpoint names:
    - w6e-joint-primary
    - w6h-projection-retune
    - w6h-semantic-adapter
    """

    required = {
        "w6e-joint-primary",
        "w6h-projection-retune",
        "w6h-semantic-adapter",
    }
    if set(checkpoint_domains) != required:
        raise ValueError("unexpected W6j checkpoint set")

    stable = {
        name: checkpoint_stability(domains)
        for name, domains in checkpoint_domains.items()
    }

    base = stable["w6e-joint-primary"]
    if not bool(base["stable"]):
        return {
            "outcome": UNRESOLVED,
            "stable_classification": None,
            "rescue_lane_authorized": False,
            "checkpoint_stability": stable,
        }

    label = str(base["classification"])
    trained = [
        stable["w6h-projection-retune"],
        stable["w6h-semantic-adapter"],
    ]
    agreeing = [
        row
        for row in trained
        if bool(row["stable"]) and str(row["classification"]) == label
    ]
    opposing = [
        row
        for row in trained
        if bool(row["stable"]) and str(row["classification"]) != label
    ]

    if agreeing and not opposing:
        return {
            "outcome": "STABLE_HIGH_CARDINALITY_ARCHITECTURE",
            "stable_classification": label,
            "rescue_lane_authorized": True,
            "checkpoint_stability": stable,
        }

    if opposing:
        return {
            "outcome": MIXED,
            "stable_classification": None,
            "rescue_lane_authorized": False,
            "checkpoint_stability": stable,
        }

    return {
        "outcome": UNRESOLVED,
        "stable_classification": None,
        "rescue_lane_authorized": False,
        "checkpoint_stability": stable,
    }
