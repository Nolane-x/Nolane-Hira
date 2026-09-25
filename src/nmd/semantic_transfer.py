from __future__ import annotations

from collections import Counter
from typing import Mapping

LABELS = (
    "SCHEMA_LABEL_INTERFACE_LIMIT",
    "SYNTHETIC_FORMAT_DEPENDENCE",
    "STATE_SCHEMA_ALIGNMENT_LIMIT",
    "GENERAL_SEMANTIC_TRANSFER_LIMIT",
    "CARDINALITY_AMPLIFICATION_AFTER_TRANSFER",
)
DOMAIN_MIXED = "MIXED_SEMANTIC_TRANSFER_FAILURE"
DOMAIN_UNRESOLVED = "SEMANTIC_TRANSFER_UNRESOLVED"

OUTCOME_STABLE = "STABLE_SEMANTIC_TRANSFER_LOCALIZATION"
OUTCOME_MIXED = "MIXED_SEMANTIC_TRANSFER_LOCALIZATION"
OUTCOME_UNRESOLVED = "SEMANTIC_TRANSFER_UNRESOLVED"

CHECKPOINTS = (
    "frozen-w6e-control",
    "typed-only-retune",
    "typed-plus-pair-primary",
    "typed-plus-pair-replica",
)
DOMAINS = ("AU", "AV", "AW", "AX")


def _f(metrics: Mapping[str, float], key: str) -> float:
    try:
        return float(metrics[key])
    except KeyError as exc:
        raise ValueError(f"missing W8 metric: {key}") from exc


def transfer_classification(
    metrics: Mapping[str, float],
) -> dict[str, object]:
    v0 = _f(metrics, "v0_k4_top1")
    v1 = _f(metrics, "v1_k4_top1")
    v2 = _f(metrics, "v2_k4_top1")
    v3 = _f(metrics, "v3_k4_top1")

    natural_view = "V1" if v1 >= v3 else "V3"
    natural_k4 = max(v1, v3)
    natural_k16 = _f(
        metrics,
        "v1_k16_top1" if natural_view == "V1" else "v3_k16_top1",
    )
    k16_coarse_error_share = _f(
        metrics,
        "k16_coarse_wrong_given_final_error_rate",
    )

    active: list[str] = []

    if (
        v0 < 0.55
        and (v1 - v0) >= 0.20
        and v1 >= 0.65
        and (v2 - v1) < 0.15
        and (v3 - v1) < 0.15
    ):
        active.append("SCHEMA_LABEL_INTERFACE_LIMIT")

    synthetic_format = (
        (v2 - v1) >= 0.20
        and v2 >= 0.70
        and v0 < 0.65
        and v1 < 0.65
    )
    if synthetic_format:
        active.append("SYNTHETIC_FORMAT_DEPENDENCE")

    if (
        (v3 - v1) >= 0.20
        and v3 >= 0.70
        and not synthetic_format
    ):
        active.append("STATE_SCHEMA_ALIGNMENT_LIMIT")

    if max(v0, v1, v2, v3) < 0.60:
        active.append("GENERAL_SEMANTIC_TRANSFER_LIMIT")

    if (
        natural_k4 >= 0.75
        and natural_k16 <= natural_k4 - 0.20
        and k16_coarse_error_share >= 0.70
    ):
        active.append("CARDINALITY_AMPLIFICATION_AFTER_TRANSFER")

    if len(active) == 1:
        label = active[0]
    elif len(active) > 1:
        label = DOMAIN_MIXED
    else:
        label = DOMAIN_UNRESOLVED

    return {
        "classification": label,
        "active_rules": active,
        "best_natural_view": natural_view,
        "best_natural_k4_top1": natural_k4,
        "best_natural_k16_top1": natural_k16,
    }


def checkpoint_stability(
    per_domain: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if set(per_domain) != set(DOMAINS):
        raise ValueError("unexpected W8 domain set")

    labels = {
        domain: str(row["classification"])
        for domain, row in per_domain.items()
    }
    counts = Counter(
        label
        for label in labels.values()
        if label not in {DOMAIN_MIXED, DOMAIN_UNRESOLVED}
    )
    if not counts:
        return {
            "stable": False,
            "classification": DOMAIN_UNRESOLVED,
            "domains": [],
        }

    top_count = max(counts.values())
    winners = [
        label for label, count in counts.items()
        if count == top_count
    ]
    if top_count < 3 or len(winners) != 1:
        return {
            "stable": False,
            "classification": (
                DOMAIN_MIXED if len(winners) > 1 else DOMAIN_UNRESOLVED
            ),
            "domains": [],
        }

    label = winners[0]
    domains = sorted(
        domain for domain, row_label in labels.items()
        if row_label == label
    )
    return {
        "stable": True,
        "classification": label,
        "domains": domains,
    }


def transfer_stability(
    checkpoint_domains: Mapping[
        str,
        Mapping[str, Mapping[str, object]],
    ],
) -> dict[str, object]:
    if set(checkpoint_domains) != set(CHECKPOINTS):
        raise ValueError("unexpected W8 checkpoint set")

    stable = {
        checkpoint: checkpoint_stability(domains)
        for checkpoint, domains in checkpoint_domains.items()
    }

    base = stable["frozen-w6e-control"]
    if not bool(base["stable"]):
        opposing = {
            str(row["classification"])
            for name, row in stable.items()
            if name != "frozen-w6e-control" and bool(row["stable"])
        }
        return {
            "outcome": (
                OUTCOME_MIXED if len(opposing) > 1 else OUTCOME_UNRESOLVED
            ),
            "stable_classification": None,
            "checkpoint_stability": stable,
        }

    label = str(base["classification"])
    retuned = [
        stable["typed-only-retune"],
        stable["typed-plus-pair-primary"],
        stable["typed-plus-pair-replica"],
    ]
    agreeing = [
        row for row in retuned
        if bool(row["stable"]) and str(row["classification"]) == label
    ]
    opposing = [
        row for row in retuned
        if bool(row["stable"]) and str(row["classification"]) != label
    ]

    if agreeing and not opposing:
        return {
            "outcome": OUTCOME_STABLE,
            "stable_classification": label,
            "checkpoint_stability": stable,
        }
    if opposing:
        return {
            "outcome": OUTCOME_MIXED,
            "stable_classification": None,
            "checkpoint_stability": stable,
        }
    return {
        "outcome": OUTCOME_UNRESOLVED,
        "stable_classification": None,
        "checkpoint_stability": stable,
    }
