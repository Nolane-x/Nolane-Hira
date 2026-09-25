from __future__ import annotations

from typing import Mapping


CANDIDATES = (
    "frozen-w6e-control",
    "projection-semantic-control",
    "shared-bridge-semantic-control",
    "asymmetric-bridge-primary",
    "asymmetric-bridge-replica",
)


def _domain_metrics(
    result: Mapping[str, object],
    domain: str,
) -> dict[str, float]:
    per_domain = result["per_domain"][domain]
    alignment = result["alignment"]["per_domain"][domain]
    views = per_domain["views"]
    return {
        "definition_k4_final": float(
            views["definition"]["4"]["final_top1"]
        ),
        "definition_k16_final": float(
            views["definition"]["16"]["final_top1"]
        ),
        "definition_k4_coarse": float(
            views["definition"]["4"]["coarse_top1"]
        ),
        "label_k4_final": float(
            views["label"]["4"]["final_top1"]
        ),
        "alignment_top1": float(alignment["top1"]),
        "probability_mass_max_error": float(
            result["probability_mass_max_error"]
        ),
        "state_encodes_per_base": float(
            result["state_encodes_per_base"]
        ),
    }


def primary_absolute_gates(
    primary: Mapping[str, object],
    frozen: Mapping[str, object],
    domain: str,
) -> dict[str, bool]:
    p = _domain_metrics(primary, domain)
    f = _domain_metrics(frozen, domain)
    return {
        "definition_k4": p["definition_k4_final"] >= 0.75,
        "definition_k16": p["definition_k16_final"] >= 0.55,
        "definition_k4_coarse": p["definition_k4_coarse"] >= 0.72,
        "alignment_top1": p["alignment_top1"] >= 0.70,
        "label_k4": p["label_k4_final"] >= 0.45,
        "probability_mass": p["probability_mass_max_error"] <= 1e-6,
        "state_once": p["state_encodes_per_base"] == 1.0,
        "definition_k4_gain_vs_frozen": (
            p["definition_k4_final"] - f["definition_k4_final"] >= 0.15
        ),
        "definition_k16_gain_vs_frozen": (
            p["definition_k16_final"] - f["definition_k16_final"] >= 0.15
        ),
        "alignment_gain_vs_frozen": (
            p["alignment_top1"] - f["alignment_top1"] >= 0.15
        ),
    }


def replica_gates(
    replica: Mapping[str, object],
    domain: str,
) -> dict[str, bool]:
    r = _domain_metrics(replica, domain)
    return {
        "definition_k4": r["definition_k4_final"] >= 0.70,
        "definition_k16": r["definition_k16_final"] >= 0.50,
        "alignment_top1": r["alignment_top1"] >= 0.65,
    }


def architecture_causal_gates(
    primary: Mapping[str, object],
    projection: Mapping[str, object],
    shared: Mapping[str, object],
    domain: str,
) -> dict[str, bool]:
    p = _domain_metrics(primary, domain)
    q = _domain_metrics(projection, domain)
    s = _domain_metrics(shared, domain)
    return {
        "k4_vs_projection": (
            p["definition_k4_final"] >= q["definition_k4_final"] + 0.05
        ),
        "alignment_vs_projection": (
            p["alignment_top1"] >= q["alignment_top1"] + 0.05
        ),
        "k16_vs_projection": (
            p["definition_k16_final"] >= q["definition_k16_final"] - 0.03
        ),
        "k4_vs_shared": (
            p["definition_k4_final"] >= s["definition_k4_final"] + 0.03
        ),
        "alignment_vs_shared": (
            p["alignment_top1"] >= s["alignment_top1"] + 0.03
        ),
        "k16_vs_shared": (
            p["definition_k16_final"] >= s["definition_k16_final"] - 0.03
        ),
    }


def control_rescue_gates(
    control: Mapping[str, object],
    frozen: Mapping[str, object],
    domain: str,
) -> dict[str, bool]:
    c = _domain_metrics(control, domain)
    f = _domain_metrics(frozen, domain)
    return {
        "definition_k4": c["definition_k4_final"] >= 0.75,
        "definition_k16": c["definition_k16_final"] >= 0.55,
        "alignment_top1": c["alignment_top1"] >= 0.70,
        "definition_k4_gain_vs_frozen": (
            c["definition_k4_final"] - f["definition_k4_final"] >= 0.15
        ),
        "definition_k16_gain_vs_frozen": (
            c["definition_k16_final"] - f["definition_k16_final"] >= 0.15
        ),
    }


def _partial_signal(
    primary: Mapping[str, object],
    replica: Mapping[str, object],
    frozen: Mapping[str, object],
    domain: str,
) -> dict[str, bool]:
    p = _domain_metrics(primary, domain)
    r = _domain_metrics(replica, domain)
    f = _domain_metrics(frozen, domain)
    return {
        "primary_k4_gain": (
            p["definition_k4_final"] - f["definition_k4_final"] >= 0.10
        ),
        "primary_k16_gain": (
            p["definition_k16_final"] - f["definition_k16_final"] >= 0.08
        ),
        "replica_k4_gain": (
            r["definition_k4_final"] - f["definition_k4_final"] >= 0.07
        ),
    }


def semantic_alignment_verdict(
    results: Mapping[str, Mapping[str, object]],
) -> tuple[str, dict[str, object]]:
    if set(results) != set(CANDIDATES):
        raise ValueError("W9 verdict requires exact candidate set")

    frozen = results["frozen-w6e-control"]
    projection = results["projection-semantic-control"]
    shared = results["shared-bridge-semantic-control"]
    primary = results["asymmetric-bridge-primary"]
    replica = results["asymmetric-bridge-replica"]

    details: dict[str, object] = {
        "primary_absolute": {},
        "replica": {},
        "architecture_causal": {},
        "projection_control": {},
        "shared_control": {},
        "partial": {},
    }

    full_primary = True
    full_replica = True
    full_causal = True
    projection_rescue = True
    shared_rescue = True
    partial = True
    asym_not_far_below_projection = True
    asym_not_far_below_shared = True

    for domain in ("BD", "BE"):
        absolute = primary_absolute_gates(primary, frozen, domain)
        replica_gate = replica_gates(replica, domain)
        causal = architecture_causal_gates(
            primary,
            projection,
            shared,
            domain,
        )
        projection_gate = control_rescue_gates(
            projection,
            frozen,
            domain,
        )
        shared_gate = control_rescue_gates(
            shared,
            frozen,
            domain,
        )
        partial_gate = _partial_signal(
            primary,
            replica,
            frozen,
            domain,
        )

        details["primary_absolute"][domain] = absolute
        details["replica"][domain] = replica_gate
        details["architecture_causal"][domain] = causal
        details["projection_control"][domain] = projection_gate
        details["shared_control"][domain] = shared_gate
        details["partial"][domain] = partial_gate

        full_primary = full_primary and all(absolute.values())
        full_replica = full_replica and all(replica_gate.values())
        full_causal = full_causal and all(causal.values())
        projection_rescue = projection_rescue and all(
            projection_gate.values()
        )
        shared_rescue = shared_rescue and all(shared_gate.values())
        partial = partial and all(partial_gate.values())

        p = _domain_metrics(primary, domain)
        q = _domain_metrics(projection, domain)
        s = _domain_metrics(shared, domain)
        asym_not_far_below_projection = (
            asym_not_far_below_projection
            and p["definition_k4_final"] >= q["definition_k4_final"] - 0.05
        )
        asym_not_far_below_shared = (
            asym_not_far_below_shared
            and p["definition_k4_final"] >= s["definition_k4_final"] - 0.05
        )

    details["full_primary"] = full_primary
    details["full_replica"] = full_replica
    details["full_causal"] = full_causal
    details["projection_rescue"] = projection_rescue
    details["shared_rescue"] = shared_rescue

    if full_primary and full_replica and full_causal:
        return "ASYMMETRIC_SEMANTIC_BRIDGE_RESCUE", details

    control_rescue = (
        projection_rescue and asym_not_far_below_projection
    ) or (
        shared_rescue and asym_not_far_below_shared
    )
    if control_rescue:
        details["rescuing_controls"] = [
            name
            for name, active in (
                ("projection-semantic-control", projection_rescue),
                ("shared-bridge-semantic-control", shared_rescue),
            )
            if active
        ]
        return "SEMANTIC_ALIGNMENT_RESCUE_NO_ASYMMETRY", details

    if partial:
        return "SEMANTIC_ALIGNMENT_PARTIAL", details

    return "SEMANTIC_ALIGNMENT_FAIL", details
