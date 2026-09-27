from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .contracts import LogicalOption

FACTOR_IDS = ("F0", "F1", "F2")
PRIMITIVES = ("choice", "score", "noul")

PARTITION_DOMAINS = {
    "qualification": ("RA", "RB"),
    "train": ("RC", "RD", "RE", "RF"),
    "dev": ("RG",),
    "confirm": ("RH", "RI"),
}

DOMAIN_CONTEXT = {
    "RA": "district heritage-pass assistance desk",
    "RB": "regional community-course enrollment desk",
    "RC": "city pet-license renewal service",
    "RD": "county sports-facility booking service",
    "RE": "public ferry-card replacement desk",
    "RF": "university equipment-loan support desk",
    "RG": "municipal market-stall permit service",
    "RH": "regional public-wifi account support desk",
    "RI": "community meal-delivery volunteer coordination desk",
}

DOMAIN_STYLE = {
    "RA": "qual_ra",
    "RB": "qual_rb",
    "RC": "train_rc",
    "RD": "train_rd",
    "RE": "train_re",
    "RF": "train_rf",
    "RG": "dev_rg",
    "RH": "confirm_rh",
    "RI": "confirm_ri",
}

STYLE_TERMS = {
    "qual_ra": ("routine service lane", "replacement handling route", "decisive operating function", "requested service result", "brief intake pause", "response allowance", "serious short-horizon consequence"),
    "qual_rb": ("established assistance lane", "substitute handling route", "indispensable enabling function", "central assistance result", "short coordination interval", "action allowance", "grave near-term consequence"),
    "train_rc": ("ordinary processing lane", "alternate processing route", "key enabling function", "main processing result", "small preparation pause", "required action window", "severe immediate consequence"),
    "train_rd": ("standard service lane", "material fallback route", "critical service function", "principal service result", "brief scheduling pause", "binding response window", "grave proximate consequence"),
    "train_re": ("normal handling lane", "replacement execution route", "pivotal operational function", "primary handling result", "short setup pause", "permitted response window", "serious immediate consequence"),
    "train_rf": ("customary workflow lane", "substitute workflow route", "essential working function", "core workflow result", "brief handoff pause", "mandatory action window", "grave short-horizon consequence"),
    "dev_rg": ("regular operating lane", "material alternate route", "decisive enabling function", "principal operating result", "small staging pause", "applicable response window", "severe near-term consequence"),
    "confirm_rh": ("primary service lane", "substitute service route", "critical enabling function", "central service result", "brief readiness pause", "enforced action window", "grave immediate consequence"),
    "confirm_ri": ("normal delivery lane", "material replacement route", "pivotal delivery function", "main delivery result", "short dispatch pause", "required response window", "serious imminent consequence"),
}

SEVERITY_FACTORS = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (1, 1, 0),
    3: (1, 1, 1),
}
SEVERITY_EVIDENCE = {
    0: {"F0": 0, "F1": 0, "U": 0, "C": 0},
    1: {"F0": 1, "F1": 0, "U": 1, "C": 0},
    2: {"F0": 1, "F1": 1, "U": 0, "C": 1},
    3: {"F0": 1, "F1": 1, "U": 1, "C": 1},
}


def _terms(style: str):
    route, alternate, capability, objective, pause, timing, harm = STYLE_TERMS[style]
    return {
        "route": route,
        "alternate": alternate,
        "capability": capability,
        "objective": objective,
        "pause": pause,
        "timing": timing,
        "harm": harm,
    }


def _factor_fragments(style: str, factor: str, value: int) -> tuple[str, ...]:
    t = _terms(style)
    if factor == "F0":
        if value == 0:
            return (
                f"the {t['route']} remains workable without moving to the {t['alternate']}",
                f"the task can remain inside the {t['route']} rather than switch to the {t['alternate']}",
            )
        return (
            f"the {t['route']} is no longer workable unless the task moves to the {t['alternate']}",
            f"the task must leave the {t['route']} and switch to the {t['alternate']}",
        )
    if factor == "F1":
        if value == 0:
            return (
                f"the {t['capability']} remains available for the {t['objective']}",
                f"the {t['objective']} stays achievable because the {t['capability']} is intact",
                f"the service still provides the {t['capability']} needed for the {t['objective']}",
            )
        return (
            f"the {t['capability']} is unavailable for the {t['objective']}",
            f"the {t['objective']} is blocked because the {t['capability']} is no longer intact",
            f"the service no longer provides the {t['capability']} needed for the {t['objective']}",
        )
    if factor == "U":
        if value == 0:
            return (
                f"the {t['timing']} still permits a {t['pause']} before action begins",
                f"action may start after a {t['pause']} without missing the {t['timing']}",
            )
        return (
            f"the {t['timing']} permits no {t['pause']} before action begins",
            f"action must start now because a {t['pause']} would miss the {t['timing']}",
        )
    if factor == "C":
        if value == 0:
            return (
                f"a {t['pause']} would not by itself cause a {t['harm']}",
                f"waiting through the {t['pause']} alone does not create a {t['harm']}",
            )
        return (
            f"a {t['pause']} would by itself cause a {t['harm']}",
            f"waiting through the {t['pause']} alone creates a {t['harm']}",
        )
    raise ValueError(f"unknown W32 evidence factor: {factor}")


def _schema_views(style: str, factor: str, value: int) -> tuple[str, str, str]:
    t = _terms(style)
    if factor == "F0":
        return (
            (
                f"The {t['route']} remains practically usable.",
                f"No switch to the {t['alternate']} is required.",
                f"Execution can stay on the {t['route']}.",
            )
            if value == 0
            else (
                f"The {t['route']} is no longer practically usable.",
                f"A switch to the {t['alternate']} is required.",
                f"Execution must leave the {t['route']}.",
            )
        )
    if factor == "F1":
        return (
            (
                f"The {t['capability']} remains present.",
                f"The {t['objective']} still has the function it requires.",
                f"Functional support for the {t['objective']} is retained.",
            )
            if value == 0
            else (
                f"The {t['capability']} is absent.",
                f"The {t['objective']} has lost the function it requires.",
                f"Functional support for the {t['objective']} is lost.",
            )
        )
    if factor == "F2":
        return (
            (
                f"The case does not combine zero waiting time with a {t['harm']}.",
                f"The {t['timing']} and delay consequence do not jointly create critical immediacy.",
                "The immediate-critical conjunction is not satisfied.",
            )
            if value == 0
            else (
                f"The case combines zero waiting time with a {t['harm']}.",
                f"The {t['timing']} and delay consequence jointly create critical immediacy.",
                "The immediate-critical conjunction is satisfied.",
            )
        )
    raise ValueError(f"unknown W32 factor: {factor}")


QUESTION_TEXT = {
    "choice": {
        "F0": "Which workflow-continuity condition is supported by the case?",
        "F1": "Which enabling-function condition is supported by the case?",
        "F2": "Which combined immediacy condition is supported by the case?",
    },
    "score": {
        "F0": "What numeric workflow-continuity condition is supported?",
        "F1": "What numeric enabling-function condition is supported?",
        "F2": "What numeric combined-immediacy condition is supported?",
    },
    "noul": {
        "F0": "Does the case require a materially different workflow route?",
        "F1": "Has the required enabling function been lost?",
        "F2": "Are zero-delay timing and severe delay consequence jointly present?",
    },
}

REFERENCE_HYPOTHESES = {
    "F0": (
        "The normal work route can still be used without adopting a materially different route.",
        "The normal work route cannot be used unless a materially different route replaces it.",
    ),
    "F1": (
        "The function necessary for the main result remains available.",
        "The function necessary for the main result is unavailable and prevents completion.",
    ),
    "U": (
        "The response timing still allows a brief interval before action starts.",
        "The response timing requires action to start immediately with no brief interval.",
    ),
    "C": (
        "A brief wait alone does not cause a severe consequence in the near term.",
        "A brief wait alone causes a severe consequence in the near term.",
    ),
    "F2": (
        "Immediate criticality is absent because no-delay timing and severe delay harm are not both present.",
        "Immediate criticality is present because no-delay timing and severe delay harm are both present.",
    ),
}


@dataclass(frozen=True)
class W32TransferCase:
    case_id: str
    domain_id: str
    partition: str
    style_id: str
    severity: int
    variant: int
    factor_vector: tuple[int, int, int]
    evidence_vector: tuple[int, int, int, int]
    state_text: str


def partition_for_domain(domain_id: str) -> str:
    for partition, domains in PARTITION_DOMAINS.items():
        if domain_id in domains:
            return partition
    raise ValueError(f"unknown W32 domain: {domain_id}")


def compose_f2(u: int, c: int) -> int:
    return int(bool(int(u)) and bool(int(c)))


def compose_severity(factors: Iterable[int]) -> int | None:
    vector = tuple(int(x) for x in factors)
    return {value: key for key, value in SEVERITY_FACTORS.items()}.get(vector)


def factor_options(domain_id: str, factor_id: str) -> tuple[LogicalOption, LogicalOption]:
    if domain_id not in DOMAIN_STYLE:
        raise ValueError(f"unknown W32 domain: {domain_id}")
    if factor_id not in FACTOR_IDS:
        raise ValueError(f"unknown W32 factor: {factor_id}")
    style = DOMAIN_STYLE[domain_id]
    options = []
    for value in (0, 1):
        views = _schema_views(style, factor_id, value)
        options.append(
            LogicalOption(
                option_id=f"{factor_id.lower()}32-{value}",
                criterion_text=views[0],
                aliases=(views[1],),
                exemplars=(views[2],),
                value=float(value),
            )
        )
    return tuple(options)


def _phrases(style: str, severity: int) -> tuple[str, ...]:
    evidence = SEVERITY_EVIDENCE[int(severity)]
    rows = tuple(
        f"{f0}; {f1}; {u}; and {c}."
        for f0, f1, u, c in product(
            _factor_fragments(style, "F0", evidence["F0"]),
            _factor_fragments(style, "F1", evidence["F1"]),
            _factor_fragments(style, "U", evidence["U"]),
            _factor_fragments(style, "C", evidence["C"]),
        )
    )
    if len(rows) != 24 or len(set(rows)) != 24:
        raise RuntimeError("W32 requires exactly 24 unique phrases/severity/style")
    return rows


def generate_w32_domains(domains: Iterable[str]) -> tuple[W32TransferCase, ...]:
    rows: list[W32TransferCase] = []
    for domain_id in tuple(domains):
        if domain_id not in DOMAIN_CONTEXT:
            raise ValueError(f"unknown W32 domain: {domain_id}")
        partition = partition_for_domain(domain_id)
        style = DOMAIN_STYLE[domain_id]
        for severity in range(4):
            vector = SEVERITY_FACTORS[severity]
            evidence = SEVERITY_EVIDENCE[severity]
            if vector[2] != compose_f2(evidence["U"], evidence["C"]):
                raise RuntimeError("W32 F2 composition changed")
            for variant, phrase in enumerate(_phrases(style, severity)):
                rows.append(
                    W32TransferCase(
                        case_id=f"{domain_id.lower()}-s{severity}-v{variant}",
                        domain_id=domain_id,
                        partition=partition,
                        style_id=style,
                        severity=severity,
                        variant=variant,
                        factor_vector=vector,
                        evidence_vector=(
                            evidence["F0"],
                            evidence["F1"],
                            evidence["U"],
                            evidence["C"],
                        ),
                        state_text=f"At the {DOMAIN_CONTEXT[domain_id]}, {phrase}",
                    )
                )

    for domain_id in tuple(domains):
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != 96:
            raise RuntimeError("W32 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W32 severity balance changed")
    return tuple(rows)


def generate_w32_partition(partition: str) -> tuple[W32TransferCase, ...]:
    if partition not in PARTITION_DOMAINS:
        raise ValueError(f"unknown W32 partition: {partition}")
    return generate_w32_domains(PARTITION_DOMAINS[partition])


def all_w32_query_texts() -> set[str]:
    values = set()
    for partition in PARTITION_DOMAINS:
        values.update(row.state_text for row in generate_w32_partition(partition))
    return values


def all_w32_schema_texts() -> set[str]:
    values = set()
    for style in STYLE_TERMS:
        for factor in FACTOR_IDS:
            for value in (0, 1):
                values.update(_schema_views(style, factor, value))
    for primitive in QUESTION_TEXT.values():
        values.update(primitive.values())
    for hypotheses in REFERENCE_HYPOTHESES.values():
        values.update(hypotheses)
    return values


def all_w32_text_atoms() -> set[str]:
    return all_w32_query_texts() | all_w32_schema_texts()


__all__ = [
    "DOMAIN_CONTEXT",
    "DOMAIN_STYLE",
    "FACTOR_IDS",
    "PARTITION_DOMAINS",
    "PRIMITIVES",
    "QUESTION_TEXT",
    "REFERENCE_HYPOTHESES",
    "SEVERITY_EVIDENCE",
    "SEVERITY_FACTORS",
    "STYLE_TERMS",
    "W32TransferCase",
    "all_w32_query_texts",
    "all_w32_schema_texts",
    "all_w32_text_atoms",
    "compose_f2",
    "compose_severity",
    "factor_options",
    "generate_w32_domains",
    "generate_w32_partition",
    "partition_for_domain",
]
