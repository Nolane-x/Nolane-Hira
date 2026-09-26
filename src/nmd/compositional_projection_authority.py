from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .contracts import LogicalOption

FACTOR_IDS = ("F0", "F1", "F2")
REFERENCE_ATOMS = ("F0", "F1", "U", "C")
VIEW_IDS = ("D0", "D1", "D2")

PARTITION_DOMAINS = {
    "qualification": ("EN", "EO"),
    "train": ("EP", "EQ", "ER", "ES"),
    "dev": ("ET",),
    "confirm": ("EU", "EV"),
}
DOMAIN_SEEDS = {
    "EN": 471101,
    "EO": 471107,
    "EP": 472201,
    "EQ": 472207,
    "ER": 472219,
    "ES": 472231,
    "ET": 473307,
    "EU": 474401,
    "EV": 474409,
}
DOMAIN_CONTEXT = {
    "EN": "municipal emergency-housing intake support review",
    "EO": "nonprofit medication-delivery coordination review",
    "EP": "university laboratory-access support review",
    "EQ": "regional waste-transfer scheduling support",
    "ER": "public cultural-event registration incident review",
    "ES": "community utility-install appointment support",
    "ET": "municipal records-transfer support review",
    "EU": "public prescription-collection coordination review",
    "EV": "regional access-pass support incident review",
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

F0_FRAGMENTS = {
    0: (
        "ordinary workflow remains essentially unchanged",
        "people can continue through the expected process without a workaround",
    ),
    1: (
        "normal use is materially disrupted and requires a nontrivial adaptation",
        "the usual process no longer works smoothly and users must follow an alternate operational path",
    ),
}
F1_FRAGMENTS = {
    0: (
        "the important service capability remains practically available",
        "the central task can still be completed because core functionality is retained",
        "users can still depend on the main service function",
    ),
    1: (
        "an important service capability is unavailable and blocks the central task",
        "core functionality has materially failed so the main activity cannot be completed normally",
        "users can no longer rely on a key service function for the central outcome",
    ),
}
U_FRAGMENTS = {
    0: (
        "a short response delay remains acceptable before intervention begins",
        "the case can wait briefly without violating its response-time requirement",
    ),
    1: (
        "intervention must begin immediately and cannot tolerate a meaningful delay",
        "the response window is effectively zero and action has to start now",
    ),
}
C_FRAGMENTS = {
    0: (
        "waiting briefly does not create an immediate serious consequence",
        "a short wait does not trigger a severe near-term outcome",
    ),
    1: (
        "waiting briefly creates a serious near-term consequence",
        "a short delay would trigger an immediate severe outcome",
    ),
}

FACTOR_DEFINITIONS = {
    "F0": (
        (
            "Choose the workflow-intact side when the normal process remains practically unchanged and no substantive adaptation is required.",
            "This side represents an event that does not force a meaningful change in how the activity is performed.",
            "Use this side when users can continue through the expected operational route.",
        ),
        (
            "Choose the workflow-disrupted side when normal use materially changes and a workaround, alternate path, or nontrivial adjustment is required.",
            "This side represents a real operational disturbance rather than a peripheral nuisance.",
            "Use this side when users must materially adapt how they perform the activity.",
        ),
    ),
    "F1": (
        (
            "Choose the capability-retained side when the important service function remains practically usable for the central task.",
            "This side means major functionality is still available despite any disruption.",
            "Use this side when users can still depend on the core capability.",
        ),
        (
            "Choose the capability-lost side when an important function is unavailable enough to prevent the central task from being completed normally.",
            "This side means major functional loss has occurred.",
            "Use this side when users cannot rely on a key service capability.",
        ),
    ),
    "F2": (
        (
            "Choose the noncritical-timing side when immediate criticality is absent and the case does not require emergency-level action at once.",
            "This side means the situation is not simultaneously no-delay and immediately consequential.",
            "Use this side when the event is not an immediate critical response condition.",
        ),
        (
            "Choose the immediate-critical side when action cannot wait and delay would also create a serious near-term consequence.",
            "This side means both no-delay timing and immediate serious consequence are present.",
            "Use this side when the situation requires action now because waiting is seriously consequential.",
        ),
    ),
}

FACTOR_QUESTIONS = {
    "F0": "Which side best describes whether normal workflow is materially disrupted?",
    "F1": "Which side best describes whether important service capability has been lost?",
    "F2": "Which side best describes whether this is an immediate critical response condition?",
}

FACTOR_OPTIONS = {
    factor_id: tuple(
        LogicalOption(
            option_id=f"{factor_id.lower()}28-{value}",
            criterion_text=FACTOR_DEFINITIONS[factor_id][value][0],
        )
        for value in (0, 1)
    )
    for factor_id in FACTOR_IDS
}

REFERENCE_HYPOTHESES = {
    "F0": (
        "Normal workflow remains essentially intact and does not require substantive adaptation.",
        "Normal workflow is materially disrupted and requires a workaround, alternate path, or meaningful adaptation.",
    ),
    "F1": (
        "The important service capability remains practically available for the central task.",
        "An important service capability is unavailable enough to block the central task.",
    ),
    "U": (
        "A short response delay remains acceptable before intervention begins.",
        "Intervention must begin immediately and cannot tolerate a meaningful delay.",
    ),
    "C": (
        "A brief delay does not create an immediate serious consequence.",
        "A brief delay creates a serious near-term consequence.",
    ),
    "F2": (
        "The situation is not an immediate critical no-delay event.",
        "The situation is an immediate critical no-delay event.",
    ),
}


@dataclass(frozen=True)
class ProjectionCompositionCase:
    case_id: str
    domain_id: str
    partition: str
    severity: int
    variant: int
    factor_vector: tuple[int, int, int]
    evidence_vector: tuple[int, int, int, int]
    severity_field: str


def partition_for_domain(domain_id: str) -> str:
    for partition, domains in PARTITION_DOMAINS.items():
        if domain_id in domains:
            return partition
    raise ValueError(f"unknown W28 domain: {domain_id}")


def compose_f2(u: int, c: int) -> int:
    return int(bool(int(u)) and bool(int(c)))


def compose_severity(factors: Iterable[int]) -> int | None:
    vector = tuple(int(x) for x in factors)
    inverse = {value: key for key, value in SEVERITY_FACTORS.items()}
    return inverse.get(vector)


def _phrases_for_severity(severity: int) -> tuple[str, ...]:
    evidence = SEVERITY_EVIDENCE[int(severity)]
    phrases = []
    for f0, f1, u, c in product(
        F0_FRAGMENTS[evidence["F0"]],
        F1_FRAGMENTS[evidence["F1"]],
        U_FRAGMENTS[evidence["U"]],
        C_FRAGMENTS[evidence["C"]],
    ):
        phrases.append(
            f"{f0}; {f1}; {u}; and {c}."
        )
    if len(phrases) != 24 or len(set(phrases)) != 24:
        raise RuntimeError("W28 requires exactly 24 unique phrases/severity")
    return tuple(phrases)


SEVERITY_VARIANTS = tuple(_phrases_for_severity(s) for s in range(4))


def generate_w28_domains(domains: Iterable[str]) -> tuple[ProjectionCompositionCase, ...]:
    rows: list[ProjectionCompositionCase] = []
    for domain_id in tuple(domains):
        if domain_id not in DOMAIN_CONTEXT:
            raise ValueError(f"unknown W28 domain: {domain_id}")
        context = DOMAIN_CONTEXT[domain_id]
        partition = partition_for_domain(domain_id)
        for severity in range(4):
            factor_vector = SEVERITY_FACTORS[severity]
            evidence = SEVERITY_EVIDENCE[severity]
            if factor_vector[2] != compose_f2(evidence["U"], evidence["C"]):
                raise RuntimeError("W28 F2/evidence composition changed")
            for variant, phrase in enumerate(SEVERITY_VARIANTS[severity]):
                rows.append(
                    ProjectionCompositionCase(
                        case_id=f"{domain_id.lower()}-s{severity}-v{variant}",
                        domain_id=domain_id,
                        partition=partition,
                        severity=severity,
                        variant=variant,
                        factor_vector=factor_vector,
                        evidence_vector=(
                            evidence["F0"],
                            evidence["F1"],
                            evidence["U"],
                            evidence["C"],
                        ),
                        severity_field=f"Within the {context}, {phrase}",
                    )
                )
    for domain_id in tuple(domains):
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != 96:
            raise RuntimeError("W28 requires 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W28 severity balance changed")
    return tuple(rows)


def generate_w28_partition(partition: str) -> tuple[ProjectionCompositionCase, ...]:
    if partition not in PARTITION_DOMAINS:
        raise ValueError(f"unknown W28 partition: {partition}")
    return generate_w28_domains(PARTITION_DOMAINS[partition])


def all_w28_query_texts() -> set[str]:
    values: set[str] = set()
    for partition in PARTITION_DOMAINS:
        values.update(row.severity_field for row in generate_w28_partition(partition))
    return values


def all_w28_schema_texts() -> set[str]:
    values: set[str] = set()
    for factor in FACTOR_DEFINITIONS.values():
        for side in factor:
            values.update(side)
    values.update(FACTOR_QUESTIONS.values())
    for options in REFERENCE_HYPOTHESES.values():
        values.update(options)
    return values


def all_w28_text_atoms() -> set[str]:
    return all_w28_query_texts() | all_w28_schema_texts()


__all__ = [
    "C_FRAGMENTS",
    "DOMAIN_CONTEXT",
    "DOMAIN_SEEDS",
    "FACTOR_DEFINITIONS",
    "FACTOR_IDS",
    "FACTOR_OPTIONS",
    "FACTOR_QUESTIONS",
    "F0_FRAGMENTS",
    "F1_FRAGMENTS",
    "PARTITION_DOMAINS",
    "ProjectionCompositionCase",
    "REFERENCE_ATOMS",
    "REFERENCE_HYPOTHESES",
    "SEVERITY_EVIDENCE",
    "SEVERITY_FACTORS",
    "SEVERITY_VARIANTS",
    "U_FRAGMENTS",
    "VIEW_IDS",
    "all_w28_query_texts",
    "all_w28_schema_texts",
    "all_w28_text_atoms",
    "compose_f2",
    "compose_severity",
    "generate_w28_domains",
    "generate_w28_partition",
    "partition_for_domain",
]
