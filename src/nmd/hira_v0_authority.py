from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .contracts import LogicalOption

DOMAINS = ("EW", "EX", "EY", "EZ")
DOMAIN_SEEDS = {
    "EW": 481101,
    "EX": 481107,
    "EY": 481119,
    "EZ": 481131,
}
DOMAIN_CONTEXT = {
    "EW": "municipal library-access restoration review",
    "EX": "nonprofit meal-delivery routing incident review",
    "EY": "university equipment-loan service review",
    "EZ": "regional transit-pass support review",
}

FACTOR_IDS = ("F0", "F1", "F2")
PRIMITIVES = ("choice", "score", "noul")

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
        "the standard operating path stays usable without a meaningful detour",
        "routine handling continues through the normal route rather than a workaround",
    ),
    1: (
        "the standard operating path is broken enough that people must switch to a substantive alternate route",
        "routine handling is materially interrupted and users must change how the task is carried out",
    ),
}
F1_FRAGMENTS = {
    0: (
        "the central service function is still available for the important task",
        "the main capability remains usable even though secondary friction may exist",
        "people can still complete the core service objective with the important function intact",
    ),
    1: (
        "the central service function is unavailable and the important task cannot be completed normally",
        "a key capability has failed enough to block the main service objective",
        "people cannot complete the core task because an important function is no longer available",
    ),
}
U_FRAGMENTS = {
    0: (
        "response work may begin after a brief coordination window without violating the timing requirement",
        "a short wait before intervention starts remains acceptable",
    ),
    1: (
        "response work has to begin at once because even a short scheduling delay is unacceptable",
        "the timing window has collapsed and intervention must start immediately",
    ),
}
C_FRAGMENTS = {
    0: (
        "a brief wait does not itself cause a severe near-term consequence",
        "postponing action for a short interval does not trigger immediate serious harm",
    ),
    1: (
        "a brief wait itself would cause a severe near-term consequence",
        "postponing action for a short interval would trigger immediate serious harm",
    ),
}

FACTOR_VIEW_TEXTS = {
    "F0": {
        0: (
            "The ordinary operating path remains practically usable.",
            "No substantive alternate workflow is required.",
            "People can continue the task through the expected route.",
        ),
        1: (
            "The ordinary operating path is materially disrupted.",
            "A substantive alternate workflow or workaround is required.",
            "People must materially change how the task is carried out.",
        ),
    },
    "F1": {
        0: (
            "The important service capability remains available.",
            "The central task can still rely on its key function.",
            "Major functional capacity is retained.",
        ),
        1: (
            "An important service capability has been lost.",
            "The central task is blocked because a key function is unavailable.",
            "Major functional capacity is no longer retained.",
        ),
    },
    "F2": {
        0: (
            "The situation is not an immediate critical no-delay condition.",
            "Immediate criticality is absent because no-delay timing and serious consequence are not both present.",
            "The case does not simultaneously require action now and carry serious harm from waiting.",
        ),
        1: (
            "The situation is an immediate critical no-delay condition.",
            "Immediate criticality is present because action cannot wait and delay causes serious near-term harm.",
            "The case simultaneously requires action now and carries serious harm from waiting.",
        ),
    },
}

QUESTION_TEXT = {
    "choice": {
        "F0": "Which workflow-disruption condition applies?",
        "F1": "Which functional-capability condition applies?",
        "F2": "Which immediate-criticality condition applies?",
    },
    "score": {
        "F0": "What numeric workflow-disruption state applies?",
        "F1": "What numeric functional-loss state applies?",
        "F2": "What numeric immediate-criticality state applies?",
    },
    "noul": {
        "F0": "Is meaningful workflow disruption present?",
        "F1": "Is major functional loss present?",
        "F2": "Is immediate criticality present?",
    },
}

REFERENCE_HYPOTHESES = {
    "F0": (
        "The normal operating route remains practically usable without a substantive workaround.",
        "The normal operating route is materially disrupted and requires a substantive workaround or alternate path.",
    ),
    "F1": (
        "The key service capability remains available for the central task.",
        "A key service capability is unavailable enough to block the central task.",
    ),
    "U": (
        "A brief coordination delay remains acceptable before response work begins.",
        "Response work must begin immediately and cannot tolerate a short delay.",
    ),
    "C": (
        "Waiting briefly does not itself cause a severe near-term consequence.",
        "Waiting briefly itself causes a severe near-term consequence.",
    ),
    "F2": (
        "The case is not an immediate critical no-delay condition.",
        "The case is an immediate critical no-delay condition.",
    ),
}


@dataclass(frozen=True)
class HiraV0Case:
    case_id: str
    domain_id: str
    severity: int
    variant: int
    factor_vector: tuple[int, int, int]
    evidence_vector: tuple[int, int, int, int]
    state_text: str


def compose_f2(u: int, c: int) -> int:
    return int(bool(int(u)) and bool(int(c)))


def compose_severity(factors: Iterable[int]) -> int | None:
    vector = tuple(int(x) for x in factors)
    inverse = {value: key for key, value in SEVERITY_FACTORS.items()}
    return inverse.get(vector)


def factor_options(factor_id: str) -> tuple[LogicalOption, LogicalOption]:
    if factor_id not in FACTOR_IDS:
        raise ValueError(f"unknown W29 factor: {factor_id}")
    out = []
    for value in (0, 1):
        views = FACTOR_VIEW_TEXTS[factor_id][value]
        out.append(
            LogicalOption(
                option_id=f"{factor_id.lower()}-semantic-{value}",
                criterion_text=views[0],
                aliases=(views[1],),
                exemplars=(views[2],),
                value=float(value),
            )
        )
    return tuple(out)


def _phrases_for_severity(severity: int) -> tuple[str, ...]:
    evidence = SEVERITY_EVIDENCE[int(severity)]
    phrases = []
    for f0, f1, u, c in product(
        F0_FRAGMENTS[evidence["F0"]],
        F1_FRAGMENTS[evidence["F1"]],
        U_FRAGMENTS[evidence["U"]],
        C_FRAGMENTS[evidence["C"]],
    ):
        phrases.append(f"{f0}; {f1}; {u}; and {c}.")
    if len(phrases) != 24 or len(set(phrases)) != 24:
        raise RuntimeError("W29 requires exactly 24 unique variants/severity")
    return tuple(phrases)


SEVERITY_VARIANTS = tuple(_phrases_for_severity(s) for s in range(4))


def generate_w29_domains(domains: Iterable[str] = DOMAINS) -> tuple[HiraV0Case, ...]:
    rows: list[HiraV0Case] = []
    for domain_id in tuple(domains):
        if domain_id not in DOMAIN_CONTEXT:
            raise ValueError(f"unknown W29 domain: {domain_id}")
        context = DOMAIN_CONTEXT[domain_id]
        for severity in range(4):
            factor_vector = SEVERITY_FACTORS[severity]
            evidence = SEVERITY_EVIDENCE[severity]
            if factor_vector[2] != compose_f2(evidence["U"], evidence["C"]):
                raise RuntimeError("W29 F2 composition changed")
            for variant, phrase in enumerate(SEVERITY_VARIANTS[severity]):
                rows.append(
                    HiraV0Case(
                        case_id=f"{domain_id.lower()}-s{severity}-v{variant}",
                        domain_id=domain_id,
                        severity=severity,
                        variant=variant,
                        factor_vector=factor_vector,
                        evidence_vector=(
                            evidence["F0"],
                            evidence["F1"],
                            evidence["U"],
                            evidence["C"],
                        ),
                        state_text=f"Within the {context}, {phrase}",
                    )
                )

    for domain_id in tuple(domains):
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != 96:
            raise RuntimeError("W29 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W29 severity balance changed")
    return tuple(rows)


def all_w29_query_texts() -> set[str]:
    return {row.state_text for row in generate_w29_domains()}


def all_w29_schema_texts() -> set[str]:
    values = set(REFERENCE_HYPOTHESES["F0"])
    for options in REFERENCE_HYPOTHESES.values():
        values.update(options)
    for factor in FACTOR_IDS:
        for value in (0, 1):
            values.update(FACTOR_VIEW_TEXTS[factor][value])
    for primitive in PRIMITIVES:
        values.update(QUESTION_TEXT[primitive].values())
    return values


def all_w29_text_atoms() -> set[str]:
    return all_w29_query_texts() | all_w29_schema_texts()


__all__ = [
    "DOMAINS",
    "DOMAIN_CONTEXT",
    "DOMAIN_SEEDS",
    "FACTOR_IDS",
    "FACTOR_VIEW_TEXTS",
    "HiraV0Case",
    "PRIMITIVES",
    "QUESTION_TEXT",
    "REFERENCE_HYPOTHESES",
    "SEVERITY_EVIDENCE",
    "SEVERITY_FACTORS",
    "SEVERITY_VARIANTS",
    "all_w29_query_texts",
    "all_w29_schema_texts",
    "all_w29_text_atoms",
    "compose_f2",
    "compose_severity",
    "factor_options",
    "generate_w29_domains",
]
