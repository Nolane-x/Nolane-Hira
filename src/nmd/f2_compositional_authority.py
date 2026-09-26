from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

DOMAIN_SEEDS = {
    "EJ": 461101,
    "EK": 461107,
    "EL": 461119,
    "EM": 461131,
}
DOMAINS = ("EJ", "EK", "EL", "EM")
DOMAIN_CONTEXT = {
    "EJ": "municipal senior-transport scheduling incident review",
    "EK": "nonprofit food-pantry pickup coordination review",
    "EL": "university exam-access support incident review",
    "EM": "regional public-works dispatch support review",
}
CASES_PER_DOMAIN = 96

ATOM_IDS = ("U", "C")
TASK_IDS = ("U", "C", "F2")

TIMING_CLAUSES = {
    0: (
        "The case can remain in the normal response queue for a short interval without requiring action to start immediately.",
        "A brief wait before intervention is acceptable, so prompt handling is sufficient rather than must-act-now escalation.",
        "The situation allows a limited response window and does not require work to begin this instant.",
        "Ordinary prioritized handling is fast enough because a short delay remains operationally tolerable.",
        "The response can start after a brief interval without violating the timing needs of the situation.",
        "There is still enough time for normal coordination before intervention begins.",
    ),
    1: (
        "Intervention must begin immediately because the response window is effectively zero.",
        "The situation cannot tolerate a meaningful wait, so action has to start now rather than later in the queue.",
        "The timing requirement is immediate and work must begin without a short scheduling delay.",
        "A no-delay response is required, with intervention starting at once.",
        "The available response window has collapsed to the present moment and action must start now.",
        "Even a brief wait is unacceptable under the timing constraints, so intervention must begin immediately.",
    ),
}

CONSEQUENCE_CLAUSES = {
    0: (
        "Waiting briefly does not create a serious near-term consequence beyond the existing inconvenience.",
        "A short delay does not materially worsen the situation or create an immediate severe outcome.",
        "There is no serious consequence expected to occur in the near term solely because intervention waits briefly.",
        "The condition remains bounded during a short wait and does not produce an immediate major harm from delay.",
    ),
    1: (
        "Waiting briefly would create a serious near-term consequence that is not present if action starts now.",
        "A short delay would materially worsen the situation and trigger an immediate severe outcome.",
        "Serious consequences would occur in the near term specifically because intervention was postponed.",
        "The condition will produce an immediate major harm if action is delayed even briefly.",
    ),
}

REFERENCE_HYPOTHESES = {
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
class F2AuthorityCase:
    case_id: str
    domain_id: str
    timing_atom: int
    consequence_atom: int
    f2: int
    timing_variant: int
    consequence_variant: int
    text: str


def compose_f2(timing_atom: int, consequence_atom: int) -> int:
    return int(bool(int(timing_atom)) and bool(int(consequence_atom)))


def generate_w27_domains(domains: Iterable[str] = DOMAINS) -> tuple[F2AuthorityCase, ...]:
    rows: list[F2AuthorityCase] = []
    for domain_id in tuple(domains):
        if domain_id not in DOMAIN_CONTEXT:
            raise ValueError(f"unknown W27 domain: {domain_id}")
        context = DOMAIN_CONTEXT[domain_id]
        for timing_atom, consequence_atom in product((0, 1), repeat=2):
            f2 = compose_f2(timing_atom, consequence_atom)
            variant = 0
            for timing_variant, timing_clause in enumerate(TIMING_CLAUSES[timing_atom]):
                for consequence_variant, consequence_clause in enumerate(
                    CONSEQUENCE_CLAUSES[consequence_atom]
                ):
                    rows.append(
                        F2AuthorityCase(
                            case_id=(
                                f"{domain_id.lower()}-u{timing_atom}-c{consequence_atom}"
                                f"-t{timing_variant}-k{consequence_variant}"
                            ),
                            domain_id=domain_id,
                            timing_atom=timing_atom,
                            consequence_atom=consequence_atom,
                            f2=f2,
                            timing_variant=timing_variant,
                            consequence_variant=consequence_variant,
                            text=(
                                f"Within the {context}, {timing_clause} "
                                f"{consequence_clause}"
                            ),
                        )
                    )
                    variant += 1
            if variant != 24:
                raise RuntimeError("W27 requires exactly 24 variants/evidence cell")

    for domain_id in tuple(domains):
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != CASES_PER_DOMAIN:
            raise RuntimeError("W27 requires exactly 96 cases/domain")
        cells = {
            (u, c): sum(
                row.timing_atom == u and row.consequence_atom == c for row in subset
            )
            for u, c in product((0, 1), repeat=2)
        }
        if any(count != 24 for count in cells.values()):
            raise RuntimeError("W27 evidence-cell balance changed")
        if sum(row.f2 == 1 for row in subset) != 24:
            raise RuntimeError("W27 F2 positive count changed")
    return tuple(rows)


def all_w27_query_texts() -> set[str]:
    return {row.text for row in generate_w27_domains()}


def all_w27_schema_texts() -> set[str]:
    values: set[str] = set()
    for options in REFERENCE_HYPOTHESES.values():
        values.update(options)
    return values


def all_w27_text_atoms() -> set[str]:
    return all_w27_query_texts() | all_w27_schema_texts()


__all__ = [
    "ATOM_IDS",
    "CASES_PER_DOMAIN",
    "CONSEQUENCE_CLAUSES",
    "DOMAIN_CONTEXT",
    "DOMAIN_SEEDS",
    "DOMAINS",
    "F2AuthorityCase",
    "REFERENCE_HYPOTHESES",
    "TASK_IDS",
    "TIMING_CLAUSES",
    "all_w27_query_texts",
    "all_w27_schema_texts",
    "all_w27_text_atoms",
    "compose_f2",
    "generate_w27_domains",
]
