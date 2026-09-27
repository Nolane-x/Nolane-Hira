from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

M2_SEMANTIC_PARTITIONS = {
    "train": {
        4: "HKA",
        8: "HKB",
        16: "HKC",
        32: "HKD",
    },
    "dev": {
        64: "HKE",
        128: "HKF",
    },
    "confirm": {
        255: "HKG",
    },
}

M2_SEMANTIC_CASES = {
    "train": 16,
    "dev": 16,
    "confirm": 24,
}

M2_SEMANTIC_SEEDS = {
    "HKA": 21401,
    "HKB": 21402,
    "HKC": 21403,
    "HKD": 21404,
    "HKE": 21405,
    "HKF": 21406,
    "HKG": 21407,
}

AREAS = (
    "harbor",
    "orchard",
    "library",
    "workshop",
    "terrace",
    "station",
    "gallery",
    "market",
    "garden",
    "archive",
    "marina",
    "courtyard",
    "studio",
    "pavilion",
    "depot",
    "meadow",
    "atrium",
)

OBJECTS = (
    "permit",
    "parcel",
    "badge",
    "record",
    "key",
    "pass",
    "locker",
    "cart",
    "folder",
    "token",
    "ticket",
    "crate",
    "label",
    "voucher",
    "kit",
    "card",
    "packet",
)

ACTIONS = (
    "register",
    "deliver",
    "renew",
    "inspect",
    "reserve",
    "return",
    "verify",
    "collect",
    "transfer",
    "issue",
    "review",
    "store",
    "release",
    "route",
    "check",
    "prepare",
)

QUALIFIERS = (
    "priority",
    "standard",
    "supervised",
    "scheduled",
    "verified",
    "temporary",
    "public",
    "staff",
)


@dataclass(frozen=True)
class M2HighKSemanticCase:
    case_id: str
    partition: str
    domain_id: str
    k: int
    state_text: str
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int


def _concept_space() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (area, obj, action, qualifier)
        for area in AREAS
        for obj in OBJECTS
        for action in ACTIONS
        for qualifier in QUALIFIERS
    )


CONCEPT_SPACE = _concept_space()


def _criterion(concept: tuple[str, str, str, str]) -> str:
    area, obj, action, qualifier = concept
    return (
        f"Route requests that {action} the {obj} in the {area} "
        f"under {qualifier} handling."
    )


def _alias(concept: tuple[str, str, str, str]) -> str:
    area, obj, action, qualifier = concept
    return f"{qualifier} {area} lane for {action} {obj}"


def _exemplar(concept: tuple[str, str, str, str]) -> str:
    area, obj, action, qualifier = concept
    return (
        f"A request needs to {action} a {obj} at the {area} "
        f"with {qualifier} processing."
    )


def _state(
    concept: tuple[str, str, str, str],
    *,
    style: int,
    case_index: int,
) -> str:
    area, obj, action, qualifier = concept
    if style == 0:
        return (
            f"Fresh routing record {case_index:02d}: the request needs to "
            f"{action} the {obj} in the {area}, using {qualifier} handling."
        )
    if style == 1:
        return (
            f"Routing note {case_index:02d} assigns {qualifier} processing "
            f"for a request to {action} a {obj} at the {area}."
        )
    return (
        f"Service record {case_index:02d} concerns the {area}. "
        f"It explicitly asks staff to {action} the {obj}; "
        f"the handling class is {qualifier}."
    )


def generate_m2_high_k_semantic(
    partition: str,
    *,
    allow_sealed: bool = False,
) -> tuple[M2HighKSemanticCase, ...]:
    if partition not in M2_SEMANTIC_PARTITIONS:
        raise ValueError(f"unknown M2 semantic partition: {partition}")
    if partition == "confirm" and not allow_sealed:
        raise RuntimeError("M2 K=255 semantic confirmation remains sealed")

    rows: list[M2HighKSemanticCase] = []
    cases_per_k = M2_SEMANTIC_CASES[partition]

    for k, domain_id in M2_SEMANTIC_PARTITIONS[partition].items():
        rng = random.Random(M2_SEMANTIC_SEEDS[domain_id])
        for case_index in range(cases_per_k):
            indices = rng.sample(range(len(CONCEPT_SPACE)), k)
            concepts = [CONCEPT_SPACE[index] for index in indices]
            gold_index = rng.randrange(k)
            gold = concepts[gold_index]

            options = tuple(
                LogicalOption(
                    option_id=f"{domain_id.lower()}-route-{index:03d}",
                    criterion_text=_criterion(concept),
                    aliases=(_alias(concept),),
                    exemplars=(_exemplar(concept),),
                )
                for index, concept in enumerate(concepts)
            )
            rows.append(
                M2HighKSemanticCase(
                    case_id=(
                        f"m2-{partition}-{domain_id.lower()}-"
                        f"k{k}-{case_index:02d}"
                    ),
                    partition=partition,
                    domain_id=domain_id,
                    k=k,
                    state_text=_state(
                        gold,
                        style=case_index % 3,
                        case_index=case_index,
                    ),
                    question_text=(
                        "Which routing criterion best matches the service record?"
                    ),
                    options=options,
                    gold_index=gold_index,
                )
            )
    return tuple(rows)


def all_m2_high_k_text_atoms(*, include_sealed: bool = True) -> set[str]:
    values: set[str] = set()
    for partition in M2_SEMANTIC_PARTITIONS:
        if partition == "confirm" and not include_sealed:
            continue
        rows = generate_m2_high_k_semantic(
            partition,
            allow_sealed=partition == "confirm",
        )
        for row in rows:
            values.add(row.state_text)
            values.add(row.question_text)
            for option in row.options:
                values.add(option.criterion_text)
                values.update(option.aliases)
                values.update(option.exemplars)
    return values


__all__ = [
    "ACTIONS",
    "AREAS",
    "M2HighKSemanticCase",
    "M2_SEMANTIC_CASES",
    "M2_SEMANTIC_PARTITIONS",
    "M2_SEMANTIC_SEEDS",
    "OBJECTS",
    "QUALIFIERS",
    "all_m2_high_k_text_atoms",
    "generate_m2_high_k_semantic",
]
