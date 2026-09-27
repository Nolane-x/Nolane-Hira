from __future__ import annotations

import random

from .contracts import LogicalOption
from .mainline_m2_authority import (
    CONCEPT_SPACE,
    M2HighKSemanticCase,
)

M2_R1_DEV_PARTITIONS = {
    64: "HKH",
    128: "HKI",
}

M2_R1_DEV_SEEDS = {
    "HKH": 21501,
    "HKI": 21502,
}

M2_R1_DEV_CASES_PER_K = 24


def _criterion(concept: tuple[str, str, str, str]) -> str:
    area, obj, action, qualifier = concept
    return (
        f"Handling rule: a {qualifier} request at the {area} "
        f"is the lane that must {action} the {obj}."
    )


def _alias(concept: tuple[str, str, str, str]) -> str:
    area, obj, action, qualifier = concept
    return (
        f"Choose the {area} {qualifier} route when staff must "
        f"{action} the {obj}."
    )


def _exemplar(concept: tuple[str, str, str, str]) -> str:
    area, obj, action, qualifier = concept
    return (
        f"Example: staff at the {area} need to {action} a {obj}; "
        f"the request is marked {qualifier}."
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
            f"Fresh service ticket R1-{case_index:02d}: at the {area}, "
            f"staff must {action} the {obj}. The processing label is "
            f"{qualifier}."
        )
    if style == 1:
        return (
            f"R1 routing entry {case_index:02d} names the {area} and asks "
            f"for the {obj} to be handled by the action {action}; "
            f"its class is {qualifier}."
        )
    return (
        f"New routing memo R1-{case_index:02d}: {qualifier} handling applies. "
        f"The location is the {area}, and the requested task is to "
        f"{action} the {obj}."
    )


def generate_m2_r1_dev() -> tuple[M2HighKSemanticCase, ...]:
    rows: list[M2HighKSemanticCase] = []
    for k, domain_id in M2_R1_DEV_PARTITIONS.items():
        rng = random.Random(M2_R1_DEV_SEEDS[domain_id])
        for case_index in range(M2_R1_DEV_CASES_PER_K):
            indices = rng.sample(range(len(CONCEPT_SPACE)), k)
            concepts = [CONCEPT_SPACE[index] for index in indices]
            gold_index = rng.randrange(k)
            gold = concepts[gold_index]

            options = tuple(
                LogicalOption(
                    option_id=f"{domain_id.lower()}-r1-route-{index:03d}",
                    criterion_text=_criterion(concept),
                    aliases=(_alias(concept),),
                    exemplars=(_exemplar(concept),),
                )
                for index, concept in enumerate(concepts)
            )
            rows.append(
                M2HighKSemanticCase(
                    case_id=f"m2-r1-dev-{domain_id.lower()}-k{k}-{case_index:02d}",
                    partition="r1-dev",
                    domain_id=domain_id,
                    k=k,
                    state_text=_state(
                        gold,
                        style=case_index % 3,
                        case_index=case_index,
                    ),
                    question_text=(
                        "Which handling rule exactly matches this new service ticket?"
                    ),
                    options=options,
                    gold_index=gold_index,
                )
            )
    return tuple(rows)


def all_m2_r1_dev_text_atoms() -> set[str]:
    values: set[str] = set()
    for row in generate_m2_r1_dev():
        values.add(row.state_text)
        values.add(row.question_text)
        for option in row.options:
            values.add(option.criterion_text)
            values.update(option.aliases)
            values.update(option.exemplars)
    return values


__all__ = [
    "M2_R1_DEV_CASES_PER_K",
    "M2_R1_DEV_PARTITIONS",
    "M2_R1_DEV_SEEDS",
    "all_m2_r1_dev_text_atoms",
    "generate_m2_r1_dev",
]
