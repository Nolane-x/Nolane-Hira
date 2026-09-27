from __future__ import annotations

from dataclasses import dataclass

from .contracts import LogicalOption, Primitive

R2_PARTITIONS = {
    "train": ("UJ", "UK", "UL", "UM"),
    "dev": ("UN", "UO"),
}

R2_DOMAIN_CONTEXT = {
    "UJ": "municipal reading-room access desk",
    "UK": "regional craft-market stall office",
    "UL": "county garden-plot registration counter",
    "UM": "public rehearsal-room booking service",
    "UN": "district volunteer-shift scheduling desk",
    "UO": "community equipment-label request office",
}

R2_DOMAIN_ROUTES = {
    "UJ": ("harbor lane", "copper lane", "fern lane"),
    "UK": ("meadow lane", "ink lane", "stone lane"),
    "UL": ("pine lane", "glass lane", "river lane"),
    "UM": ("clover lane", "bronze lane", "cloud lane"),
    "UN": ("heather lane", "silver lane", "oak lane"),
    "UO": ("sand lane", "indigo lane", "grove lane"),
}

R2_CONFIDENCE_BANDS = ("clear", "qualified", "sparse")
R2_CONFIDENCE_MASS = {
    "clear": 0.90,
    "qualified": 0.75,
    "sparse": 0.60,
}


@dataclass(frozen=True)
class M1R2AuthorityCase:
    case_id: str
    partition: str
    domain_id: str
    primitive: Primitive
    state_text: str
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int
    gold_probabilities: tuple[float, ...]
    confidence_band: str
    is_ood: bool = False
    ood_kind: str | None = None


def _soft_target(k: int, gold: int, mass: float) -> tuple[float, ...]:
    other = (1.0 - mass) / float(k - 1)
    return tuple(mass if index == gold else other for index in range(k))


def _options(domain: str, primitive: Primitive) -> tuple[LogicalOption, ...]:
    if primitive == "choice":
        return tuple(
            LogicalOption(
                option_id=f"{domain.lower()}-r2-route-{index}",
                criterion_text=f"The request is assigned to the {route}.",
                aliases=(f"Process this request through the {route}.",),
            )
            for index, route in enumerate(R2_DOMAIN_ROUTES[domain])
        )
    if primitive == "score":
        return (
            LogicalOption(
                option_id=f"{domain.lower()}-r2-load-low",
                criterion_text="The stated workload level is low.",
                aliases=("Workload band zero is low.",),
                value=0.0,
            ),
            LogicalOption(
                option_id=f"{domain.lower()}-r2-load-mid",
                criterion_text="The stated workload level is medium.",
                aliases=("Workload band one is medium.",),
                value=1.0,
            ),
            LogicalOption(
                option_id=f"{domain.lower()}-r2-load-high",
                criterion_text="The stated workload level is high.",
                aliases=("Workload band two is high.",),
                value=2.0,
            ),
        )
    if primitive == "noul":
        return (
            LogicalOption(
                option_id=f"{domain.lower()}-r2-check-no",
                criterion_text="A second staff check is not required.",
                aliases=("The request may proceed without a second staff check.",),
                value=0.0,
            ),
            LogicalOption(
                option_id=f"{domain.lower()}-r2-check-yes",
                criterion_text="A second staff check is required.",
                aliases=("The request must receive a second staff check.",),
                value=1.0,
            ),
        )
    raise ValueError(f"unknown R2 primitive: {primitive}")


def _question(primitive: Primitive) -> str:
    if primitive == "choice":
        return "Which handling lane is explicitly assigned in this record?"
    if primitive == "score":
        return "Which workload level is explicitly stated in this record?"
    if primitive == "noul":
        return "Does this record explicitly require a second staff check?"
    raise ValueError(f"unknown R2 primitive: {primitive}")


def _state(
    domain: str,
    primitive: Primitive,
    target: int,
    confidence: str,
    variant: int,
) -> str:
    context = R2_DOMAIN_CONTEXT[domain]
    certainty = {
        "clear": "The entry is direct and contains no competing assignment.",
        "qualified": "The entry is explicit but also notes one irrelevant administrative detail.",
        "sparse": "The entry is short and gives only the decision-bearing statement.",
    }[confidence]

    if primitive == "choice":
        routes = R2_DOMAIN_ROUTES[domain]
        fact = (
            f"The request is assigned to the {routes[target]}; "
            "the remaining listed lanes are not assigned."
        )
    elif primitive == "score":
        label = ("low", "medium", "high")[target]
        fact = (
            f"The record explicitly states that the workload level is {label}; "
            "no different workload level is stated."
        )
    elif primitive == "noul":
        fact = (
            "The record explicitly requires a second staff check before processing."
            if target == 1
            else "The record explicitly states that a second staff check is not required."
        )
    else:
        raise ValueError(f"unknown R2 primitive: {primitive}")

    return (
        f"At the {context}, fresh entry {variant:02d} records a request. "
        f"{fact} {certainty}"
    )


def _generate_domain(domain: str, partition: str) -> tuple[M1R2AuthorityCase, ...]:
    rows: list[M1R2AuthorityCase] = []
    primitives: tuple[Primitive, ...] = ("choice", "score", "noul")
    for primitive_index, primitive in enumerate(primitives):
        k = 2 if primitive == "noul" else 3
        for confidence_index, confidence in enumerate(R2_CONFIDENCE_BANDS):
            for repeat in range(4):
                target = (repeat + 2 * confidence_index + primitive_index) % k
                variant = primitive_index * 12 + confidence_index * 4 + repeat
                options = _options(domain, primitive)
                rows.append(
                    M1R2AuthorityCase(
                        case_id=f"m1r2-{domain.lower()}-{primitive}-{variant:02d}",
                        partition=partition,
                        domain_id=domain,
                        primitive=primitive,
                        state_text=_state(
                            domain,
                            primitive,
                            target,
                            confidence,
                            variant,
                        ),
                        question_text=_question(primitive),
                        options=options,
                        gold_index=target,
                        gold_probabilities=_soft_target(
                            len(options),
                            target,
                            R2_CONFIDENCE_MASS[confidence],
                        ),
                        confidence_band=confidence,
                    )
                )
    if len(rows) != 36:
        raise RuntimeError("M1-R2 authority requires 36 cases/domain")
    return tuple(rows)


def generate_m1_r2_authority(partition: str) -> tuple[M1R2AuthorityCase, ...]:
    if partition not in R2_PARTITIONS:
        raise ValueError(f"unknown M1-R2 partition: {partition}")
    rows: list[M1R2AuthorityCase] = []
    for domain in R2_PARTITIONS[partition]:
        rows.extend(_generate_domain(domain, partition))
    return tuple(rows)


def all_m1_r2_text_atoms() -> set[str]:
    values: set[str] = set()
    for partition in R2_PARTITIONS:
        for row in generate_m1_r2_authority(partition):
            values.add(row.state_text)
            values.add(row.question_text)
            for option in row.options:
                values.add(option.criterion_text)
                values.update(option.aliases)
    return values


__all__ = [
    "M1R2AuthorityCase",
    "R2_CONFIDENCE_BANDS",
    "R2_CONFIDENCE_MASS",
    "R2_DOMAIN_CONTEXT",
    "R2_DOMAIN_ROUTES",
    "R2_PARTITIONS",
    "all_m1_r2_text_atoms",
    "generate_m1_r2_authority",
]
