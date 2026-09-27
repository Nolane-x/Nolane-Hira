from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import LogicalOption, Primitive

AUTHORITY_PARTITIONS = {
    "cal_train": ("UA", "UB", "UC"),
    "cal_dev": ("UD",),
    "selective_confirm": ("UE",),
    "ood_train": ("UF", "UG"),
    "ood_dev": ("UH",),
    "ood_confirm": ("UI",),
}

SEALED_PARTITIONS = {"selective_confirm", "ood_confirm"}

DOMAIN_CONTEXT = {
    "UA": "district archive-copy request office",
    "UB": "regional community-hall booking counter",
    "UC": "municipal bicycle-locker permit service",
    "UD": "public shoreline-access registration desk",
    "UE": "county civic-workshop scheduling office",
    "UF": "regional seed-library checkout counter",
    "UG": "municipal art-cart reservation desk",
    "UH": "public trail-pass support office",
    "UI": "community tool-lending registration desk",
}

DOMAIN_ROUTES = {
    "UA": ("cedar route", "linen route", "quartz route"),
    "UB": ("amber route", "willow route", "granite route"),
    "UC": ("marsh route", "cobalt route", "birch route"),
    "UD": ("orchid route", "slate route", "juniper route"),
    "UE": ("maple route", "coral route", "basalt route"),
    "UF": ("spruce route", "pearl route", "clay route"),
    "UG": ("violet route", "elm route", "flint route"),
    "UH": ("cypress route", "opal route", "reed route"),
    "UI": ("ash route", "ivory route", "mica route"),
}

DOMAIN_OOD_TOPICS = {
    "UF": (
        "a mural uses alternating blue and silver pigments",
        "a recipe lists lentils, cumin, and roasted pepper",
        "a telescope log records a faint star near the horizon",
    ),
    "UG": (
        "a violin score changes from adagio to allegro",
        "a garden note describes moss growing beside a stone wall",
        "a weather diary records cloud cover before sunrise",
    ),
    "UH": (
        "a pottery kiln reaches a new glaze temperature",
        "a chess notebook analyzes a rook endgame",
        "a bakery card lists rye flour and sesame seeds",
    ),
    "UI": (
        "a bird survey counts swallows above a wetland",
        "a fabric guide compares linen weave densities",
        "a music rehearsal changes the percussion pattern",
    ),
}

CONFIDENCE_BANDS = ("strong", "mixed", "thin")
CONFIDENCE_MASS = {
    "strong": 0.90,
    "mixed": 0.75,
    "thin": 0.60,
}
OOD_KINDS = ("topic_mismatch", "foreign_task", "no_relevant_evidence")


@dataclass(frozen=True)
class M1AuthorityCase:
    case_id: str
    partition: str
    domain_id: str
    primitive: Primitive
    state_text: str
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int | None
    gold_probabilities: tuple[float, ...] | None
    confidence_band: str | None
    is_ood: bool
    ood_kind: str | None


def partition_for_domain(domain_id: str) -> str:
    for partition, domains in AUTHORITY_PARTITIONS.items():
        if domain_id in domains:
            return partition
    raise ValueError(f"unknown M1 authority domain: {domain_id}")


def _soft_target(k: int, gold: int, mass: float) -> tuple[float, ...]:
    if k < 2 or not 0 <= gold < k or not 0.0 < mass <= 1.0:
        raise ValueError("invalid M1 soft-target parameters")
    other = (1.0 - mass) / float(k - 1)
    return tuple(mass if i == gold else other for i in range(k))


def _choice_options(domain: str) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=f"{domain.lower()}-route-{index}",
            criterion_text=f"The request belongs to the {route}.",
            aliases=(f"Use the {route} for this request.",),
        )
        for index, route in enumerate(DOMAIN_ROUTES[domain])
    )


def _score_options(domain: str) -> tuple[LogicalOption, ...]:
    labels = (
        ("light", "The stated impact is light.", 0.0),
        ("moderate", "The stated impact is moderate.", 1.0),
        ("severe", "The stated impact is severe.", 2.0),
    )
    return tuple(
        LogicalOption(
            option_id=f"{domain.lower()}-impact-{name}",
            criterion_text=text,
            aliases=(f"Impact band {value:.0f} corresponds to {name}.",),
            value=value,
        )
        for name, text, value in labels
    )


def _noul_options(domain: str) -> tuple[LogicalOption, ...]:
    return (
        LogicalOption(
            option_id=f"{domain.lower()}-review-no",
            criterion_text="Independent review is not required.",
            aliases=("The request may proceed without independent review.",),
            value=0.0,
        ),
        LogicalOption(
            option_id=f"{domain.lower()}-review-yes",
            criterion_text="Independent review is required.",
            aliases=("The request must receive independent review.",),
            value=1.0,
        ),
    )


def _question(primitive: Primitive) -> str:
    if primitive == "choice":
        return "Which processing route is explicitly assigned to this request?"
    if primitive == "score":
        return "Which impact band is explicitly stated for this request?"
    if primitive == "noul":
        return "Does the record explicitly require independent review?"
    raise ValueError(f"unknown primitive: {primitive}")


def _id_state(
    domain: str,
    primitive: Primitive,
    target: int,
    confidence: str,
    variant: int,
) -> str:
    context = DOMAIN_CONTEXT[domain]
    routes = DOMAIN_ROUTES[domain]
    evidence = {
        "strong": "The record is direct and internally consistent.",
        "mixed": "The record is usable but includes one secondary caveat.",
        "thin": "The record is brief and supplies only the essential statement.",
    }[confidence]

    if primitive == "choice":
        fact = (
            f"The request is explicitly assigned to the {routes[target]}; "
            f"the other listed routes are not assigned."
        )
    elif primitive == "score":
        label = ("light", "moderate", "severe")[target]
        fact = (
            f"The request explicitly states a {label} impact band; "
            "no different impact band is stated."
        )
    elif primitive == "noul":
        if target == 1:
            fact = (
                "The record explicitly requires independent review before the "
                "request may proceed."
            )
        else:
            fact = (
                "The record explicitly says independent review is not required "
                "before the request proceeds."
            )
    else:
        raise ValueError(f"unknown primitive: {primitive}")

    return (
        f"At the {context}, entry {variant:02d} records the request. "
        f"{fact} {evidence}"
    )


def _ood_state(
    domain: str,
    primitive: Primitive,
    kind: str,
    variant: int,
) -> str:
    context = DOMAIN_CONTEXT[domain]
    topics = DOMAIN_OOD_TOPICS[domain]
    topic = topics[variant % len(topics)]

    if kind == "topic_mismatch":
        tail = (
            f"The note only says that {topic}. "
            "It contains no processing-route, impact-band, or review requirement."
        )
    elif kind == "foreign_task":
        tail = (
            f"The entry discusses an unrelated observation: {topic}. "
            "It does not describe the service request asked about by the schema."
        )
    elif kind == "no_relevant_evidence":
        tail = (
            f"The record contains only this unrelated detail: {topic}. "
            "No evidence relevant to the requested decision is present."
        )
    else:
        raise ValueError(f"unknown OOD kind: {kind}")

    return (
        f"At the {context}, out-of-scope entry {variant:02d} is attached by mistake. "
        f"{tail}"
    )


def _primitive_options(domain: str, primitive: Primitive) -> tuple[LogicalOption, ...]:
    if primitive == "choice":
        return _choice_options(domain)
    if primitive == "score":
        return _score_options(domain)
    if primitive == "noul":
        return _noul_options(domain)
    raise ValueError(f"unknown primitive: {primitive}")


def _generate_id_domain(domain: str, partition: str) -> tuple[M1AuthorityCase, ...]:
    rows: list[M1AuthorityCase] = []
    primitives: tuple[Primitive, ...] = ("choice", "score", "noul")
    for p_index, primitive in enumerate(primitives):
        k = 2 if primitive == "noul" else 3
        for confidence_index, confidence in enumerate(CONFIDENCE_BANDS):
            for repeat in range(4):
                target = (repeat + confidence_index + p_index) % k
                variant = p_index * 12 + confidence_index * 4 + repeat
                options = _primitive_options(domain, primitive)
                rows.append(
                    M1AuthorityCase(
                        case_id=f"m1-{domain.lower()}-{primitive}-{variant:02d}",
                        partition=partition,
                        domain_id=domain,
                        primitive=primitive,
                        state_text=_id_state(
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
                            CONFIDENCE_MASS[confidence],
                        ),
                        confidence_band=confidence,
                        is_ood=False,
                        ood_kind=None,
                    )
                )
    if len(rows) != 36:
        raise RuntimeError("M1 ID authority requires 36 cases/domain")
    return tuple(rows)


def _generate_ood_domain(domain: str, partition: str) -> tuple[M1AuthorityCase, ...]:
    rows: list[M1AuthorityCase] = []
    primitives: tuple[Primitive, ...] = ("choice", "score", "noul")
    for p_index, primitive in enumerate(primitives):
        for kind_index, kind in enumerate(OOD_KINDS):
            for repeat in range(4):
                variant = p_index * 12 + kind_index * 4 + repeat
                rows.append(
                    M1AuthorityCase(
                        case_id=f"m1-{domain.lower()}-{primitive}-ood-{variant:02d}",
                        partition=partition,
                        domain_id=domain,
                        primitive=primitive,
                        state_text=_ood_state(
                            domain,
                            primitive,
                            kind,
                            variant,
                        ),
                        question_text=_question(primitive),
                        options=_primitive_options(domain, primitive),
                        gold_index=None,
                        gold_probabilities=None,
                        confidence_band=None,
                        is_ood=True,
                        ood_kind=kind,
                    )
                )
    if len(rows) != 36:
        raise RuntimeError("M1 OOD authority requires 36 cases/domain")
    return tuple(rows)


def generate_m1_authority(
    partition: str,
    *,
    allow_sealed: bool = False,
) -> tuple[M1AuthorityCase, ...]:
    if partition not in AUTHORITY_PARTITIONS:
        raise ValueError(f"unknown M1 authority partition: {partition}")
    if partition in SEALED_PARTITIONS and not allow_sealed:
        raise RuntimeError(
            f"M1 {partition} authority is sealed until pre-registered exposure"
        )

    rows: list[M1AuthorityCase] = []
    for domain in AUTHORITY_PARTITIONS[partition]:
        if partition.startswith("ood_"):
            rows.extend(_generate_ood_domain(domain, partition))
        else:
            rows.extend(_generate_id_domain(domain, partition))
    return tuple(rows)


def all_m1_authority_text_atoms(*, include_sealed: bool = True) -> set[str]:
    values: set[str] = set()
    for partition in AUTHORITY_PARTITIONS:
        if partition in SEALED_PARTITIONS and not include_sealed:
            continue
        rows = generate_m1_authority(
            partition,
            allow_sealed=partition in SEALED_PARTITIONS,
        )
        for row in rows:
            values.add(row.state_text)
            values.add(row.question_text)
            for option in row.options:
                values.add(option.criterion_text)
                values.update(option.aliases)
    return values


def authority_case_ids(
    partitions: Iterable[str],
    *,
    allow_sealed: bool = False,
) -> set[str]:
    values: set[str] = set()
    for partition in partitions:
        values.update(
            row.case_id
            for row in generate_m1_authority(
                partition,
                allow_sealed=allow_sealed,
            )
        )
    return values


__all__ = [
    "AUTHORITY_PARTITIONS",
    "CONFIDENCE_BANDS",
    "CONFIDENCE_MASS",
    "DOMAIN_CONTEXT",
    "DOMAIN_OOD_TOPICS",
    "DOMAIN_ROUTES",
    "M1AuthorityCase",
    "OOD_KINDS",
    "SEALED_PARTITIONS",
    "all_m1_authority_text_atoms",
    "authority_case_ids",
    "generate_m1_authority",
    "partition_for_domain",
]
