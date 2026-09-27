from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .contracts import LogicalOption

FACTOR_IDS = ("F0", "F1", "F2")
PRIMITIVES = ("choice", "score", "noul")

PARTITION_DOMAINS = {
    "qualification": ("QH", "QI"),
    "train": ("QJ", "QK", "QL", "QM"),
    "dev": ("QN",),
    "confirm": ("QO", "QP"),
}

DOMAIN_CONTEXT = {
    "QH": "city archive-request assistance review",
    "QI": "community childcare-registration support review",
    "QJ": "regional parking-permit service review",
    "QK": "public museum-membership support review",
    "QL": "college transcript-delivery operations review",
    "QM": "municipal garden-allotment support review",
    "QN": "county library-card renewal support review",
    "QO": "regional bicycle-share account support review",
    "QP": "community event-volunteer scheduling review",
}

DOMAIN_STYLE = {
    "QH": "qual_a",
    "QI": "qual_b",
    "QJ": "train_a",
    "QK": "train_a",
    "QL": "train_b",
    "QM": "train_b",
    "QN": "dev",
    "QO": "confirm_a",
    "QP": "confirm_b",
}

STYLE_TERMS = {
    "qual_a": {
        "route": "default service procedure",
        "alternate": "substitute operating procedure",
        "capability": "indispensable service function",
        "objective": "primary service outcome",
        "pause": "brief preparation interval",
        "timing": "required response window",
        "harm": "grave near-term consequence",
    },
    "qual_b": {
        "route": "established handling channel",
        "alternate": "material fallback channel",
        "capability": "essential enabling capability",
        "objective": "central completion goal",
        "pause": "short coordination pause",
        "timing": "mandatory action window",
        "harm": "serious immediate result",
    },
    "train_a": {
        "route": "ordinary execution path",
        "alternate": "substantial replacement workflow",
        "capability": "key operational function",
        "objective": "main task objective",
        "pause": "small scheduling delay",
        "timing": "allowed intervention window",
        "harm": "severe short-term consequence",
    },
    "train_b": {
        "route": "standard processing route",
        "alternate": "meaningful detour process",
        "capability": "crucial service capacity",
        "objective": "core completion target",
        "pause": "brief handoff period",
        "timing": "binding response deadline",
        "harm": "grave proximate outcome",
    },
    "dev": {
        "route": "customary operational channel",
        "alternate": "material replacement path",
        "capability": "pivotal functional resource",
        "objective": "principal activity goal",
        "pause": "short staging interval",
        "timing": "applicable response limit",
        "harm": "serious near-term effect",
    },
    "confirm_a": {
        "route": "primary delivery procedure",
        "alternate": "substitute execution route",
        "capability": "decisive service capability",
        "objective": "principal service objective",
        "pause": "brief setup interval",
        "timing": "enforced timing boundary",
        "harm": "grave immediate consequence",
    },
    "confirm_b": {
        "route": "normal task pathway",
        "alternate": "material alternate method",
        "capability": "critical enabling function",
        "objective": "central task result",
        "pause": "short readiness interval",
        "timing": "required start-time boundary",
        "harm": "severe imminent outcome",
    },
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


def _factor_fragments(style: str, factor: str, value: int) -> tuple[str, ...]:
    t = STYLE_TERMS[style]
    if factor == "F0":
        if value == 0:
            return (
                f"the {t['route']} remains usable without switching to a {t['alternate']}",
                f"work can continue through the {t['route']} rather than adopting a {t['alternate']}",
            )
        return (
            f"the {t['route']} is no longer usable unless work switches to a {t['alternate']}",
            f"work must leave the {t['route']} and adopt a {t['alternate']}",
        )
    if factor == "F1":
        if value == 0:
            return (
                f"the {t['capability']} remains available for the {t['objective']}",
                f"the {t['objective']} remains achievable because its {t['capability']} still works",
                f"the service retains the {t['capability']} needed by the {t['objective']}",
            )
        return (
            f"the {t['capability']} is unavailable for the {t['objective']}",
            f"the {t['objective']} is blocked because its {t['capability']} has failed",
            f"the service has lost the {t['capability']} needed by the {t['objective']}",
        )
    if factor == "U":
        if value == 0:
            return (
                f"the {t['timing']} still permits a {t['pause']} before action starts",
                f"action may begin after a {t['pause']} without violating the {t['timing']}",
            )
        return (
            f"the {t['timing']} permits no {t['pause']} before action starts",
            f"action must start at once because a {t['pause']} would violate the {t['timing']}",
        )
    if factor == "C":
        if value == 0:
            return (
                f"that {t['pause']} would not itself cause a {t['harm']}",
                f"no {t['harm']} follows merely from using the {t['pause']}",
            )
        return (
            f"that {t['pause']} would itself cause a {t['harm']}",
            f"a {t['harm']} follows merely from using the {t['pause']}",
        )
    raise ValueError(f"unknown W31 evidence factor: {factor}")


def _schema_views(style: str, factor: str, value: int) -> tuple[str, str, str]:
    t = STYLE_TERMS[style]
    if factor == "F0":
        if value == 0:
            return (
                f"The {t['route']} remains practically usable.",
                f"No {t['alternate']} is required.",
                f"The activity stays on the {t['route']}.",
            )
        return (
            f"The {t['route']} is materially disrupted.",
            f"A {t['alternate']} is required.",
            f"The activity must leave the {t['route']}.",
        )
    if factor == "F1":
        if value == 0:
            return (
                f"The {t['capability']} is retained.",
                f"The {t['objective']} still has its required function.",
                f"Functional capacity for the {t['objective']} remains present.",
            )
        return (
            f"The {t['capability']} is lost.",
            f"The {t['objective']} no longer has its required function.",
            f"Functional capacity for the {t['objective']} is absent.",
        )
    if factor == "F2":
        if value == 0:
            return (
                f"The case does not jointly require zero delay and produce a {t['harm']} from waiting.",
                f"The {t['timing']} and delay consequence do not jointly form an immediate-critical condition.",
                f"Immediate criticality is absent because both required components are not present together.",
            )
        return (
            f"The case jointly requires zero delay and produces a {t['harm']} from waiting.",
            f"The {t['timing']} and delay consequence jointly form an immediate-critical condition.",
            f"Immediate criticality is present because both required components are present together.",
        )
    raise ValueError(f"unknown W31 factor: {factor}")


QUESTION_TEXT = {
    "choice": {
        "F0": "Which process-continuity state is supported?",
        "F1": "Which functional-capacity state is supported?",
        "F2": "Which joint critical-timing state is supported?",
    },
    "score": {
        "F0": "What numeric process-continuity state is supported?",
        "F1": "What numeric functional-capacity state is supported?",
        "F2": "What numeric joint critical-timing state is supported?",
    },
    "noul": {
        "F0": "Is a material process substitution required?",
        "F1": "Is essential functional capacity lost?",
        "F2": "Is the joint no-delay critical condition present?",
    },
}

REFERENCE_HYPOTHESES = {
    "F0": (
        "The expected process remains viable without replacing it with a materially different procedure.",
        "The expected process is disrupted enough that a materially different procedure must replace it.",
    ),
    "F1": (
        "The enabling function required for the principal task remains available.",
        "The enabling function required for the principal task is unavailable and blocks completion.",
    ),
    "U": (
        "A short preparation period is still compatible with the required response timing.",
        "The required response timing allows no short preparation period before action.",
    ),
    "C": (
        "A short wait does not itself produce a grave consequence in the near term.",
        "A short wait itself produces a grave consequence in the near term.",
    ),
    "F2": (
        "The combined condition of no-delay timing and grave harm from waiting is not met.",
        "The combined condition of no-delay timing and grave harm from waiting is met.",
    ),
}


@dataclass(frozen=True)
class W31TransferCase:
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
    raise ValueError(f"unknown W31 domain: {domain_id}")


def compose_f2(u: int, c: int) -> int:
    return int(bool(int(u)) and bool(int(c)))


def compose_severity(factors: Iterable[int]) -> int | None:
    vector = tuple(int(x) for x in factors)
    return {value: key for key, value in SEVERITY_FACTORS.items()}.get(vector)


def factor_options(domain_id: str, factor_id: str) -> tuple[LogicalOption, LogicalOption]:
    if domain_id not in DOMAIN_STYLE:
        raise ValueError(f"unknown W31 domain: {domain_id}")
    if factor_id not in FACTOR_IDS:
        raise ValueError(f"unknown W31 factor: {factor_id}")
    style = DOMAIN_STYLE[domain_id]
    out = []
    for value in (0, 1):
        views = _schema_views(style, factor_id, value)
        out.append(
            LogicalOption(
                option_id=f"{factor_id.lower()}31-{value}",
                criterion_text=views[0],
                aliases=(views[1],),
                exemplars=(views[2],),
                value=float(value),
            )
        )
    return tuple(out)


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
        raise RuntimeError("W31 requires exactly 24 unique phrases/severity/style")
    return rows


def generate_w31_domains(domains: Iterable[str]) -> tuple[W31TransferCase, ...]:
    rows: list[W31TransferCase] = []
    for domain_id in tuple(domains):
        if domain_id not in DOMAIN_CONTEXT:
            raise ValueError(f"unknown W31 domain: {domain_id}")
        partition = partition_for_domain(domain_id)
        style = DOMAIN_STYLE[domain_id]
        for severity in range(4):
            factor_vector = SEVERITY_FACTORS[severity]
            evidence = SEVERITY_EVIDENCE[severity]
            if factor_vector[2] != compose_f2(evidence["U"], evidence["C"]):
                raise RuntimeError("W31 F2 composition changed")
            for variant, phrase in enumerate(_phrases(style, severity)):
                rows.append(
                    W31TransferCase(
                        case_id=f"{domain_id.lower()}-s{severity}-v{variant}",
                        domain_id=domain_id,
                        partition=partition,
                        style_id=style,
                        severity=severity,
                        variant=variant,
                        factor_vector=factor_vector,
                        evidence_vector=(
                            evidence["F0"],
                            evidence["F1"],
                            evidence["U"],
                            evidence["C"],
                        ),
                        state_text=f"Within the {DOMAIN_CONTEXT[domain_id]}, {phrase}",
                    )
                )

    for domain_id in tuple(domains):
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != 96:
            raise RuntimeError("W31 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W31 severity balance changed")
    return tuple(rows)


def generate_w31_partition(partition: str) -> tuple[W31TransferCase, ...]:
    if partition not in PARTITION_DOMAINS:
        raise ValueError(f"unknown W31 partition: {partition}")
    return generate_w31_domains(PARTITION_DOMAINS[partition])


def all_w31_query_texts() -> set[str]:
    values: set[str] = set()
    for partition in PARTITION_DOMAINS:
        values.update(row.state_text for row in generate_w31_partition(partition))
    return values


def all_w31_schema_texts() -> set[str]:
    values: set[str] = set()
    for style in STYLE_TERMS:
        for factor in FACTOR_IDS:
            for value in (0, 1):
                values.update(_schema_views(style, factor, value))
    for primitive in QUESTION_TEXT.values():
        values.update(primitive.values())
    for hypotheses in REFERENCE_HYPOTHESES.values():
        values.update(hypotheses)
    return values


def all_w31_text_atoms() -> set[str]:
    return all_w31_query_texts() | all_w31_schema_texts()


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
    "W31TransferCase",
    "all_w31_query_texts",
    "all_w31_schema_texts",
    "all_w31_text_atoms",
    "compose_f2",
    "compose_severity",
    "factor_options",
    "generate_w31_domains",
    "generate_w31_partition",
    "partition_for_domain",
]
