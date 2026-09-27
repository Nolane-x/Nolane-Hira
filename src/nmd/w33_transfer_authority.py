from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .contracts import LogicalOption

FACTOR_IDS = ("F0", "F1", "F2")
PRIMITIVES = ("choice", "score", "noul")

PARTITION_DOMAINS = {
    "qualification": ("SA", "SB"),
    "train": ("SC", "SD", "SE", "SF"),
    "dev": ("SG",),
    "confirm": ("SH", "SI"),
}

DOMAIN_CONTEXT = {
    "SA": "borough cultural-workshop registration desk",
    "SB": "county public-map request service",
    "SC": "city tree-planting permit support desk",
    "SD": "regional library-room reservation service",
    "SE": "public marina access-card support desk",
    "SF": "college laboratory-key checkout service",
    "SG": "municipal neighborhood-grant application desk",
    "SH": "regional community-bike repair booking desk",
    "SI": "public food-pantry pickup coordination desk",
}

DOMAIN_STYLE = {
    "SA": "qual_sa",
    "SB": "qual_sb",
    "SC": "train_sc",
    "SD": "train_sd",
    "SE": "train_se",
    "SF": "train_sf",
    "SG": "dev_sg",
    "SH": "confirm_sh",
    "SI": "confirm_si",
}

STYLE_TERMS = {
    "qual_sa": {
        "route": "usual request channel",
        "alternate": "replacement handling channel",
        "capability": "deciding service capability",
        "objective": "requested completion result",
        "pause": "short intake interval",
        "timing": "response timing allowance",
        "harm": "major near-term consequence",
    },
    "qual_sb": {
        "route": "established response channel",
        "alternate": "substitute response channel",
        "capability": "necessary enabling resource",
        "objective": "central request result",
        "pause": "brief coordination interval",
        "timing": "required action allowance",
        "harm": "serious near-horizon result",
    },
    "train_sc": {
        "route": "ordinary request pathway",
        "alternate": "replacement processing pathway",
        "capability": "key execution resource",
        "objective": "main service result",
        "pause": "small preparation interval",
        "timing": "permitted action window",
        "harm": "major immediate result",
    },
    "train_sd": {
        "route": "standard handling pathway",
        "alternate": "material detour pathway",
        "capability": "crucial operating resource",
        "objective": "primary handling result",
        "pause": "brief scheduling interval",
        "timing": "binding start window",
        "harm": "grave short-term result",
    },
    "train_se": {
        "route": "normal service pathway",
        "alternate": "substitute execution pathway",
        "capability": "pivotal enabling resource",
        "objective": "principal service result",
        "pause": "short setup interval",
        "timing": "allowed start window",
        "harm": "serious immediate result",
    },
    "train_sf": {
        "route": "customary processing pathway",
        "alternate": "replacement workflow pathway",
        "capability": "essential execution resource",
        "objective": "core completion result",
        "pause": "brief handoff interval",
        "timing": "mandatory response window",
        "harm": "grave near-horizon result",
    },
    "dev_sg": {
        "route": "regular operating pathway",
        "alternate": "material fallback pathway",
        "capability": "decisive functional resource",
        "objective": "principal operating result",
        "pause": "small staging interval",
        "timing": "applicable action boundary",
        "harm": "major near-term result",
    },
    "confirm_sh": {
        "route": "primary service pathway",
        "alternate": "substitute operating pathway",
        "capability": "critical functional resource",
        "objective": "central service result",
        "pause": "brief readiness interval",
        "timing": "enforced response boundary",
        "harm": "grave immediate result",
    },
    "confirm_si": {
        "route": "normal delivery pathway",
        "alternate": "material alternative pathway",
        "capability": "pivotal delivery resource",
        "objective": "main delivery result",
        "pause": "short dispatch interval",
        "timing": "required start boundary",
        "harm": "serious imminent result",
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
                f"the {t['route']} can still carry the work without a move to the {t['alternate']}",
                f"the task remains executable through the {t['route']} and does not need the {t['alternate']}",
            )
        return (
            f"the {t['route']} cannot carry the work unless it moves to the {t['alternate']}",
            f"the task must leave the {t['route']} and proceed through the {t['alternate']}",
        )
    if factor == "F1":
        if value == 0:
            return (
                f"the {t['capability']} is still present for the {t['objective']}",
                f"the {t['objective']} remains possible because the {t['capability']} continues to function",
                f"the service keeps the {t['capability']} on which the {t['objective']} depends",
            )
        return (
            f"the {t['capability']} is no longer present for the {t['objective']}",
            f"the {t['objective']} is blocked because the {t['capability']} no longer functions",
            f"the service has lost the {t['capability']} on which the {t['objective']} depends",
        )
    if factor == "U":
        if value == 0:
            return (
                f"the {t['timing']} leaves room for a {t['pause']} before action starts",
                f"action can begin after a {t['pause']} and still remain inside the {t['timing']}",
            )
        return (
            f"the {t['timing']} leaves no room for a {t['pause']} before action starts",
            f"action has to start immediately because a {t['pause']} would exceed the {t['timing']}",
        )
    if factor == "C":
        if value == 0:
            return (
                f"the {t['pause']} alone would not produce a {t['harm']}",
                f"merely waiting through the {t['pause']} does not create a {t['harm']}",
            )
        return (
            f"the {t['pause']} alone would produce a {t['harm']}",
            f"merely waiting through the {t['pause']} creates a {t['harm']}",
        )
    raise ValueError(f"unknown W33 evidence factor: {factor}")


def _schema_views(style: str, factor: str, value: int) -> tuple[str, str, str]:
    t = STYLE_TERMS[style]
    if factor == "F0":
        if value == 0:
            return (
                f"The {t['route']} continues to support execution.",
                f"Execution does not require the {t['alternate']}.",
                f"The case keeps its work inside the {t['route']}.",
            )
        return (
            f"The {t['route']} no longer supports execution.",
            f"Execution requires the {t['alternate']}.",
            f"The case must move its work out of the {t['route']}.",
        )
    if factor == "F1":
        if value == 0:
            return (
                f"The {t['capability']} is retained.",
                f"The {t['objective']} continues to possess its required resource.",
                f"The resource needed to achieve the {t['objective']} remains functional.",
            )
        return (
            f"The {t['capability']} is lost.",
            f"The {t['objective']} no longer possesses its required resource.",
            f"The resource needed to achieve the {t['objective']} is nonfunctional.",
        )
    if factor == "F2":
        if value == 0:
            return (
                f"The case lacks the joint condition of no waiting room plus a {t['harm']}.",
                f"The {t['timing']} and delay effect do not together establish critical immediacy.",
                "At least one of the two required immediate-critical evidence pieces is absent.",
            )
        return (
            f"The case has the joint condition of no waiting room plus a {t['harm']}.",
            f"The {t['timing']} and delay effect together establish critical immediacy.",
            "Both required immediate-critical evidence pieces are simultaneously present.",
        )
    raise ValueError(f"unknown W33 factor: {factor}")


QUESTION_TEXT = {
    "choice": {
        "F0": "Which execution-route condition best matches this record?",
        "F1": "Which required-resource condition best matches this record?",
        "F2": "Which two-part critical-immediacy condition best matches this record?",
    },
    "score": {
        "F0": "What numeric execution-route condition follows from this record?",
        "F1": "What numeric required-resource condition follows from this record?",
        "F2": "What numeric two-part critical-immediacy condition follows from this record?",
    },
    "noul": {
        "F0": "Must execution leave its ordinary route?",
        "F1": "Is the resource required for completion absent?",
        "F2": "Are both immediate-start necessity and severe waiting consequence present together?",
    },
}

REFERENCE_HYPOTHESES = {
    "F0": (
        "Work can remain on its ordinary execution route without changing to a materially different route.",
        "Work cannot remain on its ordinary execution route and must change to a materially different route.",
    ),
    "F1": (
        "The resource needed to complete the principal activity is still available.",
        "The resource needed to complete the principal activity is unavailable and completion is blocked.",
    ),
    "U": (
        "The timing requirement still permits a short interval before action begins.",
        "The timing requirement permits no short interval and action must begin at once.",
    ),
    "C": (
        "A short wait by itself does not create a major consequence soon.",
        "A short wait by itself creates a major consequence soon.",
    ),
    "F2": (
        "The record does not contain both immediate-start necessity and major harm from waiting.",
        "The record contains both immediate-start necessity and major harm from waiting.",
    ),
}


@dataclass(frozen=True)
class W33TransferCase:
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
    raise ValueError(f"unknown W33 domain: {domain_id}")


def compose_f2(u: int, c: int) -> int:
    return int(bool(int(u)) and bool(int(c)))


def compose_severity(factors: Iterable[int]) -> int | None:
    vector = tuple(int(x) for x in factors)
    return {value: key for key, value in SEVERITY_FACTORS.items()}.get(vector)


def factor_options(domain_id: str, factor_id: str) -> tuple[LogicalOption, LogicalOption]:
    if domain_id not in DOMAIN_STYLE:
        raise ValueError(f"unknown W33 domain: {domain_id}")
    if factor_id not in FACTOR_IDS:
        raise ValueError(f"unknown W33 factor: {factor_id}")
    style = DOMAIN_STYLE[domain_id]
    out = []
    for value in (0, 1):
        views = _schema_views(style, factor_id, value)
        out.append(
            LogicalOption(
                option_id=f"{factor_id.lower()}33-{value}",
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
        raise RuntimeError("W33 requires exactly 24 unique phrases/severity/style")
    return rows


def generate_w33_domains(domains: Iterable[str]) -> tuple[W33TransferCase, ...]:
    rows: list[W33TransferCase] = []
    for domain_id in tuple(domains):
        if domain_id not in DOMAIN_CONTEXT:
            raise ValueError(f"unknown W33 domain: {domain_id}")
        partition = partition_for_domain(domain_id)
        style = DOMAIN_STYLE[domain_id]
        for severity in range(4):
            vector = SEVERITY_FACTORS[severity]
            evidence = SEVERITY_EVIDENCE[severity]
            if vector[2] != compose_f2(evidence["U"], evidence["C"]):
                raise RuntimeError("W33 F2 composition changed")
            for variant, phrase in enumerate(_phrases(style, severity)):
                rows.append(
                    W33TransferCase(
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
                        state_text=f"During review at the {DOMAIN_CONTEXT[domain_id]}, {phrase}",
                    )
                )

    for domain_id in tuple(domains):
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != 96:
            raise RuntimeError("W33 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W33 severity balance changed")
    return tuple(rows)


def generate_w33_partition(partition: str) -> tuple[W33TransferCase, ...]:
    if partition not in PARTITION_DOMAINS:
        raise ValueError(f"unknown W33 partition: {partition}")
    return generate_w33_domains(PARTITION_DOMAINS[partition])


def all_w33_query_texts() -> set[str]:
    values: set[str] = set()
    for partition in PARTITION_DOMAINS:
        values.update(row.state_text for row in generate_w33_partition(partition))
    return values


def all_w33_schema_texts() -> set[str]:
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


def all_w33_text_atoms() -> set[str]:
    return all_w33_query_texts() | all_w33_schema_texts()


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
    "W33TransferCase",
    "all_w33_query_texts",
    "all_w33_schema_texts",
    "all_w33_text_atoms",
    "compose_f2",
    "compose_severity",
    "factor_options",
    "generate_w33_domains",
    "generate_w33_partition",
    "partition_for_domain",
]
