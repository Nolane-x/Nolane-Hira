from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .contracts import LogicalOption

FACTOR_IDS = ("F0", "F1", "F2")
PRIMITIVES = ("choice", "score", "noul")

PARTITION_DOMAINS = {
    "qualification": ("TA", "TB"),
    "train": ("TC", "TD", "TE", "TF"),
    "dev": ("TG",),
    "confirm": ("TH", "TI"),
}

DOMAIN_CONTEXT = {
    "TA": "regional visitor-parking permit counter",
    "TB": "city public-event equipment request desk",
    "TC": "municipal recycling-bin replacement service",
    "TD": "county language-course enrollment office",
    "TE": "regional transit lost-card support counter",
    "TF": "university media-kit loan desk",
    "TG": "municipal street-vending license office",
    "TH": "regional public-charging account desk",
    "TI": "community grocery-delivery coordination office",
}

DOMAIN_STYLE = {
    "TA": "qual_ta",
    "TB": "qual_tb",
    "TC": "train_tc",
    "TD": "train_td",
    "TE": "train_te",
    "TF": "train_tf",
    "TG": "dev_tg",
    "TH": "confirm_th",
    "TI": "confirm_ti",
}

STYLE_TERMS = {
    "qual_ta": ("standard permit route", "alternate permit route", "required permit function", "requested permit result", "brief delay", "response window", "serious near-term harm"),
    "qual_tb": ("standard equipment route", "alternate equipment route", "required equipment function", "requested equipment result", "short delay", "action window", "serious near-term harm"),
    "train_tc": ("standard replacement route", "alternate replacement route", "required replacement function", "requested replacement result", "brief delay", "response window", "serious short-term harm"),
    "train_td": ("standard enrollment route", "alternate enrollment route", "required enrollment function", "requested enrollment result", "short delay", "action window", "serious short-term harm"),
    "train_te": ("standard card-support route", "alternate card-support route", "required card-support function", "requested card result", "brief delay", "response window", "serious immediate harm"),
    "train_tf": ("standard loan route", "alternate loan route", "required loan function", "requested loan result", "short delay", "action window", "serious immediate harm"),
    "dev_tg": ("standard licensing route", "alternate licensing route", "required licensing function", "requested licensing result", "brief delay", "response window", "serious near-term harm"),
    "confirm_th": ("standard account route", "alternate account route", "required account function", "requested account result", "short delay", "action window", "serious immediate harm"),
    "confirm_ti": ("standard delivery route", "alternate delivery route", "required delivery function", "requested delivery result", "brief delay", "response window", "serious immediate harm"),
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


def _terms(style: str) -> dict[str, str]:
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
                f"the {t['route']} is usable and the {t['alternate']} is not required",
                f"work can stay on the {t['route']} without switching to the {t['alternate']}",
            )
        return (
            f"the {t['route']} is unusable and the {t['alternate']} is required",
            f"work cannot continue on the {t['route']} and must switch to the {t['alternate']}",
        )
    if factor == "F1":
        if value == 0:
            return (
                f"the {t['capability']} is available and the {t['objective']} can be completed",
                f"the {t['objective']} can be completed because the {t['capability']} is available",
                f"the service still has the {t['capability']} needed for the {t['objective']}",
            )
        return (
            f"the {t['capability']} is unavailable and the {t['objective']} cannot be completed",
            f"the {t['objective']} cannot be completed because the {t['capability']} is unavailable",
            f"the service lacks the {t['capability']} needed for the {t['objective']}",
        )
    if factor == "U":
        if value == 0:
            return (
                f"the {t['timing']} allows a {t['pause']} before action starts",
                f"action may wait for a {t['pause']} and still meet the {t['timing']}",
            )
        return (
            f"the {t['timing']} allows no {t['pause']} and action must start immediately",
            f"action must start immediately because any {t['pause']} would miss the {t['timing']}",
        )
    if factor == "C":
        if value == 0:
            return (
                f"a {t['pause']} will not cause {t['harm']}",
                f"waiting for a {t['pause']} does not produce {t['harm']}",
            )
        return (
            f"a {t['pause']} will cause {t['harm']}",
            f"waiting for a {t['pause']} produces {t['harm']}",
        )
    raise ValueError(f"unknown W34 evidence factor: {factor}")


def _schema_views(style: str, factor: str, value: int) -> tuple[str, str, str]:
    t = _terms(style)
    if factor == "F0":
        return (
            (
                f"The {t['route']} remains usable for the requested work.",
                f"The {t['alternate']} does not need to replace the {t['route']}.",
                f"Execution may remain on the {t['route']}.",
            )
            if value == 0
            else (
                f"The {t['route']} cannot be used for the requested work.",
                f"The {t['alternate']} must replace the {t['route']}.",
                f"Execution must leave the {t['route']}.",
            )
        )
    if factor == "F1":
        return (
            (
                f"The {t['capability']} is present for the {t['objective']}.",
                f"The {t['objective']} can proceed because the {t['capability']} is present.",
                f"Required support from the {t['capability']} remains available.",
            )
            if value == 0
            else (
                f"The {t['capability']} is absent for the {t['objective']}.",
                f"The {t['objective']} is blocked because the {t['capability']} is absent.",
                f"Required support from the {t['capability']} is unavailable.",
            )
        )
    if factor == "F2":
        return (
            (
                f"The {t['timing']} and {t['harm']} do not jointly create the two-part urgent condition.",
                f"At least one required urgency component is absent for the {t['objective']}.",
                f"The record does not jointly require immediate action and produce {t['harm']} from a {t['pause']}.",
            )
            if value == 0
            else (
                f"The {t['timing']} and {t['harm']} jointly create the two-part urgent condition.",
                f"Both required urgency components are present for the {t['objective']}.",
                f"The record jointly requires immediate action and produces {t['harm']} from a {t['pause']}.",
            )
        )
    raise ValueError(f"unknown W34 factor: {factor}")


QUESTION_TEXT = {
    "choice": {
        "F0": "Select the W34 ordinary-path status supported by this record.",
        "F1": "Select the W34 required-capability status supported by this record.",
        "F2": "Select the W34 two-evidence urgency status supported by this record.",
    },
    "score": {
        "F0": "Return the W34 numeric ordinary-path status supported by this record.",
        "F1": "Return the W34 numeric required-capability status supported by this record.",
        "F2": "Return the W34 numeric two-evidence urgency status supported by this record.",
    },
    "noul": {
        "F0": "For W34, must the ordinary path be replaced by an alternate path?",
        "F1": "For W34, is the capability required for completion absent?",
        "F2": "For W34, are no-delay action and serious delay harm jointly true?",
    },
}

REFERENCE_HYPOTHESES = {
    "F0": (
        "Reference judgment: the normal processing path remains usable, so replacement by another path is unnecessary.",
        "Reference judgment: the normal processing path is unusable, so replacement by another path is necessary.",
    ),
    "F1": (
        "Reference judgment: the capability required for the main result is available, so completion remains possible.",
        "Reference judgment: the capability required for the main result is unavailable, so completion is blocked.",
    ),
    "U": (
        "Reference judgment: the timing rule permits a brief wait before action starts.",
        "Reference judgment: the timing rule permits no brief wait and action must start immediately.",
    ),
    "C": (
        "Reference judgment: a brief wait does not itself produce serious near-term harm.",
        "Reference judgment: a brief wait itself produces serious near-term harm.",
    ),
    "F2": (
        "Reference judgment: immediate action and serious harm from waiting are not jointly present.",
        "Reference judgment: immediate action and serious harm from waiting are jointly present.",
    ),
}


@dataclass(frozen=True)
class W34TransferCase:
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
    raise ValueError(f"unknown W34 domain: {domain_id}")


def compose_f2(u: int, c: int) -> int:
    return int(bool(int(u)) and bool(int(c)))


def compose_severity(factors: Iterable[int]) -> int | None:
    vector = tuple(int(x) for x in factors)
    return {value: key for key, value in SEVERITY_FACTORS.items()}.get(vector)


def factor_options(domain_id: str, factor_id: str) -> tuple[LogicalOption, LogicalOption]:
    if domain_id not in DOMAIN_STYLE:
        raise ValueError(f"unknown W34 domain: {domain_id}")
    if factor_id not in FACTOR_IDS:
        raise ValueError(f"unknown W34 factor: {factor_id}")
    style = DOMAIN_STYLE[domain_id]
    options = []
    for value in (0, 1):
        views = _schema_views(style, factor_id, value)
        options.append(
            LogicalOption(
                option_id=f"{factor_id.lower()}34-{value}",
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
        raise RuntimeError("W34 requires exactly 24 unique phrases/severity/style")
    return rows


def generate_w34_domains(domains: Iterable[str]) -> tuple[W34TransferCase, ...]:
    rows: list[W34TransferCase] = []
    for domain_id in tuple(domains):
        if domain_id not in DOMAIN_CONTEXT:
            raise ValueError(f"unknown W34 domain: {domain_id}")
        partition = partition_for_domain(domain_id)
        style = DOMAIN_STYLE[domain_id]
        for severity in range(4):
            vector = SEVERITY_FACTORS[severity]
            evidence = SEVERITY_EVIDENCE[severity]
            if vector[2] != compose_f2(evidence["U"], evidence["C"]):
                raise RuntimeError("W34 F2 composition changed")
            for variant, phrase in enumerate(_phrases(style, severity)):
                rows.append(
                    W34TransferCase(
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
            raise RuntimeError("W34 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W34 severity balance changed")
    return tuple(rows)


def generate_w34_partition(partition: str) -> tuple[W34TransferCase, ...]:
    if partition not in PARTITION_DOMAINS:
        raise ValueError(f"unknown W34 partition: {partition}")
    return generate_w34_domains(PARTITION_DOMAINS[partition])


def all_w34_query_texts() -> set[str]:
    values = set()
    for partition in PARTITION_DOMAINS:
        values.update(row.state_text for row in generate_w34_partition(partition))
    return values


def all_w34_schema_texts() -> set[str]:
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


def all_w34_text_atoms() -> set[str]:
    return all_w34_query_texts() | all_w34_schema_texts()


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
    "W34TransferCase",
    "all_w34_query_texts",
    "all_w34_schema_texts",
    "all_w34_text_atoms",
    "compose_f2",
    "compose_severity",
    "factor_options",
    "generate_w34_domains",
    "generate_w34_partition",
    "partition_for_domain",
]
