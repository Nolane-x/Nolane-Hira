from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .contracts import LogicalOption

FACTOR_IDS = ("F0", "F1", "F2")
PRIMITIVES = ("choice", "score", "noul")

PARTITION_DOMAINS = {
    "train": ("FA", "FB", "FC", "FD"),
    "dev": ("FE",),
    "confirm": ("FF", "FG"),
}

DOMAIN_CONTEXT = {
    "FA": "county permit-renewal support desk",
    "FB": "community clinic appointment-rescheduling service",
    "FC": "university room-booking operations review",
    "FD": "regional parcel-collection support service",
    "FE": "public recreation-pass access review",
    "FF": "municipal recycling-pickup support review",
    "FG": "nonprofit volunteer-dispatch coordination review",
}

DOMAIN_STYLE = {
    "FA": "train_a",
    "FB": "train_a",
    "FC": "train_b",
    "FD": "train_b",
    "FE": "dev",
    "FF": "confirm_a",
    "FG": "confirm_b",
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

STYLE_FRAGMENTS = {
    "train_a": {
        "F0": {
            0: (
                "the usual service route can still be followed without changing the operating method",
                "work proceeds through the expected process with no substantial alternate procedure",
            ),
            1: (
                "the usual service route no longer works as expected and a materially different procedure is needed",
                "work cannot proceed in the standard way without switching to a substantial workaround",
            ),
        },
        "F1": {
            0: (
                "the essential function needed for the main objective remains usable",
                "the principal capability still supports completion of the core task",
                "the key service function continues to be available for practical completion",
            ),
            1: (
                "the essential function needed for the main objective is unavailable",
                "the principal capability has failed enough to prevent completion of the core task",
                "the key service function can no longer support the main objective",
            ),
        },
        "U": {
            0: (
                "response can start after a short coordination delay and still meet its timing obligation",
                "the case tolerates a brief scheduling interval before action begins",
            ),
            1: (
                "response must start now because the timing obligation allows no brief scheduling delay",
                "the case cannot tolerate even a short interval before action begins",
            ),
        },
        "C": {
            0: (
                "a short delay by itself does not create a serious near-term outcome",
                "briefly postponing action does not itself produce severe immediate consequences",
            ),
            1: (
                "a short delay by itself creates a serious near-term outcome",
                "briefly postponing action itself produces severe immediate consequences",
            ),
        },
    },
    "train_b": {
        "F0": {
            0: (
                "the regular workflow remains serviceable and no meaningful rerouting is necessary",
                "people can stay on the normal operational path rather than adopting a replacement process",
            ),
            1: (
                "the regular workflow is no longer serviceable without meaningful rerouting",
                "people must leave the normal operational path and adopt a replacement process",
            ),
        },
        "F1": {
            0: (
                "the function that matters for the primary outcome is still operational",
                "core capability is preserved sufficiently for the main job to finish",
                "the service retains the major function on which the central result depends",
            ),
            1: (
                "the function that matters for the primary outcome is no longer operational",
                "core capability is lost sufficiently to stop the main job from finishing",
                "the service has lost the major function on which the central result depends",
            ),
        },
        "U": {
            0: (
                "starting after a modest handoff interval remains within the allowed response window",
                "there is room for a brief handoff before intervention needs to begin",
            ),
            1: (
                "starting after a modest handoff interval would miss the allowed response window",
                "there is no room for a brief handoff before intervention must begin",
            ),
        },
        "C": {
            0: (
                "waiting through that brief handoff would not itself trigger grave near-term effects",
                "the short handoff interval does not itself lead to a severe immediate result",
            ),
            1: (
                "waiting through that brief handoff would itself trigger grave near-term effects",
                "the short handoff interval itself leads to a severe immediate result",
            ),
        },
    },
    "dev": {
        "F0": {
            0: (
                "the established operating channel remains practical without a substantial detour",
                "the task can stay on its established route instead of moving to an alternate workflow",
            ),
            1: (
                "the established operating channel is impractical unless the task takes a substantial detour",
                "the task must leave its established route and move to an alternate workflow",
            ),
        },
        "F1": {
            0: (
                "the capability required for the central outcome is retained",
                "the main objective remains achievable because its crucial function still works",
                "the service still supplies the function essential to completing the primary activity",
            ),
            1: (
                "the capability required for the central outcome has been lost",
                "the main objective is no longer achievable because its crucial function has failed",
                "the service no longer supplies the function essential to completing the primary activity",
            ),
        },
        "U": {
            0: (
                "a small coordination pause is permissible before the response commences",
                "the response window remains open after a brief coordination pause",
            ),
            1: (
                "a small coordination pause is impermissible because the response must commence at once",
                "the response window would be missed by even a brief coordination pause",
            ),
        },
        "C": {
            0: (
                "that brief pause would not itself produce a grave consequence in the near term",
                "no severe near-term effect arises merely from the short pause",
            ),
            1: (
                "that brief pause would itself produce a grave consequence in the near term",
                "a severe near-term effect arises merely from the short pause",
            ),
        },
    },
    "confirm_a": {
        "F0": {
            0: (
                "the default way of carrying out the service remains viable without material procedural substitution",
                "the activity can remain on its default execution route with no substantial replacement path",
            ),
            1: (
                "the default way of carrying out the service is no longer viable without material procedural substitution",
                "the activity must abandon its default execution route for a substantial replacement path",
            ),
        },
        "F1": {
            0: (
                "the indispensable capability for the primary job is still functioning",
                "the primary job remains completable because its indispensable function is intact",
                "the service preserves the capability whose loss would stop the central objective",
            ),
            1: (
                "the indispensable capability for the primary job is no longer functioning",
                "the primary job cannot be completed because its indispensable function is gone",
                "the service has lost the capability whose absence stops the central objective",
            ),
        },
        "U": {
            0: (
                "the response deadline still permits a short preparation interval before work starts",
                "work may begin after brief preparation without breaching the required timing",
            ),
            1: (
                "the response deadline permits no short preparation interval before work starts",
                "work has to begin immediately to avoid breaching the required timing",
            ),
        },
        "C": {
            0: (
                "using that short preparation interval would not itself cause severe consequences soon",
                "the brief preparation period alone carries no grave near-term result",
            ),
            1: (
                "using that short preparation interval would itself cause severe consequences soon",
                "the brief preparation period alone carries a grave near-term result",
            ),
        },
    },
    "confirm_b": {
        "F0": {
            0: (
                "the primary operating route remains workable without replacing the method of execution",
                "normal task execution can continue without shifting onto a materially different route",
            ),
            1: (
                "the primary operating route has become unworkable unless the method of execution is replaced",
                "normal task execution cannot continue without shifting onto a materially different route",
            ),
        },
        "F1": {
            0: (
                "the pivotal function for accomplishing the central task remains available",
                "the central task can still finish because the pivotal service function is intact",
                "major functional capacity needed by the primary objective remains present",
            ),
            1: (
                "the pivotal function for accomplishing the central task is unavailable",
                "the central task cannot finish because the pivotal service function is absent",
                "major functional capacity needed by the primary objective is missing",
            ),
        },
        "U": {
            0: (
                "a brief staging period remains compatible with the required response timing",
                "the response may be staged for a short interval before action starts",
            ),
            1: (
                "a brief staging period is incompatible with the required response timing",
                "the response cannot be staged even briefly before action starts",
            ),
        },
        "C": {
            0: (
                "that short staging period would not itself bring about a severe near-term consequence",
                "no grave immediate result follows merely from the brief staging period",
            ),
            1: (
                "that short staging period would itself bring about a severe near-term consequence",
                "a grave immediate result follows merely from the brief staging period",
            ),
        },
    },
}

SCHEMA_VIEWS = {
    "train_a": {
        "F0": {
            0: ("The normal operating method remains usable.", "No substantial replacement workflow is necessary.", "The task stays on its expected procedural route."),
            1: ("The normal operating method is materially disrupted.", "A substantial replacement workflow is necessary.", "The task must move away from its expected procedural route."),
        },
        "F1": {
            0: ("The essential service capability remains usable.", "The main task retains the function it depends on.", "Core functional capacity is preserved."),
            1: ("The essential service capability has been lost.", "The main task has lost the function it depends on.", "Core functional capacity is absent."),
        },
        "F2": {
            0: ("The case is not both no-delay and seriously consequential.", "Immediate criticality is absent because the timing-and-consequence conjunction is false.", "The response condition does not combine zero waiting time with grave harm from delay."),
            1: ("The case is both no-delay and seriously consequential.", "Immediate criticality is present because the timing-and-consequence conjunction is true.", "The response condition combines zero waiting time with grave harm from delay."),
        },
    },
    "train_b": {
        "F0": {
            0: ("The regular execution route remains practically intact.", "A meaningful reroute of the workflow is unnecessary.", "Operations can continue using their ordinary path."),
            1: ("The regular execution route has been practically broken.", "A meaningful reroute of the workflow is required.", "Operations cannot continue using their ordinary path."),
        },
        "F1": {
            0: ("The major function needed for the central outcome is retained.", "The primary objective still has its key enabling capability.", "Important functional capacity remains operational."),
            1: ("The major function needed for the central outcome is lost.", "The primary objective no longer has its key enabling capability.", "Important functional capacity is no longer operational."),
        },
        "F2": {
            0: ("The condition does not require action now with severe harm caused by waiting.", "At least one part of the no-delay plus grave-consequence condition is absent.", "The situation falls outside the combined immediate-critical threshold."),
            1: ("The condition requires action now with severe harm caused by waiting.", "Both parts of the no-delay plus grave-consequence condition are present.", "The situation meets the combined immediate-critical threshold."),
        },
    },
    "dev": {
        "F0": {
            0: ("The established process remains viable without a major detour.", "Material procedural substitution is not required.", "The normal execution channel remains practical."),
            1: ("The established process is not viable without a major detour.", "Material procedural substitution is required.", "The normal execution channel is no longer practical."),
        },
        "F1": {
            0: ("The crucial capability for the central activity is retained.", "The core objective can still use its necessary function.", "The major enabling function remains present."),
            1: ("The crucial capability for the central activity is lost.", "The core objective can no longer use its necessary function.", "The major enabling function is absent."),
        },
        "F2": {
            0: ("The case is outside the immediate-critical conjunction.", "No-delay timing and grave consequence are not jointly present.", "The event does not satisfy both urgency and severe delay-harm together."),
            1: ("The case is inside the immediate-critical conjunction.", "No-delay timing and grave consequence are jointly present.", "The event satisfies both urgency and severe delay-harm together."),
        },
    },
    "confirm_a": {
        "F0": {
            0: ("The default service procedure remains workable as-is.", "The activity needs no material substitute execution path.", "Ordinary procedural flow remains practically available."),
            1: ("The default service procedure is no longer workable as-is.", "The activity needs a material substitute execution path.", "Ordinary procedural flow is no longer practically available."),
        },
        "F1": {
            0: ("The indispensable function behind the primary job remains intact.", "The central objective still has the capability required to finish.", "The service retains its pivotal functional capacity."),
            1: ("The indispensable function behind the primary job is gone.", "The central objective lacks the capability required to finish.", "The service has lost its pivotal functional capacity."),
        },
        "F2": {
            0: ("The case does not meet the combined act-now and severe-delay-harm condition.", "Immediate criticality is false because both required components are not present together.", "Zero-delay necessity and grave waiting consequences do not coincide."),
            1: ("The case meets the combined act-now and severe-delay-harm condition.", "Immediate criticality is true because both required components are present together.", "Zero-delay necessity and grave waiting consequences coincide."),
        },
    },
    "confirm_b": {
        "F0": {
            0: ("The primary operational route remains fit for use.", "No materially different workflow has to replace the normal one.", "Standard task execution remains practically available."),
            1: ("The primary operational route is no longer fit for use.", "A materially different workflow has to replace the normal one.", "Standard task execution is no longer practically available."),
        },
        "F1": {
            0: ("The pivotal capability needed to accomplish the main task remains available.", "The main objective still possesses its required core function.", "Essential functional capacity has been retained."),
            1: ("The pivotal capability needed to accomplish the main task is unavailable.", "The main objective no longer possesses its required core function.", "Essential functional capacity has been lost."),
        },
        "F2": {
            0: ("The situation falls short of the joint immediate-critical condition.", "The need to act without delay and severe harm from waiting are not both true.", "The combined urgency-consequence threshold is not satisfied."),
            1: ("The situation satisfies the joint immediate-critical condition.", "The need to act without delay and severe harm from waiting are both true.", "The combined urgency-consequence threshold is satisfied."),
        },
    },
}

QUESTION_TEXT = {
    "choice": {
        "F0": "Which operational-path state is supported?",
        "F1": "Which essential-capability state is supported?",
        "F2": "Which combined immediate-critical state is supported?",
    },
    "score": {
        "F0": "What numeric operational-path state is supported?",
        "F1": "What numeric essential-capability state is supported?",
        "F2": "What numeric immediate-critical state is supported?",
    },
    "noul": {
        "F0": "Is material operational rerouting present?",
        "F1": "Is loss of an essential capability present?",
        "F2": "Is the joint immediate-critical condition present?",
    },
}

REFERENCE_HYPOTHESES = {
    "F0": (
        "The ordinary operating method remains viable without a meaningful replacement path.",
        "The ordinary operating method is disrupted enough to require a meaningful replacement path.",
    ),
    "F1": (
        "The function essential to the central task remains available.",
        "The function essential to the central task is unavailable and blocks completion.",
    ),
    "U": (
        "A short preparation interval remains acceptable before response begins.",
        "Response must begin at once and cannot accept a short preparation interval.",
    ),
    "C": (
        "A short wait does not itself cause a grave near-term consequence.",
        "A short wait itself causes a grave near-term consequence.",
    ),
    "F2": (
        "The joint no-delay and grave-consequence condition is not satisfied.",
        "The joint no-delay and grave-consequence condition is satisfied.",
    ),
}


@dataclass(frozen=True)
class W30TransferCase:
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
    raise ValueError(f"unknown W30 domain: {domain_id}")


def compose_f2(u: int, c: int) -> int:
    return int(bool(int(u)) and bool(int(c)))


def compose_severity(factors: Iterable[int]) -> int | None:
    vector = tuple(int(x) for x in factors)
    return {value: key for key, value in SEVERITY_FACTORS.items()}.get(vector)


def factor_options(domain_id: str, factor_id: str) -> tuple[LogicalOption, LogicalOption]:
    if domain_id not in DOMAIN_STYLE:
        raise ValueError(f"unknown W30 domain: {domain_id}")
    if factor_id not in FACTOR_IDS:
        raise ValueError(f"unknown W30 factor: {factor_id}")
    style = DOMAIN_STYLE[domain_id]
    result = []
    for value in (0, 1):
        views = SCHEMA_VIEWS[style][factor_id][value]
        result.append(
            LogicalOption(
                option_id=f"{factor_id.lower()}30-{value}",
                criterion_text=views[0],
                aliases=(views[1],),
                exemplars=(views[2],),
                value=float(value),
            )
        )
    return tuple(result)


def _phrases(style_id: str, severity: int) -> tuple[str, ...]:
    evidence = SEVERITY_EVIDENCE[int(severity)]
    fragments = STYLE_FRAGMENTS[style_id]
    rows = tuple(
        f"{f0}; {f1}; {u}; and {c}."
        for f0, f1, u, c in product(
            fragments["F0"][evidence["F0"]],
            fragments["F1"][evidence["F1"]],
            fragments["U"][evidence["U"]],
            fragments["C"][evidence["C"]],
        )
    )
    if len(rows) != 24 or len(set(rows)) != 24:
        raise RuntimeError("W30 requires exactly 24 unique phrases/severity/style")
    return rows


def generate_w30_domains(domains: Iterable[str]) -> tuple[W30TransferCase, ...]:
    rows: list[W30TransferCase] = []
    for domain_id in tuple(domains):
        if domain_id not in DOMAIN_CONTEXT:
            raise ValueError(f"unknown W30 domain: {domain_id}")
        partition = partition_for_domain(domain_id)
        style = DOMAIN_STYLE[domain_id]
        context = DOMAIN_CONTEXT[domain_id]
        for severity in range(4):
            factor_vector = SEVERITY_FACTORS[severity]
            evidence = SEVERITY_EVIDENCE[severity]
            if factor_vector[2] != compose_f2(evidence["U"], evidence["C"]):
                raise RuntimeError("W30 F2 composition changed")
            for variant, phrase in enumerate(_phrases(style, severity)):
                rows.append(
                    W30TransferCase(
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
                        state_text=f"During the {context}, {phrase}",
                    )
                )
    for domain_id in tuple(domains):
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != 96:
            raise RuntimeError("W30 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W30 severity balance changed")
    return tuple(rows)


def generate_w30_partition(partition: str) -> tuple[W30TransferCase, ...]:
    if partition not in PARTITION_DOMAINS:
        raise ValueError(f"unknown W30 partition: {partition}")
    return generate_w30_domains(PARTITION_DOMAINS[partition])


def all_w30_query_texts() -> set[str]:
    values: set[str] = set()
    for partition in PARTITION_DOMAINS:
        values.update(row.state_text for row in generate_w30_partition(partition))
    return values


def all_w30_schema_texts() -> set[str]:
    values: set[str] = set()
    for style in SCHEMA_VIEWS.values():
        for factor in style.values():
            for views in factor.values():
                values.update(views)
    for primitive in QUESTION_TEXT.values():
        values.update(primitive.values())
    for hypotheses in REFERENCE_HYPOTHESES.values():
        values.update(hypotheses)
    return values


def all_w30_text_atoms() -> set[str]:
    return all_w30_query_texts() | all_w30_schema_texts()


__all__ = [
    "DOMAIN_CONTEXT",
    "DOMAIN_STYLE",
    "FACTOR_IDS",
    "PARTITION_DOMAINS",
    "PRIMITIVES",
    "QUESTION_TEXT",
    "REFERENCE_HYPOTHESES",
    "SCHEMA_VIEWS",
    "SEVERITY_EVIDENCE",
    "SEVERITY_FACTORS",
    "STYLE_FRAGMENTS",
    "W30TransferCase",
    "all_w30_query_texts",
    "all_w30_schema_texts",
    "all_w30_text_atoms",
    "compose_f2",
    "compose_severity",
    "factor_options",
    "generate_w30_domains",
    "generate_w30_partition",
    "partition_for_domain",
]
