from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import LogicalOption

FACTOR_IDS = ("F0", "F1", "F2")
VIEW_IDS = ("D0", "D1", "D2")
SEVERITY_TO_FACTORS = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (1, 1, 0),
    3: (1, 1, 1),
}
FACTORS_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_FACTORS.items()}

PARTITION_DOMAINS = {
    "train": ("DT", "DU", "DV", "DW"),
    "dev": ("DX",),
    "confirm": ("DY", "DZ"),
}
DOMAINS_ORDER = tuple(
    domain for partition in ("train", "dev", "confirm")
    for domain in PARTITION_DOMAINS[partition]
)
DOMAIN_SEEDS = {
    "DT": 441101,
    "DU": 441107,
    "DV": 441119,
    "DW": 441131,
    "DX": 442207,
    "DY": 443311,
    "DZ": 443323,
}
DOMAIN_CONTEXT = {
    "DT": "municipal street-light service incident assessment",
    "DU": "nonprofit meal-delivery coordination triage",
    "DV": "university equipment-loan service assessment",
    "DW": "regional ferry-ticket support triage",
    "DX": "community permit-inspection scheduling service",
    "DY": "public clinic transport-booking support",
    "DZ": "regional library-material transfer service",
}
CASES_PER_DOMAIN = 96
TRAIN_CASES = 384
DEV_CASES = 96
CONFIRM_CASES = 192

# Twenty-four fresh narratives per severity class. Domain context is prepended
# at materialization time; none of these sentences are copied from W24.
SEVERITY_VARIANTS = (
    (
        "A small visual irregularity appears, yet the intended task proceeds normally, no workaround is needed, and routine handling is fully sufficient.",
        "Users notice a narrow cosmetic defect without any practical change to the workflow, service availability, or response timing.",
        "One nonessential indicator behaves oddly while the useful activity continues exactly as expected and nobody needs to adapt.",
        "The report concerns a minor presentation issue that leaves normal completion untouched and carries no operational consequence.",
        "A slight local inconsistency is visible but does not alter the task, remove useful capability, or require special follow-up.",
        "The issue is noticeable only at the surface level; ordinary use continues without adaptation and standard service timing is appropriate.",
        "A tiny defect affects appearance rather than function, so users continue normally and no accelerated action is justified.",
        "There is a small nuisance with no meaningful workflow disruption, no loss of important capability, and no time-sensitive consequence.",
        "An optional display element is imperfect while all practical actions remain available and the matter can remain in the routine queue.",
        "The observed flaw is easy to ignore in practice because the expected work completes normally and no special operational response is needed.",
        "A minor localized problem exists without forcing any alternate process or affecting an important service function.",
        "Normal activity is essentially unchanged despite the reported issue, and there is no reason to move the case ahead of ordinary handling.",
        "The condition causes negligible practical impact: users do not change behavior, important function stays intact, and timing is noncritical.",
        "A superficial inconsistency is present but the central workflow is unaffected and no meaningful intervention is required.",
        "The issue remains below the threshold of operational disruption; expected use proceeds normally with no workaround.",
        "A small annoyance is reported, though it neither blocks nor materially complicates the intended activity.",
        "The service behaves normally for practical purposes despite a minor imperfection that can be handled through ordinary maintenance.",
        "There is no substantive loss or adaptation requirement; only a limited nuisance is visible to users.",
        "The reported condition has trivial practical effect and does not change how the task is completed.",
        "A noncritical detail is imperfect while core use remains fully intact and the case can wait for standard review.",
        "The event is minor enough that users keep their normal process, important capability stays available, and no prompt escalation is warranted.",
        "Only a small local defect is present; it does not create operational follow-up beyond routine housekeeping.",
        "The issue has cosmetic or peripheral impact only, with no meaningful effect on normal service use.",
        "Users can proceed exactly as usual despite a tiny imperfection, so ordinary handling remains adequate.",
    ),
    (
        "The problem materially changes normal use and requires a supported workaround, while important service capability remains available and no urgent escalation is needed.",
        "Users must adapt their usual process because of the disruption, but the central task can still be completed through an alternate path.",
        "The incident creates a real operational inconvenience that warrants follow-up without removing the important function itself.",
        "Normal workflow is disrupted enough to require an alternate procedure, yet the core capability remains practically usable.",
        "Staff need to address a persistent service disturbance because users must change behavior, although no major function has been lost.",
        "The issue is more than cosmetic and forces a manageable workaround, but the essential activity remains possible under standard support timing.",
        "A meaningful disruption affects routine use and requires adaptation, while the important service function continues to operate.",
        "Users cannot follow the ordinary path and need a practical substitute process, but the central outcome is still achievable.",
        "The event produces sustained inconvenience requiring operational attention, though it stops short of major functional loss.",
        "A nontrivial service disturbance changes how people complete the task, but core capability is still available and delay is tolerable.",
        "The normal process is materially inconvenient and a workaround is necessary, yet important function has not failed.",
        "Operational follow-up is justified because routine use is disrupted, while the main service remains accessible.",
        "The problem forces users to adjust their workflow in a meaningful way but does not block the essential task.",
        "A persistent issue requires adaptation and planned support intervention, although the core service can still be relied on.",
        "Normal use is materially altered by the incident, but users retain a workable route to the intended result.",
        "The disruption is substantial enough to demand a workaround and follow-up, while major capability remains preserved.",
        "Users encounter a real process obstacle that needs attention, but important functionality continues and emergency handling is unnecessary.",
        "The issue materially interferes with the normal route, though a supported alternative keeps the central task available.",
        "A practical workaround is required for reliable use, but the incident has not progressed to loss of a core function.",
        "The service remains usable only with adaptation, making the issue operationally meaningful but not a major outage.",
        "People must change their usual steps because of the problem, while the important capability itself remains intact.",
        "The case deserves scheduled operational action because normal activity is disrupted, yet no central function is unavailable.",
        "The incident is a genuine workflow disruption rather than a nuisance, but the essential outcome remains reachable.",
        "Users need a nontrivial alternate path to proceed, although important service capability is still present and the situation is not acute.",
    ),
    (
        "An important service function is materially unavailable, so normal completion of a central task is blocked and prompt restoration is warranted without emergency-level immediacy.",
        "Users have lost dependable access to a core capability and need accelerated repair, although a short delay does not create an acute critical consequence.",
        "The incident removes an important part of the service, producing major functional loss that should be handled promptly rather than routinely.",
        "A central workflow cannot be completed normally because a key function is unavailable, but immediate emergency intervention is not required.",
        "Important capability is heavily impaired and ordinary operation is no longer reliable, calling for prompt corrective action.",
        "The problem has advanced beyond workaround-level inconvenience into major functional loss, while remaining short of a must-act-now emergency.",
        "Users cannot depend on a core service function and significant activity is blocked until restoration occurs.",
        "A key operational capability is down or severely impaired, so the case should be prioritized even though a brief delay remains tolerable.",
        "The incident causes serious loss of useful function and blocks an important task under normal conditions.",
        "Core service availability is materially reduced, creating a major operational impact that warrants accelerated response.",
        "An essential part of the workflow is unavailable and users cannot complete the important task through ordinary operation.",
        "The service has suffered substantial functional loss that requires prompt intervention but not immediate emergency action.",
        "A major capability is no longer reliably available, making the event significantly more severe than a manageable disruption.",
        "Users face a blocked central function and need restoration on an accelerated timescale, though the condition is not acutely time critical.",
        "The issue prevents normal completion of an important activity because a key function has materially failed.",
        "A central service feature is unavailable to users, producing serious operational loss that should move ahead of routine cases.",
        "The problem removes dependable access to important capability, while still allowing a short response window before consequences become critical.",
        "Normal work is significantly blocked by loss of a key function, so prompt repair is necessary.",
        "The incident represents major functional impairment rather than mere adaptation difficulty, but it does not yet require immediate emergency escalation.",
        "An important workflow component has failed and users cannot rely on the service for a central task.",
        "The event creates substantial operational loss by removing a core capability, warranting prioritized restoration.",
        "A key function is effectively unavailable and the intended task cannot be completed normally until service is restored.",
        "The incident causes a serious capability gap that requires prompt action while remaining noncritical in immediate timing.",
        "Important service function is materially lost, producing major disruption without a present must-act-now consequence.",
    ),
    (
        "A core service function has failed under conditions where delay creates immediate serious consequences, so intervention must begin at once.",
        "Important capability is unavailable and the situation is acutely time critical, making ordinary or merely prompt handling insufficient.",
        "The incident combines major functional loss with immediate consequence, requiring emergency-level action without delay.",
        "A central workflow is down and even a short wait is unacceptable because serious impact is occurring now.",
        "Users have lost essential capability in a condition that demands immediate restoration rather than entry into any normal queue.",
        "The service failure is both major and acutely critical, so action must start now to prevent immediate serious consequences.",
        "An important function is unavailable in a time-sensitive situation where postponing intervention is not acceptable.",
        "The event has escalated to critical functional loss with immediate impact, requiring response at once.",
        "Core capability has collapsed and the consequences of waiting are severe enough that emergency handling is required now.",
        "A major service breakdown is occurring under acute conditions that cannot tolerate normal support timing.",
        "The central task is blocked by functional failure and the surrounding situation makes any meaningful delay unacceptable.",
        "Important functionality is lost and immediate intervention is required because serious consequences are already imminent.",
        "The issue represents a critical outage: major capability is unavailable and action must begin without delay.",
        "A core service failure has immediate high-impact consequences, so it cannot be managed through routine or merely accelerated response.",
        "Users face loss of essential function in a situation that requires emergency restoration now.",
        "The incident combines severe operational loss with a strict time constraint that makes delayed action unacceptable.",
        "A central capability is unavailable and the consequences are acutely critical, requiring immediate escalation.",
        "The service has entered a must-act-now condition because important function is lost and serious impact is immediate.",
        "A major functional failure is unfolding with no tolerable response delay, so intervention must start at once.",
        "Important service capability has failed under emergency-level circumstances where waiting would cause immediate harm or serious consequence.",
        "The core workflow is unavailable and the event is time critical enough that ordinary prioritization is insufficient.",
        "A severe outage is paired with acute urgency, making immediate corrective action mandatory.",
        "Users have lost a central capability and the surrounding conditions leave no safe or acceptable delay before intervention.",
        "The incident is critically severe because major function is unavailable and response must be immediate.",
    ),
)

FACTOR_DEFINITIONS = {
    "F0": (
        (
            "Choose the low-impact side when practical use remains essentially unchanged and no meaningful workaround or operational adaptation is required.",
            "This side describes an issue whose effect stays peripheral enough that users continue the ordinary process.",
            "Use this side when the event does not create a substantive workflow disturbance.",
        ),
        (
            "Choose the disruption-present side when normal use is materially altered and users or operators must adapt, use an alternate path, or perform nontrivial follow-up.",
            "This side describes a real operational disturbance beyond a minor nuisance.",
            "Use this side when the event meaningfully changes how the work is carried out.",
        ),
    ),
    "F1": (
        (
            "Choose the core-capability-retained side when the important service function remains practically available despite any inconvenience.",
            "This side means the central task can still be completed because major functionality has not been lost.",
            "Use this side when important capability remains dependable enough for practical use.",
        ),
        (
            "Choose the core-capability-lost side when an important function is unavailable or impaired enough that a central task cannot be completed normally.",
            "This side means major functional loss has occurred rather than mere inconvenience.",
            "Use this side when users can no longer rely on a key service capability.",
        ),
    ),
    "F2": (
        (
            "Choose the delay-tolerable side when the case can wait for ordinary or prompt handling without immediate serious consequence.",
            "This side means accelerated or emergency action is not required at once.",
            "Use this side when a short response delay remains acceptable.",
        ),
        (
            "Choose the immediate-action side when the condition cannot tolerate delay and intervention must begin now because serious consequences are immediate.",
            "This side means the situation is acutely time critical rather than simply important.",
            "Use this side when emergency-level response timing is required.",
        ),
    ),
}

FACTOR_REFERENCE_HYPOTHESES = {
    "F0": (
        "The event does not materially change the normal workflow or require substantive adaptation.",
        "The event materially changes normal use and requires adaptation, an alternate path, or meaningful operational follow-up.",
    ),
    "F1": (
        "The important service capability remains practically available for the central task.",
        "An important service capability is unavailable or impaired enough to block the central task.",
    ),
    "F2": (
        "A short response delay is acceptable and immediate intervention is not required.",
        "The situation cannot tolerate delay and requires immediate intervention.",
    ),
}

FACTOR_QUESTIONS = {
    "F0": "Which side best captures whether this event creates a substantive workflow disruption?",
    "F1": "Which side best captures whether important service capability has been materially lost?",
    "F2": "Which side best captures whether immediate intervention is required?",
}

FACTOR_OPTIONS = {
    factor_id: tuple(
        LogicalOption(
            option_id=f"{factor_id.lower()}25-{value}",
            criterion_text=FACTOR_DEFINITIONS[factor_id][value][0],
        )
        for value in (0, 1)
    )
    for factor_id in FACTOR_IDS
}


@dataclass(frozen=True)
class AtomicGeometryCase:
    case_id: str
    domain_id: str
    partition: str
    severity: int
    variant: int
    factor_vector: tuple[int, int, int]
    severity_field: str


def severity_factors(severity: int) -> tuple[int, int, int]:
    if int(severity) not in SEVERITY_TO_FACTORS:
        raise ValueError(f"invalid W25 severity: {severity}")
    return SEVERITY_TO_FACTORS[int(severity)]


def compose_severity(factors: Iterable[int]) -> int | None:
    return FACTORS_TO_SEVERITY.get(tuple(int(x) for x in factors))


def domain_partition(domain_id: str) -> str:
    for partition, domains in PARTITION_DOMAINS.items():
        if domain_id in domains:
            return partition
    raise ValueError(f"unknown W25 domain: {domain_id}")


def generate_w25_domains(domains: Iterable[str]) -> tuple[AtomicGeometryCase, ...]:
    rows: list[AtomicGeometryCase] = []
    for domain_id in tuple(domains):
        context = DOMAIN_CONTEXT[domain_id]
        partition = domain_partition(domain_id)
        for severity in range(4):
            for variant, phrase in enumerate(SEVERITY_VARIANTS[severity]):
                rows.append(
                    AtomicGeometryCase(
                        case_id=f"{domain_id.lower()}-s{severity}-v{variant}",
                        domain_id=domain_id,
                        partition=partition,
                        severity=severity,
                        variant=variant,
                        factor_vector=severity_factors(severity),
                        severity_field=f"For the {context}, {phrase}",
                    )
                )
    for domain_id in tuple(domains):
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != CASES_PER_DOMAIN:
            raise RuntimeError("W25 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W25 severity balance changed")
    return tuple(rows)


def generate_w25_partition(partition: str) -> tuple[AtomicGeometryCase, ...]:
    if partition not in PARTITION_DOMAINS:
        raise ValueError(f"unknown W25 partition: {partition}")
    return generate_w25_domains(PARTITION_DOMAINS[partition])


def all_w25_query_texts(partitions: Iterable[str] = ("train", "dev", "confirm")) -> set[str]:
    values: set[str] = set()
    for partition in tuple(partitions):
        values.update(row.severity_field for row in generate_w25_partition(partition))
    return values


def all_w25_schema_texts() -> set[str]:
    values: set[str] = set()
    for factor_id in FACTOR_IDS:
        values.add(FACTOR_QUESTIONS[factor_id])
        values.update(FACTOR_REFERENCE_HYPOTHESES[factor_id])
        for side in FACTOR_DEFINITIONS[factor_id]:
            values.update(side)
    return values


def all_w25_text_atoms(partitions: Iterable[str] = ("train", "dev", "confirm")) -> set[str]:
    return all_w25_query_texts(partitions) | all_w25_schema_texts()


__all__ = [
    "CASES_PER_DOMAIN",
    "CONFIRM_CASES",
    "DEV_CASES",
    "DOMAIN_CONTEXT",
    "DOMAIN_SEEDS",
    "DOMAINS_ORDER",
    "FACTOR_DEFINITIONS",
    "FACTOR_IDS",
    "FACTOR_OPTIONS",
    "FACTOR_QUESTIONS",
    "FACTOR_REFERENCE_HYPOTHESES",
    "FACTORS_TO_SEVERITY",
    "PARTITION_DOMAINS",
    "SEVERITY_TO_FACTORS",
    "TRAIN_CASES",
    "VIEW_IDS",
    "AtomicGeometryCase",
    "all_w25_query_texts",
    "all_w25_schema_texts",
    "all_w25_text_atoms",
    "compose_severity",
    "domain_partition",
    "generate_w25_domains",
    "generate_w25_partition",
    "severity_factors",
]
