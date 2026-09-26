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
    "qualification": ("EA", "EB"),
    "train": ("EC", "ED", "EE", "EF"),
    "dev": ("EG",),
    "confirm": ("EH", "EI"),
}
DOMAIN_SEEDS = {
    "EA": 451101,
    "EB": 451107,
    "EC": 452201,
    "ED": 452207,
    "EE": 452219,
    "EF": 452231,
    "EG": 453307,
    "EH": 454401,
    "EI": 454409,
}
DOMAIN_CONTEXT = {
    "EA": "municipal community-room reservation incident assessment",
    "EB": "nonprofit volunteer-shift coordination incident assessment",
    "EC": "university campus-shuttle support incident assessment",
    "ED": "regional recycling-pickup support triage",
    "EE": "public arts-program registration service assessment",
    "EF": "community broadband-install scheduling support",
    "EG": "municipal document-delivery service assessment",
    "EH": "public pharmacy pickup coordination support",
    "EI": "regional mobility-pass support incident assessment",
}
CASES_PER_DOMAIN = 96

# Fresh W26 surface language. These sentences intentionally avoid exact W25
# wording while preserving the same preregistered atomic factor semantics.
SEVERITY_VARIANTS = (
    (
        "A small peripheral irregularity is visible, but people complete the intended activity in the usual way and no operational adjustment is needed.",
        "The report concerns a minor presentation defect while practical service use remains unchanged and standard handling is sufficient.",
        "One optional detail is imperfect, yet the useful task proceeds normally without an alternate process or accelerated response.",
        "A narrow nuisance is present without changing the workflow, removing important capability, or creating time pressure.",
        "Users can follow the normal path from start to finish despite a slight nonessential inconsistency.",
        "The condition affects appearance or convenience only at a negligible level, leaving practical operation intact.",
        "A localized issue can be noted for routine maintenance while ordinary activity continues without adaptation.",
        "The observed defect has no substantive effect on how people carry out the task and does not warrant special prioritization.",
        "A small side issue exists, but there is no meaningful interruption and no reason to alter the standard process.",
        "The service remains practically normal; only a limited nonfunctional imperfection has been reported.",
        "A minor inconsistency is noticeable without creating a workaround, loss of useful function, or urgent consequence.",
        "The issue stays below the level of operational impact, so users continue exactly through the expected route.",
        "A superficial flaw is present while the central activity remains fully available and routine response timing is appropriate.",
        "The report describes a low-impact annoyance that neither blocks nor materially complicates the intended use.",
        "There is a small local problem, but no one needs to change behavior and no important service capability is affected.",
        "Normal completion remains straightforward even though a noncritical detail is not behaving perfectly.",
        "A limited peripheral defect exists without altering practical use, requiring follow-up, or increasing response urgency.",
        "The condition is operationally negligible: the normal route works, important function remains intact, and delay is harmless.",
        "Only a cosmetic or optional element is affected while the expected activity remains fully usable.",
        "Users encounter a tiny imperfection but can ignore it operationally because the task proceeds as designed.",
        "The event produces no meaningful change in service use and can remain in the ordinary support queue.",
        "A nonessential detail is wrong, yet all useful actions remain available and no alternate workflow is required.",
        "The reported problem is minor enough that practical operation, capability, and timing all remain normal.",
        "A slight irregularity is present with no substantive workflow consequence and no need for exceptional handling.",
    ),
    (
        "Normal use is disrupted enough that people must follow an alternate procedure, although the important service capability remains available.",
        "Users need a practical workaround because the usual route is unreliable, but the central task can still be completed.",
        "The incident creates a genuine operational inconvenience that deserves follow-up without removing the core function.",
        "People must change their normal steps to proceed, while the important outcome remains reachable through a supported alternative.",
        "The standard process no longer works smoothly and requires adaptation, yet major functionality is still present.",
        "A meaningful disturbance changes how the task is performed, but users retain dependable access to the central capability.",
        "Operational attention is warranted because routine use is materially altered, although the essential service remains usable.",
        "The issue forces a nontrivial workaround while stopping short of blocking the important task itself.",
        "Users cannot rely on the ordinary path and must adapt, but a workable route to the intended result is still available.",
        "The event has practical operational impact beyond a nuisance, though the core service function remains intact.",
        "A persistent disturbance requires people to modify their workflow, but no major capability has been lost.",
        "Normal activity is materially less convenient and needs an alternate method, while the central function continues to operate.",
        "The problem changes everyday use enough to justify operational follow-up, yet it does not prevent the main task.",
        "A real process obstacle requires adaptation, but users can still accomplish the essential objective.",
        "The service remains usable only after a meaningful change in procedure, without any immediate critical consequence.",
        "Routine behavior must be adjusted because of the incident, although important function is still practically available.",
        "The issue produces sustained workflow disruption that requires attention while preserving the main service capability.",
        "A supported substitute path is needed for reliable use, but the essential task is not blocked.",
        "People must work around the problem in a substantive way, though the service still provides its central function.",
        "The incident is operationally meaningful because the normal route is impaired, yet major functional loss has not occurred.",
        "A nontrivial alternate process is necessary, but users retain access to the important capability and emergency handling is unnecessary.",
        "The issue causes enough disruption to require planned intervention while the core task remains achievable.",
        "Users need to depart from the usual workflow to proceed, although the service still supports the central outcome.",
        "The event is more than a nuisance and changes normal operation, but it has not become a major outage.",
    ),
    (
        "A key service capability is unavailable, preventing normal completion of an important task, but a short response delay remains tolerable.",
        "Users have lost practical access to a central function and need prioritized restoration without a must-act-now emergency.",
        "The incident blocks an important activity because a core capability has materially failed, while immediate intervention is not yet required.",
        "A major functional loss prevents normal service completion and warrants prompt repair rather than routine scheduling.",
        "An important part of the service is down, leaving users unable to complete the central task through normal operation.",
        "The problem has progressed beyond workaround-level disruption into loss of a key function, though a brief delay is acceptable.",
        "Core functionality is materially impaired and important work is blocked until restoration occurs.",
        "A central capability cannot be relied upon, so the case requires accelerated handling without emergency-level timing.",
        "The service has lost an important function and users cannot complete a primary activity in the usual way.",
        "A key operational feature is unavailable, producing serious functional impact that should be addressed promptly.",
        "The issue removes dependable access to central capability while still leaving a limited response window.",
        "Important functionality has failed and normal completion is blocked, but the situation can tolerate prompt rather than immediate action.",
        "Users face a substantial capability gap because a core part of the service is unavailable.",
        "The event causes major functional impairment and should move ahead of routine cases, although it is not acutely time critical.",
        "A central task cannot be completed because an important service component has materially failed.",
        "The service is no longer practically complete: a key function is missing and prioritized restoration is required.",
        "Normal work is blocked by loss of an important capability, while the consequences do not demand intervention this instant.",
        "A major service function is unavailable and users need restoration on an accelerated but nonemergency schedule.",
        "The incident creates serious operational loss by removing a central capability while allowing a short delay before action.",
        "Users cannot rely on the service for an important task because a key function is down.",
        "A substantial functional outage has occurred, requiring prompt corrective work but not immediate emergency escalation.",
        "An important workflow component is unavailable, preventing normal completion until service is restored.",
        "The issue causes major capability loss and prioritized response is appropriate even though immediate action is not mandatory.",
        "A core function has materially failed, creating serious disruption without a present no-delay consequence.",
    ),
    (
        "A central service capability is unavailable in a situation where even a short delay would create immediate serious consequences.",
        "Users have lost an important function under acutely time-sensitive conditions, so intervention must begin at once.",
        "The incident combines major functional failure with a no-delay consequence that requires emergency-level response.",
        "A core task is blocked and the surrounding conditions make any meaningful wait unacceptable.",
        "Important service capability has failed in a situation that cannot safely tolerate ordinary or merely prompt handling.",
        "The outage is both functionally major and immediately critical, requiring action now rather than placement in a queue.",
        "A central function is unavailable and serious consequences are imminent unless restoration begins immediately.",
        "The event has escalated to major capability loss with acute timing pressure that demands intervention without delay.",
        "Users cannot perform the essential task and the consequence of waiting is severe enough to require immediate escalation.",
        "A key service breakdown is occurring under conditions where postponing action is not acceptable.",
        "Core functionality is down and immediate response is necessary because the impact is already time critical.",
        "The incident removes an essential capability while leaving no reasonable delay before corrective action must start.",
        "An important function has failed in a must-act-now situation with immediate serious operational consequences.",
        "The service outage requires emergency handling because central capability is lost and the response window is effectively zero.",
        "Users face a blocked core task together with acute consequences that make immediate intervention mandatory.",
        "A major functional failure is unfolding in circumstances where normal prioritization would be too slow.",
        "Important capability is unavailable and the case has become critically time sensitive, requiring action at once.",
        "The event combines severe operational loss and immediate consequence, so delayed handling is not acceptable.",
        "A central service function has collapsed and urgent intervention must start now to avoid immediate serious impact.",
        "The problem is critically severe because users have lost key capability under conditions that cannot tolerate delay.",
        "A major outage is paired with acute timing constraints that require emergency restoration immediately.",
        "The essential workflow is unavailable and the present circumstances make even brief postponement unacceptable.",
        "Users have lost a central capability in an emergency-level situation where action must begin without delay.",
        "The incident is both a major functional outage and an immediate critical event, so response must start now.",
    ),
)

FACTOR_DEFINITIONS = {
    "F0": (
        (
            "Select the minimal-disruption side when practical activity follows the normal route and no meaningful adaptation is required.",
            "This side represents conditions that remain operationally peripheral rather than changing how the work is carried out.",
            "Use this side when users do not need a substantive alternate process or operational adjustment.",
        ),
        (
            "Select the material-disruption side when ordinary use is changed enough to require adaptation, a workaround, or meaningful follow-up.",
            "This side represents a real workflow disturbance rather than a minor nuisance.",
            "Use this side when people must materially change how they perform the activity.",
        ),
    ),
    "F1": (
        (
            "Select the capability-retained side when the important service function remains practically available for the central task.",
            "This side means major functionality is still usable even if the workflow is inconvenient.",
            "Use this side when users can still depend on the core capability.",
        ),
        (
            "Select the capability-lost side when an important service function is unavailable or impaired enough to block the central task.",
            "This side represents major functional loss rather than workflow inconvenience alone.",
            "Use this side when users cannot practically rely on a key capability.",
        ),
    ),
    "F2": (
        (
            "Select the delay-tolerable side when ordinary or prompt handling is sufficient and a short wait does not create immediate serious consequence.",
            "This side means the case is not in a must-act-now state.",
            "Use this side when intervention can occur after a limited response interval.",
        ),
        (
            "Select the no-delay side when serious consequences are immediate enough that intervention must begin now.",
            "This side represents acute critical timing rather than simple importance or priority.",
            "Use this side when even a short delay is unacceptable.",
        ),
    ),
}

FACTOR_REFERENCE_HYPOTHESES = {
    "F0": (
        "The situation leaves the normal workflow essentially unchanged and does not require meaningful adaptation.",
        "The situation materially changes normal use and requires adaptation, a workaround, or substantial operational follow-up.",
    ),
    "F1": (
        "The important service capability remains practically usable for the central activity.",
        "An important service capability is unavailable or impaired enough to prevent the central activity.",
    ),
    "F2": (
        "The situation can tolerate a short response delay without immediate serious consequence.",
        "The situation cannot tolerate delay and requires intervention to begin immediately.",
    ),
}

FACTOR_QUESTIONS = {
    "F0": "Which alternative best describes the degree of workflow disruption in this situation?",
    "F1": "Which alternative best describes whether an important service capability is still practically available?",
    "F2": "Which alternative best describes whether the response can tolerate any meaningful delay?",
}

FACTOR_OPTIONS = {
    factor_id: tuple(
        LogicalOption(
            option_id=f"{factor_id.lower()}26-{value}",
            criterion_text=FACTOR_DEFINITIONS[factor_id][value][0],
        )
        for value in (0, 1)
    )
    for factor_id in FACTOR_IDS
}


@dataclass(frozen=True)
class ProjectionReplicationCase:
    case_id: str
    domain_id: str
    partition: str
    severity: int
    variant: int
    factor_vector: tuple[int, int, int]
    severity_field: str


def severity_factors(severity: int) -> tuple[int, int, int]:
    if int(severity) not in SEVERITY_TO_FACTORS:
        raise ValueError(f"invalid W26 severity: {severity}")
    return SEVERITY_TO_FACTORS[int(severity)]


def compose_severity(factors: Iterable[int]) -> int | None:
    return FACTORS_TO_SEVERITY.get(tuple(int(x) for x in factors))


def domain_partition(domain_id: str) -> str:
    for partition, domains in PARTITION_DOMAINS.items():
        if domain_id in domains:
            return partition
    raise ValueError(f"unknown W26 domain: {domain_id}")


def generate_w26_domains(domains: Iterable[str]) -> tuple[ProjectionReplicationCase, ...]:
    rows: list[ProjectionReplicationCase] = []
    for domain_id in tuple(domains):
        if domain_id not in DOMAIN_CONTEXT:
            raise ValueError(f"unknown W26 domain: {domain_id}")
        context = DOMAIN_CONTEXT[domain_id]
        partition = domain_partition(domain_id)
        for severity in range(4):
            for variant, phrase in enumerate(SEVERITY_VARIANTS[severity]):
                rows.append(
                    ProjectionReplicationCase(
                        case_id=f"{domain_id.lower()}-s{severity}-v{variant}",
                        domain_id=domain_id,
                        partition=partition,
                        severity=severity,
                        variant=variant,
                        factor_vector=severity_factors(severity),
                        severity_field=f"Within the {context}, {phrase}",
                    )
                )
    for domain_id in tuple(domains):
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != CASES_PER_DOMAIN:
            raise RuntimeError("W26 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W26 severity balance changed")
    return tuple(rows)


def generate_w26_partition(partition: str) -> tuple[ProjectionReplicationCase, ...]:
    if partition not in PARTITION_DOMAINS:
        raise ValueError(f"unknown W26 partition: {partition}")
    return generate_w26_domains(PARTITION_DOMAINS[partition])


def all_w26_query_texts(partitions: Iterable[str] = ("qualification", "train", "dev", "confirm")) -> set[str]:
    values: set[str] = set()
    for partition in tuple(partitions):
        values.update(row.severity_field for row in generate_w26_partition(partition))
    return values


def all_w26_schema_texts() -> set[str]:
    values: set[str] = set()
    for factor_id in FACTOR_IDS:
        values.add(FACTOR_QUESTIONS[factor_id])
        values.update(FACTOR_REFERENCE_HYPOTHESES[factor_id])
        for side in FACTOR_DEFINITIONS[factor_id]:
            values.update(side)
    return values


def all_w26_text_atoms(partitions: Iterable[str] = ("qualification", "train", "dev", "confirm")) -> set[str]:
    return all_w26_query_texts(partitions) | all_w26_schema_texts()


__all__ = [
    "CASES_PER_DOMAIN",
    "DOMAIN_CONTEXT",
    "DOMAIN_SEEDS",
    "FACTOR_DEFINITIONS",
    "FACTOR_IDS",
    "FACTOR_OPTIONS",
    "FACTOR_QUESTIONS",
    "FACTOR_REFERENCE_HYPOTHESES",
    "FACTORS_TO_SEVERITY",
    "PARTITION_DOMAINS",
    "SEVERITY_TO_FACTORS",
    "VIEW_IDS",
    "ProjectionReplicationCase",
    "all_w26_query_texts",
    "all_w26_schema_texts",
    "all_w26_text_atoms",
    "compose_severity",
    "domain_partition",
    "generate_w26_domains",
    "generate_w26_partition",
    "severity_factors",
]
