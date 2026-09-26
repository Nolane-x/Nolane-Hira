from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAINS_ORDER = ("DP", "DQ", "DR", "DS")
DOMAIN_SEEDS = {
    "DP": 431101,
    "DQ": 431107,
    "DR": 431119,
    "DS": 431131,
}
CASES_PER_DOMAIN = 96
TOTAL_CASES = 384
VIEW_IDS = ("D0", "D1", "D2")
PROTOTYPES_PER_CLASS = 3
FACTOR_IDS = ("F0", "F1", "F2")

DOMAIN_CONTEXT = {
    "DP": "municipal recreation-facility access service assessment",
    "DQ": "nonprofit appointment-scheduling service triage",
    "DR": "university print-and-copy service incident assessment",
    "DS": "public EV-charging information service triage",
}

SEVERITY_TO_FACTORS = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (1, 1, 0),
    3: (1, 1, 1),
}
FACTORS_TO_SEVERITY = {value: key for key, value in SEVERITY_TO_FACTORS.items()}

SEVERITY_QUERY_PHRASES = (
    (
        "A tiny presentation flaw is visible, but the normal task finishes unchanged, no workaround is needed, important function remains available, and ordinary handling is sufficient.",
        "The service shows a slight local nuisance while users proceed normally without adaptation, no core capability is lost, and there is no need for accelerated response.",
        "A minor cosmetic irregularity appears without changing how anyone completes the intended task; important functions stay available and routine handling is enough.",
        "Users notice a small imperfection that requires no workaround or operational follow-up, leaves the useful service intact, and carries no urgent consequence.",
        "The disturbance is negligible in practice: normal use continues as expected, important capability is preserved, and standard timing is entirely adequate.",
        "A narrow annoyance occurs but does not materially disrupt the workflow, remove any important function, or justify response outside ordinary handling.",
        "The reported issue is easy to absorb with no meaningful adaptation; the central service remains available and nothing requires prompt or immediate escalation.",
        "A very small defect is present, yet the expected activity proceeds normally, no significant function is impaired, and regular support timing is sufficient.",
    ),
    (
        "The disruption is meaningful enough that users need a supported workaround and staff follow-up, but the important service remains available and immediate escalation is unnecessary.",
        "People must adapt their normal process because the issue materially interferes with use, while the core function still works and routine support can resolve it.",
        "The problem creates a real operational inconvenience requiring an alternate path, yet no major capability is lost and standard response timing remains acceptable.",
        "Users cannot proceed in the usual way and need a manageable workaround, but the essential task is still possible and the situation is not acutely critical.",
        "The incident warrants scheduled attention because it materially changes normal use, although important service capability remains practically available and no emergency action is needed.",
        "A persistent disruption forces users to adjust how they complete the task, while the core function survives and ordinary support procedures remain sufficient.",
        "The condition is more than a trivial nuisance and requires operational follow-up, but it does not remove an important service function or demand immediate intervention.",
        "Normal activity is noticeably disrupted and a workaround is required, yet central functionality remains usable and the response can stay within standard handling.",
    ),
    (
        "An important service function is materially unavailable, creating a major disruption that needs prompt repair, but the situation can still wait for accelerated non-emergency handling.",
        "Users have lost a core capability and cannot complete an important part of the task normally; prompt intervention is warranted, though immediate emergency action is not required.",
        "The incident causes substantial functional loss across a key workflow, so it should move ahead of routine work while remaining short of an acute critical state.",
        "A central capability is heavily impaired and users lose dependable access to it, requiring accelerated response but not immediate emergency escalation.",
        "The problem removes an important function and creates serious operational consequences; repair should be prompt, yet a short delay does not create an acute emergency.",
        "A major part of the service cannot be used reliably and normal activity is significantly blocked, while the response remains urgent rather than immediate.",
        "The disruption has progressed to meaningful loss of a core function, demanding prompt restoration without crossing into a critical must-act-now condition.",
        "Important service capability is substantially unavailable and the impact is severe, but emergency-level intervention is not yet necessary.",
    ),
    (
        "A core service capability has failed in an acutely critical way, normal use is seriously blocked, and delaying intervention is unacceptable.",
        "Important function is unavailable and the consequences are immediate enough that emergency action must begin now rather than enter any ordinary queue.",
        "The service has suffered major functional loss together with acute criticality, so users face serious consequences unless intervention happens immediately.",
        "A central capability has effectively collapsed and the situation cannot tolerate delay; immediate restoration is required.",
        "The incident combines severe loss of important function with time-critical consequences that require action now.",
        "Users have lost essential capability and the failure is acutely consequential, making ordinary or merely prompt handling insufficient.",
        "A major service breakdown is occurring with immediate serious impact, so emergency-level intervention is required without delay.",
        "The core workflow is unavailable under a critical condition where even a short wait is unacceptable and action must be immediate.",
    ),
)

CONFIDENCE_QUERY_PHRASES = (
    (
        "The account is supported only by incomplete indications, with unresolved conflicts and no dependable independent confirmation.",
        "Key evidence is missing or inconsistent, so the reported facts remain materially uncertain.",
        "Available records do not yet corroborate the claim and important discrepancies remain open.",
        "The evidence base is fragmentary enough that no reliable conclusion should be treated as established.",
        "The report lacks completed verification and competing details have not been reconciled.",
        "Current information remains ambiguous because dependable corroboration has not been obtained.",
        "Only weak partial support exists, leaving the account genuinely unresolved.",
        "Important checks are missing and the available evidence does not yet justify a dependable factual conclusion.",
    ),
    (
        "Multiple credible indications support the account, but a final independent verification step has not yet been completed.",
        "The evidence is coherent and persuasive enough for a provisional conclusion while one confirmation gap remains.",
        "Reliable preliminary records point to the same account, although complete corroboration is still pending.",
        "The claim has substantial support from credible sources but has not reached fully completed verification.",
        "Available evidence supports acting provisionally, with a remaining independent check still open.",
        "The factual account is reasonably well supported even though the verification process is not finished.",
        "Several dependable signals agree, but final confirmation has not yet closed every evidential gap.",
        "The report is persuasive and provisionally usable while one material corroboration step remains incomplete.",
    ),
    (
        "Independent checks have been completed and dependable sources consistently establish the reported facts.",
        "The account is fully corroborated by finished verification with no material conflict remaining.",
        "Separate reliable records agree after all important confirmation steps have concluded.",
        "Completed cross-checking from dependable sources establishes the factual account consistently.",
        "Every material verification step has finished and the evidence converges on the same conclusion.",
        "The report is confirmed through completed independent corroboration without an unresolved evidential gap.",
        "Reliable sources have finished validating the account and no meaningful contradiction remains.",
        "The factual basis is established because independent confirmation is complete and mutually consistent.",
    ),
)

DIRECT_SEVERITY_DEFINITIONS = (
    (
        "Select the negligible-impact class when normal activity continues without meaningful adaptation or operational follow-up.",
        "This option represents a small nuisance with important service capability fully preserved.",
        "Use this class when ordinary handling is enough because practical function is essentially unchanged.",
    ),
    (
        "Select the managed-disruption class when users need a workaround or follow-up while important function remains available.",
        "This option represents material inconvenience that changes normal use without major functional loss.",
        "Use this class when the disruption matters but remains manageable within standard support timing.",
    ),
    (
        "Select the major-loss class when an important function is materially unavailable and prompt restoration is warranted.",
        "This option represents serious functional impairment that should move ahead of routine work without requiring immediate emergency action.",
        "Use this class when core capability is substantially lost but a short delay is still tolerable.",
    ),
    (
        "Select the critical-loss class when major functional failure is paired with consequences that cannot tolerate delay.",
        "This option represents acute service breakdown requiring immediate intervention.",
        "Use this class when important capability is unavailable and emergency-level action must begin now.",
    ),
)

CONFIDENCE_DEFINITIONS = (
    (
        "Select the unresolved-support class when evidence is incomplete, conflicting, or not independently corroborated.",
        "This option represents an account that remains too uncertain for dependable factual acceptance.",
        "Use this class when material verification gaps are still open.",
    ),
    (
        "Select the provisional-support class when credible evidence aligns but final verification is not complete.",
        "This option represents substantial preliminary backing with a remaining confirmation gap.",
        "Use this class when a tentative conclusion is justified but not yet fully established.",
    ),
    (
        "Select the confirmed-support class when independent verification is complete and the evidence consistently agrees.",
        "This option represents facts established by finished dependable corroboration.",
        "Use this class when no material confirmation gap remains.",
    ),
)

FACTOR_DEFINITIONS = {
    "F0": (
        (
            "Select the no-meaningful-disruption option when the issue is only a trivial nuisance and normal activity needs no substantive adaptation.",
            "This option means users can complete the expected task essentially normally without a workaround or material operational follow-up.",
            "Use this side when practical workflow disruption is negligible.",
        ),
        (
            "Select the meaningful-disruption option when the issue materially changes normal use and requires adaptation, workaround, or operational follow-up.",
            "This option means the problem creates a real workflow disruption beyond a trivial nuisance.",
            "Use this side when users or operators must meaningfully respond to the disruption.",
        ),
    ),
    "F1": (
        (
            "Select the important-function-preserved option when core service capability remains practically available.",
            "This option means no important or central function has been materially lost.",
            "Use this side when the essential capability can still be relied on.",
        ),
        (
            "Select the major-functional-loss option when an important or core capability is materially unavailable or heavily impaired.",
            "This option means a central service function can no longer be relied on for normal use.",
            "Use this side when serious functional loss has occurred.",
        ),
    ),
    "F2": (
        (
            "Select the no-immediate-criticality option when ordinary or prompt handling is sufficient and a short delay is tolerable.",
            "This option means emergency action is not required right now.",
            "Use this side when the situation can be handled without immediate intervention.",
        ),
        (
            "Select the immediate-criticality option when delay is unacceptable and intervention must begin immediately.",
            "This option means the consequences are acutely critical enough to require action now.",
            "Use this side when ordinary or merely prompt handling is insufficient.",
        ),
    ),
}

FACTOR_REFERENCE_HYPOTHESES = {
    "F0": (
        "The situation is only a trivial nuisance and does not materially disrupt normal activity.",
        "The situation materially disrupts normal activity and requires adaptation, workaround, or operational follow-up.",
    ),
    "F1": (
        "Important service capability remains practically available.",
        "An important or core service capability is materially unavailable or heavily impaired.",
    ),
    "F2": (
        "The situation does not require immediate intervention and can tolerate ordinary or prompt handling.",
        "The situation is acutely critical and requires immediate intervention without delay.",
    ),
}

SEVERITY_PROTOTYPE_PHRASES = (
    (
        "A small display inconsistency appears while users complete the normal task with no workaround and no operational consequence.",
        "One optional indicator is briefly inaccurate, yet all important capability remains available and routine handling is enough.",
        "A minor local defect is noticed but normal use proceeds unchanged and nobody needs to adapt.",
    ),
    (
        "Users switch to a supported alternate path for part of the day while the important service function remains available.",
        "A persistent issue requires staff follow-up and a manageable workaround, but the central task can still be completed.",
        "Normal use is materially inconvenient and users must adapt, although no important capability has been lost.",
    ),
    (
        "A key function is unavailable for many users and prompt repair is needed, but the incident does not require immediate emergency action.",
        "An important workflow is materially impaired and should receive accelerated restoration while a short delay remains tolerable.",
        "Core capability is substantially lost, causing serious disruption that warrants prompt rather than immediate intervention.",
    ),
    (
        "An important service function is down under time-critical conditions where any delay creates immediate serious consequences.",
        "A core workflow has failed completely and emergency restoration must begin now.",
        "Major functional loss is paired with acute criticality, making immediate intervention mandatory.",
    ),
)

CONFIDENCE_PROTOTYPE_PHRASES = (
    (
        "A single partial report conflicts with another record and no independent verification resolves the discrepancy.",
        "Important details are missing and the available evidence does not provide dependable corroboration.",
        "Early indications are inconsistent and no completed check establishes the claimed facts.",
    ),
    (
        "Two credible indications support the same account while one independent confirmation step remains pending.",
        "The available records align and provide strong preliminary support, but final verification is unfinished.",
        "Reliable preliminary evidence points consistently in one direction while a closing corroboration step remains open.",
    ),
    (
        "Independent records and a completed review agree with no material verification gap remaining.",
        "Multiple dependable sources have been fully cross-checked and consistently confirm the same facts.",
        "All relevant confirmation steps are complete and separate reliable evidence establishes the account.",
    ),
)

DIRECT_SEVERITY_QUESTION = "Which frozen four-level operational consequence class fits this isolated severity narrative?"
CONFIDENCE_QUESTION = "Which frozen evidential-support class fits this isolated confidence narrative?"
SEVERITY_PROTOTYPE_QUESTION = "Which fixed operational consequence examples are closest to this isolated severity narrative?"
CONFIDENCE_PROTOTYPE_QUESTION = "Which fixed evidential examples are closest to this isolated confidence narrative?"
FACTOR_QUESTIONS = {
    "F0": "Does this isolated severity narrative indicate a meaningful operational disruption?",
    "F1": "Does this isolated severity narrative indicate major loss of important function?",
    "F2": "Does this isolated severity narrative indicate immediate criticality?",
}


@dataclass(frozen=True)
class AtomicSeverityAuthorityCase:
    case_id: str
    domain_id: str
    severity: int
    confidence_index: int
    variant: int
    factor_vector: tuple[int, int, int]
    severity_field: str
    confidence_field: str


DIRECT_SEVERITY_OPTIONS = tuple(
    LogicalOption(option_id=f"severity24-{index}", criterion_text=views[0])
    for index, views in enumerate(DIRECT_SEVERITY_DEFINITIONS)
)
CONFIDENCE_OPTIONS = tuple(
    LogicalOption(option_id=f"confidence24-{index}", criterion_text=views[0])
    for index, views in enumerate(CONFIDENCE_DEFINITIONS)
)
FACTOR_OPTIONS = {
    factor_id: tuple(
        LogicalOption(
            option_id=f"{factor_id.lower()}24-{value}",
            criterion_text=FACTOR_DEFINITIONS[factor_id][value][0],
        )
        for value in (0, 1)
    )
    for factor_id in FACTOR_IDS
}


def severity_prototypes(domain_id: str) -> tuple[tuple[str, ...], ...]:
    context = DOMAIN_CONTEXT[domain_id]
    return tuple(
        tuple(f"Within {context}, {phrase}" for phrase in class_phrases)
        for class_phrases in SEVERITY_PROTOTYPE_PHRASES
    )


def confidence_prototypes(domain_id: str) -> tuple[tuple[str, ...], ...]:
    context = DOMAIN_CONTEXT[domain_id]
    return tuple(
        tuple(f"Within {context}, {phrase}" for phrase in class_phrases)
        for class_phrases in CONFIDENCE_PROTOTYPE_PHRASES
    )


def severity_prototype_options(domain_id: str) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=f"{domain_id.lower()}-severity24-s{class_index}-p{prototype_index}",
            criterion_text=text,
        )
        for class_index, prototypes in enumerate(severity_prototypes(domain_id))
        for prototype_index, text in enumerate(prototypes)
    )


def confidence_prototype_options(domain_id: str) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=f"{domain_id.lower()}-confidence24-c{class_index}-p{prototype_index}",
            criterion_text=text,
        )
        for class_index, prototypes in enumerate(confidence_prototypes(domain_id))
        for prototype_index, text in enumerate(prototypes)
    )


def severity_factors(severity: int) -> tuple[int, int, int]:
    try:
        return SEVERITY_TO_FACTORS[int(severity)]
    except KeyError as exc:
        raise ValueError(f"invalid W24 severity: {severity}") from exc


def compose_severity(factors: tuple[int, int, int]) -> int | None:
    return FACTORS_TO_SEVERITY.get(tuple(int(x) for x in factors))


def generate_all_w24() -> tuple[AtomicSeverityAuthorityCase, ...]:
    rows = []
    for domain_id in DOMAINS_ORDER:
        context = DOMAIN_CONTEXT[domain_id]
        rng = random.Random(DOMAIN_SEEDS[domain_id])
        for severity in range(4):
            for confidence in range(3):
                confidence_variants = list(range(8))
                rng.shuffle(confidence_variants)
                for variant in range(8):
                    confidence_variant = confidence_variants[variant]
                    rows.append(
                        AtomicSeverityAuthorityCase(
                            case_id=f"{domain_id.lower()}-s{severity}-c{confidence}-v{variant}",
                            domain_id=domain_id,
                            severity=severity,
                            confidence_index=confidence,
                            variant=variant,
                            factor_vector=severity_factors(severity),
                            severity_field=(
                                f"For the {context}, "
                                f"{SEVERITY_QUERY_PHRASES[severity][variant]}"
                            ),
                            confidence_field=(
                                f"For the {context}, "
                                f"{CONFIDENCE_QUERY_PHRASES[confidence][confidence_variant]}"
                            ),
                        )
                    )
    if len(rows) != TOTAL_CASES:
        raise RuntimeError("W24 requires exactly 384 query cases")
    for domain_id in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != CASES_PER_DOMAIN:
            raise RuntimeError("W24 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W24 severity balance changed")
        if [sum(row.confidence_index == c for row in subset) for c in range(3)] != [32] * 3:
            raise RuntimeError("W24 confidence balance changed")
    return tuple(rows)


def all_w24_query_texts() -> set[str]:
    values = set()
    for row in generate_all_w24():
        values.add(row.severity_field)
        values.add(row.confidence_field)
    return values


def all_w24_prototype_texts() -> set[str]:
    values = set()
    for domain_id in DOMAINS_ORDER:
        for class_values in severity_prototypes(domain_id):
            values.update(class_values)
        for class_values in confidence_prototypes(domain_id):
            values.update(class_values)
    return values


def all_w24_text_atoms() -> set[str]:
    values = all_w24_query_texts() | all_w24_prototype_texts()
    for group in (DIRECT_SEVERITY_DEFINITIONS, CONFIDENCE_DEFINITIONS):
        for views in group:
            values.update(views)
    for factor_id in FACTOR_IDS:
        for side_views in FACTOR_DEFINITIONS[factor_id]:
            values.update(side_views)
        values.update(FACTOR_REFERENCE_HYPOTHESES[factor_id])
        values.add(FACTOR_QUESTIONS[factor_id])
    values.update(
        (
            DIRECT_SEVERITY_QUESTION,
            CONFIDENCE_QUESTION,
            SEVERITY_PROTOTYPE_QUESTION,
            CONFIDENCE_PROTOTYPE_QUESTION,
        )
    )
    return values


__all__ = [
    "CASES_PER_DOMAIN",
    "CONFIDENCE_DEFINITIONS",
    "CONFIDENCE_OPTIONS",
    "CONFIDENCE_PROTOTYPE_QUESTION",
    "CONFIDENCE_QUESTION",
    "DIRECT_SEVERITY_DEFINITIONS",
    "DIRECT_SEVERITY_OPTIONS",
    "DIRECT_SEVERITY_QUESTION",
    "DOMAIN_SEEDS",
    "DOMAINS_ORDER",
    "FACTOR_DEFINITIONS",
    "FACTOR_IDS",
    "FACTOR_OPTIONS",
    "FACTOR_QUESTIONS",
    "FACTOR_REFERENCE_HYPOTHESES",
    "FACTORS_TO_SEVERITY",
    "PROTOTYPES_PER_CLASS",
    "SEVERITY_PROTOTYPE_QUESTION",
    "SEVERITY_TO_FACTORS",
    "TOTAL_CASES",
    "VIEW_IDS",
    "AtomicSeverityAuthorityCase",
    "all_w24_prototype_texts",
    "all_w24_query_texts",
    "all_w24_text_atoms",
    "compose_severity",
    "confidence_prototype_options",
    "confidence_prototypes",
    "generate_all_w24",
    "severity_factors",
    "severity_prototype_options",
    "severity_prototypes",
]
