from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAINS_ORDER = ("DH", "DI", "DJ", "DK")
DOMAIN_SEEDS = {
    "DH": 411101,
    "DI": 411107,
    "DJ": 411119,
    "DK": 411131,
}
CASES_PER_DOMAIN = 96
TOTAL_CASES = 384
VIEW_IDS = ("D0", "D1", "D2")
PROTOTYPES_PER_CLASS = 3

DOMAIN_CONTEXT = {
    "DH": "municipal document-delivery incident assessment",
    "DI": "community food-distribution service triage",
    "DJ": "university media-equipment service assessment",
    "DK": "regional bicycle-share support triage",
}

SEVERITY_QUERY_PHRASES = (
    (
        "The disturbance is barely consequential; the intended service remains essentially intact.",
        "A small nuisance is present, yet people can complete the normal task without changing plans.",
        "Only a narrow inconvenience appears and the main workflow continues normally.",
        "The issue is mild enough that routine activity proceeds with almost no functional loss.",
        "Service quality shows a slight imperfection but ordinary use remains fully practical.",
        "The reported effect is limited, localized, and easy to absorb through normal operation.",
        "Users notice a minor flaw without losing meaningful access to the expected service.",
        "The situation can remain in ordinary handling because its practical impact is very small.",
    ),
    (
        "The disruption is clearly felt and requires follow-up, though the core service still works.",
        "People need a manageable workaround while ordinary response procedures remain sufficient.",
        "The problem reduces service quality in a meaningful but contained way.",
        "Routine use is inconvenienced enough to matter, yet no accelerated escalation is necessary.",
        "The condition interferes with normal activity while staying within standard operating capacity.",
        "A persistent service problem needs attention but does not seriously disable the main function.",
        "Users must adapt around the issue, although the essential workflow remains available.",
        "The impact is moderate: more than a minor flaw, less than a major operational loss.",
    ),
    (
        "The disruption removes a major part of expected function and prompt intervention is warranted.",
        "Normal activity is substantially impaired, making ordinary scheduling too slow for the situation.",
        "The issue causes serious operational loss that should be handled ahead of routine work.",
        "A core service function is unreliable enough that delayed action would prolong significant harm.",
        "The practical effect is severe and users cannot depend on normal operation until repair occurs.",
        "The problem blocks an important workflow and requires accelerated response.",
        "Service capability is heavily degraded, producing major consequences short of emergency failure.",
        "The condition is serious enough that prompt escalation is appropriate even though it is not acute.",
    ),
    (
        "The disruption has become an acute failure that requires immediate intervention.",
        "Essential operation is effectively unavailable and emergency escalation is justified now.",
        "The consequences are extreme enough that waiting for ordinary handling is unacceptable.",
        "A critical service breakdown is occurring and response cannot be deferred.",
        "Normal operation has collapsed at the highest-impact end of the scale.",
        "The issue creates immediate serious consequences that require urgent action.",
        "Ordinary response channels are insufficient because the failure is now critical.",
        "The situation represents maximum operational severity and needs immediate escalation.",
    ),
)

CONFIDENCE_QUERY_PHRASES = (
    (
        "The report relies on incomplete indications and unresolved contradictions remain.",
        "Available evidence is too weak or inconsistent to support a dependable conclusion.",
        "Important details have not been corroborated, leaving the account materially doubtful.",
        "The factual basis remains uncertain because supporting records are missing or conflicting.",
        "Current information could still reflect a mistaken account and should be treated cautiously.",
        "Support for the claim is fragmentary and not reliable enough to settle what occurred.",
        "The evidence remains ambiguous, with no completed check establishing the report.",
        "The account is not yet dependable because key confirmation steps have not produced consistent support.",
    ),
    (
        "Credible information supports the report, but one or more final checks remain unfinished.",
        "The evidence is coherent enough for a provisional conclusion while complete confirmation is still open.",
        "Meaningful corroboration exists, although the account has not reached final verification.",
        "The claim has plausible support from reliable indications but still needs a closing validation step.",
        "Available records point consistently toward the report without fully establishing it.",
        "There is substantial preliminary backing, with independent confirmation not yet complete.",
        "The evidence supports acting provisionally while preserving an unresolved verification gap.",
        "The account is reasonably supported but remains short of completed corroboration.",
    ),
    (
        "Independent checks are complete and consistently confirm the reported facts.",
        "The account is backed by finished corroboration from dependable records.",
        "All material verification steps have concluded with evidence supporting the same account.",
        "Reliable sources have completed cross-checking and no meaningful confirmation gap remains.",
        "The factual basis is established through completed independent validation.",
        "The report has strong corroboration and every important check has been resolved.",
        "Evidence from dependable sources agrees after full verification.",
        "The account can be treated as established because confirmation is complete and consistent.",
    ),
)

ABSTRACT_SEVERITY_DEFINITIONS = (
    (
        "Select the minimal-impact category for a localized issue with almost no loss of useful function.",
        "This option represents consequences that ordinary activity can absorb with little adjustment.",
        "Use this option when service remains effectively normal despite a small disturbance.",
    ),
    (
        "Select the managed-impact category for a noticeable problem that standard procedures can still handle.",
        "This option represents meaningful inconvenience without major functional loss.",
        "Use this option when the disruption matters but remains contained.",
    ),
    (
        "Select the major-impact category for substantial service loss requiring prompt intervention.",
        "This option represents serious impairment that should move ahead of routine response.",
        "Use this option when an important function is materially compromised.",
    ),
    (
        "Select the critical-impact category for an acute breakdown requiring immediate escalation.",
        "This option represents extreme consequences beyond ordinary handling.",
        "Use this option when essential operation has reached emergency-level failure.",
    ),
)

ABSTRACT_CONFIDENCE_DEFINITIONS = (
    (
        "Select the unresolved-support category when evidence remains materially uncertain or contradictory.",
        "This option represents accounts without dependable corroboration.",
        "Use this option when the facts cannot yet be established reliably.",
    ),
    (
        "Select the provisional-support category when credible evidence exists but final confirmation is incomplete.",
        "This option represents accounts with meaningful preliminary backing.",
        "Use this option when the evidence supports a tentative conclusion while verification remains open.",
    ),
    (
        "Select the confirmed-support category when independent corroboration has been completed.",
        "This option represents accounts established by dependable finished checks.",
        "Use this option when the relevant facts have been fully verified.",
    ),
)

SEVERITY_PROTOTYPE_PHRASES = (
    (
        "A single delivery arrives a few minutes late, but every requested document is still received and no plan changes.",
        "One kiosk button responds slowly while all required document services remain available through the usual process.",
        "A brief queue forms at one counter and clears without preventing anyone from completing the intended transaction.",
    ),
    (
        "Several users must use a temporary alternate pickup point for the day, but the service remains available through normal staff support.",
        "A recurring delay stretches routine completion times and needs follow-up, while users can still finish the core task.",
        "One service channel is unavailable for several hours, forcing a manageable workaround through another standard channel.",
    ),
    (
        "A major processing function is unavailable to many users, preventing normal completion until staff perform an urgent repair.",
        "An important workflow repeatedly fails across the service, causing substantial disruption that needs prompt intervention.",
        "Most users cannot complete a core transaction because a serious service fault has removed a key capability.",
    ),
    (
        "An essential service has stopped entirely during a time-sensitive operation and immediate emergency action is required.",
        "A critical system failure blocks all essential processing and ordinary escalation would be too slow.",
        "The service is in a complete acute breakdown with immediate serious consequences if operation is not restored.",
    ),
)

CONFIDENCE_PROTOTYPE_PHRASES = (
    (
        "One person's recollection conflicts with the available log and no independent record has confirmed either version.",
        "A claim is based on partial notes with missing timestamps and contradictory details that remain unresolved.",
        "Early indications suggest a possible incident, but no dependable source has yet corroborated the account.",
    ),
    (
        "Two credible records support the same explanation, while a planned independent check has not yet finished.",
        "The available evidence is internally consistent and persuasive, but one final confirmation source is still pending.",
        "A reliable preliminary record and a second supporting indication agree, although completed verification is not yet available.",
    ),
    (
        "Independent logs and a completed secondary review agree on the same facts with no material verification gap.",
        "Multiple dependable records have been cross-checked and all important confirmation steps are complete.",
        "The report has been fully corroborated by separate reliable sources that consistently establish what happened.",
    ),
)

ABSTRACT_SEVERITY_QUESTION = "Which consequence category matches this isolated service-impact description?"
ABSTRACT_CONFIDENCE_QUESTION = "Which evidence-status category matches this isolated support description?"
SEVERITY_PROTOTYPE_QUESTION = "Which concrete service-impact examples are semantically closest to this isolated consequence?"
CONFIDENCE_PROTOTYPE_QUESTION = "Which concrete evidence examples are semantically closest to this isolated support statement?"


@dataclass(frozen=True)
class ReferencePanelAuthorityCase:
    case_id: str
    domain_id: str
    severity: int
    confidence_index: int
    variant: int
    severity_field: str
    confidence_field: str


ABSTRACT_SEVERITY_OPTIONS = tuple(
    LogicalOption(option_id=f"severity22-{index}", criterion_text=views[0])
    for index, views in enumerate(ABSTRACT_SEVERITY_DEFINITIONS)
)
ABSTRACT_CONFIDENCE_OPTIONS = tuple(
    LogicalOption(option_id=f"confidence22-{index}", criterion_text=views[0])
    for index, views in enumerate(ABSTRACT_CONFIDENCE_DEFINITIONS)
)


def severity_prototypes(domain_id: str) -> tuple[tuple[str, ...], ...]:
    context = DOMAIN_CONTEXT[domain_id]
    return tuple(
        tuple(f"In the setting of {context}, {phrase}" for phrase in class_phrases)
        for class_phrases in SEVERITY_PROTOTYPE_PHRASES
    )


def confidence_prototypes(domain_id: str) -> tuple[tuple[str, ...], ...]:
    context = DOMAIN_CONTEXT[domain_id]
    return tuple(
        tuple(f"In the setting of {context}, {phrase}" for phrase in class_phrases)
        for class_phrases in CONFIDENCE_PROTOTYPE_PHRASES
    )


def severity_prototype_options(domain_id: str) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=f"{domain_id.lower()}-severity22-s{class_index}-p{prototype_index}",
            criterion_text=text,
        )
        for class_index, prototypes in enumerate(severity_prototypes(domain_id))
        for prototype_index, text in enumerate(prototypes)
    )


def confidence_prototype_options(domain_id: str) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=f"{domain_id.lower()}-confidence22-c{class_index}-p{prototype_index}",
            criterion_text=text,
        )
        for class_index, prototypes in enumerate(confidence_prototypes(domain_id))
        for prototype_index, text in enumerate(prototypes)
    )


def generate_all_w22() -> tuple[ReferencePanelAuthorityCase, ...]:
    rows = []
    for domain_id in DOMAINS_ORDER:
        context = DOMAIN_CONTEXT[domain_id]
        rng = random.Random(DOMAIN_SEEDS[domain_id])
        for severity in range(4):
            for confidence in range(3):
                c_variants = list(range(8))
                rng.shuffle(c_variants)
                for variant in range(8):
                    c_variant = c_variants[variant]
                    rows.append(
                        ReferencePanelAuthorityCase(
                            case_id=f"{domain_id.lower()}-s{severity}-c{confidence}-v{variant}",
                            domain_id=domain_id,
                            severity=severity,
                            confidence_index=confidence,
                            variant=variant,
                            severity_field=(
                                f"For this {context}, "
                                f"{SEVERITY_QUERY_PHRASES[severity][variant]}"
                            ),
                            confidence_field=(
                                f"For this {context}, "
                                f"{CONFIDENCE_QUERY_PHRASES[confidence][c_variant]}"
                            ),
                        )
                    )
    if len(rows) != TOTAL_CASES:
        raise RuntimeError("W22 requires exactly 384 query cases")
    for domain_id in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != CASES_PER_DOMAIN:
            raise RuntimeError("W22 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W22 severity balance changed")
        if [sum(row.confidence_index == c for row in subset) for c in range(3)] != [32] * 3:
            raise RuntimeError("W22 confidence balance changed")
    return tuple(rows)


def all_w22_query_texts() -> set[str]:
    values = set()
    for row in generate_all_w22():
        values.add(row.severity_field)
        values.add(row.confidence_field)
    return values


def all_w22_prototype_texts() -> set[str]:
    values = set()
    for domain_id in DOMAINS_ORDER:
        for class_values in severity_prototypes(domain_id):
            values.update(class_values)
        for class_values in confidence_prototypes(domain_id):
            values.update(class_values)
    return values


def all_w22_text_atoms() -> set[str]:
    values = all_w22_query_texts() | all_w22_prototype_texts()
    for group in (ABSTRACT_SEVERITY_DEFINITIONS, ABSTRACT_CONFIDENCE_DEFINITIONS):
        for views in group:
            values.update(views)
    values.update(
        (
            ABSTRACT_SEVERITY_QUESTION,
            ABSTRACT_CONFIDENCE_QUESTION,
            SEVERITY_PROTOTYPE_QUESTION,
            CONFIDENCE_PROTOTYPE_QUESTION,
        )
    )
    return values


__all__ = [
    "ABSTRACT_CONFIDENCE_DEFINITIONS",
    "ABSTRACT_CONFIDENCE_OPTIONS",
    "ABSTRACT_CONFIDENCE_QUESTION",
    "ABSTRACT_SEVERITY_DEFINITIONS",
    "ABSTRACT_SEVERITY_OPTIONS",
    "ABSTRACT_SEVERITY_QUESTION",
    "CASES_PER_DOMAIN",
    "CONFIDENCE_PROTOTYPE_QUESTION",
    "DOMAIN_SEEDS",
    "DOMAINS_ORDER",
    "PROTOTYPES_PER_CLASS",
    "ReferencePanelAuthorityCase",
    "SEVERITY_PROTOTYPE_QUESTION",
    "TOTAL_CASES",
    "VIEW_IDS",
    "all_w22_prototype_texts",
    "all_w22_query_texts",
    "all_w22_text_atoms",
    "confidence_prototype_options",
    "confidence_prototypes",
    "generate_all_w22",
    "severity_prototype_options",
    "severity_prototypes",
]
