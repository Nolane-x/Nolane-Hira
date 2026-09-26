from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAINS_ORDER = ("DL", "DM", "DN", "DO")
DOMAIN_SEEDS = {
    "DL": 421101,
    "DM": 421107,
    "DN": 421119,
    "DO": 421131,
}
CASES_PER_DOMAIN = 96
TOTAL_CASES = 384
VIEW_IDS = ("D0", "D1", "D2")
PROTOTYPES_PER_CLASS = 3

DOMAIN_CONTEXT = {
    "DL": "public museum reservation service incident assessment",
    "DM": "neighborhood waste-collection information service triage",
    "DN": "university room-booking service assessment",
    "DO": "regional parcel-locker support triage",
}

SEVERITY_QUERY_PHRASES = (
    (
        "The fault is barely noticeable and the intended task remains available without meaningful adjustment.",
        "A small imperfection appears, but normal use continues and nobody loses practical access.",
        "The incident creates only a light inconvenience that routine operation absorbs immediately.",
        "Users can proceed as planned because the disturbance has almost no functional consequence.",
        "The service is essentially intact, with only a minor localized annoyance.",
        "Ordinary handling is sufficient because useful operation remains nearly unchanged.",
        "The issue is limited enough that the expected workflow completes normally.",
        "There is a slight service blemish, yet the practical outcome is unaffected.",
    ),
    (
        "The problem is significant enough to require follow-up, while the central service remains usable.",
        "A temporary workaround is needed, but standard operating procedures can contain the disruption.",
        "Users experience a material inconvenience without losing the core function.",
        "The service is degraded in a noticeable but manageable way that does not require urgent escalation.",
        "Normal activity is disrupted enough to matter, though routine response capacity is still adequate.",
        "The condition needs scheduled attention because it interferes with use without causing major failure.",
        "People must adapt around the issue, but the essential task can still be completed.",
        "The impact is meaningful and persistent while remaining within ordinary support handling.",
    ),
    (
        "A major function is unavailable and prompt intervention is needed to restore dependable service.",
        "The incident substantially blocks normal activity and should be escalated ahead of routine work.",
        "Users have lost an important capability, creating serious operational disruption.",
        "The service has degraded enough that delay would prolong a major functional loss.",
        "A key workflow cannot be completed reliably until accelerated repair occurs.",
        "The consequences are severe, with substantial access or capability removed.",
        "Normal support timing is no longer appropriate because the disruption is materially serious.",
        "The incident causes major impairment short of a complete emergency-level breakdown.",
    ),
    (
        "The service has entered a critical failure state and immediate intervention is required.",
        "Essential operation is unavailable, making ordinary response timing unacceptable.",
        "The incident creates acute consequences that demand emergency-level escalation now.",
        "A complete high-impact breakdown is preventing essential use and cannot be deferred.",
        "The situation has reached the maximum consequence tier with immediate action necessary.",
        "Normal operation has effectively collapsed under a critical service failure.",
        "The failure is acute enough that standard queues are insufficient for the required response.",
        "Immediate restoration is necessary because the disruption has become extreme and essential.",
    ),
)

CONFIDENCE_QUERY_PHRASES = (
    (
        "The account rests on fragmentary evidence and important contradictions have not been resolved.",
        "Available information is incomplete enough that the reported facts remain genuinely uncertain.",
        "No dependable independent check has yet established which version of events is correct.",
        "The report remains doubtful because key records are missing, inconsistent, or unverified.",
        "Current indications are too weak to support a reliable conclusion about what happened.",
        "Evidence is sparse and conflicting, so the claim should still be treated as unresolved.",
        "The factual picture remains ambiguous because corroboration has not been obtained.",
        "Important verification gaps remain and the available support is not dependable enough to settle the claim.",
    ),
    (
        "Several credible indications support the account, although one final verification step remains open.",
        "The evidence is coherent enough for a provisional conclusion but has not reached completed confirmation.",
        "Reliable preliminary support exists while independent corroboration is still unfinished.",
        "The report is reasonably supported, with a remaining check preventing final verification.",
        "Available records point in the same direction but the full confirmation process is not complete.",
        "The claim has substantial credible backing without yet being fully established.",
        "Evidence supports acting provisionally while preserving a clear unresolved verification gap.",
        "The account is persuasive enough for a tentative judgment but still awaits final corroboration.",
    ),
    (
        "Independent verification has finished and the relevant sources consistently confirm the account.",
        "Completed cross-checks from dependable records establish the reported facts.",
        "All important confirmation steps have concluded with mutually consistent evidence.",
        "Reliable independent sources agree and no material verification gap remains.",
        "The factual basis has been fully established through completed corroboration.",
        "Separate dependable checks consistently validate the same account.",
        "Verification is complete and the supporting records converge without unresolved conflict.",
        "The report can be treated as confirmed because all material checks are finished and consistent.",
    ),
)

ABSTRACT_SEVERITY_DEFINITIONS = (
    (
        "Choose the low-consequence option when useful operation is almost entirely preserved.",
        "This category describes a small localized disturbance with negligible functional loss.",
        "Use this option when the issue is absorbed by normal activity with little or no adjustment.",
    ),
    (
        "Choose the contained-disruption option when the problem matters but ordinary support can manage it.",
        "This category describes noticeable interference without loss of the central function.",
        "Use this option when a workaround or scheduled follow-up is sufficient.",
    ),
    (
        "Choose the major-disruption option when an important capability is substantially impaired.",
        "This category describes serious functional loss that warrants accelerated intervention.",
        "Use this option when prompt repair is needed because ordinary response would be too slow.",
    ),
    (
        "Choose the acute-failure option when essential operation has broken down and immediate escalation is required.",
        "This category describes the highest consequence level beyond normal support handling.",
        "Use this option when delay is unacceptable because a critical function is unavailable.",
    ),
)

ABSTRACT_CONFIDENCE_DEFINITIONS = (
    (
        "Choose the unresolved-evidence option when the account lacks dependable corroboration.",
        "This category describes materially incomplete or contradictory support.",
        "Use this option when the facts remain too uncertain for a reliable conclusion.",
    ),
    (
        "Choose the provisional-evidence option when credible support exists but confirmation is unfinished.",
        "This category describes a persuasive account with a remaining verification gap.",
        "Use this option when a tentative conclusion is justified but not fully established.",
    ),
    (
        "Choose the established-evidence option when independent verification is complete and consistent.",
        "This category describes an account supported by finished corroboration.",
        "Use this option when dependable checks have fully confirmed the relevant facts.",
    ),
)

SEVERITY_PROTOTYPE_PHRASES = (
    (
        "A visitor sees a brief interface delay, then completes the intended task normally without changing plans.",
        "One optional display is slow for a few minutes while every essential service remains available.",
        "A short-lived queue adds a small inconvenience but nobody loses access to the core service.",
    ),
    (
        "Users rely on a standard alternate channel for several hours while the main service remains available.",
        "A recurring delay lengthens completion time and needs staff follow-up, but the task can still be finished.",
        "One normal access path is unavailable, forcing a manageable workaround through another supported path.",
    ),
    (
        "A central workflow repeatedly fails for many users, preventing normal completion until urgent repair.",
        "A major service capability is unavailable and a large share of users cannot complete an important task.",
        "Serious repeated faults remove a key function and require prompt intervention ahead of routine work.",
    ),
    (
        "The essential service is completely unavailable during a time-sensitive period and immediate action is required.",
        "A critical failure blocks all normal completion paths and emergency restoration is necessary.",
        "The core system has stopped at the highest-impact level, creating immediate serious consequences.",
    ),
)

CONFIDENCE_PROTOTYPE_PHRASES = (
    (
        "A single unverified report conflicts with system records and no independent source resolves the discrepancy.",
        "Partial notes omit key details while two available accounts disagree about what occurred.",
        "An early claim has not been corroborated by any dependable record or completed check.",
    ),
    (
        "Two credible sources support the same explanation while one independent verification step is still pending.",
        "The available records agree and provide strong preliminary support, but final confirmation is unfinished.",
        "A dependable initial check and a second supporting source align while complete verification remains open.",
    ),
    (
        "Independent records and a completed secondary review agree on the same facts without a remaining gap.",
        "Multiple dependable sources have been fully cross-checked and consistently establish the account.",
        "All planned verification steps are complete, with separate reliable evidence confirming the same conclusion.",
    ),
)

ABSTRACT_SEVERITY_QUESTION = "Which consequence level best fits this isolated operational-impact statement?"
ABSTRACT_CONFIDENCE_QUESTION = "Which evidential status best fits this isolated verification statement?"
SEVERITY_PROTOTYPE_QUESTION = "Which fixed operational-impact examples best match this isolated consequence?"
CONFIDENCE_PROTOTYPE_QUESTION = "Which fixed verification examples best match this isolated evidence statement?"


@dataclass(frozen=True)
class CrossEncoderAuthorityCase:
    case_id: str
    domain_id: str
    severity: int
    confidence_index: int
    variant: int
    severity_field: str
    confidence_field: str


ABSTRACT_SEVERITY_OPTIONS = tuple(
    LogicalOption(option_id=f"severity23-{index}", criterion_text=views[0])
    for index, views in enumerate(ABSTRACT_SEVERITY_DEFINITIONS)
)
ABSTRACT_CONFIDENCE_OPTIONS = tuple(
    LogicalOption(option_id=f"confidence23-{index}", criterion_text=views[0])
    for index, views in enumerate(ABSTRACT_CONFIDENCE_DEFINITIONS)
)


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
            option_id=f"{domain_id.lower()}-severity23-s{class_index}-p{prototype_index}",
            criterion_text=text,
        )
        for class_index, prototypes in enumerate(severity_prototypes(domain_id))
        for prototype_index, text in enumerate(prototypes)
    )


def confidence_prototype_options(domain_id: str) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=f"{domain_id.lower()}-confidence23-c{class_index}-p{prototype_index}",
            criterion_text=text,
        )
        for class_index, prototypes in enumerate(confidence_prototypes(domain_id))
        for prototype_index, text in enumerate(prototypes)
    )


def generate_all_w23() -> tuple[CrossEncoderAuthorityCase, ...]:
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
                        CrossEncoderAuthorityCase(
                            case_id=f"{domain_id.lower()}-s{severity}-c{confidence}-v{variant}",
                            domain_id=domain_id,
                            severity=severity,
                            confidence_index=confidence,
                            variant=variant,
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
        raise RuntimeError("W23 requires exactly 384 query cases")
    for domain_id in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != CASES_PER_DOMAIN:
            raise RuntimeError("W23 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W23 severity balance changed")
        if [sum(row.confidence_index == c for row in subset) for c in range(3)] != [32] * 3:
            raise RuntimeError("W23 confidence balance changed")
    return tuple(rows)


def all_w23_query_texts() -> set[str]:
    values = set()
    for row in generate_all_w23():
        values.add(row.severity_field)
        values.add(row.confidence_field)
    return values


def all_w23_prototype_texts() -> set[str]:
    values = set()
    for domain_id in DOMAINS_ORDER:
        for class_values in severity_prototypes(domain_id):
            values.update(class_values)
        for class_values in confidence_prototypes(domain_id):
            values.update(class_values)
    return values


def all_w23_text_atoms() -> set[str]:
    values = all_w23_query_texts() | all_w23_prototype_texts()
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
    "CrossEncoderAuthorityCase",
    "DOMAIN_SEEDS",
    "DOMAINS_ORDER",
    "PROTOTYPES_PER_CLASS",
    "SEVERITY_PROTOTYPE_QUESTION",
    "TOTAL_CASES",
    "VIEW_IDS",
    "all_w23_prototype_texts",
    "all_w23_query_texts",
    "all_w23_text_atoms",
    "confidence_prototype_options",
    "confidence_prototypes",
    "generate_all_w23",
    "severity_prototype_options",
    "severity_prototypes",
]
