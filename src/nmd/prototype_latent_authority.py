from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAINS_ORDER = ("DD", "DE", "DF", "DG")
DOMAIN_SEEDS = {
    "DD": 401101,
    "DE": 401107,
    "DF": 401119,
    "DG": 401131,
}
CASES_PER_DOMAIN = 96
TOTAL_CASES = 384
VIEW_IDS = ("D0", "D1", "D2")
PROTOTYPES_PER_CLASS = 3

DOMAIN_CONTEXT = {
    "DD": "regional archive access disruption assessment",
    "DE": "community cooling-center service triage",
    "DF": "research equipment booking incident assessment",
    "DG": "local transit information service triage",
}

SEVERITY_QUERY_PHRASES = (
    (
        "The issue causes only a faint inconvenience and normal use continues with essentially no loss of function.",
        "Service remains usable in the ordinary way, with just a small localized nuisance.",
        "The practical effect is limited enough that people can proceed almost as usual.",
        "Only a minor disturbance is present and routine operations absorb it easily.",
        "The problem has little operational consequence beyond a brief low-level inconvenience.",
        "Normal activity remains intact despite a small imperfection in service.",
        "The interruption is slight, narrow in scope, and does not materially change routine use.",
        "The condition stays at a mild level that can wait for ordinary handling.",
    ),
    (
        "The issue creates a noticeable interruption that matters, but ordinary service can still continue.",
        "People must adjust around the problem, although the situation remains manageable through standard response.",
        "The practical effect is meaningful without causing serious loss of normal function.",
        "The disruption deserves routine follow-up because it is clearly felt but still contained.",
        "Service quality is reduced in a visible way while remaining within normal operating capacity.",
        "The condition produces a sustained inconvenience that does not require rapid escalation.",
        "Normal use is affected enough to need attention, yet the impact remains controllable.",
        "The problem is more than minor but can still be handled through regular service procedures.",
    ),
    (
        "The issue seriously interferes with normal use and should be handled on an accelerated basis.",
        "A major loss of ordinary function is occurring and prompt response is warranted.",
        "The disruption has become pronounced enough that routine scheduling is no longer appropriate.",
        "Service is substantially impaired and delayed action would prolong significant operational harm.",
        "The practical consequences are severe, with normal activity no longer proceeding reliably.",
        "The condition demands rapid attention because a large part of expected service is unavailable.",
        "The interruption is major and creates clear operational harm short of an emergency condition.",
        "Normal use is strongly compromised and the matter should move ahead of ordinary follow-up.",
    ),
    (
        "The issue has reached an acute state where immediate intervention is required.",
        "Service has effectively broken down at the most serious level and emergency escalation is justified.",
        "The practical consequences are extreme enough that waiting for normal handling would be unsafe or untenable.",
        "The condition represents a critical operational failure demanding immediate response.",
        "Normal use is overwhelmed by the disruption and urgent intervention cannot be deferred.",
        "The interruption is at the maximum end of seriousness with exceptional consequences.",
        "Ordinary response procedures are inadequate because the situation is now acute.",
        "The problem has become an emergency-level failure requiring immediate escalation.",
    ),
)

CONFIDENCE_QUERY_PHRASES = (
    (
        "The account rests on uncertain indications and important parts of the story remain uncorroborated.",
        "Available information is too inconsistent to support a dependable conclusion yet.",
        "The evidence leaves substantial doubt and could still point to a mistaken account.",
        "Key details remain unsupported, so the report should be treated cautiously.",
        "The current record is fragmentary and does not reliably establish what happened.",
        "The claim lacks strong corroboration and remains materially uncertain.",
        "Evidence is sparse or conflicting enough that confidence in the account should stay low.",
        "The information available so far is not dependable enough to settle the facts.",
    ),
    (
        "The account has credible support, although a final verification step has not been completed.",
        "Evidence is reasonably consistent and supports a provisional conclusion rather than a final one.",
        "The report has meaningful backing but still lacks complete independent corroboration.",
        "Available information is persuasive enough to act provisionally while confirmation remains open.",
        "The claim is supported by credible indications that have not yet been fully checked.",
        "There is solid preliminary evidence, with some validation work still outstanding.",
        "The current record supports the account in a plausible way but does not close every verification gap.",
        "Evidence points coherently toward the report while stopping short of complete confirmation.",
    ),
    (
        "The account is supported by completed independent checks that consistently establish the facts.",
        "All important verification steps are finished and the evidence confirms the report.",
        "The claim has dependable corroboration from completed review rather than provisional indications.",
        "Available information has been fully validated and supports the account without a material open question.",
        "Independent sources have completed confirmation and agree on the reported facts.",
        "The evidence is established, cross-checked, and no significant verification step remains.",
        "The report has passed full corroboration through reliable completed checks.",
        "The factual basis is confirmed strongly enough to treat the account as established.",
    ),
)

ABSTRACT_SEVERITY_DEFINITIONS = (
    (
        "Choose the low-consequence category for a small service imperfection with almost no functional loss.",
        "This category covers effects that remain easy to absorb during ordinary use.",
        "Use this category when the disruption stays mild and operationally limited.",
    ),
    (
        "Choose the contained-impact category for a meaningful but manageable service interruption.",
        "This category covers noticeable consequences that remain within standard response capacity.",
        "Use this category when normal use is affected but not seriously impaired.",
    ),
    (
        "Choose the serious-impact category for substantial functional loss requiring prompt handling.",
        "This category covers major disruption that materially compromises ordinary operation.",
        "Use this category when accelerated response is appropriate but emergency handling is not yet necessary.",
    ),
    (
        "Choose the acute-impact category for the most serious operational failure requiring immediate intervention.",
        "This category covers extreme consequences beyond ordinary response procedures.",
        "Use this category when the situation has reached emergency-level seriousness.",
    ),
)

ABSTRACT_CONFIDENCE_DEFINITIONS = (
    (
        "Choose the unresolved-evidence category when the account remains materially doubtful.",
        "This category covers claims without dependable corroboration.",
        "Use this category when available information cannot yet establish the facts.",
    ),
    (
        "Choose the supported-but-open category when credible evidence exists but final verification is incomplete.",
        "This category covers claims with meaningful preliminary backing.",
        "Use this category when a provisional conclusion is justified while confirmation remains open.",
    ),
    (
        "Choose the established-evidence category when reliable corroboration has been completed.",
        "This category covers claims whose important facts have been independently confirmed.",
        "Use this category when verification is complete and dependable.",
    ),
)

SEVERITY_PROTOTYPE_PHRASES = (
    (
        "A visitor sees a short delay at one desk, but every requested service remains available and plans continue normally.",
        "One small feature behaves imperfectly for a few minutes while the main service works without meaningful interruption.",
        "A minor inconvenience affects a narrow part of the process and users can continue without changing what they intended to do.",
    ),
    (
        "Several users must take a simple workaround for the afternoon, but the service remains available and routine staff can manage it.",
        "A recurring interruption slows ordinary activity and needs follow-up, although people can still complete the core task.",
        "Part of the service is temporarily inconvenient enough to matter, yet normal procedures still provide a workable path.",
    ),
    (
        "A major function is unavailable for many users, blocking normal activity until staff intervene promptly.",
        "The service is seriously degraded across an important workflow and postponing response would cause substantial disruption.",
        "Users cannot reliably complete a core task because a large operational failure requires accelerated repair.",
    ),
    (
        "A critical service has stopped entirely at a moment when continued failure creates immediate serious consequences.",
        "The system is in an acute breakdown state where ordinary escalation channels are too slow and intervention must begin now.",
        "A maximum-impact failure prevents essential operation and requires emergency response without delay.",
    ),
)

CONFIDENCE_PROTOTYPE_PHRASES = (
    (
        "One unverified account conflicts with another source, and no dependable record has yet confirmed which version is correct.",
        "A claim is based on incomplete recollection with missing documentation and unresolved contradictory details.",
        "Initial indications suggest something may have happened, but the evidence is too sparse to establish the account reliably.",
    ),
    (
        "Two credible indications support the same account, while one planned verification check has not yet been completed.",
        "The available record is consistent and persuasive, but final independent confirmation is still pending.",
        "A reliable preliminary source supports the report and secondary evidence agrees, though the last validation step remains open.",
    ),
    (
        "Independent records and a completed secondary check agree on the same facts with no material verification gap remaining.",
        "The account has been cross-checked against reliable sources and every important confirmation step is complete.",
        "Multiple dependable records have finished corroborating the report and consistently establish what occurred.",
    ),
)

ABSTRACT_SEVERITY_QUESTION = "Which impact category best matches the isolated operational consequence?"
ABSTRACT_CONFIDENCE_QUESTION = "Which evidence category best matches the isolated support status?"
SEVERITY_PROTOTYPE_QUESTION = "Which concrete impact prototypes are most semantically compatible with this isolated consequence?"
CONFIDENCE_PROTOTYPE_QUESTION = "Which concrete evidence prototypes are most semantically compatible with this isolated support statement?"


@dataclass(frozen=True)
class PrototypeLatentAuthorityCase:
    case_id: str
    domain_id: str
    severity: int
    confidence_index: int
    variant: int
    severity_field: str
    confidence_field: str


ABSTRACT_SEVERITY_OPTIONS = tuple(
    LogicalOption(option_id=f"severity21-{index}", criterion_text=views[0])
    for index, views in enumerate(ABSTRACT_SEVERITY_DEFINITIONS)
)
ABSTRACT_CONFIDENCE_OPTIONS = tuple(
    LogicalOption(option_id=f"confidence21-{index}", criterion_text=views[0])
    for index, views in enumerate(ABSTRACT_CONFIDENCE_DEFINITIONS)
)


def severity_prototypes(domain_id: str) -> tuple[tuple[str, ...], ...]:
    context = DOMAIN_CONTEXT[domain_id]
    return tuple(
        tuple(f"For this {context}, {phrase}" for phrase in class_phrases)
        for class_phrases in SEVERITY_PROTOTYPE_PHRASES
    )


def confidence_prototypes(domain_id: str) -> tuple[tuple[str, ...], ...]:
    context = DOMAIN_CONTEXT[domain_id]
    return tuple(
        tuple(f"For this {context}, {phrase}" for phrase in class_phrases)
        for class_phrases in CONFIDENCE_PROTOTYPE_PHRASES
    )


def severity_prototype_options(domain_id: str) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=f"{domain_id.lower()}-severity21-s{class_index}-p{prototype_index}",
            criterion_text=text,
        )
        for class_index, prototypes in enumerate(severity_prototypes(domain_id))
        for prototype_index, text in enumerate(prototypes)
    )


def confidence_prototype_options(domain_id: str) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=f"{domain_id.lower()}-confidence21-c{class_index}-p{prototype_index}",
            criterion_text=text,
        )
        for class_index, prototypes in enumerate(confidence_prototypes(domain_id))
        for prototype_index, text in enumerate(prototypes)
    )


def generate_all_w21() -> tuple[PrototypeLatentAuthorityCase, ...]:
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
                        PrototypeLatentAuthorityCase(
                            case_id=f"{domain_id.lower()}-s{severity}-c{confidence}-v{variant}",
                            domain_id=domain_id,
                            severity=severity,
                            confidence_index=confidence,
                            variant=variant,
                            severity_field=(
                                f"Within this {context}, "
                                f"{SEVERITY_QUERY_PHRASES[severity][variant]}"
                            ),
                            confidence_field=(
                                f"Regarding this {context}, "
                                f"{CONFIDENCE_QUERY_PHRASES[confidence][c_variant]}"
                            ),
                        )
                    )
    if len(rows) != TOTAL_CASES:
        raise RuntimeError("W21 requires exactly 384 query cases")
    for domain_id in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != CASES_PER_DOMAIN:
            raise RuntimeError("W21 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W21 severity balance changed")
        if [sum(row.confidence_index == c for row in subset) for c in range(3)] != [32] * 3:
            raise RuntimeError("W21 confidence balance changed")
    return tuple(rows)


def all_w21_query_texts() -> set[str]:
    values = set()
    for row in generate_all_w21():
        values.add(row.severity_field)
        values.add(row.confidence_field)
    return values


def all_w21_prototype_texts() -> set[str]:
    values = set()
    for domain_id in DOMAINS_ORDER:
        for class_values in severity_prototypes(domain_id):
            values.update(class_values)
        for class_values in confidence_prototypes(domain_id):
            values.update(class_values)
    return values


def all_w21_text_atoms() -> set[str]:
    values = all_w21_query_texts() | all_w21_prototype_texts()
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
    "PrototypeLatentAuthorityCase",
    "SEVERITY_PROTOTYPE_QUESTION",
    "TOTAL_CASES",
    "VIEW_IDS",
    "all_w21_prototype_texts",
    "all_w21_query_texts",
    "all_w21_text_atoms",
    "confidence_prototype_options",
    "confidence_prototypes",
    "generate_all_w21",
    "severity_prototype_options",
    "severity_prototypes",
]
