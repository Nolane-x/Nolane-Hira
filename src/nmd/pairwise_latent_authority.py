from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAINS_ORDER = ("CZ", "DA", "DB", "DC")
DOMAIN_SEEDS = {
    "CZ": 391101,
    "DA": 391107,
    "DB": 391119,
    "DC": 391131,
}
CASES_PER_DOMAIN = 96
TOTAL_CASES = 384
VIEW_IDS = ("D0", "D1", "D2")

DOMAIN_CONTEXT = {
    "CZ": "public-library service interruption report",
    "DA": "community energy service incident assessment",
    "DB": "campus laboratory support disruption report",
    "DC": "municipal water-service issue assessment",
}

SEVERITY_FIELD_PHRASES = (
    (
        "The reported effect is barely disruptive and routine activity proceeds with almost no practical interference.",
        "Only a small inconvenience is present, with ordinary operation remaining essentially intact.",
        "The condition has a limited footprint and does not materially disturb normal service.",
        "The impact is minor enough that routine handling can absorb it without urgency.",
        "Normal activity remains broadly unaffected despite a small local disturbance.",
        "The consequence is slight and stays well within everyday operating tolerance.",
        "The interruption is low-impact and creates little practical loss of function.",
        "The situation remains mild, localized, and easy to accommodate through ordinary service.",
    ),
    (
        "The reported effect causes a clear but manageable interruption that deserves normal follow-up.",
        "Routine activity is noticeably affected, although service can still continue in a controlled way.",
        "The condition creates a meaningful inconvenience without seriously disabling ordinary operation.",
        "The impact is moderate enough to require attention but not accelerated intervention.",
        "Normal service is disrupted in a visible yet contained manner.",
        "The consequence is material but remains within standard operating capacity.",
        "The interruption is more than minor while still being manageable through routine response.",
        "The situation creates a sustained inconvenience that should be addressed without emergency escalation.",
    ),
    (
        "The reported effect substantially impairs routine activity and calls for prompt intervention.",
        "Normal service is seriously degraded and delaying response would carry meaningful operational cost.",
        "The condition creates a major interruption that requires accelerated handling.",
        "The impact is strong enough to compromise ordinary operation in an important way.",
        "Routine activity cannot proceed normally because the disruption has become pronounced.",
        "The consequence is severe and warrants a prompt response rather than standard scheduling.",
        "The interruption causes major loss of function but has not reached the most extreme state.",
        "The situation is highly disruptive and should be escalated for rapid handling.",
    ),
    (
        "The reported effect has become extreme and immediate intervention is necessary.",
        "Normal service is effectively overwhelmed by the condition and emergency-level escalation is warranted.",
        "The interruption sits at the highest-impact end of the scale with acute operational consequences.",
        "The condition is critical enough that routine or delayed handling is no longer appropriate.",
        "The impact is exceptionally serious and demands immediate response.",
        "Ordinary procedures are insufficient because the disruption has reached an extreme state.",
        "The situation represents a maximum-severity breakdown requiring urgent escalation.",
        "The consequence is acute, system-threatening, and cannot safely remain in normal service flow.",
    ),
)

CONFIDENCE_FIELD_PHRASES = (
    (
        "The account is weakly supported and important details remain uncertain or conflicting.",
        "Available evidence is incomplete enough that the report cannot yet be trusted confidently.",
        "The information lacks dependable corroboration and still leaves substantial doubt.",
        "Support for the account is fragmentary, with unresolved uncertainty in the evidence.",
        "The report remains tentative because the available indications are ambiguous.",
        "The evidence is too unreliable to establish the facts with confidence.",
        "Key claims remain unverified and the current support could still be mistaken.",
        "The available record leaves material doubt about whether the account is correct.",
    ),
    (
        "The account has credible preliminary support, but a final confirmation step is still open.",
        "Available evidence is reasonably consistent while full corroboration has not yet been completed.",
        "The report has meaningful support and can be treated provisionally, though not as fully verified.",
        "The information is plausible and backed by evidence that still requires final checking.",
        "There is dependable preliminary backing without complete independent confirmation.",
        "The account is supported well enough for a provisional conclusion but remains short of full verification.",
        "Evidence points consistently toward the report while one or more confirmation steps remain pending.",
        "The facts have credible support, although complete corroboration is not yet available.",
    ),
    (
        "The account has been fully corroborated through dependable checks and the facts are established.",
        "Available evidence has completed verification and consistently confirms the report.",
        "The information is backed by strong independent confirmation with no important validation step pending.",
        "The report has dependable completed corroboration from reliable evidence.",
        "The facts are established by fully checked support rather than a provisional indication.",
        "Independent verification has been completed and confirms the account consistently.",
        "The evidence is conclusive enough to treat the report as fully established.",
        "All material confirmation steps are complete and support the account reliably.",
    ),
)

FLAT_SEVERITY_DEFINITIONS = (
    (
        "Select the mild-impact option for effects that leave normal activity almost entirely intact.",
        "This option describes only limited practical interference with routine operation.",
        "Use this option when consequences remain small, localized, and easy to absorb.",
    ),
    (
        "Select the managed-disruption option for noticeable effects that still fit routine response.",
        "This option describes meaningful interruption without major loss of normal function.",
        "Use this option when service is clearly affected but remains controllable through standard handling.",
    ),
    (
        "Select the major-disruption option for effects that seriously impair normal operation.",
        "This option describes pronounced interruption requiring accelerated response.",
        "Use this option when routine activity is materially compromised and prompt handling is needed.",
    ),
    (
        "Select the extreme-impact option for effects requiring immediate emergency-level intervention.",
        "This option describes the most acute operational condition where ordinary handling is insufficient.",
        "Use this option when consequences are exceptionally serious and immediate escalation is required.",
    ),
)

FLAT_CONFIDENCE_DEFINITIONS = (
    (
        "Select the doubtful-evidence option when support remains ambiguous, incomplete, or unreliable.",
        "This option describes an account whose facts are not yet dependably established.",
        "Use this option when unresolved uncertainty prevents a confident conclusion.",
    ),
    (
        "Select the provisionally-supported option when credible evidence exists but final corroboration is incomplete.",
        "This option describes an account with meaningful backing that still awaits complete verification.",
        "Use this option when a provisional conclusion is justified but not a fully confirmed one.",
    ),
    (
        "Select the fully-confirmed option when dependable corroboration is complete.",
        "This option describes an account whose evidence has been independently checked and established.",
        "Use this option when no material confirmation step remains pending.",
    ),
)

SEVERITY_PAIR_DEFINITIONS = (
    (
        (
            "This side describes a mild effect with almost no interference to ordinary activity.",
            "Choose this side when the consequence is small and routine operation remains essentially intact.",
            "This side represents limited disruption that is easy to absorb.",
        ),
        (
            "This side describes a clearly noticeable but still manageable interruption.",
            "Choose this side when normal service is affected enough to require routine follow-up.",
            "This side represents meaningful disruption that remains within standard handling.",
        ),
    ),
    (
        (
            "This side describes a manageable interruption that remains within ordinary response capacity.",
            "Choose this side when service is noticeably affected but not seriously impaired.",
            "This side represents a contained disruption without major operational loss.",
        ),
        (
            "This side describes a major interruption that materially impairs routine operation.",
            "Choose this side when the situation warrants accelerated handling because consequences are pronounced.",
            "This side represents serious disruption requiring prompt response.",
        ),
    ),
    (
        (
            "This side describes a major but non-extreme disruption requiring prompt handling.",
            "Choose this side when normal operation is seriously impaired but emergency-level intervention is not yet required.",
            "This side represents severe consequences short of the maximum condition.",
        ),
        (
            "This side describes an extreme condition requiring immediate intervention.",
            "Choose this side when ordinary handling is insufficient and emergency-level escalation is warranted.",
            "This side represents the highest-impact operational state.",
        ),
    ),
)

CONFIDENCE_PAIR_DEFINITIONS = (
    (
        (
            "This side describes evidence that remains too uncertain for a dependable conclusion.",
            "Choose this side when support is ambiguous or insufficiently corroborated.",
            "This side represents an account that is still materially doubtful.",
        ),
        (
            "This side describes evidence with credible preliminary support despite incomplete final confirmation.",
            "Choose this side when the account is reasonably backed but still provisional.",
            "This side represents meaningful support that has not yet reached complete verification.",
        ),
    ),
    (
        (
            "This side describes credible provisional support with at least one confirmation step still open.",
            "Choose this side when the evidence is meaningful but not yet fully corroborated.",
            "This side represents a supported account that remains short of complete verification.",
        ),
        (
            "This side describes evidence whose corroboration is complete and independently established.",
            "Choose this side when the account has passed all material confirmation steps.",
            "This side represents fully verified support.",
        ),
    ),
)

FLAT_SEVERITY_QUESTION = "Which overall impact description best matches this isolated service-effect statement?"
FLAT_CONFIDENCE_QUESTION = "Which overall evidence-status description best matches this isolated support statement?"
SEVERITY_PAIR_QUESTIONS = (
    "Which of these two neighboring impact descriptions better matches the isolated effect?",
    "Which of these two adjacent disruption descriptions better matches the isolated effect?",
    "Which of these two upper-impact descriptions better matches the isolated effect?",
)
CONFIDENCE_PAIR_QUESTIONS = (
    "Which of these two neighboring evidence-status descriptions better matches the isolated support?",
    "Which of these two higher-certainty descriptions better matches the isolated support?",
)


@dataclass(frozen=True)
class PairwiseLatentAuthorityCase:
    case_id: str
    domain_id: str
    severity: int
    confidence_index: int
    variant: int
    severity_field: str
    confidence_field: str


FLAT_SEVERITY_OPTIONS = tuple(
    LogicalOption(option_id=f"severity20-{index}", criterion_text=views[0])
    for index, views in enumerate(FLAT_SEVERITY_DEFINITIONS)
)
FLAT_CONFIDENCE_OPTIONS = tuple(
    LogicalOption(option_id=f"confidence20-{index}", criterion_text=views[0])
    for index, views in enumerate(FLAT_CONFIDENCE_DEFINITIONS)
)


def _pair_options(prefix: str, definitions):
    return tuple(
        LogicalOption(option_id=f"{prefix}-{index}", criterion_text=views[0])
        for index, views in enumerate(definitions)
    )


SEVERITY_PAIR_OPTIONS = tuple(
    _pair_options(f"severity20-p{index}", definitions)
    for index, definitions in enumerate(SEVERITY_PAIR_DEFINITIONS)
)
CONFIDENCE_PAIR_OPTIONS = tuple(
    _pair_options(f"confidence20-p{index}", definitions)
    for index, definitions in enumerate(CONFIDENCE_PAIR_DEFINITIONS)
)


def generate_all_w20() -> tuple[PairwiseLatentAuthorityCase, ...]:
    rows = []
    for domain_id in DOMAINS_ORDER:
        context = DOMAIN_CONTEXT[domain_id]
        rng = random.Random(DOMAIN_SEEDS[domain_id])
        for severity in range(4):
            for confidence in range(3):
                confidence_variants = list(range(8))
                rng.shuffle(confidence_variants)
                for variant in range(8):
                    c_variant = confidence_variants[variant]
                    rows.append(
                        PairwiseLatentAuthorityCase(
                            case_id=f"{domain_id.lower()}-s{severity}-c{confidence}-v{variant}",
                            domain_id=domain_id,
                            severity=severity,
                            confidence_index=confidence,
                            variant=variant,
                            severity_field=(
                                f"In this {context}, "
                                f"{SEVERITY_FIELD_PHRASES[severity][variant]}"
                            ),
                            confidence_field=(
                                f"For this {context}, "
                                f"{CONFIDENCE_FIELD_PHRASES[confidence][c_variant]}"
                            ),
                        )
                    )
    if len(rows) != TOTAL_CASES:
        raise RuntimeError("W20 requires exactly 384 cases")
    for domain_id in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != CASES_PER_DOMAIN:
            raise RuntimeError("W20 requires exactly 96 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [24] * 4:
            raise RuntimeError("W20 severity balance changed")
        if [sum(row.confidence_index == c for row in subset) for c in range(3)] != [32] * 3:
            raise RuntimeError("W20 confidence balance changed")
    return tuple(rows)


def all_w20_text_atoms() -> set[str]:
    values = set()
    for row in generate_all_w20():
        values.add(row.severity_field)
        values.add(row.confidence_field)
    for group in (FLAT_SEVERITY_DEFINITIONS, FLAT_CONFIDENCE_DEFINITIONS):
        for views in group:
            values.update(views)
    for pair_group in (SEVERITY_PAIR_DEFINITIONS, CONFIDENCE_PAIR_DEFINITIONS):
        for pair in pair_group:
            for option_views in pair:
                values.update(option_views)
    values.update(
        (
            FLAT_SEVERITY_QUESTION,
            FLAT_CONFIDENCE_QUESTION,
            *SEVERITY_PAIR_QUESTIONS,
            *CONFIDENCE_PAIR_QUESTIONS,
        )
    )
    return values


__all__ = [
    "CASES_PER_DOMAIN",
    "CONFIDENCE_PAIR_DEFINITIONS",
    "CONFIDENCE_PAIR_OPTIONS",
    "CONFIDENCE_PAIR_QUESTIONS",
    "DOMAIN_SEEDS",
    "DOMAINS_ORDER",
    "FLAT_CONFIDENCE_DEFINITIONS",
    "FLAT_CONFIDENCE_OPTIONS",
    "FLAT_CONFIDENCE_QUESTION",
    "FLAT_SEVERITY_DEFINITIONS",
    "FLAT_SEVERITY_OPTIONS",
    "FLAT_SEVERITY_QUESTION",
    "PairwiseLatentAuthorityCase",
    "SEVERITY_PAIR_DEFINITIONS",
    "SEVERITY_PAIR_OPTIONS",
    "SEVERITY_PAIR_QUESTIONS",
    "TOTAL_CASES",
    "VIEW_IDS",
    "all_w20_text_atoms",
    "generate_all_w20",
]
