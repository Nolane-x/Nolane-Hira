from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption
from .typed_decisions import TypedDecision, TypedDecisionCase
from .typed_reliability_authority import (
    CONFIDENCE_MASS,
    _categorical_distribution,
    _expected_score,
    _ordinal_distribution,
)

K_VALUES = (4, 8, 16)
PARAPHRASE_VIEWS = ("D0", "D1", "D2")
INTENTS_PER_DOMAIN = 16
STATE_VARIANTS_PER_INTENT = 6
CASES_PER_DOMAIN = 96
DOMAINS_ORDER = ("CR", "CS", "CT", "CU")

DOMAIN_SEEDS = {
    "CR": 371101,
    "CS": 371107,
    "CT": 371119,
    "CU": 371131,
}

SEVERITY_KEYS = ("minimal", "moderate", "high", "critical")
CONFIDENCE_KEYS = ("verified", "provisional", "uncertain")


@dataclass(frozen=True)
class SubjectSpec:
    key: str
    label: str


@dataclass(frozen=True)
class ActionSpec:
    key: str
    definitions: tuple[str, str, str]
    states: tuple[str, str, str, str, str, str]


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    workflow: str
    actor: str
    question: str
    subjects: tuple[SubjectSpec, SubjectSpec, SubjectSpec, SubjectSpec]
    actions: tuple[ActionSpec, ActionSpec, ActionSpec, ActionSpec]


@dataclass(frozen=True)
class IntentSpec:
    intent_id: str
    definitions: tuple[str, str, str]
    state_texts: tuple[str, str, str, str, str, str]


@dataclass(frozen=True)
class LatentCompositionAuthorityCase:
    typed: TypedDecisionCase
    domain_id: str
    diagnosis_k: int
    severity: int
    confidence: str
    intent_field: str
    severity_field: str
    confidence_field: str
    option_definitions: tuple[
        tuple[tuple[str, str, str], ...],
        ...,
    ]


def _actions(actor: str) -> tuple[ActionSpec, ActionSpec, ActionSpec, ActionSpec]:
    return (
        ActionSpec(
            "establish",
            (
                f"The {actor} wants to establish new {{subject}}.",
                "This message concerns putting {subject} into effect for the first time.",
                "The requested operation is creation of a new {subject}.",
            ),
            (
                "I need a new {subject} established.",
                "Please arrange {subject} that does not exist yet.",
                "Nothing is active for {subject}; I want one created.",
                "Can you put a fresh {subject} in place?",
                "I am setting up {subject} for the first time.",
                "How do I get new {subject} established?",
            ),
        ),
        ActionSpec(
            "adjust",
            (
                f"The {actor} wants to adjust existing {{subject}}.",
                "This message concerns changing details attached to current {subject}.",
                "The requested operation is modification of an already active {subject}.",
            ),
            (
                "I need to adjust details on my existing {subject}.",
                "Please change information attached to the current {subject}.",
                "The {subject} already exists, but some details need alteration.",
                "Can you revise what is recorded for {subject}?",
                "I want to modify my present {subject} without replacing it.",
                "How do I alter the active {subject}?",
            ),
        ),
        ActionSpec(
            "retire",
            (
                f"The {actor} wants to retire existing {{subject}}.",
                "This message concerns ending or removing current {subject}.",
                "The requested operation is closure of an active {subject}.",
            ),
            (
                "I need my existing {subject} retired.",
                "Please end the {subject} that is active now.",
                "I want the current {subject} removed.",
                "Can you close the {subject} I already have?",
                "The present {subject} should no longer stay active.",
                "How can I discontinue my existing {subject}?",
            ),
        ),
        ActionSpec(
            "inspect",
            (
                f"The {actor} wants to inspect the present state of {{subject}}.",
                "This message asks for current progress or status of {subject}.",
                "The requested operation is checking where existing {subject} stands.",
            ),
            (
                "I need to inspect the current state of my {subject}.",
                "Please tell me where the existing {subject} stands.",
                "I want the latest progress for {subject}.",
                "Can you show what has happened with my current {subject}?",
                "Has the active {subject} changed recently?",
                "How far along is my existing {subject}?",
            ),
        ),
    )


def _domain(
    domain_id: str,
    workflow: str,
    actor: str,
    question: str,
    subjects: tuple[tuple[str, str], ...],
) -> DomainSpec:
    subject_specs = tuple(SubjectSpec(*row) for row in subjects)
    if len(subject_specs) != 4:
        raise ValueError("W18 requires four subjects/domain")
    return DomainSpec(
        domain_id=domain_id,
        workflow=workflow,
        actor=actor,
        question=question,
        subjects=subject_specs,  # type: ignore[arg-type]
        actions=_actions(actor),
    )


DOMAINS = {
    spec.domain_id: spec
    for spec in (
        _domain(
            "CR",
            "w18-condominium-package-room",
            "resident",
            "Which condominium package-room request matches the resident message?",
            (
                ("locker", "parcel-locker access profile"),
                ("delivery", "courier delivery authorization"),
                ("pickup", "resident parcel pickup credential"),
                ("notice", "package arrival notification profile"),
            ),
        ),
        _domain(
            "CS",
            "w18-farm-share-membership",
            "member",
            "Which regional farm-share request matches the member message?",
            (
                ("produce", "weekly produce share"),
                ("eggs", "farm egg-share add-on"),
                ("fruit", "seasonal fruit-share add-on"),
                ("pickup", "farm-share pickup-site profile"),
            ),
        ),
        _domain(
            "CT",
            "w18-youth-sports-registration",
            "guardian",
            "Which youth sports league request matches the guardian message?",
            (
                ("soccer", "youth soccer registration"),
                ("basketball", "youth basketball registration"),
                ("swim", "youth swim-clinic enrollment"),
                ("equipment", "league equipment-rental registration"),
            ),
        ),
        _domain(
            "CU",
            "w18-smart-meter-support",
            "householder",
            "Which household smart-meter request matches the householder message?",
            (
                ("dashboard", "smart-meter usage dashboard access"),
                ("alerts", "smart-meter outage-alert profile"),
                ("export", "interval-usage data export"),
                ("visit", "smart-meter service appointment record"),
            ),
        ),
    )
}


def _intent_specs(domain: DomainSpec) -> tuple[IntentSpec, ...]:
    rows: list[IntentSpec] = []
    for subject in domain.subjects:
        for action in domain.actions:
            rows.append(
                IntentSpec(
                    intent_id=f"{domain.domain_id.lower()}-{subject.key}-{action.key}",
                    definitions=tuple(
                        template.format(subject=subject.label)
                        for template in action.definitions
                    ),
                    state_texts=tuple(
                        template.format(subject=subject.label)
                        for template in action.states
                    ),
                )
            )
    if len(rows) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W18 requires exactly 16 intents/domain")
    if len({row.intent_id for row in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W18 intent IDs must be unique")
    all_defs = [text for row in rows for text in row.definitions]
    if len(set(all_defs)) != INTENTS_PER_DOMAIN * 3:
        raise RuntimeError("W18 intent definitions must be unique")
    return tuple(rows)


INTENTS = {
    domain_id: _intent_specs(spec)
    for domain_id, spec in DOMAINS.items()
}


SEVERITY_FIELD_TEXTS = (
    (
        "The reported effect is barely noticeable.",
        "Consequences remain very limited in scope.",
        "Only a slight practical effect is present.",
        "The situation causes little meaningful disruption.",
        "Observed consequences are negligible.",
        "The issue produces only a minor disturbance.",
    ),
    (
        "The situation has a noticeable but manageable effect.",
        "Consequences are meaningful yet still contained.",
        "The issue causes some disruption without severe consequences.",
        "There is a clear effect, but normal operations can still continue.",
        "The impact is material while remaining manageable.",
        "The disruption is real but stays within ordinary control.",
    ),
    (
        "The situation is causing substantial disruption.",
        "Consequences are serious and need prompt attention.",
        "The issue has a strong operational effect.",
        "There is major disruption that should be addressed quickly.",
        "The consequences are pronounced and difficult to ignore.",
        "The situation is materially severe and needs accelerated handling.",
    ),
    (
        "The situation is causing the most severe level of disruption.",
        "Consequences are extreme and demand immediate intervention.",
        "The issue has reached an emergency level of impact.",
        "The disruption is extreme enough that ordinary handling is insufficient.",
        "The consequences are at the maximum seriousness level.",
        "The situation presents an extreme operational impact requiring immediate action.",
    ),
)

CONFIDENCE_FIELD_TEXTS = (
    (
        "The supporting evidence has been independently confirmed.",
        "Available evidence has been checked and corroborated.",
        "The reported facts are backed by confirmed evidence.",
        "The evidence has passed a complete confirmation step.",
        "The information is supported by fully confirmed evidence.",
        "Independent checks support the reported facts.",
    ),
    (
        "The evidence is incomplete and still awaiting final confirmation.",
        "Available support is preliminary rather than fully confirmed.",
        "The report has some evidence, but confirmation is still pending.",
        "The information is partly supported while verification remains unfinished.",
        "Evidence exists, although the confirmation process is not complete.",
        "The reported facts have preliminary support only.",
    ),
    (
        "The available evidence is ambiguous and confidence is low.",
        "Support for the report is unclear and cannot be relied on yet.",
        "The facts remain poorly established because the evidence conflicts.",
        "Evidence is too ambiguous to support a confident conclusion.",
        "The report lacks dependable support at this point.",
        "The available information leaves substantial doubt about the facts.",
    ),
)


SEVERITY_LATENT_DEFINITIONS = (
    (
        "Minimal severity means consequences are slight and negligible.",
        "This severity category covers only very limited practical impact.",
        "Choose the lowest severity when disruption is barely meaningful.",
    ),
    (
        "Moderate severity means consequences are noticeable but manageable.",
        "This severity category covers meaningful yet contained impact.",
        "Choose moderate severity when disruption is real without becoming severe.",
    ),
    (
        "High severity means consequences are substantial and serious.",
        "This severity category covers major disruption needing prompt attention.",
        "Choose high severity when impact is pronounced but below the emergency maximum.",
    ),
    (
        "Critical severity means consequences are extreme and immediate.",
        "This severity category covers the maximum level of operational impact.",
        "Choose critical severity when emergency intervention is warranted.",
    ),
)

CONFIDENCE_LATENT_DEFINITIONS = (
    (
        "Verified confidence means the evidence has been confirmed and corroborated.",
        "This confidence category applies when independent checks support the facts.",
        "Choose verified confidence for fully confirmed evidence.",
    ),
    (
        "Provisional confidence means evidence exists but final confirmation is pending.",
        "This confidence category applies to preliminary or partly supported information.",
        "Choose provisional confidence when verification is still incomplete.",
    ),
    (
        "Uncertain confidence means the evidence is ambiguous or unreliable.",
        "This confidence category applies when the facts remain poorly established.",
        "Choose uncertain confidence when available support leaves substantial doubt.",
    ),
)


RESPONSE_DEFINITIONS = (
    (
        "Use the ordinary handling track for the lowest impact condition.",
        "The first response category is routine handling for slight consequences.",
        "Choose response zero when no accelerated operational action is needed.",
    ),
    (
        "Use the scheduled handling track for a manageable but meaningful condition.",
        "The second response category is planned follow-up for contained disruption.",
        "Choose response one when attention should be arranged but not accelerated.",
    ),
    (
        "Use the accelerated handling track for a serious disruptive condition.",
        "The third response category is prompt escalation for substantial impact.",
        "Choose response two when operational action should happen quickly.",
    ),
    (
        "Use the emergency handling track for an extreme condition.",
        "The fourth response category is immediate escalation for maximum impact.",
        "Choose response three when intervention cannot wait.",
    ),
)

REVIEW_DEFINITIONS = (
    (
        "Keep the case automated when evidence is dependable enough and impact is below the maximum level.",
        "Choose no human review unless evidence is doubtful or consequences are extreme.",
        "The automated path is allowed when neither the evidence nor impact triggers escalation.",
    ),
    (
        "Send the case to a person when evidence is doubtful or consequences reach the maximum level.",
        "Choose human review if support is unreliable or the impact is extreme.",
        "A reviewer is required when either evidence quality or severity crosses its escalation condition.",
    ),
)

RISK_DEFINITIONS = (
    (
        "Risk category zero corresponds to only slight consequences.",
        "Assign the first risk category for negligible practical impact.",
        "Use risk zero for the lowest impact condition.",
    ),
    (
        "Risk category one corresponds to manageable but meaningful consequences.",
        "Assign the second risk category for contained disruption.",
        "Use risk one for the next impact level above slight.",
    ),
    (
        "Risk category two corresponds to substantial serious consequences.",
        "Assign the third risk category for pronounced disruption.",
        "Use risk two for a condition needing prompt attention.",
    ),
    (
        "Risk category three corresponds to extreme consequences.",
        "Assign the fourth risk category for emergency-level impact.",
        "Use risk three for the maximum impact condition.",
    ),
)

URGENCY_DEFINITIONS = (
    (
        "Urgency category zero permits ordinary timing.",
        "The first urgency category needs no accelerated timing.",
        "Choose urgency zero when routine scheduling is sufficient.",
    ),
    (
        "Urgency category one calls for scheduled attention.",
        "The second urgency category means action should be planned.",
        "Choose urgency one when follow-up should be arranged.",
    ),
    (
        "Urgency category two calls for prompt attention.",
        "The third urgency category means action should be accelerated.",
        "Choose urgency two when intervention should occur quickly.",
    ),
    (
        "Urgency category three calls for immediate attention.",
        "The fourth urgency category means action cannot wait.",
        "Choose urgency three for the fastest response level.",
    ),
)


def _logical_options(
    prefix: str,
    definitions: tuple[tuple[str, str, str], ...],
    *,
    values: tuple[float | None, ...] | None = None,
) -> tuple[LogicalOption, ...]:
    if values is None:
        values = tuple(None for _ in definitions)
    if len(values) != len(definitions):
        raise ValueError("W18 values/definitions mismatch")
    return tuple(
        LogicalOption(
            option_id=f"{prefix}-{index}",
            criterion_text=definition[0],
            value=value,
        )
        for index, (definition, value) in enumerate(zip(definitions, values))
    )


RESPONSE_OPTIONS = _logical_options("response18", RESPONSE_DEFINITIONS)
REVIEW_OPTIONS = _logical_options(
    "review18",
    REVIEW_DEFINITIONS,
    values=(0.0, 1.0),
)
RISK_OPTIONS = _logical_options(
    "risk18",
    RISK_DEFINITIONS,
    values=(0.0, 1.0, 2.0, 3.0),
)
URGENCY_OPTIONS = _logical_options(
    "urgency18",
    URGENCY_DEFINITIONS,
    values=(0.0, 1.0, 2.0, 3.0),
)

SEVERITY_LATENT_OPTIONS = _logical_options(
    "severity18",
    SEVERITY_LATENT_DEFINITIONS,
)
CONFIDENCE_LATENT_OPTIONS = _logical_options(
    "confidence18",
    CONFIDENCE_LATENT_DEFINITIONS,
)


def _candidate_order(
    domain_id: str,
    intent_index: int,
    state_variant: int,
) -> tuple[list[str], dict[str, int]]:
    intents = INTENTS[domain_id]
    gold = intents[intent_index].intent_id
    negatives = [row.intent_id for row in intents if row.intent_id != gold]
    seed = DOMAIN_SEEDS[domain_id] + intent_index * 1031 + state_variant * 113
    random.Random(seed).shuffle(negatives)
    membership = [gold, *negatives]
    presentation = list(membership)
    random.Random(seed + 81_181).shuffle(presentation)
    return membership, {
        item: index for index, item in enumerate(presentation)
    }


def _diagnosis_k(intent_index: int, state_variant: int) -> int:
    return K_VALUES[
        (intent_index * STATE_VARIANTS_PER_INTENT + state_variant) % 3
    ]


def _case(
    domain_id: str,
    intent_index: int,
    state_variant: int,
) -> LatentCompositionAuthorityCase:
    domain = DOMAINS[domain_id]
    intents = INTENTS[domain_id]
    by_id = {row.intent_id: row for row in intents}
    intent = intents[intent_index]

    k = _diagnosis_k(intent_index, state_variant)
    membership, rank = _candidate_order(
        domain_id,
        intent_index,
        state_variant,
    )
    selected = membership[:k]
    presented = sorted(selected, key=lambda item: rank[item])
    diagnosis_gold = presented.index(intent.intent_id)
    diagnosis_defs = tuple(
        by_id[option_id].definitions for option_id in presented
    )
    diagnosis_options = tuple(
        LogicalOption(
            option_id=option_id,
            criterion_text=defs[0],
        )
        for option_id, defs in zip(presented, diagnosis_defs)
    )

    rng = random.Random(
        DOMAIN_SEEDS[domain_id]
        + intent_index * 4153
        + state_variant * 541
        + 37
    )
    severity = rng.randrange(4)
    confidence = CONFIDENCE_KEYS[rng.randrange(3)]
    confidence_index = CONFIDENCE_KEYS.index(confidence)
    gold_mass = CONFIDENCE_MASS[confidence]

    severity_variant = (
        intent_index * 3 + state_variant
    ) % len(SEVERITY_FIELD_TEXTS[severity])
    confidence_variant = (
        intent_index * 5 + state_variant
    ) % len(CONFIDENCE_FIELD_TEXTS[confidence_index])

    intent_field = intent.state_texts[state_variant]
    severity_field = SEVERITY_FIELD_TEXTS[severity][severity_variant]
    confidence_field = CONFIDENCE_FIELD_TEXTS[confidence_index][
        confidence_variant
    ]

    response_gold = severity
    review_gold = int(confidence == "uncertain" or severity == 3)
    risk_gold = severity
    urgency_gold = min(
        3,
        severity + int(confidence == "uncertain"),
    )

    diagnosis_prob = _categorical_distribution(
        k,
        diagnosis_gold,
        gold_mass,
    )
    response_prob = _categorical_distribution(
        4,
        response_gold,
        gold_mass,
    )
    review_prob = _categorical_distribution(
        2,
        review_gold,
        gold_mass,
    )
    risk_prob = _ordinal_distribution(
        4,
        risk_gold,
        gold_mass,
    )
    urgency_prob = _ordinal_distribution(
        4,
        urgency_gold,
        gold_mass,
    )

    decisions = (
        TypedDecision(
            question_id="diagnosis",
            primitive="choice",
            question_text=domain.question,
            options=diagnosis_options,
            gold_index=diagnosis_gold,
            gold_probabilities=diagnosis_prob,
        ),
        TypedDecision(
            question_id="response",
            primitive="choice",
            question_text="Which operational response category follows from the extracted impact level?",
            options=RESPONSE_OPTIONS,
            gold_index=response_gold,
            gold_probabilities=response_prob,
        ),
        TypedDecision(
            question_id="needs_review",
            primitive="noul",
            question_text="Does the extracted impact or evidence state require human review?",
            options=REVIEW_OPTIONS,
            gold_index=review_gold,
            gold_probabilities=review_prob,
        ),
        TypedDecision(
            question_id="risk",
            primitive="score",
            question_text="Which risk category follows from the extracted impact level?",
            options=RISK_OPTIONS,
            gold_index=risk_gold,
            gold_probabilities=risk_prob,
            gold_score=_expected_score(risk_prob),
        ),
        TypedDecision(
            question_id="urgency",
            primitive="score",
            question_text="Which urgency category follows from the extracted impact and evidence states?",
            options=URGENCY_OPTIONS,
            gold_index=urgency_gold,
            gold_probabilities=urgency_prob,
            gold_score=_expected_score(urgency_prob),
        ),
    )

    option_definitions = (
        diagnosis_defs,
        RESPONSE_DEFINITIONS,
        REVIEW_DEFINITIONS,
        RISK_DEFINITIONS,
        URGENCY_DEFINITIONS,
    )
    for decision, definitions in zip(decisions, option_definitions):
        if len(decision.options) != len(definitions):
            raise RuntimeError("W18 decision/definition width mismatch")
        for option, views in zip(decision.options, definitions):
            if option.criterion_text != views[0]:
                raise RuntimeError("W18 D0 must equal criterion text")

    base_id = f"w18-{domain_id.lower()}-{intent_index:02d}-{state_variant}"
    typed = TypedDecisionCase(
        case_id=f"{base_id}-k{k}",
        workflow=domain.workflow,
        state_text=(
            f"{intent_field} {severity_field} {confidence_field}"
        ),
        decisions=decisions,
    )

    return LatentCompositionAuthorityCase(
        typed=typed,
        domain_id=domain_id,
        diagnosis_k=k,
        severity=severity,
        confidence=confidence,
        intent_field=intent_field,
        severity_field=severity_field,
        confidence_field=confidence_field,
        option_definitions=option_definitions,
    )


def generate_w18_domain(
    domain_id: str,
) -> list[LatentCompositionAuthorityCase]:
    if domain_id not in DOMAINS:
        raise ValueError("W18 domain must be CR/CS/CT/CU")
    rows = [
        _case(domain_id, intent_index, state_variant)
        for intent_index in range(INTENTS_PER_DOMAIN)
        for state_variant in range(STATE_VARIANTS_PER_INTENT)
    ]
    if len(rows) != CASES_PER_DOMAIN:
        raise RuntimeError("W18 domain case count mismatch")
    counts = {
        k: sum(row.diagnosis_k == k for row in rows)
        for k in K_VALUES
    }
    if counts != {4: 32, 8: 32, 16: 32}:
        raise RuntimeError(f"W18 K balance changed: {counts}")
    return rows


def generate_all_w18() -> list[LatentCompositionAuthorityCase]:
    rows: list[LatentCompositionAuthorityCase] = []
    for domain_id in DOMAINS_ORDER:
        rows.extend(generate_w18_domain(domain_id))
    if len(rows) != 4 * CASES_PER_DOMAIN:
        raise RuntimeError("W18 authority count mismatch")
    return rows


def all_w18_text_atoms() -> set[str]:
    values: set[str] = set()
    for domain in DOMAINS.values():
        values.add(domain.question)
    for intents in INTENTS.values():
        for intent in intents:
            values.update(intent.definitions)
            values.update(intent.state_texts)
    for group in SEVERITY_FIELD_TEXTS:
        values.update(group)
    for group in CONFIDENCE_FIELD_TEXTS:
        values.update(group)
    for group in SEVERITY_LATENT_DEFINITIONS:
        values.update(group)
    for group in CONFIDENCE_LATENT_DEFINITIONS:
        values.update(group)
    for family in (
        RESPONSE_DEFINITIONS,
        REVIEW_DEFINITIONS,
        RISK_DEFINITIONS,
        URGENCY_DEFINITIONS,
    ):
        for group in family:
            values.update(group)
    values.update(
        (
            "Which operational response category follows from the extracted impact level?",
            "Does the extracted impact or evidence state require human review?",
            "Which risk category follows from the extracted impact level?",
            "Which urgency category follows from the extracted impact and evidence states?",
            "Which latent severity category matches the isolated impact evidence?",
            "Which latent confidence category matches the isolated evidence-quality statement?",
        )
    )
    return values


__all__ = [
    "CASES_PER_DOMAIN",
    "CONFIDENCE_KEYS",
    "CONFIDENCE_LATENT_DEFINITIONS",
    "CONFIDENCE_LATENT_OPTIONS",
    "DOMAIN_SEEDS",
    "DOMAINS",
    "DOMAINS_ORDER",
    "INTENTS",
    "K_VALUES",
    "LatentCompositionAuthorityCase",
    "PARAPHRASE_VIEWS",
    "SEVERITY_KEYS",
    "SEVERITY_LATENT_DEFINITIONS",
    "SEVERITY_LATENT_OPTIONS",
    "all_w18_text_atoms",
    "generate_all_w18",
    "generate_w18_domain",
]
