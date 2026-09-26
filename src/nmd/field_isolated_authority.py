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

TRAIN_DOMAINS = ("CK", "CL", "CM", "CN")
DEV_DOMAIN = "CO"
CONFIRM_DOMAINS = ("CP", "CQ")

DOMAIN_SEEDS = {
    "CK": 361201,
    "CL": 361207,
    "CM": 361219,
    "CN": 361227,
    "CO": 362331,
    "CP": 363441,
    "CQ": 364557,
}

SEVERITY_LABELS = ("minimal", "moderate", "high", "critical")
CONFIDENCE_LEVELS = ("verified", "provisional", "uncertain")


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
class FieldIsolatedAuthorityCase:
    typed: TypedDecisionCase
    split: str
    domain_id: str
    diagnosis_k: int
    severity: int
    confidence: str
    intent_field: str
    severity_field: str
    confidence_field: str
    full_state_text: str
    # decisions -> options -> D0/D1/D2
    option_definitions: tuple[
        tuple[tuple[str, str, str], ...],
        ...,
    ]


def _actions(actor: str) -> tuple[ActionSpec, ActionSpec, ActionSpec, ActionSpec]:
    return (
        ActionSpec(
            "initiate",
            (
                f"The {actor} is initiating a new {{subject}}.",
                "This request concerns establishing {subject} that is not active yet.",
                "The intended operation is first-time activation of {subject}.",
            ),
            (
                "I need to initiate a new {subject}.",
                "Can you establish {subject} for me?",
                "Nothing is active yet; I want {subject} started.",
                "Please activate {subject} as a fresh request.",
                "I am arranging {subject} for the first time.",
                "How can I create a new {subject}?",
            ),
        ),
        ActionSpec(
            "amend",
            (
                f"The {actor} is amending information on an active {{subject}}.",
                "This request concerns modifying details of existing {subject}.",
                "The intended operation is a revision to current {subject}.",
            ),
            (
                "I need to amend information on my active {subject}.",
                "Can you modify details attached to the current {subject}?",
                "The {subject} exists already and needs a revision.",
                "Please correct the recorded details for {subject}.",
                "I want to alter my current {subject} without replacing it.",
                "How can I update the existing {subject}?",
            ),
        ),
        ActionSpec(
            "terminate",
            (
                f"The {actor} is terminating an active {{subject}}.",
                "This request concerns ending existing {subject}.",
                "The intended operation is withdrawal or termination of {subject}.",
            ),
            (
                "I need to terminate my active {subject}.",
                "Please end the existing {subject}.",
                "I want to withdraw the current {subject}.",
                "Can you deactivate {subject} that is already active?",
                "The present {subject} should be discontinued.",
                "How can I remove my active {subject}?",
            ),
        ),
        ActionSpec(
            "track",
            (
                f"The {actor} is tracking progress of an existing {{subject}}.",
                "This request asks for the present progress of {subject}.",
                "The intended operation is checking the latest state of {subject}.",
            ),
            (
                "I need the current progress of my {subject}.",
                "Can you show the latest state of the existing {subject}?",
                "I want to track what has happened with {subject}.",
                "Please report where my current {subject} stands.",
                "Has the active {subject} moved forward?",
                "How far has my {subject} progressed?",
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
        raise ValueError("W17 requires four subjects/domain")
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
            "CK",
            "w17-community-garden-administration",
            "gardener",
            "Which community-garden administration request matches the gardener message?",
            (
                ("plot", "community garden plot agreement"),
                ("greenhouse", "shared greenhouse access"),
                ("compost", "community compost allocation"),
                ("locker", "garden tool-locker assignment"),
            ),
        ),
        _domain(
            "CL",
            "w17-heatpump-support",
            "homeowner",
            "Which residential heat-pump support request matches the homeowner message?",
            (
                ("survey", "heat-pump property assessment"),
                ("service", "heat-pump maintenance plan"),
                ("control", "heat-pump control integration"),
                ("rebate", "heat-pump rebate claim"),
            ),
        ),
        _domain(
            "CM",
            "w17-campus-media-lending",
            "student",
            "Which campus media-equipment request matches the student message?",
            (
                ("camera", "video-camera equipment loan"),
                ("lighting", "portable lighting-kit loan"),
                ("audio", "microphone recording-kit loan"),
                ("editing", "editing-workstation reservation"),
            ),
        ),
        _domain(
            "CN",
            "w17-ferry-commuter-account",
            "commuter",
            "Which ferry commuter-account request matches the commuter message?",
            (
                ("pass", "ferry commuter travel pass"),
                ("vehicle", "ferry vehicle-account add-on"),
                ("bicycle", "ferry bicycle-account add-on"),
                ("payment", "ferry account payment profile"),
            ),
        ),
        _domain(
            "CO",
            "w17-workshop-membership-dev",
            "member",
            "Which neighborhood workshop membership request matches the member message?",
            (
                ("wood", "woodworking-room membership"),
                ("electronics", "electronics-bench membership"),
                ("ceramics", "ceramics-studio membership"),
                ("textile", "textile-room membership"),
            ),
        ),
        _domain(
            "CP",
            "w17-appliance-recycling-confirm",
            "resident",
            "Which appliance recycling-pickup request matches the resident message?",
            (
                ("fridge", "refrigerator recycling pickup"),
                ("washer", "washing-machine recycling pickup"),
                ("television", "television recycling pickup"),
                ("cooler", "air-conditioner recycling pickup"),
            ),
        ),
        _domain(
            "CQ",
            "w17-marina-berth-confirm",
            "boater",
            "Which public marina permit request matches the boater message?",
            (
                ("seasonal", "seasonal marina berth authorization"),
                ("visitor", "visitor marina berth authorization"),
                ("launch", "marina launch-ramp authorization"),
                ("storage", "marina storage-rack authorization"),
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
                        text.format(subject=subject.label)
                        for text in action.definitions
                    ),
                    state_texts=tuple(
                        text.format(subject=subject.label)
                        for text in action.states
                    ),
                )
            )
    if len(rows) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W17 requires 16 intents/domain")
    if len({row.intent_id for row in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W17 intent IDs must be unique")
    all_defs = [text for row in rows for text in row.definitions]
    if len(set(all_defs)) != INTENTS_PER_DOMAIN * 3:
        raise RuntimeError("W17 diagnosis definitions must be unique")
    return tuple(rows)


INTENTS = {domain_id: _intent_specs(spec) for domain_id, spec in DOMAINS.items()}


RESPONSE_DEFINITIONS = (
    (
        "Use ordinary handling when the impact tier is minimal.",
        "A minimal impact tier calls for routine handling.",
        "Choose the routine operational path for minimal impact.",
    ),
    (
        "Schedule planned handling when the impact tier is moderate.",
        "A moderate impact tier calls for planned follow-up.",
        "Choose scheduled operational attention for moderate impact.",
    ),
    (
        "Escalate handling promptly when the impact tier is high.",
        "A high impact tier calls for prompt escalation.",
        "Choose accelerated operational handling for high impact.",
    ),
    (
        "Use emergency handling when the impact tier is critical.",
        "A critical impact tier calls for immediate escalation.",
        "Choose the emergency operational path for critical impact.",
    ),
)

REVIEW_DEFINITIONS = (
    (
        "Human review is not required when evidence is not uncertain and impact is below critical.",
        "Choose no review unless evidence status is uncertain or impact tier is critical.",
        "The case can remain automated when confidence is adequate and impact is noncritical.",
    ),
    (
        "Human review is required when evidence is uncertain or impact is critical.",
        "Choose review for uncertain evidence status or a critical impact tier.",
        "The case needs a person when confidence is uncertain or severity reaches critical.",
    ),
)

RISK_DEFINITIONS = (
    (
        "Risk level zero corresponds to a minimal impact tier.",
        "Assign risk zero for minimal impact.",
        "The lowest risk level maps to minimal impact.",
    ),
    (
        "Risk level one corresponds to a moderate impact tier.",
        "Assign risk one for moderate impact.",
        "The second risk level maps to moderate impact.",
    ),
    (
        "Risk level two corresponds to a high impact tier.",
        "Assign risk two for high impact.",
        "The third risk level maps to high impact.",
    ),
    (
        "Risk level three corresponds to a critical impact tier.",
        "Assign risk three for critical impact.",
        "The highest risk level maps to critical impact.",
    ),
)

URGENCY_DEFINITIONS = (
    (
        "Urgency zero means routine timing is sufficient.",
        "Assign urgency zero when no accelerated response is needed.",
        "The lowest urgency level permits routine timing.",
    ),
    (
        "Urgency one means planned attention is appropriate.",
        "Assign urgency one when attention should be scheduled.",
        "The second urgency level calls for planned follow-up.",
    ),
    (
        "Urgency two means prompt attention is required.",
        "Assign urgency two when intervention should happen promptly.",
        "The third urgency level calls for accelerated attention.",
    ),
    (
        "Urgency three means immediate attention is required.",
        "Assign urgency three when intervention cannot wait.",
        "The highest urgency level calls for immediate action.",
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
    return tuple(
        LogicalOption(
            option_id=f"{prefix}-{index}",
            criterion_text=definition[0],
            value=value,
        )
        for index, (definition, value) in enumerate(zip(definitions, values))
    )


RESPONSE_OPTIONS = _logical_options("response17", RESPONSE_DEFINITIONS)
REVIEW_OPTIONS = _logical_options("review17", REVIEW_DEFINITIONS, values=(0.0, 1.0))
RISK_OPTIONS = _logical_options(
    "risk17", RISK_DEFINITIONS, values=(0.0, 1.0, 2.0, 3.0)
)
URGENCY_OPTIONS = _logical_options(
    "urgency17", URGENCY_DEFINITIONS, values=(0.0, 1.0, 2.0, 3.0)
)


def _candidate_order(
    domain_id: str,
    intent_index: int,
    state_variant: int,
) -> tuple[list[str], dict[str, int]]:
    intents = INTENTS[domain_id]
    gold = intents[intent_index].intent_id
    negatives = [row.intent_id for row in intents if row.intent_id != gold]
    seed = DOMAIN_SEEDS[domain_id] + intent_index * 1021 + state_variant * 109
    random.Random(seed).shuffle(negatives)
    membership = [gold, *negatives]
    presentation = list(membership)
    random.Random(seed + 71_177).shuffle(presentation)
    return membership, {item: idx for idx, item in enumerate(presentation)}


def _diagnosis_k(intent_index: int, state_variant: int) -> int:
    return K_VALUES[(intent_index * STATE_VARIANTS_PER_INTENT + state_variant) % 3]


def _case(
    domain_id: str,
    intent_index: int,
    state_variant: int,
    *,
    split: str,
) -> FieldIsolatedAuthorityCase:
    domain = DOMAINS[domain_id]
    intents = INTENTS[domain_id]
    by_id = {row.intent_id: row for row in intents}
    intent = intents[intent_index]

    k = _diagnosis_k(intent_index, state_variant)
    membership, rank = _candidate_order(domain_id, intent_index, state_variant)
    selected = membership[:k]
    presented = sorted(selected, key=lambda item: rank[item])
    diagnosis_gold = presented.index(intent.intent_id)
    diagnosis_defs = tuple(by_id[option_id].definitions for option_id in presented)
    diagnosis_options = tuple(
        LogicalOption(option_id=option_id, criterion_text=defs[0])
        for option_id, defs in zip(presented, diagnosis_defs)
    )

    rng = random.Random(
        DOMAIN_SEEDS[domain_id] + intent_index * 4127 + state_variant * 521 + 29
    )
    severity = rng.randrange(4)
    confidence = CONFIDENCE_LEVELS[rng.randrange(3)]
    gold_mass = CONFIDENCE_MASS[confidence]

    response_gold = severity
    review_gold = int(confidence == "uncertain" or severity == 3)
    risk_gold = severity
    urgency_gold = min(3, severity + int(confidence == "uncertain"))

    intent_field = intent.state_texts[state_variant]
    severity_field = f"Impact tier is {SEVERITY_LABELS[severity]}."
    confidence_field = f"Evidence status is {confidence}."
    full_state = (
        f"Request detail: {intent_field} "
        f"{severity_field} {confidence_field}"
    )

    diagnosis_prob = _categorical_distribution(k, diagnosis_gold, gold_mass)
    response_prob = _categorical_distribution(4, response_gold, gold_mass)
    review_prob = _categorical_distribution(2, review_gold, gold_mass)
    risk_prob = _ordinal_distribution(4, risk_gold, gold_mass)
    urgency_prob = _ordinal_distribution(4, urgency_gold, gold_mass)

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
            question_text="Which handling path matches the impact tier?",
            options=RESPONSE_OPTIONS,
            gold_index=response_gold,
            gold_probabilities=response_prob,
        ),
        TypedDecision(
            question_id="needs_review",
            primitive="noul",
            question_text="Does impact tier or evidence status require human review?",
            options=REVIEW_OPTIONS,
            gold_index=review_gold,
            gold_probabilities=review_prob,
        ),
        TypedDecision(
            question_id="risk",
            primitive="score",
            question_text="Which risk level corresponds to the impact tier?",
            options=RISK_OPTIONS,
            gold_index=risk_gold,
            gold_probabilities=risk_prob,
            gold_score=_expected_score(risk_prob),
        ),
        TypedDecision(
            question_id="urgency",
            primitive="score",
            question_text="Which urgency level follows from impact tier and evidence status?",
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
            raise RuntimeError("W17 decision/definition width mismatch")
        for option, views in zip(decision.options, definitions):
            if option.criterion_text != views[0]:
                raise RuntimeError("W17 D0 must equal production criterion")

    base_id = f"w17-{domain_id.lower()}-{intent_index:02d}-{state_variant}"
    typed = TypedDecisionCase(
        case_id=f"{base_id}-k{k}",
        workflow=domain.workflow,
        state_text=full_state,
        decisions=decisions,
    )
    return FieldIsolatedAuthorityCase(
        typed=typed,
        split=split,
        domain_id=domain_id,
        diagnosis_k=k,
        severity=severity,
        confidence=confidence,
        intent_field=intent_field,
        severity_field=severity_field,
        confidence_field=confidence_field,
        full_state_text=full_state,
        option_definitions=option_definitions,
    )


def _generate_domain(domain_id: str, *, split: str) -> list[FieldIsolatedAuthorityCase]:
    if domain_id not in DOMAINS:
        raise ValueError("unknown W17 domain")
    rows = [
        _case(domain_id, intent_index, state_variant, split=split)
        for intent_index in range(INTENTS_PER_DOMAIN)
        for state_variant in range(STATE_VARIANTS_PER_INTENT)
    ]
    if len(rows) != CASES_PER_DOMAIN:
        raise RuntimeError("W17 domain case count mismatch")
    counts = {k: sum(row.diagnosis_k == k for row in rows) for k in K_VALUES}
    if counts != {4: 32, 8: 32, 16: 32}:
        raise RuntimeError(f"W17 K balance changed: {counts}")
    return rows


def generate_w17_train() -> list[FieldIsolatedAuthorityCase]:
    rows: list[FieldIsolatedAuthorityCase] = []
    for domain in TRAIN_DOMAINS:
        rows.extend(_generate_domain(domain, split="train"))
    if len(rows) != 384:
        raise RuntimeError("W17 TRAIN count changed")
    return rows


def generate_w17_dev() -> list[FieldIsolatedAuthorityCase]:
    rows = _generate_domain(DEV_DOMAIN, split="dev-co")
    if len(rows) != 96:
        raise RuntimeError("W17 DEV count changed")
    return rows


def generate_w17_confirm(
    domain_id: str,
    *,
    allow_confirm: bool = False,
) -> list[FieldIsolatedAuthorityCase]:
    if not allow_confirm:
        raise RuntimeError("W17 CONFIRM CP/CQ sealed before pre-confirm freeze")
    if domain_id not in CONFIRM_DOMAINS:
        raise ValueError("W17 CONFIRM domain must be CP/CQ")
    rows = _generate_domain(domain_id, split=f"confirm-{domain_id.lower()}")
    if len(rows) != 96:
        raise RuntimeError("W17 CONFIRM count changed")
    return rows


def all_w17_text_atoms() -> set[str]:
    values: set[str] = set()
    for domain in DOMAINS.values():
        values.add(domain.question)
    for intents in INTENTS.values():
        for intent in intents:
            values.update(intent.definitions)
            values.update(intent.state_texts)
    for group in (
        RESPONSE_DEFINITIONS,
        REVIEW_DEFINITIONS,
        RISK_DEFINITIONS,
        URGENCY_DEFINITIONS,
    ):
        for option in group:
            values.update(option)
    values.update(
        (
            "Which handling path matches the impact tier?",
            "Does impact tier or evidence status require human review?",
            "Which risk level corresponds to the impact tier?",
            "Which urgency level follows from impact tier and evidence status?",
        )
    )
    values.update(f"Impact tier is {label}." for label in SEVERITY_LABELS)
    values.update(f"Evidence status is {label}." for label in CONFIDENCE_LEVELS)
    return values


__all__ = [
    "CASES_PER_DOMAIN",
    "CONFIRM_DOMAINS",
    "DEV_DOMAIN",
    "DOMAIN_SEEDS",
    "DOMAINS",
    "FieldIsolatedAuthorityCase",
    "INTENTS",
    "K_VALUES",
    "PARAPHRASE_VIEWS",
    "TRAIN_DOMAINS",
    "all_w17_text_atoms",
    "generate_w17_confirm",
    "generate_w17_dev",
    "generate_w17_train",
]
