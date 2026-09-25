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
CASES_PER_DOMAIN = INTENTS_PER_DOMAIN * STATE_VARIANTS_PER_INTENT

TRAIN_DOMAINS = ("BZ", "CA", "CB", "CC")
DEV_DOMAIN = "CD"
CONFIRM_DOMAINS = ("CE", "CF")

DOMAIN_SEEDS = {
    "BZ": 341201,
    "CA": 341207,
    "CB": 341219,
    "CC": 341227,
    "CD": 342331,
    "CE": 343441,
    "CF": 344557,
}

SEVERITY_LABELS = (
    "minimal",
    "moderate",
    "high",
    "critical",
)
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
class AnchorPreservingAuthorityCase:
    typed: TypedDecisionCase
    split: str
    domain_id: str
    diagnosis_k: int
    severity: int
    confidence: str
    # decisions -> options -> D0/D1/D2
    option_definitions: tuple[
        tuple[tuple[str, str, str], ...],
        ...,
    ]


def _actions(actor: str) -> tuple[ActionSpec, ActionSpec, ActionSpec, ActionSpec]:
    return (
        ActionSpec(
            "open",
            (
                f"The {actor} wants to create a new {{subject}}.",
                "The message asks to begin {subject} that does not yet exist.",
                "The requested action is to open {subject} for the first time.",
            ),
            (
                "I want to open a new {subject}.",
                "Please help me begin {subject} from scratch.",
                "There is no active {subject}; I need to create one.",
                "How do I get a new {subject} started?",
                "I am trying to set up {subject} for the first time.",
                "Can you start a fresh {subject} request for me?",
            ),
        ),
        ActionSpec(
            "revise",
            (
                f"The {actor} wants to revise details on an existing {{subject}}.",
                "The message concerns changing information attached to current {subject}.",
                "The requested action is to update an already active {subject}.",
            ),
            (
                "I need to revise details on my existing {subject}.",
                "Please update information connected with the current {subject}.",
                "The {subject} already exists, but some details must change.",
                "How can I correct the information on my {subject}?",
                "I want to edit the current {subject} without opening a new one.",
                "Can you change the details recorded for my {subject}?",
            ),
        ),
        ActionSpec(
            "close",
            (
                f"The {actor} wants an existing {{subject}} closed or withdrawn.",
                "The message asks to end {subject} that is currently active.",
                "The requested action is to cancel the existing {subject}.",
            ),
            (
                "I want my existing {subject} closed.",
                "Please cancel the active {subject}.",
                "I need to withdraw the {subject} I already have.",
                "How do I end the current {subject}?",
                "The {subject} should no longer remain active.",
                "Please stop my present {subject} request.",
            ),
        ),
        ActionSpec(
            "check",
            (
                f"The {actor} wants the latest status of an existing {{subject}}.",
                "The message asks where the current {subject} stands in its process.",
                "The requested action is to check progress on {subject}.",
            ),
            (
                "What is the latest status of my {subject}?",
                "I want to check progress on the existing {subject}.",
                "Has anything changed with my current {subject}?",
                "Please tell me where the {subject} stands now.",
                "How far along is the {subject} at the moment?",
                "I need an update about my active {subject}.",
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
        raise ValueError("W15 domains require exactly four subjects")
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
            "BZ",
            "w15-museum-member-services",
            "visitor",
            "Which museum account or ticket request best matches the visitor message?",
            (
                ("member", "museum annual membership"),
                ("event", "museum event ticket booking"),
                ("tour", "guided-tour reservation"),
                ("donor", "museum donor-benefits profile"),
            ),
        ),
        _domain(
            "CA",
            "w15-residential-solar-support",
            "homeowner",
            "Which residential solar support request best matches the homeowner message?",
            (
                ("survey", "solar site-survey request"),
                ("install", "solar installation schedule"),
                ("monitor", "inverter monitoring service"),
                ("meter", "net-metering paperwork request"),
            ),
        ),
        _domain(
            "CB",
            "w15-port-credential-administration",
            "port user",
            "Which maritime port credential request best matches the port-user message?",
            (
                ("gate", "terminal gate-access credential"),
                ("crew", "vessel crew access pass"),
                ("contractor", "port contractor permit"),
                ("vehicle", "terminal vehicle badge"),
            ),
        ),
        _domain(
            "CC",
            "w15-mealkit-subscription-support",
            "subscriber",
            "Which meal-kit subscription request best matches the subscriber message?",
            (
                ("plan", "meal-kit weekly plan"),
                ("delivery", "meal-kit delivery schedule"),
                ("diet", "dietary preference profile"),
                ("billing", "meal-kit billing method"),
            ),
        ),
        _domain(
            "CD",
            "w15-fleet-maintenance-dev",
            "fleet manager",
            "Which commercial fleet maintenance request best matches the fleet-manager message?",
            (
                ("preventive", "preventive maintenance contract"),
                ("roadside", "roadside assistance agreement"),
                ("tires", "fleet tire-service agreement"),
                ("telematics", "telematics maintenance plan"),
            ),
        ),
        _domain(
            "CE",
            "w15-community-arts-grant-confirm",
            "applicant",
            "Which community arts grant request best matches the applicant message?",
            (
                ("project", "community arts project grant"),
                ("equipment", "arts equipment grant"),
                ("venue", "community venue-support grant"),
                ("residency", "artist residency grant"),
            ),
        ),
        _domain(
            "CF",
            "w15-language-exchange-confirm",
            "participant",
            "Which language-exchange program request best matches the participant message?",
            (
                ("partner", "language partner-matching request"),
                ("group", "conversation-group enrollment"),
                ("session", "guided practice-session registration"),
                ("assessment", "language proficiency assessment appointment"),
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
                    intent_id=(
                        f"{domain.domain_id.lower()}-{subject.key}-{action.key}"
                    ),
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
        raise RuntimeError("W15 domain must contain exactly 16 intents")
    if len({row.intent_id for row in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W15 intent IDs must be unique")
    all_definitions = [text for row in rows for text in row.definitions]
    if len(set(all_definitions)) != INTENTS_PER_DOMAIN * 3:
        raise RuntimeError("W15 diagnosis paraphrases must be unique")
    return tuple(rows)


INTENTS = {
    domain_id: _intent_specs(spec)
    for domain_id, spec in DOMAINS.items()
}


RESPONSE_DEFINITIONS = (
    (
        "Continue routine observation because the reported impact is minimal.",
        "Minimal impact calls for ordinary monitoring without special intervention.",
        "Choose routine observation when the current impact remains minimal.",
    ),
    (
        "Arrange planned follow-up because the reported impact is moderate.",
        "Moderate impact calls for scheduled attention rather than immediate intervention.",
        "Choose planned follow-up when the current impact is moderate.",
    ),
    (
        "Escalate for prompt intervention because the reported impact is high.",
        "High impact calls for prompt operational intervention.",
        "Choose prompt escalation when the current impact is high.",
    ),
    (
        "Trigger immediate intervention because the reported impact is critical.",
        "Critical impact requires immediate operational action.",
        "Choose immediate intervention when the current impact is critical.",
    ),
)

REVIEW_DEFINITIONS = (
    (
        "Human review is not required when evidence is sufficiently reliable and impact is not critical.",
        "Choose no review when the evidence is reliable enough and the case is below critical impact.",
        "The case can proceed without human review if evidence is not uncertain and impact is not critical.",
    ),
    (
        "Human review is required when evidence is uncertain or the impact is critical.",
        "Choose review whenever confidence is uncertain or the case reaches critical impact.",
        "A human must review the case if uncertainty is high or the impact is critical.",
    ),
)

RISK_DEFINITIONS = (
    (
        "Risk score zero represents minimal impact.",
        "Assign risk level zero to the minimal-impact condition.",
        "The lowest risk score corresponds to minimal reported impact.",
    ),
    (
        "Risk score one represents moderate impact.",
        "Assign risk level one to the moderate-impact condition.",
        "The second risk score corresponds to moderate reported impact.",
    ),
    (
        "Risk score two represents high impact.",
        "Assign risk level two to the high-impact condition.",
        "The third risk score corresponds to high reported impact.",
    ),
    (
        "Risk score three represents critical impact.",
        "Assign risk level three to the critical-impact condition.",
        "The highest risk score corresponds to critical reported impact.",
    ),
)

URGENCY_DEFINITIONS = (
    (
        "Urgency score zero permits routine timing.",
        "Assign urgency zero when routine timing is sufficient.",
        "The lowest urgency level means no accelerated response is needed.",
    ),
    (
        "Urgency score one calls for planned attention.",
        "Assign urgency one when planned attention is appropriate.",
        "The second urgency level means follow-up should be scheduled.",
    ),
    (
        "Urgency score two calls for prompt attention.",
        "Assign urgency two when prompt attention is required.",
        "The third urgency level means intervention should occur promptly.",
    ),
    (
        "Urgency score three calls for immediate attention.",
        "Assign urgency three when immediate attention is required.",
        "The highest urgency level means intervention should happen immediately.",
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
        raise ValueError("W15 values/definitions mismatch")
    return tuple(
        LogicalOption(
            option_id=f"{prefix}-{index}",
            criterion_text=definition[0],
            value=value,
        )
        for index, (definition, value) in enumerate(zip(definitions, values))
    )


RESPONSE_OPTIONS = _logical_options("response", RESPONSE_DEFINITIONS)
REVIEW_OPTIONS = _logical_options(
    "review",
    REVIEW_DEFINITIONS,
    values=(0.0, 1.0),
)
RISK_OPTIONS = _logical_options(
    "risk",
    RISK_DEFINITIONS,
    values=(0.0, 1.0, 2.0, 3.0),
)
URGENCY_OPTIONS = _logical_options(
    "urgency",
    URGENCY_DEFINITIONS,
    values=(0.0, 1.0, 2.0, 3.0),
)


def _candidate_order(
    domain_id: str,
    intent_index: int,
    state_variant: int,
) -> tuple[list[str], dict[str, int]]:
    intents = INTENTS[domain_id]
    gold = intents[intent_index].intent_id
    negatives = [row.intent_id for row in intents if row.intent_id != gold]
    seed = DOMAIN_SEEDS[domain_id] + intent_index * 1009 + state_variant * 103
    random.Random(seed).shuffle(negatives)
    membership = [gold, *negatives]
    presentation = list(membership)
    random.Random(seed + 51_173).shuffle(presentation)
    return membership, {
        intent_id: index for index, intent_id in enumerate(presentation)
    }


def _diagnosis_k(intent_index: int, state_variant: int) -> int:
    index = (intent_index * STATE_VARIANTS_PER_INTENT + state_variant) % 3
    return K_VALUES[index]


def _state_with_reliability(
    state_text: str,
    *,
    severity: int,
    confidence: str,
) -> str:
    return (
        f"{state_text} "
        f"Reported impact is {SEVERITY_LABELS[severity]}; "
        f"evidence confidence is {confidence}."
    )


def _case(
    domain_id: str,
    intent_index: int,
    state_variant: int,
    *,
    split: str,
) -> AnchorPreservingAuthorityCase:
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

    diagnosis_definitions = tuple(
        by_id[option_id].definitions for option_id in presented
    )
    diagnosis_options = tuple(
        LogicalOption(
            option_id=option_id,
            criterion_text=definitions[0],
        )
        for option_id, definitions in zip(presented, diagnosis_definitions)
    )

    reliability_rng = random.Random(
        DOMAIN_SEEDS[domain_id]
        + intent_index * 4_099
        + state_variant * 509
        + 17
    )
    severity = reliability_rng.randrange(4)
    confidence = CONFIDENCE_LEVELS[
        reliability_rng.randrange(len(CONFIDENCE_LEVELS))
    ]
    gold_mass = CONFIDENCE_MASS[confidence]

    response_gold = severity
    review_gold = int(confidence == "uncertain" or severity == 3)
    risk_gold = severity
    urgency_gold = min(3, severity + int(confidence == "uncertain"))

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

    state_text = _state_with_reliability(
        intent.state_texts[state_variant],
        severity=severity,
        confidence=confidence,
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
            question_text=(
                "Which operational response matches the reported impact?"
            ),
            options=RESPONSE_OPTIONS,
            gold_index=response_gold,
            gold_probabilities=response_prob,
        ),
        TypedDecision(
            question_id="needs_review",
            primitive="noul",
            question_text=(
                "Does this case require human review because confidence is uncertain or impact is critical?"
            ),
            options=REVIEW_OPTIONS,
            gold_index=review_gold,
            gold_probabilities=review_prob,
        ),
        TypedDecision(
            question_id="risk",
            primitive="score",
            question_text="What risk score matches the reported impact?",
            options=RISK_OPTIONS,
            gold_index=risk_gold,
            gold_probabilities=risk_prob,
            gold_score=_expected_score(risk_prob),
        ),
        TypedDecision(
            question_id="urgency",
            primitive="score",
            question_text=(
                "What urgency score follows from impact and evidence confidence?"
            ),
            options=URGENCY_OPTIONS,
            gold_index=urgency_gold,
            gold_probabilities=urgency_prob,
            gold_score=_expected_score(urgency_prob),
        ),
    )

    option_definitions = (
        diagnosis_definitions,
        RESPONSE_DEFINITIONS,
        REVIEW_DEFINITIONS,
        RISK_DEFINITIONS,
        URGENCY_DEFINITIONS,
    )
    for decision, definitions in zip(decisions, option_definitions):
        if len(decision.options) != len(definitions):
            raise RuntimeError("W15 decision/schema option count mismatch")
        for option, views in zip(decision.options, definitions):
            if option.criterion_text != views[0]:
                raise RuntimeError("W15 D0 must equal production criterion text")

    base_id = (
        f"w15-{domain_id.lower()}-{intent_index:02d}-{state_variant}"
    )
    typed = TypedDecisionCase(
        case_id=f"{base_id}-k{k}",
        workflow=domain.workflow,
        state_text=state_text,
        decisions=decisions,
    )
    return AnchorPreservingAuthorityCase(
        typed=typed,
        split=split,
        domain_id=domain_id,
        diagnosis_k=k,
        severity=severity,
        confidence=confidence,
        option_definitions=option_definitions,
    )


def _generate_domain(
    domain_id: str,
    *,
    split: str,
) -> list[AnchorPreservingAuthorityCase]:
    if domain_id not in DOMAINS:
        raise ValueError("unknown W15 domain")
    rows = [
        _case(
            domain_id,
            intent_index,
            state_variant,
            split=split,
        )
        for intent_index in range(INTENTS_PER_DOMAIN)
        for state_variant in range(STATE_VARIANTS_PER_INTENT)
    ]
    if len(rows) != CASES_PER_DOMAIN:
        raise RuntimeError("W15 domain case count mismatch")
    counts = {
        k: sum(row.diagnosis_k == k for row in rows)
        for k in K_VALUES
    }
    if counts != {4: 32, 8: 32, 16: 32}:
        raise RuntimeError(f"W15 K balance changed: {counts}")
    return rows


def generate_w15_train() -> list[AnchorPreservingAuthorityCase]:
    rows: list[AnchorPreservingAuthorityCase] = []
    for domain_id in TRAIN_DOMAINS:
        rows.extend(_generate_domain(domain_id, split="train"))
    if len(rows) != 384:
        raise RuntimeError("W15 TRAIN must contain 384 cases")
    return rows


def generate_w15_dev() -> list[AnchorPreservingAuthorityCase]:
    rows = _generate_domain(DEV_DOMAIN, split="dev-cd")
    if len(rows) != 96:
        raise RuntimeError("W15 DEV-CD must contain 96 cases")
    return rows


def generate_w15_confirm(
    domain_id: str,
    *,
    allow_confirm: bool = False,
) -> list[AnchorPreservingAuthorityCase]:
    if not allow_confirm:
        raise RuntimeError(
            "W15 CONFIRM-CE/CF sealed until every trainable candidate is DEV-frozen"
        )
    if domain_id not in CONFIRM_DOMAINS:
        raise ValueError("W15 confirm domain must be CE or CF")
    rows = _generate_domain(
        domain_id,
        split=f"confirm-{domain_id.lower()}",
    )
    if len(rows) != 96:
        raise RuntimeError("W15 CONFIRM domain must contain 96 cases")
    return rows


def all_w15_text_atoms() -> set[str]:
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
            "Which operational response matches the reported impact?",
            "Does this case require human review because confidence is uncertain or impact is critical?",
            "What risk score matches the reported impact?",
            "What urgency score follows from impact and evidence confidence?",
        )
    )
    return values


__all__ = [
    "AnchorPreservingAuthorityCase",
    "CASES_PER_DOMAIN",
    "CONFIRM_DOMAINS",
    "DEV_DOMAIN",
    "DOMAIN_SEEDS",
    "DOMAINS",
    "INTENTS",
    "K_VALUES",
    "PARAPHRASE_VIEWS",
    "TRAIN_DOMAINS",
    "all_w15_text_atoms",
    "generate_w15_confirm",
    "generate_w15_dev",
    "generate_w15_train",
]
