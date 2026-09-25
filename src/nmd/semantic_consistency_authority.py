from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAIN_SEEDS = {
    "BR": 321001,
    "BS": 321013,
    "BT": 321019,
    "BU": 321031,
}
K_VALUES = (4, 8, 16)
PARAPHRASE_VIEWS = ("D0", "D1", "D2")
INTENTS_PER_DOMAIN = 16
STATE_VARIANTS_PER_INTENT = 4
BASES_PER_DOMAIN = 64


@dataclass(frozen=True)
class SubjectSpec:
    key: str
    label: str


@dataclass(frozen=True)
class ActionSpec:
    key: str
    definition_templates: tuple[str, str, str]
    state_templates: tuple[str, str, str, str]


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
    state_texts: tuple[str, str, str, str]


@dataclass(frozen=True)
class SemanticConsistencyView:
    case_id: str
    base_id: str
    domain_id: str
    intent_id: str
    state_text: str
    question_text: str
    view_id: str
    diagnosis_k: int
    option_ids: tuple[str, ...]
    option_texts: tuple[str, ...]
    gold_index: int

    def logical_options(self) -> tuple[LogicalOption, ...]:
        return tuple(
            LogicalOption(option_id=option_id, criterion_text=text)
            for option_id, text in zip(self.option_ids, self.option_texts)
        )


def _actions(actor: str) -> tuple[ActionSpec, ActionSpec, ActionSpec, ActionSpec]:
    return (
        ActionSpec(
            "open",
            (
                f"The {actor} is asking to establish a new {{subject}}.",
                "This message concerns initiating {subject} that is not yet active.",
                "The request is to set up {subject} for the first time.",
            ),
            (
                "I need to arrange {subject} for the first time.",
                "Please tell me how to get a new {subject} started.",
                "Nothing is active yet; I want to set up {subject}.",
                "I am trying to open {subject} as a new request.",
            ),
        ),
        ActionSpec(
            "revise",
            (
                f"The {actor} wants to revise details on an existing {{subject}}.",
                "This message is about altering information for {subject} that already exists.",
                "The request concerns making a change to current {subject}.",
            ),
            (
                "Some details on my current {subject} need revising.",
                "How can I alter information attached to my existing {subject}?",
                "I already have {subject}, but something in it must be changed.",
                "Please help me correct the current {subject} details.",
            ),
        ),
        ActionSpec(
            "end",
            (
                f"The {actor} wants an existing {{subject}} to be ended or withdrawn.",
                "This message concerns stopping {subject} that is currently active.",
                "The request is to close out or cancel the existing {subject}.",
            ),
            (
                "I need the existing {subject} ended.",
                "Please stop the {subject} that is active now.",
                "I want to withdraw my current {subject}.",
                "How do I close out the {subject} I already have?",
            ),
        ),
        ActionSpec(
            "track",
            (
                f"The {actor} wants to learn the present progress of an existing {{subject}}.",
                "This message asks where the current {subject} stands in its process.",
                "The request concerns checking the latest progress or status of {subject}.",
            ),
            (
                "Where does my current {subject} stand right now?",
                "I want to check how far along the {subject} is.",
                "Has there been any progress on my existing {subject}?",
                "Please give me the latest update for the {subject}.",
            ),
        ),
    )


PARKING = DomainSpec(
    "BR",
    "w13-municipal-parking-permit-administration",
    "resident",
    "Which municipal parking-permit request best matches the resident message?",
    (
        SubjectSpec("resident", "resident street-parking permit"),
        SubjectSpec("guest", "visitor parking permit"),
        SubjectSpec("construction", "temporary construction-zone parking permit"),
        SubjectSpec("accessible", "accessible parking permit"),
    ),
    _actions("resident"),
)

EDUCATION = DomainSpec(
    "BS",
    "w13-continuing-education-enrollment-services",
    "learner",
    "Which continuing-education enrollment request best matches the learner message?",
    (
        SubjectSpec("certificate", "certificate-course enrollment"),
        SubjectSpec("workshop", "skills workshop registration"),
        SubjectSpec("language", "language-course enrollment"),
        SubjectSpec("seminar", "professional seminar registration"),
    ),
    _actions("learner"),
)

INSURANCE = DomainSpec(
    "BT",
    "w13-household-insurance-claim-administration",
    "policyholder",
    "Which household-insurance claim request best matches the policyholder message?",
    (
        SubjectSpec("water", "water-damage claim"),
        SubjectSpec("theft", "household theft claim"),
        SubjectSpec("appliance", "appliance-damage claim"),
        SubjectSpec("liability", "household liability claim"),
    ),
    _actions("policyholder"),
)

GROCERY = DomainSpec(
    "BU",
    "w13-grocery-delivery-subscription-support",
    "subscriber",
    "Which grocery-delivery subscription request best matches the subscriber message?",
    (
        SubjectSpec("produce", "weekly produce-box subscription"),
        SubjectSpec("pantry", "pantry-staples subscription"),
        SubjectSpec("meals", "prepared-meals subscription"),
        SubjectSpec("essentials", "household-essentials subscription"),
    ),
    _actions("subscriber"),
)

DOMAINS = {
    spec.domain_id: spec
    for spec in (PARKING, EDUCATION, INSURANCE, GROCERY)
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
                        for template in action.definition_templates
                    ),
                    state_texts=tuple(
                        template.format(subject=subject.label)
                        for template in action.state_templates
                    ),
                )
            )
    if len(rows) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W13 domain must contain exactly 16 intents")
    if len({row.intent_id for row in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W13 intent IDs must be unique")
    all_definitions = [text for row in rows for text in row.definitions]
    if len(set(all_definitions)) != INTENTS_PER_DOMAIN * len(PARAPHRASE_VIEWS):
        raise RuntimeError("W13 paraphrase definitions must be unique")
    return tuple(rows)


INTENTS = {
    domain_id: _intent_specs(spec)
    for domain_id, spec in DOMAINS.items()
}


def _base_order(
    domain_id: str,
    intent_index: int,
    state_variant: int,
) -> tuple[list[str], dict[str, int]]:
    intents = INTENTS[domain_id]
    gold = intents[intent_index].intent_id
    negatives = [row.intent_id for row in intents if row.intent_id != gold]
    seed = DOMAIN_SEEDS[domain_id] + intent_index * 1013 + state_variant * 101
    random.Random(seed).shuffle(negatives)
    membership = [gold, *negatives]
    presentation = list(membership)
    random.Random(seed + 37_919).shuffle(presentation)
    rank = {intent_id: index for index, intent_id in enumerate(presentation)}
    return membership, rank


def generate_w13_domain(domain_id: str) -> list[SemanticConsistencyView]:
    if domain_id not in DOMAINS:
        raise ValueError("W13 domain must be BR, BS, BT or BU")
    domain = DOMAINS[domain_id]
    intents = INTENTS[domain_id]
    by_id = {row.intent_id: row for row in intents}
    rows: list[SemanticConsistencyView] = []

    for intent_index, intent in enumerate(intents):
        for state_variant, state_text in enumerate(intent.state_texts):
            base_id = f"w13-{domain_id.lower()}-{intent_index:02d}-{state_variant}"
            membership, rank = _base_order(domain_id, intent_index, state_variant)
            for k in K_VALUES:
                selected = membership[:k]
                presented = sorted(selected, key=lambda item: rank[item])
                gold_index = presented.index(intent.intent_id)
                for paraphrase_index, view_id in enumerate(PARAPHRASE_VIEWS):
                    option_texts = tuple(
                        by_id[option_id].definitions[paraphrase_index]
                        for option_id in presented
                    )
                    rows.append(
                        SemanticConsistencyView(
                            case_id=f"{base_id}-{view_id.lower()}-k{k}",
                            base_id=base_id,
                            domain_id=domain_id,
                            intent_id=intent.intent_id,
                            state_text=state_text,
                            question_text=domain.question,
                            view_id=view_id,
                            diagnosis_k=k,
                            option_ids=tuple(presented),
                            option_texts=option_texts,
                            gold_index=gold_index,
                        )
                    )

    expected = BASES_PER_DOMAIN * len(K_VALUES) * len(PARAPHRASE_VIEWS)
    if len(rows) != expected:
        raise RuntimeError("W13 view count mismatch")
    return rows


def generate_all_w13() -> list[SemanticConsistencyView]:
    rows: list[SemanticConsistencyView] = []
    for domain_id in DOMAIN_SEEDS:
        rows.extend(generate_w13_domain(domain_id))
    return rows


def generate_w13_diagnostics() -> list[SemanticConsistencyView]:
    return generate_all_w13()


def all_w13_text_atoms() -> set[str]:
    values: set[str] = set()
    for domain in DOMAINS.values():
        values.add(domain.question)
    for intents in INTENTS.values():
        for intent in intents:
            values.update(intent.definitions)
            values.update(intent.state_texts)
    return values
