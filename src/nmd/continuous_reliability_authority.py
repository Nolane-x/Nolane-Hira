from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAIN_SEEDS = {
    "BV": 331101,
    "BW": 331111,
    "BX": 331123,
    "BY": 331133,
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
class ContinuousReliabilityView:
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
            "start",
            (
                f"The {actor} wants to begin a new {{subject}} request.",
                "This message concerns creating {subject} that is not currently active.",
                "The request is to initiate {subject} for the first time.",
            ),
            (
                "I need to start {subject} as a new request.",
                "Please explain how to begin {subject}.",
                "There is no active {subject} yet; I want to create one.",
                "I am trying to set up {subject} for the first time.",
            ),
        ),
        ActionSpec(
            "change",
            (
                f"The {actor} wants to change details on an existing {{subject}}.",
                "This message concerns revising information attached to current {subject}.",
                "The request is to update an already existing {subject}.",
            ),
            (
                "I need to change some details on my current {subject}.",
                "How can I revise information for the existing {subject}?",
                "The {subject} already exists, but its details need updating.",
                "Please help me correct information on my {subject}.",
            ),
        ),
        ActionSpec(
            "stop",
            (
                f"The {actor} wants an existing {{subject}} stopped, closed, or withdrawn.",
                "This message concerns ending {subject} that is currently active.",
                "The request is to cancel or close the existing {subject}.",
            ),
            (
                "I want the existing {subject} stopped.",
                "Please close the {subject} that is active now.",
                "I need to withdraw my current {subject}.",
                "How do I cancel the {subject} I already have?",
            ),
        ),
        ActionSpec(
            "status",
            (
                f"The {actor} wants the current status of an existing {{subject}}.",
                "This message asks where the present {subject} stands in its process.",
                "The request concerns checking progress or the latest state of {subject}.",
            ),
            (
                "What is the current status of my {subject}?",
                "I want to know how far along the {subject} is.",
                "Has there been any progress on the existing {subject}?",
                "Please give me the latest update about my {subject}.",
            ),
        ),
    )


WASTE = DomainSpec(
    "BV",
    "w14-municipal-waste-collection-services",
    "resident",
    "Which municipal waste-collection service request best matches the resident message?",
    (
        SubjectSpec("trash", "household trash-bin service"),
        SubjectSpec("recycling", "curbside recycling service"),
        SubjectSpec("yard", "yard-waste collection service"),
        SubjectSpec("bulky", "bulky-item pickup service"),
    ),
    _actions("resident"),
)

TRANSCRIPT = DomainSpec(
    "BW",
    "w14-university-transcript-administration",
    "student",
    "Which university transcript administration request best matches the student message?",
    (
        SubjectSpec("official", "official transcript request"),
        SubjectSpec("digital", "digital transcript delivery"),
        SubjectSpec("archive", "archived academic record request"),
        SubjectSpec("verification", "enrollment-verification record request"),
    ),
    _actions("student"),
)

INTERNET = DomainSpec(
    "BX",
    "w14-home-internet-account-support",
    "subscriber",
    "Which home internet account request best matches the subscriber message?",
    (
        SubjectSpec("plan", "home internet service plan"),
        SubjectSpec("router", "provider router service"),
        SubjectSpec("address", "service-address account record"),
        SubjectSpec("autopay", "internet account automatic payment"),
    ),
    _actions("subscriber"),
)

RECREATION = DomainSpec(
    "BY",
    "w14-community-recreation-membership-services",
    "member",
    "Which community recreation membership request best matches the member message?",
    (
        SubjectSpec("pool", "community pool membership"),
        SubjectSpec("fitness", "fitness-center membership"),
        SubjectSpec("classes", "recreation class membership"),
        SubjectSpec("courts", "sports-court membership"),
    ),
    _actions("member"),
)

DOMAINS = {
    spec.domain_id: spec
    for spec in (WASTE, TRANSCRIPT, INTERNET, RECREATION)
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
        raise RuntimeError("W14 domain must contain exactly 16 intents")
    if len({row.intent_id for row in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W14 intent IDs must be unique")
    definitions = [text for row in rows for text in row.definitions]
    if len(set(definitions)) != INTENTS_PER_DOMAIN * len(PARAPHRASE_VIEWS):
        raise RuntimeError("W14 paraphrase definitions must be unique")
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
    random.Random(seed + 41_171).shuffle(presentation)
    rank = {intent_id: index for index, intent_id in enumerate(presentation)}
    return membership, rank


def generate_w14_domain(domain_id: str) -> list[ContinuousReliabilityView]:
    if domain_id not in DOMAINS:
        raise ValueError("W14 domain must be BV, BW, BX or BY")
    domain = DOMAINS[domain_id]
    intents = INTENTS[domain_id]
    by_id = {row.intent_id: row for row in intents}
    rows: list[ContinuousReliabilityView] = []

    for intent_index, intent in enumerate(intents):
        for state_variant, state_text in enumerate(intent.state_texts):
            base_id = f"w14-{domain_id.lower()}-{intent_index:02d}-{state_variant}"
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
                        ContinuousReliabilityView(
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
        raise RuntimeError("W14 view count mismatch")
    return rows


def generate_all_w14() -> list[ContinuousReliabilityView]:
    rows: list[ContinuousReliabilityView] = []
    for domain_id in DOMAIN_SEEDS:
        rows.extend(generate_w14_domain(domain_id))
    return rows


def all_w14_text_atoms() -> set[str]:
    values: set[str] = set()
    for domain in DOMAINS.values():
        values.add(domain.question)
    for intents in INTENTS.values():
        for intent in intents:
            values.update(intent.definitions)
            values.update(intent.state_texts)
    return values


__all__ = [
    "DOMAIN_SEEDS",
    "K_VALUES",
    "PARAPHRASE_VIEWS",
    "INTENTS_PER_DOMAIN",
    "STATE_VARIANTS_PER_INTENT",
    "BASES_PER_DOMAIN",
    "DOMAINS",
    "INTENTS",
    "ContinuousReliabilityView",
    "generate_w14_domain",
    "generate_all_w14",
    "all_w14_text_atoms",
]
