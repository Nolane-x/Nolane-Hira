from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAIN_SEEDS = {
    "BN": 311901,
    "BO": 311907,
    "BP": 311919,
    "BQ": 311931,
}
K_VALUES = (4, 8, 16)
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
    label_template: str
    definition_template: str
    state_templates: tuple[str, str, str, str]


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    workflow: str
    question: str
    subjects: tuple[SubjectSpec, SubjectSpec, SubjectSpec, SubjectSpec]
    actions: tuple[ActionSpec, ActionSpec, ActionSpec, ActionSpec]


@dataclass(frozen=True)
class IntentSpec:
    intent_id: str
    terse_label: str
    natural_definition: str
    state_texts: tuple[str, str, str, str]


@dataclass(frozen=True)
class AnchorResidualView:
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
            LogicalOption(option_id=oid, criterion_text=text)
            for oid, text in zip(self.option_ids, self.option_texts)
        )


def _actions(actor: str) -> tuple[ActionSpec, ActionSpec, ActionSpec, ActionSpec]:
    return (
        ActionSpec(
            "start",
            "start {subject}",
            f"The {actor} wants to begin a new {{subject}} process.",
            (
                "I need to begin a new {subject}.",
                "How can I start the {subject} process?",
                "Please help me open a new {subject}.",
                "I am trying to create a fresh {subject} request.",
            ),
        ),
        ActionSpec(
            "change",
            "change {subject}",
            f"The {actor} wants to modify an existing {{subject}}.",
            (
                "I need to change my existing {subject}.",
                "Can I update details on the {subject} I already have?",
                "Please modify my current {subject}.",
                "Something on my existing {subject} needs an update.",
            ),
        ),
        ActionSpec(
            "stop",
            "cancel {subject}",
            f"The {actor} wants to cancel, withdraw, or end an existing {{subject}}.",
            (
                "I want to cancel my existing {subject}.",
                "Please close the {subject} I already arranged.",
                "I no longer want to continue with this {subject}.",
                "How do I withdraw the current {subject}?",
            ),
        ),
        ActionSpec(
            "status",
            "{subject} status",
            f"The {actor} wants the current status or progress of an existing {{subject}}.",
            (
                "What is the current status of my {subject}?",
                "I am checking the progress of the {subject}.",
                "Can you tell me what is happening with my {subject}?",
                "I need an update about the {subject}.",
            ),
        ),
    )


LIBRARY = DomainSpec(
    "BN",
    "w12-public-library-account-services",
    "Which public library account request best matches the patron message?",
    (
        SubjectSpec("card", "library card"),
        SubjectSpec("hold", "item hold"),
        SubjectSpec("loan", "digital loan"),
        SubjectSpec("room", "study room booking"),
    ),
    _actions("patron"),
)

VEHICLE = DomainSpec(
    "BO",
    "w12-vehicle-inspection-administration",
    "Which vehicle inspection request best matches the driver message?",
    (
        SubjectSpec("safety", "safety inspection appointment"),
        SubjectSpec("emissions", "emissions inspection appointment"),
        SubjectSpec("certificate", "inspection certificate"),
        SubjectSpec("recheck", "inspection recheck"),
    ),
    _actions("driver"),
)

BENEFITS = DomainSpec(
    "BP",
    "w12-workplace-benefits-administration",
    "Which workplace benefits request best matches the employee message?",
    (
        SubjectSpec("health", "health plan enrollment"),
        SubjectSpec("dental", "dental plan enrollment"),
        SubjectSpec("retirement", "retirement contribution election"),
        SubjectSpec("leave", "paid leave benefit request"),
    ),
    _actions("employee"),
)

VET = DomainSpec(
    "BQ",
    "w12-veterinary-appointment-support",
    "Which veterinary service request best matches the pet owner message?",
    (
        SubjectSpec("checkup", "routine pet checkup"),
        SubjectSpec("vaccine", "pet vaccination visit"),
        SubjectSpec("dental", "pet dental visit"),
        SubjectSpec("followup", "pet follow-up visit"),
    ),
    _actions("pet owner"),
)

DOMAINS = {x.domain_id: x for x in (LIBRARY, VEHICLE, BENEFITS, VET)}


def _intent_specs(domain: DomainSpec) -> tuple[IntentSpec, ...]:
    rows: list[IntentSpec] = []
    for subject in domain.subjects:
        for action in domain.actions:
            rows.append(
                IntentSpec(
                    intent_id=f"{domain.domain_id.lower()}-{subject.key}-{action.key}",
                    terse_label=action.label_template.format(subject=subject.label),
                    natural_definition=action.definition_template.format(subject=subject.label),
                    state_texts=tuple(
                        x.format(subject=subject.label) for x in action.state_templates
                    ),  # type: ignore[arg-type]
                )
            )
    if len(rows) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W12 domain must have exactly 16 intents")
    if len({x.intent_id for x in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W12 intent IDs must be unique")
    if len({x.terse_label for x in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W12 labels must be unique")
    if len({x.natural_definition for x in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W12 definitions must be unique")
    return tuple(rows)


INTENTS = {domain_id: _intent_specs(spec) for domain_id, spec in DOMAINS.items()}


def _base_order(
    domain_id: str,
    intent_index: int,
    state_variant: int,
) -> tuple[list[str], dict[str, int]]:
    intents = INTENTS[domain_id]
    gold = intents[intent_index].intent_id
    negatives = [x.intent_id for x in intents if x.intent_id != gold]
    seed = DOMAIN_SEEDS[domain_id] + intent_index * 1013 + state_variant * 101
    random.Random(seed).shuffle(negatives)
    membership = [gold, *negatives]
    presentation = list(membership)
    random.Random(seed + 31_337).shuffle(presentation)
    rank = {intent_id: i for i, intent_id in enumerate(presentation)}
    return membership, rank


def generate_w12_domain(domain_id: str) -> list[AnchorResidualView]:
    if domain_id not in DOMAINS:
        raise ValueError("W12 domain must be BN, BO, BP or BQ")
    domain = DOMAINS[domain_id]
    intents = INTENTS[domain_id]
    by_id = {x.intent_id: x for x in intents}
    rows: list[AnchorResidualView] = []

    for intent_index, intent in enumerate(intents):
        for state_variant, state_text in enumerate(intent.state_texts):
            base_id = f"w12-{domain_id.lower()}-{intent_index:02d}-{state_variant}"
            membership, rank = _base_order(domain_id, intent_index, state_variant)
            for k in K_VALUES:
                selected = membership[:k]
                presented = sorted(selected, key=lambda x: rank[x])
                gold_index = presented.index(intent.intent_id)
                for view_id in ("definition", "label"):
                    option_texts = tuple(
                        by_id[oid].natural_definition
                        if view_id == "definition"
                        else by_id[oid].terse_label
                        for oid in presented
                    )
                    rows.append(
                        AnchorResidualView(
                            case_id=f"{base_id}-{view_id}-k{k}",
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
    if len(rows) != BASES_PER_DOMAIN * len(K_VALUES) * 2:
        raise RuntimeError("W12 view count mismatch")
    return rows


def generate_all_w12() -> list[AnchorResidualView]:
    rows: list[AnchorResidualView] = []
    for domain_id in DOMAIN_SEEDS:
        rows.extend(generate_w12_domain(domain_id))
    return rows
