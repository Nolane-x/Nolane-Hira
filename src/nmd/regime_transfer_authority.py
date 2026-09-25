from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAIN_SEEDS = {
    "CG": 351101,
    "CH": 351107,
    "CI": 351121,
    "CJ": 351133,
}
K_VALUES = (4, 8, 16)
PARAPHRASE_VIEWS = ("D0", "D1", "D2")
INTENTS_PER_DOMAIN = 16
STATE_VARIANTS_PER_INTENT = 4
BASES_PER_DOMAIN = 64

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
    states: tuple[str, str, str, str]


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
class RegimeTransferView:
    case_id: str
    base_id: str
    domain_id: str
    intent_id: str
    bare_state_text: str
    decorated_state_text: str
    question_text: str
    view_id: str
    diagnosis_k: int
    option_ids: tuple[str, ...]
    option_texts: tuple[str, ...]
    gold_index: int
    severity: int
    confidence: str

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
                f"The {actor} wants to create a new {{subject}}.",
                "The message asks to begin {subject} that does not yet exist.",
                "The requested action is to open {subject} for the first time.",
            ),
            (
                "I want to open a new {subject}.",
                "Please help me begin {subject} from scratch.",
                "There is no active {subject}; I need to create one.",
                "How do I get a new {subject} started?",
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
        raise ValueError("W16 requires four subjects/domain")
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
            "CG",
            "w16-tree-permit-administration",
            "resident",
            "Which municipal tree-permit request best matches the resident message?",
            (
                ("street", "street-tree work permit"),
                ("removal", "tree-removal permit"),
                ("planting", "tree-planting permit"),
                ("inspection", "tree-safety inspection request"),
            ),
        ),
        _domain(
            "CH",
            "w16-appliance-warranty-services",
            "customer",
            "Which home-appliance warranty request best matches the customer message?",
            (
                ("washer", "washing-machine warranty case"),
                ("fridge", "refrigerator warranty case"),
                ("oven", "oven warranty case"),
                ("dishwasher", "dishwasher warranty case"),
            ),
        ),
        _domain(
            "CI",
            "w16-college-equipment-lending",
            "student",
            "Which community-college equipment-lending request best matches the student message?",
            (
                ("laptop", "student laptop loan"),
                ("camera", "media-camera loan"),
                ("calculator", "graphing-calculator loan"),
                ("audio", "portable audio-recorder loan"),
            ),
        ),
        _domain(
            "CJ",
            "w16-bikeshare-account-support",
            "rider",
            "Which bicycle-share account request best matches the rider message?",
            (
                ("pass", "bicycle-share ride pass"),
                ("billing", "bicycle-share billing profile"),
                ("dock", "preferred docking-station setting"),
                ("access", "bicycle-share access credential"),
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
        raise RuntimeError("W16 requires 16 intents/domain")
    if len({row.intent_id for row in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W16 intent IDs must be unique")
    defs = [text for row in rows for text in row.definitions]
    if len(set(defs)) != INTENTS_PER_DOMAIN * 3:
        raise RuntimeError("W16 definitions must be unique")
    return tuple(rows)


INTENTS = {domain_id: _intent_specs(spec) for domain_id, spec in DOMAINS.items()}


def _base_order(
    domain_id: str,
    intent_index: int,
    state_variant: int,
) -> tuple[list[str], dict[str, int]]:
    intents = INTENTS[domain_id]
    gold = intents[intent_index].intent_id
    negatives = [row.intent_id for row in intents if row.intent_id != gold]
    seed = DOMAIN_SEEDS[domain_id] + intent_index * 1013 + state_variant * 107
    random.Random(seed).shuffle(negatives)
    membership = [gold, *negatives]
    presentation = list(membership)
    random.Random(seed + 61_171).shuffle(presentation)
    return membership, {item: idx for idx, item in enumerate(presentation)}


def _metadata(domain_id: str, intent_index: int, state_variant: int) -> tuple[int, str]:
    rng = random.Random(
        DOMAIN_SEEDS[domain_id] + intent_index * 4099 + state_variant * 509 + 23
    )
    return rng.randrange(4), CONFIDENCE_LEVELS[rng.randrange(3)]


def _decorate(text: str, severity: int, confidence: str) -> str:
    return (
        f"{text} Reported impact is {SEVERITY_LABELS[severity]}; "
        f"evidence confidence is {confidence}."
    )


def generate_w16_domain(domain_id: str) -> list[RegimeTransferView]:
    if domain_id not in DOMAINS:
        raise ValueError("W16 domain must be CG/CH/CI/CJ")
    domain = DOMAINS[domain_id]
    intents = INTENTS[domain_id]
    by_id = {row.intent_id: row for row in intents}
    rows: list[RegimeTransferView] = []

    for intent_index, intent in enumerate(intents):
        for state_variant, bare in enumerate(intent.state_texts):
            base_id = f"w16-{domain_id.lower()}-{intent_index:02d}-{state_variant}"
            severity, confidence = _metadata(domain_id, intent_index, state_variant)
            decorated = _decorate(bare, severity, confidence)
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
                        RegimeTransferView(
                            case_id=f"{base_id}-{view_id.lower()}-k{k}",
                            base_id=base_id,
                            domain_id=domain_id,
                            intent_id=intent.intent_id,
                            bare_state_text=bare,
                            decorated_state_text=decorated,
                            question_text=domain.question,
                            view_id=view_id,
                            diagnosis_k=k,
                            option_ids=tuple(presented),
                            option_texts=option_texts,
                            gold_index=gold_index,
                            severity=severity,
                            confidence=confidence,
                        )
                    )
    expected = BASES_PER_DOMAIN * len(K_VALUES) * len(PARAPHRASE_VIEWS)
    if len(rows) != expected:
        raise RuntimeError("W16 view count mismatch")
    return rows


def generate_all_w16() -> list[RegimeTransferView]:
    rows: list[RegimeTransferView] = []
    for domain_id in DOMAIN_SEEDS:
        rows.extend(generate_w16_domain(domain_id))
    return rows


def all_w16_text_atoms() -> set[str]:
    values: set[str] = set()
    for domain in DOMAINS.values():
        values.add(domain.question)
    for intents in INTENTS.values():
        for intent in intents:
            values.update(intent.definitions)
            values.update(intent.state_texts)
    # Decoration template atoms are intentionally included in freshness accounting.
    values.update(
        f"Reported impact is {severity}; evidence confidence is {confidence}."
        for severity in SEVERITY_LABELS
        for confidence in CONFIDENCE_LEVELS
    )
    return values


__all__ = [
    "BASES_PER_DOMAIN",
    "DOMAIN_SEEDS",
    "DOMAINS",
    "INTENTS",
    "K_VALUES",
    "PARAPHRASE_VIEWS",
    "RegimeTransferView",
    "all_w16_text_atoms",
    "generate_all_w16",
    "generate_w16_domain",
]
