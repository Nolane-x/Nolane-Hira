from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

BASES_PER_DOMAIN = 64
INTENTS_PER_DOMAIN = 16
K_VALUES = (4, 8, 16)
VIEW_IDS = ("V0", "V1", "V2", "V3")

DOMAIN_SEEDS = {
    "AU": 271501,
    "AV": 271507,
    "AW": 271519,
    "AX": 271531,
}


@dataclass(frozen=True)
class SubjectSpec:
    key: str
    label: str
    state_noun: str


@dataclass(frozen=True)
class ActionSpec:
    key: str
    label_template: str
    definition_template: str
    anchor_template: str
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
    structured_criterion: str
    lexical_bridge: str
    state_texts: tuple[str, str, str, str]


@dataclass(frozen=True)
class SemanticTransferView:
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


UTILITY = DomainSpec(
    domain_id="AU",
    workflow="w8-household-utility-support",
    question="Which household utility support intent best matches the message?",
    subjects=(
        SubjectSpec("electricity", "electricity service", "electric power"),
        SubjectSpec("water", "water service", "tap water"),
        SubjectSpec("gas", "gas service", "home gas"),
        SubjectSpec("waste", "waste collection", "trash pickup"),
    ),
    actions=(
        ActionSpec(
            "interruption",
            "{subject} outage",
            "The customer reports that {state_noun} is unavailable or has stopped.",
            "{state_noun} stopped",
            (
                "My {state_noun} stopped this morning and it still has not come back.",
                "There is no {state_noun} at my home right now.",
                "We suddenly lost {state_noun} and need to report the interruption.",
                "The {state_noun} has been unavailable since earlier today.",
            ),
        ),
        ActionSpec(
            "charge",
            "{subject} unexpected charge",
            "The customer is questioning an unexpected or incorrect charge for {state_noun}.",
            "unexpected charge",
            (
                "I found an unexpected charge on my {state_noun} account.",
                "The bill for {state_noun} includes a charge I do not recognize.",
                "Why was I charged extra for {state_noun} this month?",
                "I need help disputing an incorrect {state_noun} charge.",
            ),
        ),
        ActionSpec(
            "account_change",
            "{subject} account change",
            "The customer wants to start, stop, or transfer the account for {state_noun}.",
            "change my account",
            (
                "I need to change my account for {state_noun} because I am moving.",
                "Please help me start a new {state_noun} account at my address.",
                "I want to stop the {state_noun} account at my old home.",
                "How do I transfer my {state_noun} account to a new address?",
            ),
        ),
        ActionSpec(
            "equipment",
            "{subject} equipment issue",
            "The customer reports a meter, container, connection, or other service equipment problem involving {state_noun}.",
            "equipment problem",
            (
                "There is an equipment problem with my {state_noun} service.",
                "The hardware used for my {state_noun} service looks damaged.",
                "I think the service equipment for {state_noun} is malfunctioning.",
                "I need someone to inspect an equipment problem related to {state_noun}.",
            ),
        ),
    ),
)

CLINIC = DomainSpec(
    domain_id="AV",
    workflow="w8-clinic-appointment-administration",
    question="Which clinic appointment intent best matches the message?",
    subjects=(
        SubjectSpec("dental", "dental visit", "dental appointment"),
        SubjectSpec("imaging", "imaging scan", "imaging appointment"),
        SubjectSpec("physio", "physiotherapy session", "physiotherapy appointment"),
        SubjectSpec("vaccine", "vaccination visit", "vaccination appointment"),
    ),
    actions=(
        ActionSpec(
            "book",
            "book {subject}",
            "The patient wants to schedule a new {state_noun}.",
            "schedule a new",
            (
                "I would like to schedule a new {state_noun}.",
                "Can you book me a {state_noun}?",
                "I need an available time for a new {state_noun}.",
                "Please help me arrange my first {state_noun}.",
            ),
        ),
        ActionSpec(
            "reschedule",
            "reschedule {subject}",
            "The patient wants to move an existing {state_noun} to a different time.",
            "move my appointment",
            (
                "I need to move my appointment for the {state_noun} to another day.",
                "Can I reschedule my existing {state_noun}?",
                "The current time for my {state_noun} no longer works for me.",
                "Please change the date of my booked {state_noun}.",
            ),
        ),
        ActionSpec(
            "cancel",
            "cancel {subject}",
            "The patient wants to cancel an existing {state_noun}.",
            "cancel my appointment",
            (
                "I need to cancel my appointment for the {state_noun}.",
                "Please remove my booked {state_noun} from the schedule.",
                "I cannot attend the {state_noun} and want to cancel it.",
                "How do I cancel the {state_noun} I already booked?",
            ),
        ),
        ActionSpec(
            "arrival",
            "late arrival for {subject}",
            "The patient is asking about arriving late or missing the check-in time for a {state_noun}.",
            "arriving late",
            (
                "I am arriving late for my {state_noun}; can I still be seen?",
                "I may miss the check-in time for my {state_noun}.",
                "What happens if I am late to the {state_noun}?",
                "Traffic delayed me and I will reach my {state_noun} after the scheduled time.",
            ),
        ),
    ),
)

PARCEL = DomainSpec(
    domain_id="AW",
    workflow="w8-parcel-delivery-customer-service",
    question="Which parcel-delivery intent best matches the message?",
    subjects=(
        SubjectSpec("home", "home delivery", "home delivery"),
        SubjectSpec("locker", "locker pickup", "locker parcel"),
        SubjectSpec("international", "international shipment", "international parcel"),
        SubjectSpec("return", "return shipment", "return parcel"),
    ),
    actions=(
        ActionSpec(
            "delay",
            "delayed {subject}",
            "The customer reports that the {state_noun} is late or has not arrived by the expected time.",
            "parcel is late",
            (
                "My {state_noun} is late and the expected day already passed.",
                "The tracking page says my {state_noun} is delayed.",
                "Why has the {state_noun} still not arrived?",
                "The delivery estimate for my {state_noun} has passed.",
            ),
        ),
        ActionSpec(
            "missing",
            "missing {subject}",
            "The customer cannot locate a {state_noun} that tracking says should be available or delivered.",
            "cannot find the parcel",
            (
                "I cannot find the parcel from my {state_noun} even though tracking says it is there.",
                "The {state_noun} is marked complete but the parcel is missing.",
                "My {state_noun} cannot be located anywhere.",
                "Tracking says the {state_noun} was delivered, but I cannot find it.",
            ),
        ),
        ActionSpec(
            "address",
            "change address for {subject}",
            "The customer wants to correct or change the destination information for a {state_noun}.",
            "change the delivery address",
            (
                "I need to change the delivery address for my {state_noun}.",
                "The destination on my {state_noun} is wrong.",
                "Can I correct the address before the {state_noun} arrives?",
                "Please update where my {state_noun} should be sent.",
            ),
        ),
        ActionSpec(
            "damage",
            "damaged {subject}",
            "The customer reports physical damage to the parcel or contents of a {state_noun}.",
            "parcel arrived damaged",
            (
                "The parcel arrived damaged from my {state_noun}.",
                "The contents of my {state_noun} were broken on arrival.",
                "My {state_noun} package has visible damage.",
                "I opened the {state_noun} and found damaged contents.",
            ),
        ),
    ),
)

WARRANTY = DomainSpec(
    domain_id="AX",
    workflow="w8-consumer-device-warranty-support",
    question="Which device-warranty intent best matches the message?",
    subjects=(
        SubjectSpec("phone", "phone warranty", "phone"),
        SubjectSpec("laptop", "laptop warranty", "laptop"),
        SubjectSpec("audio", "headphone warranty", "headphones"),
        SubjectSpec("watch", "smart-watch warranty", "smart watch"),
    ),
    actions=(
        ActionSpec(
            "repair",
            "{subject} repair request",
            "The customer wants a covered device inspected or repaired under the {state_noun} warranty.",
            "repair my device",
            (
                "I want to repair my device under the warranty for my {state_noun}.",
                "My {state_noun} stopped working and I need warranty repair.",
                "How can I send the {state_noun} in for a covered repair?",
                "I need service to repair a fault with my {state_noun}.",
            ),
        ),
        ActionSpec(
            "replace",
            "{subject} replacement request",
            "The customer wants a replacement device under the {state_noun} warranty.",
            "replacement device",
            (
                "Can I get a replacement device for my broken {state_noun}?",
                "I want the warranty to replace my {state_noun}.",
                "My {state_noun} failed and I am asking for a replacement.",
                "How do I request a replacement under the {state_noun} warranty?",
            ),
        ),
        ActionSpec(
            "coverage",
            "{subject} coverage question",
            "The customer wants to know whether a problem or event is covered by the {state_noun} warranty.",
            "is this covered",
            (
                "Is this covered by the warranty on my {state_noun}?",
                "I need to know whether my {state_noun} warranty covers this problem.",
                "Does the warranty apply to what happened to my {state_noun}?",
                "Can you tell me if this issue qualifies for {state_noun} warranty coverage?",
            ),
        ),
        ActionSpec(
            "status",
            "{subject} claim status",
            "The customer is checking the progress or current status of an existing {state_noun} warranty claim.",
            "claim status",
            (
                "I want an update on my {state_noun} claim status.",
                "What is happening with the warranty claim for my {state_noun}?",
                "Please tell me the progress of my existing {state_noun} claim.",
                "I am checking whether my {state_noun} warranty case has moved forward.",
            ),
        ),
    ),
)

DOMAINS = {d.domain_id: d for d in (UTILITY, CLINIC, PARCEL, WARRANTY)}


def _intent_specs(domain: DomainSpec) -> tuple[IntentSpec, ...]:
    intents: list[IntentSpec] = []
    for subject in domain.subjects:
        for action in domain.actions:
            label = action.label_template.format(subject=subject.label)
            definition = action.definition_template.format(
                subject=subject.label,
                state_noun=subject.state_noun,
            )
            anchor = action.anchor_template.format(
                subject=subject.label,
                state_noun=subject.state_noun,
            )
            structured = (
                f"service: {subject.label}; request: {action.key.replace('_', ' ')}; "
                f"criterion: {definition}"
            )
            lexical = f"{definition} Key phrase: {anchor}."
            states = tuple(
                template.format(
                    subject=subject.label,
                    state_noun=subject.state_noun,
                )
                for template in action.state_templates
            )
            intents.append(
                IntentSpec(
                    intent_id=f"{domain.domain_id.lower()}-{subject.key}-{action.key}",
                    terse_label=label,
                    natural_definition=definition,
                    structured_criterion=structured,
                    lexical_bridge=lexical,
                    state_texts=states,  # type: ignore[arg-type]
                )
            )
    if len(intents) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W8 domain must contain exactly 16 intents")
    if len({x.intent_id for x in intents}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W8 intent IDs must be unique")
    if len({x.terse_label for x in intents}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W8 terse labels must be unique")
    return tuple(intents)


INTENTS = {domain_id: _intent_specs(spec) for domain_id, spec in DOMAINS.items()}


def _view_text(intent: IntentSpec, view_id: str) -> str:
    if view_id == "V0":
        return intent.terse_label
    if view_id == "V1":
        return intent.natural_definition
    if view_id == "V2":
        return intent.structured_criterion
    if view_id == "V3":
        return intent.lexical_bridge
    raise ValueError(f"unknown W8 view: {view_id}")


def _base_candidate_orders(
    domain_id: str,
    intent_index: int,
    state_variant: int,
) -> tuple[list[str], dict[str, int]]:
    intents = INTENTS[domain_id]
    gold = intents[intent_index].intent_id
    negatives = [x.intent_id for x in intents if x.intent_id != gold]
    seed = (
        DOMAIN_SEEDS[domain_id]
        + intent_index * 1009
        + state_variant * 97
    )
    random.Random(seed).shuffle(negatives)
    membership = [gold, *negatives]
    presentation = list(membership)
    random.Random(seed + 17_171).shuffle(presentation)
    rank = {intent_id: i for i, intent_id in enumerate(presentation)}
    return membership, rank


def generate_w8_domain(domain_id: str) -> list[SemanticTransferView]:
    if domain_id not in DOMAINS:
        raise ValueError("W8 domain must be AU, AV, AW or AX")
    domain = DOMAINS[domain_id]
    intents = INTENTS[domain_id]
    by_id = {intent.intent_id: intent for intent in intents}
    rows: list[SemanticTransferView] = []

    for intent_index, intent in enumerate(intents):
        for state_variant, state_text in enumerate(intent.state_texts):
            base_id = (
                f"w8-{domain_id.lower()}-{intent_index:02d}-{state_variant}"
            )
            membership, rank = _base_candidate_orders(
                domain_id,
                intent_index,
                state_variant,
            )
            for k in K_VALUES:
                selected = membership[:k]
                presented = sorted(selected, key=lambda intent_id: rank[intent_id])
                gold_index = presented.index(intent.intent_id)
                for view_id in VIEW_IDS:
                    option_texts = tuple(
                        _view_text(by_id[intent_id], view_id)
                        for intent_id in presented
                    )
                    rows.append(
                        SemanticTransferView(
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

    expected = BASES_PER_DOMAIN * len(K_VALUES) * len(VIEW_IDS)
    if len(rows) != expected:
        raise RuntimeError(
            f"W8 domain expected {expected} views, got {len(rows)}"
        )
    return rows


def generate_w8_diagnostics() -> list[SemanticTransferView]:
    return [
        *generate_w8_domain("AU"),
        *generate_w8_domain("AV"),
        *generate_w8_domain("AW"),
        *generate_w8_domain("AX"),
    ]


def all_w8_text_atoms() -> set[str]:
    atoms: set[str] = set()
    for domain_id, intents in INTENTS.items():
        atoms.add(DOMAINS[domain_id].question)
        for intent in intents:
            atoms.update(
                {
                    intent.intent_id,
                    intent.terse_label,
                    intent.natural_definition,
                    intent.structured_criterion,
                    intent.lexical_bridge,
                    *intent.state_texts,
                }
            )
    return atoms
