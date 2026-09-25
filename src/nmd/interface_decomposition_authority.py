from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAIN_SEEDS = {
    "BJ": 301801,
    "BK": 301807,
    "BL": 301819,
    "BM": 301831,
}
K_VALUES = (4, 8, 16)
INTENTS_PER_DOMAIN = 16
STATE_VARIANTS_PER_INTENT = 4
BASES_PER_DOMAIN = INTENTS_PER_DOMAIN * STATE_VARIANTS_PER_INTENT


@dataclass(frozen=True)
class SubjectSpec:
    key: str
    label: str
    noun: str


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
class InterfaceDecompositionView:
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


HOUSING = DomainSpec(
    domain_id="BJ",
    workflow="w11-university-housing-administration",
    question="Which university housing request best matches the resident message?",
    subjects=(
        SubjectSpec("room", "room assignment", "room assignment"),
        SubjectSpec("parking", "residence parking permit", "residence parking permit"),
        SubjectSpec("repair", "housing repair visit", "housing repair visit"),
        SubjectSpec("payment", "housing payment arrangement", "housing payment arrangement"),
    ),
    actions=(
        ActionSpec(
            "request",
            "request {subject}",
            "The resident wants to create or submit a new request for a {noun}.",
            (
                "I need to submit a new request for my {noun}.",
                "How do I start a request about the {noun}?",
                "Please help me arrange a new {noun}.",
                "I want to open a request for the {noun}.",
            ),
        ),
        ActionSpec(
            "change",
            "change {subject}",
            "The resident wants to modify details of an existing {noun}.",
            (
                "I need to change the details of my existing {noun}.",
                "Can I update something on the {noun} I already have?",
                "Please modify my current {noun}.",
                "The existing {noun} needs an update.",
            ),
        ),
        ActionSpec(
            "cancel",
            "cancel {subject}",
            "The resident wants to cancel or withdraw an existing {noun}.",
            (
                "I want to cancel my existing {noun}.",
                "Please withdraw the {noun} I already requested.",
                "I no longer need the current {noun}.",
                "How can I remove my existing {noun} request?",
            ),
        ),
        ActionSpec(
            "status",
            "{subject} status",
            "The resident wants the current status or progress of an existing {noun}.",
            (
                "What is the current status of my {noun}?",
                "I am checking on the progress of the {noun}.",
                "Can you give me an update about my {noun}?",
                "I want to know what is happening with the {noun}.",
            ),
        ),
    ),
)

AGRI = DomainSpec(
    domain_id="BK",
    workflow="w11-agricultural-equipment-leasing",
    question="Which agricultural equipment leasing intent best matches the customer message?",
    subjects=(
        SubjectSpec("tractor", "tractor lease", "tractor lease"),
        SubjectSpec("harvester", "harvester lease", "harvester lease"),
        SubjectSpec("pump", "irrigation pump lease", "irrigation pump lease"),
        SubjectSpec("drone", "field drone lease", "field drone lease"),
    ),
    actions=(
        ActionSpec(
            "quote",
            "{subject} quote",
            "The customer wants a price estimate for the {noun}.",
            (
                "What would the {noun} cost for my job?",
                "Can you give me a price estimate for the {noun}?",
                "I need a quote before I take the {noun}.",
                "Please tell me the expected lease price for the {noun}.",
            ),
        ),
        ActionSpec(
            "reserve",
            "reserve {subject}",
            "The customer wants to reserve the {noun} for a future time.",
            (
                "I want to reserve the {noun} for later.",
                "Can you hold the {noun} for my planned dates?",
                "Please book the {noun} for my upcoming work.",
                "I need to secure the {noun} before the job begins.",
            ),
        ),
        ActionSpec(
            "modify",
            "modify {subject}",
            "The customer wants to change an existing reservation for the {noun}.",
            (
                "I need to change my current {noun} reservation.",
                "Can I modify the dates on the {noun} booking?",
                "Please update my existing booking for the {noun}.",
                "Something on my {noun} reservation needs changing.",
            ),
        ),
        ActionSpec(
            "return",
            "return {subject}",
            "The customer is asking about returning or ending use of the rented {noun}.",
            (
                "How do I return the {noun} when I am finished?",
                "I need instructions for bringing back the {noun}.",
                "Where should I return my rented {noun}?",
                "I am done with the {noun} and need the return process.",
            ),
        ),
    ),
)

CERT = DomainSpec(
    domain_id="BL",
    workflow="w11-professional-certification-administration",
    question="Which professional certification request best matches the applicant message?",
    subjects=(
        SubjectSpec("exam", "certification exam appointment", "certification exam appointment"),
        SubjectSpec("renewal", "credential renewal", "credential renewal"),
        SubjectSpec("report", "score report", "score report"),
        SubjectSpec("accommodation", "testing accommodation request", "testing accommodation request"),
    ),
    actions=(
        ActionSpec(
            "submit",
            "submit {subject}",
            "The applicant wants to submit or begin a new {noun}.",
            (
                "I need to submit a new {noun}.",
                "How can I start the {noun} process?",
                "Please help me create a new {noun}.",
                "I am ready to begin my {noun}.",
            ),
        ),
        ActionSpec(
            "update",
            "update {subject}",
            "The applicant wants to change information on an existing {noun}.",
            (
                "I need to update my existing {noun}.",
                "Can I change the information on the {noun}?",
                "Please correct details on my current {noun}.",
                "Some information in the {noun} needs changing.",
            ),
        ),
        ActionSpec(
            "withdraw",
            "withdraw {subject}",
            "The applicant wants to cancel or withdraw an existing {noun}.",
            (
                "I want to withdraw my existing {noun}.",
                "Please cancel the {noun} I already submitted.",
                "I no longer want to continue with this {noun}.",
                "How can I remove my current {noun}?",
            ),
        ),
        ActionSpec(
            "progress",
            "{subject} progress",
            "The applicant wants the current progress or availability status of the {noun}.",
            (
                "What is happening with my {noun} right now?",
                "I am checking the current progress of the {noun}.",
                "Can you tell me whether my {noun} is ready?",
                "I need an update about the {noun}.",
            ),
        ),
    ),
)

BAGGAGE = DomainSpec(
    domain_id="BM",
    workflow="w11-airline-baggage-support",
    question="Which airline baggage request best matches the passenger message?",
    subjects=(
        SubjectSpec("checked", "checked bag", "checked bag"),
        SubjectSpec("sports", "sports equipment bag", "sports equipment bag"),
        SubjectSpec("stroller", "checked stroller", "checked stroller"),
        SubjectSpec("instrument", "musical instrument case", "musical instrument case"),
    ),
    actions=(
        ActionSpec(
            "add",
            "add {subject}",
            "The passenger wants to add the {noun} to an existing trip.",
            (
                "I need to add a {noun} to my trip.",
                "Can I include the {noun} on my booking?",
                "Please help me add the {noun} before I fly.",
                "I want the {noun} attached to my reservation.",
            ),
        ),
        ActionSpec(
            "fee",
            "{subject} fee",
            "The passenger is asking about the fee or charge for the {noun}.",
            (
                "How much will I be charged for the {noun}?",
                "I need to know the fee for taking the {noun}.",
                "What does the airline charge for this {noun}?",
                "Can you explain the cost for my {noun}?",
            ),
        ),
        ActionSpec(
            "locate",
            "locate {subject}",
            "The passenger wants help locating or tracking the {noun}.",
            (
                "I cannot find my {noun} after the trip.",
                "Can you help me locate the {noun}?",
                "I need a tracking update for my {noun}.",
                "Where is the {noun} now?",
            ),
        ),
        ActionSpec(
            "damage",
            "damaged {subject}",
            "The passenger reports that the {noun} was damaged during handling.",
            (
                "My {noun} was damaged during the journey.",
                "I found damage on the {noun} after arrival.",
                "The airline handling caused a problem with my {noun}.",
                "I need to report damage to the {noun}.",
            ),
        ),
    ),
)

DOMAINS = {d.domain_id: d for d in (HOUSING, AGRI, CERT, BAGGAGE)}


def _intent_specs(domain: DomainSpec) -> tuple[IntentSpec, ...]:
    rows: list[IntentSpec] = []
    for subject in domain.subjects:
        for action in domain.actions:
            rows.append(
                IntentSpec(
                    intent_id=f"{domain.domain_id.lower()}-{subject.key}-{action.key}",
                    terse_label=action.label_template.format(subject=subject.label),
                    natural_definition=action.definition_template.format(
                        subject=subject.label,
                        noun=subject.noun,
                    ),
                    state_texts=tuple(
                        template.format(subject=subject.label, noun=subject.noun)
                        for template in action.state_templates
                    ),  # type: ignore[arg-type]
                )
            )
    if len(rows) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W11 domain must have exactly 16 intents")
    if len({x.intent_id for x in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W11 intent IDs must be unique")
    if len({x.terse_label for x in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W11 labels must be unique")
    if len({x.natural_definition for x in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W11 definitions must be unique")
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
    random.Random(seed + 20_021).shuffle(presentation)
    rank = {intent_id: i for i, intent_id in enumerate(presentation)}
    return membership, rank


def generate_w11_domain(domain_id: str) -> list[InterfaceDecompositionView]:
    if domain_id not in DOMAINS:
        raise ValueError("W11 domain must be BJ, BK, BL or BM")
    domain = DOMAINS[domain_id]
    intents = INTENTS[domain_id]
    by_id = {x.intent_id: x for x in intents}
    rows: list[InterfaceDecompositionView] = []

    for intent_index, intent in enumerate(intents):
        for state_variant, state_text in enumerate(intent.state_texts):
            base_id = f"w11-{domain_id.lower()}-{intent_index:02d}-{state_variant}"
            membership, rank = _base_order(domain_id, intent_index, state_variant)
            for k in K_VALUES:
                selected = membership[:k]
                presented = sorted(selected, key=lambda x: rank[x])
                gold_index = presented.index(intent.intent_id)
                for view_id in ("definition", "label"):
                    option_texts = tuple(
                        (
                            by_id[oid].natural_definition
                            if view_id == "definition"
                            else by_id[oid].terse_label
                        )
                        for oid in presented
                    )
                    rows.append(
                        InterfaceDecompositionView(
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

    expected = BASES_PER_DOMAIN * len(K_VALUES) * 2
    if len(rows) != expected:
        raise RuntimeError(f"W11 expected {expected} views, got {len(rows)}")
    return rows


def generate_w11_diagnostics() -> list[InterfaceDecompositionView]:
    return [
        *generate_w11_domain("BJ"),
        *generate_w11_domain("BK"),
        *generate_w11_domain("BL"),
        *generate_w11_domain("BM"),
    ]


def all_w11_text_atoms() -> set[str]:
    atoms: set[str] = set()
    for domain_id, intents in INTENTS.items():
        atoms.add(DOMAINS[domain_id].question)
        for intent in intents:
            atoms.update(
                {
                    intent.intent_id,
                    intent.terse_label,
                    intent.natural_definition,
                    *intent.state_texts,
                }
            )
    return atoms
