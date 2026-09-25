from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import LogicalOption

DOMAIN_SEEDS = {
    "BF": 291701,
    "BG": 291707,
    "BH": 291719,
    "BI": 291731,
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
class RepresentationCeilingView:
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


ENERGY = DomainSpec(
    domain_id="BF",
    workflow="w10-residential-energy-account-support",
    question="Which residential energy support intent best matches the message?",
    subjects=(
        SubjectSpec("meter", "meter service", "meter"),
        SubjectSpec("solar", "solar credit", "solar credit"),
        SubjectSpec("autopay", "automatic payment", "automatic payment"),
        SubjectSpec("tariff", "rate plan", "rate plan"),
    ),
    actions=(
        ActionSpec(
            "activate",
            "activate {subject}",
            "The customer wants to enable or begin {noun}.",
            (
                "I need to get my {noun} started on the account.",
                "Can you help me begin using the {noun}?",
                "I want the {noun} enabled from now on.",
                "Please tell me how to start the {noun}.",
            ),
        ),
        ActionSpec(
            "correct",
            "correct {subject}",
            "The customer reports incorrect information or an error involving {noun}.",
            (
                "Something is wrong with the information for my {noun}.",
                "The account shows an error related to the {noun}.",
                "I need a mistake involving the {noun} corrected.",
                "The details recorded for my {noun} are not right.",
            ),
        ),
        ActionSpec(
            "pause",
            "pause {subject}",
            "The customer wants to temporarily stop or suspend {noun}.",
            (
                "I want to pause the {noun} for a while.",
                "Can the {noun} be suspended temporarily?",
                "Please stop the {noun} until I ask to resume it.",
                "I need a temporary hold on the {noun}.",
            ),
        ),
        ActionSpec(
            "status",
            "{subject} status",
            "The customer is asking for the current status or progress of {noun}.",
            (
                "What is the current status of my {noun}?",
                "I am checking whether the {noun} has been processed.",
                "Can you give me an update about the {noun}?",
                "I want to know what is happening with my {noun}.",
            ),
        ),
    ),
)

LAB = DomainSpec(
    domain_id="BG",
    workflow="w10-outpatient-laboratory-administration",
    question="Which outpatient laboratory intent best matches the message?",
    subjects=(
        SubjectSpec("blood", "blood test", "blood test"),
        SubjectSpec("urine", "urine test", "urine test"),
        SubjectSpec("genetic", "genetic panel", "genetic panel"),
        SubjectSpec("pathology", "pathology sample", "pathology sample"),
    ),
    actions=(
        ActionSpec(
            "schedule",
            "schedule {subject}",
            "The patient wants to arrange a new appointment for a {noun}.",
            (
                "I need a time to come in for my {noun}.",
                "Can I arrange an appointment for the {noun}?",
                "Please help me book the {noun}.",
                "I want to schedule when the {noun} will be done.",
            ),
        ),
        ActionSpec(
            "preparation",
            "{subject} preparation",
            "The patient is asking how to prepare before the {noun}.",
            (
                "What do I need to do before my {noun}?",
                "Are there any preparation instructions for the {noun}?",
                "How should I get ready for the {noun}?",
                "I need the pre-visit instructions for the {noun}.",
            ),
        ),
        ActionSpec(
            "result",
            "{subject} result",
            "The patient is asking whether results from the {noun} are available or how to obtain them.",
            (
                "Are the results from my {noun} ready yet?",
                "How can I see the result of the {noun}?",
                "I am waiting for the report from my {noun}.",
                "Can you tell me when the {noun} result will appear?",
            ),
        ),
        ActionSpec(
            "billing",
            "{subject} billing",
            "The patient is questioning a charge or billing issue related to the {noun}.",
            (
                "I have a billing question about my {noun}.",
                "There is a charge for the {noun} that I do not understand.",
                "Why was I billed this amount for the {noun}?",
                "I need help with the invoice for my {noun}.",
            ),
        ),
    ),
)

FREIGHT = DomainSpec(
    domain_id="BH",
    workflow="w10-freight-booking-support",
    question="Which freight-booking intent best matches the message?",
    subjects=(
        SubjectSpec("pallet", "pallet shipment", "pallet shipment"),
        SubjectSpec("container", "container booking", "container booking"),
        SubjectSpec("refrigerated", "refrigerated load", "refrigerated load"),
        SubjectSpec("oversize", "oversize cargo", "oversize cargo"),
    ),
    actions=(
        ActionSpec(
            "quote",
            "{subject} quote",
            "The customer wants a price estimate for the {noun}.",
            (
                "I need a price estimate for my {noun}.",
                "How much would it cost to move this {noun}?",
                "Can you quote the {noun} before I book it?",
                "Please give me the expected charge for the {noun}.",
            ),
        ),
        ActionSpec(
            "change",
            "change {subject} booking",
            "The customer wants to modify an existing booking for the {noun}.",
            (
                "I need to change my existing {noun} booking.",
                "Can I modify the details of the {noun} I already booked?",
                "Please update the reservation for my {noun}.",
                "Some information on my {noun} booking needs to be changed.",
            ),
        ),
        ActionSpec(
            "documents",
            "{subject} documents",
            "The customer needs paperwork or documentation requirements for the {noun}.",
            (
                "Which documents do I need for the {noun}?",
                "I need help with the paperwork for my {noun}.",
                "What documentation is required before the {noun} can move?",
                "Please tell me what forms are needed for this {noun}.",
            ),
        ),
        ActionSpec(
            "tracking",
            "track {subject}",
            "The customer wants the current movement or location status of the {noun}.",
            (
                "Where is my {noun} right now?",
                "I want to check the current progress of the {noun}.",
                "Can you tell me the latest location of my {noun}?",
                "I am looking for a movement update on the {noun}.",
            ),
        ),
    ),
)

DEVICE = DomainSpec(
    domain_id="BI",
    workflow="w10-connected-device-subscription-support",
    question="Which connected-device subscription intent best matches the message?",
    subjects=(
        SubjectSpec("camera", "camera cloud plan", "camera cloud plan"),
        SubjectSpec("tracker", "tracker connectivity plan", "tracker connectivity plan"),
        SubjectSpec("hub", "home hub plan", "home hub plan"),
        SubjectSpec("sensor", "sensor monitoring plan", "sensor monitoring plan"),
    ),
    actions=(
        ActionSpec(
            "upgrade",
            "upgrade {subject}",
            "The customer wants to move the {noun} to a higher subscription tier.",
            (
                "I want more features on my {noun}.",
                "Can I move the {noun} to a higher plan?",
                "Please help me upgrade the subscription for the {noun}.",
                "I need a better tier for my {noun}.",
            ),
        ),
        ActionSpec(
            "cancel",
            "cancel {subject}",
            "The customer wants to end the subscription for the {noun}.",
            (
                "I want to end the subscription for my {noun}.",
                "Please cancel the plan attached to the {noun}.",
                "I no longer need the subscription on my {noun}.",
                "How can I stop paying for the {noun} plan?",
            ),
        ),
        ActionSpec(
            "renewal",
            "{subject} renewal",
            "The customer is asking about renewal timing or renewal status for the {noun}.",
            (
                "When will my {noun} renew?",
                "I am checking the renewal date for the {noun}.",
                "Has the subscription for my {noun} renewed already?",
                "Please tell me the renewal status of the {noun}.",
            ),
        ),
        ActionSpec(
            "access",
            "{subject} access issue",
            "The customer cannot access subscription features associated with the {noun}.",
            (
                "The paid features on my {noun} are not available.",
                "I cannot use the subscription functions for the {noun}.",
                "My {noun} says the plan is active but I cannot access its features.",
                "I need help restoring access to the paid features on my {noun}.",
            ),
        ),
    ),
)

DOMAINS = {d.domain_id: d for d in (ENERGY, LAB, FREIGHT, DEVICE)}


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
        raise RuntimeError("W10 domain must contain exactly 16 intents")
    if len({x.intent_id for x in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W10 intent IDs must be unique")
    if len({x.terse_label for x in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W10 labels must be unique")
    if len({x.natural_definition for x in rows}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W10 definitions must be unique")
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
    seed = DOMAIN_SEEDS[domain_id] + intent_index * 1009 + state_variant * 97
    random.Random(seed).shuffle(negatives)
    membership = [gold, *negatives]
    presentation = list(membership)
    random.Random(seed + 19_919).shuffle(presentation)
    rank = {intent_id: i for i, intent_id in enumerate(presentation)}
    return membership, rank


def generate_w10_domain(domain_id: str) -> list[RepresentationCeilingView]:
    if domain_id not in DOMAINS:
        raise ValueError("W10 domain must be BF, BG, BH or BI")
    domain = DOMAINS[domain_id]
    intents = INTENTS[domain_id]
    by_id = {row.intent_id: row for row in intents}
    rows: list[RepresentationCeilingView] = []

    for intent_index, intent in enumerate(intents):
        for state_variant, state_text in enumerate(intent.state_texts):
            base_id = f"w10-{domain_id.lower()}-{intent_index:02d}-{state_variant}"
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
                        RepresentationCeilingView(
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
        raise RuntimeError(f"W10 domain expected {expected} views, got {len(rows)}")
    return rows


def generate_w10_diagnostics() -> list[RepresentationCeilingView]:
    return [
        *generate_w10_domain("BF"),
        *generate_w10_domain("BG"),
        *generate_w10_domain("BH"),
        *generate_w10_domain("BI"),
    ]


def all_w10_text_atoms() -> set[str]:
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
