from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from .contracts import LogicalOption


TRAIN_DOMAINS = ("AY", "AZ", "BA", "BB")
DEV_DOMAINS = ("BC",)
CONFIRM_DOMAINS = ("BD", "BE")
ALL_DOMAINS = TRAIN_DOMAINS + DEV_DOMAINS + CONFIRM_DOMAINS
DOMAIN_SEEDS = {
    "AY": 281601,
    "AZ": 281607,
    "BA": 281619,
    "BB": 281627,
    "BC": 282731,
    "BD": 283841,
    "BE": 284953,
}
K_VALUES = (4, 8, 16)
VIEW_IDS = ("label", "definition")
INTENTS_PER_DOMAIN = 16
STATE_VARIANTS_PER_INTENT = 4
BASES_PER_DOMAIN = INTENTS_PER_DOMAIN * STATE_VARIANTS_PER_INTENT


@dataclass(frozen=True)
class SubjectSpec:
    key: str
    label: str
    state_noun: str


@dataclass(frozen=True)
class ActionSpec:
    key: str
    label: str
    definition_template: str
    state_templates: tuple[str, str, str, str]


@dataclass(frozen=True)
class IntentSpec:
    intent_id: str
    terse_label: str
    natural_definition: str
    state_texts: tuple[str, str, str, str]


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    workflow: str
    question: str
    subjects: tuple[SubjectSpec, SubjectSpec, SubjectSpec, SubjectSpec]
    actions: tuple[ActionSpec, ActionSpec, ActionSpec, ActionSpec]


@dataclass(frozen=True)
class SemanticAlignmentView:
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


def _travel() -> DomainSpec:
    return DomainSpec(
        domain_id="AY",
        workflow="w9-travel-itinerary-servicing",
        question="Which travel-service intent best matches this request?",
        subjects=(
            SubjectSpec("flight", "flight booking", "flight"),
            SubjectSpec("hotel", "hotel booking", "hotel stay"),
            SubjectSpec("car", "rental car booking", "rental car"),
            SubjectSpec("rail", "rail booking", "train journey"),
        ),
        actions=(
            ActionSpec(
                "modify",
                "change",
                "The traveler wants to change an existing {state_noun} reservation, such as its date, time, route, room, or other booked detail.",
                (
                    "I already booked the {state_noun}, but I need to move it to a different time.",
                    "Can I change a detail on my existing {state_noun} reservation?",
                    "My plans changed and I need to adjust the {state_noun} I already booked.",
                    "I want to edit the date or another detail for my booked {state_noun}.",
                ),
            ),
            ActionSpec(
                "cancel",
                "cancellation",
                "The traveler wants to cancel an existing {state_noun} reservation rather than change or use it.",
                (
                    "I will not be using my booked {state_noun} and want to cancel it.",
                    "Please tell me how to call off the {state_noun} reservation I already made.",
                    "My trip is off, so I need the {state_noun} booking cancelled.",
                    "I need to remove the existing {state_noun} reservation completely.",
                ),
            ),
            ActionSpec(
                "status",
                "booking status",
                "The traveler is checking whether an existing {state_noun} reservation is confirmed, pending, changed, or otherwise active.",
                (
                    "Can you check whether my {state_noun} reservation is actually confirmed?",
                    "I need the current status of the {state_noun} I booked.",
                    "The {state_noun} reservation is in my account, but I do not know if it is finalized.",
                    "What is happening with my existing {state_noun} booking right now?",
                ),
            ),
            ActionSpec(
                "receipt",
                "confirmation document",
                "The traveler needs a confirmation, receipt, voucher, or other booking document for an existing {state_noun} reservation.",
                (
                    "I need the confirmation document for my {state_noun} reservation.",
                    "Where can I get a receipt or voucher for the booked {state_noun}?",
                    "Please send me proof of the {state_noun} reservation.",
                    "I cannot find the document that confirms my {state_noun} booking.",
                ),
            ),
        ),
    )


def _education() -> DomainSpec:
    return DomainSpec(
        domain_id="AZ",
        workflow="w9-education-portal-support",
        question="Which education-portal intent best matches this message?",
        subjects=(
            SubjectSpec("course", "course page", "course page"),
            SubjectSpec("exam", "online exam", "online exam"),
            SubjectSpec("assignment", "assignment submission", "assignment submission"),
            SubjectSpec("account", "student account", "student account"),
        ),
        actions=(
            ActionSpec(
                "access",
                "access problem",
                "The student cannot open, sign in to, or otherwise access the {state_noun} they should be able to use.",
                (
                    "The {state_noun} will not open for me even though I should have access.",
                    "I cannot get into the {state_noun} from the portal.",
                    "The system blocks me when I try to use the {state_noun}.",
                    "I should be able to reach the {state_noun}, but it is unavailable to me.",
                ),
            ),
            ActionSpec(
                "deadline",
                "deadline question",
                "The student wants to know the due date, availability window, or timing requirement for the {state_noun}.",
                (
                    "What is the deadline connected with the {state_noun}?",
                    "I need to know when the {state_noun} closes or is due.",
                    "How much time do I have before the {state_noun} deadline?",
                    "Please tell me the date or time limit for the {state_noun}.",
                ),
            ),
            ActionSpec(
                "change",
                "change request",
                "The student wants an allowed detail of the {state_noun} changed, reset, reopened, or corrected.",
                (
                    "I need a detail of the {state_noun} changed or reset.",
                    "Can the {state_noun} be reopened or corrected for me?",
                    "Something about the {state_noun} needs to be changed after it was set up.",
                    "I am asking for an adjustment to the existing {state_noun}.",
                ),
            ),
            ActionSpec(
                "status",
                "status question",
                "The student is checking the current processing, grading, activation, or completion status of the {state_noun}.",
                (
                    "What is the current status of my {state_noun}?",
                    "I am waiting for an update about the {state_noun}.",
                    "Has the {state_noun} been processed or completed yet?",
                    "I need to know what stage the {state_noun} is at now.",
                ),
            ),
        ),
    )


def _appliance() -> DomainSpec:
    return DomainSpec(
        domain_id="BA",
        workflow="w9-home-appliance-repair",
        question="Which appliance-service intent best matches the report?",
        subjects=(
            SubjectSpec("fridge", "refrigerator", "refrigerator"),
            SubjectSpec("washer", "washing machine", "washing machine"),
            SubjectSpec("oven", "oven", "oven"),
            SubjectSpec("ac", "air conditioner", "air conditioner"),
        ),
        actions=(
            ActionSpec(
                "power",
                "power failure",
                "The {state_noun} does not power on or unexpectedly loses electrical power.",
                (
                    "My {state_noun} will not turn on at all.",
                    "The {state_noun} suddenly has no power.",
                    "I press the controls but the {state_noun} stays completely off.",
                    "The {state_noun} keeps losing power and cannot start normally.",
                ),
            ),
            ActionSpec(
                "performance",
                "performance problem",
                "The {state_noun} runs but does not perform its main job correctly or strongly enough.",
                (
                    "The {state_noun} runs, but it is not doing its job properly.",
                    "Performance from my {state_noun} has become much worse than normal.",
                    "The {state_noun} is operating but the result is not correct.",
                    "Something is wrong with how well the {state_noun} works.",
                ),
            ),
            ActionSpec(
                "leak",
                "leak or discharge",
                "The {state_noun} is leaking, dripping, or releasing liquid where it should not.",
                (
                    "There is liquid leaking from my {state_noun}.",
                    "I found a puddle around the {state_noun}.",
                    "The {state_noun} is dripping during or after use.",
                    "Something is leaking out of the {state_noun}.",
                ),
            ),
            ActionSpec(
                "noise",
                "unusual noise",
                "The {state_noun} makes an abnormal sound, vibration, rattle, or other unusual noise while operating.",
                (
                    "My {state_noun} has started making a strange noise.",
                    "There is a loud vibration coming from the {state_noun}.",
                    "The {state_noun} rattles in a way it never did before.",
                    "I hear an abnormal sound whenever the {state_noun} runs.",
                ),
            ),
        ),
    )


def _insurance() -> DomainSpec:
    return DomainSpec(
        domain_id="BB",
        workflow="w9-insurance-policy-servicing",
        question="Which insurance-service intent best matches this request?",
        subjects=(
            SubjectSpec("auto", "auto policy", "car insurance policy"),
            SubjectSpec("home", "home policy", "home insurance policy"),
            SubjectSpec("travel", "travel policy", "travel insurance policy"),
            SubjectSpec("pet", "pet policy", "pet insurance policy"),
        ),
        actions=(
            ActionSpec(
                "details",
                "update details",
                "The policyholder wants personal, contact, asset, beneficiary, or other recorded details changed on the {state_noun}.",
                (
                    "I need to change information recorded on my {state_noun}.",
                    "Some details on the {state_noun} are out of date and need updating.",
                    "How can I correct the information attached to my {state_noun}?",
                    "I want to update an existing detail on the {state_noun}.",
                ),
            ),
            ActionSpec(
                "payment",
                "payment question",
                "The policyholder has a billing, premium, payment-method, due-date, or payment-status question about the {state_noun}.",
                (
                    "I have a question about paying for my {state_noun}.",
                    "Something is unclear about the premium or payment on the {state_noun}.",
                    "I need help with a payment connected to my {state_noun}.",
                    "Can you explain the billing situation for my {state_noun}?",
                ),
            ),
            ActionSpec(
                "coverage",
                "coverage question",
                "The policyholder wants to know what protection, event, cost, or situation the {state_noun} covers.",
                (
                    "I need to know whether my {state_noun} covers this situation.",
                    "What protection is included in the {state_noun}?",
                    "Can you tell me if this type of event is covered by my {state_noun}?",
                    "I am checking what the {state_noun} actually pays for.",
                ),
            ),
            ActionSpec(
                "claim",
                "claim status",
                "The policyholder is checking the progress or current status of a claim associated with the {state_noun}.",
                (
                    "What is happening with the claim under my {state_noun}?",
                    "I am waiting for an update on a claim for the {state_noun}.",
                    "Can you check the current claim status linked to my {state_noun}?",
                    "I need to know how far my {state_noun} claim has progressed.",
                ),
            ),
        ),
    )


def _tickets() -> DomainSpec:
    return DomainSpec(
        domain_id="BC",
        workflow="w9-event-ticket-support",
        question="Which ticket-support intent best matches this message?",
        subjects=(
            SubjectSpec("concert", "concert ticket", "concert ticket"),
            SubjectSpec("sport", "sports ticket", "sports ticket"),
            SubjectSpec("theater", "theater ticket", "theater ticket"),
            SubjectSpec("festival", "festival ticket", "festival ticket"),
        ),
        actions=(
            ActionSpec(
                "transfer",
                "ticket transfer",
                "The customer wants an existing {state_noun} transferred or reassigned to another person.",
                (
                    "I need to give my {state_noun} to someone else.",
                    "How can I transfer the {state_noun} to another person?",
                    "The {state_noun} is mine now, but another person needs to use it.",
                    "I want to reassign my {state_noun} to a friend.",
                ),
            ),
            ActionSpec(
                "refund",
                "refund request",
                "The customer wants money returned for an existing {state_noun} rather than using or transferring it.",
                (
                    "I cannot attend and want a refund for the {state_noun}.",
                    "How do I get my money back for this {state_noun}?",
                    "I want to return the {state_noun} and receive a refund.",
                    "Please help me request a refund for the {state_noun}.",
                ),
            ),
            ActionSpec(
                "entry",
                "entry code problem",
                "The customer cannot use the barcode, QR code, mobile pass, or other entry credential associated with the {state_noun}.",
                (
                    "The entry code on my {state_noun} is not working.",
                    "My phone cannot show a usable barcode for the {state_noun}.",
                    "The gate will not accept the code attached to my {state_noun}.",
                    "I have the {state_noun}, but its entry credential does not work.",
                ),
            ),
            ActionSpec(
                "seat",
                "seat change",
                "The customer wants to change the seat, section, area, or assigned position for an existing {state_noun}.",
                (
                    "Can I move to a different seat with my {state_noun}?",
                    "I want to change the assigned location for the {state_noun}.",
                    "Is it possible to switch the seat or section on my {state_noun}?",
                    "I need a different assigned place for this {state_noun}.",
                ),
            ),
        ),
    )


def _permits() -> DomainSpec:
    return DomainSpec(
        domain_id="BD",
        workflow="w9-municipal-permit-assistance",
        question="Which permit-service intent best matches this request?",
        subjects=(
            SubjectSpec("renovation", "renovation permit", "renovation permit"),
            SubjectSpec("vendor", "food-vendor permit", "food-vendor permit"),
            SubjectSpec("event", "street-event permit", "street-event permit"),
            SubjectSpec("sign", "business-sign permit", "business-sign permit"),
        ),
        actions=(
            ActionSpec(
                "requirements",
                "requirements question",
                "The applicant wants to know eligibility rules, required documents, prerequisites, or other requirements for the {state_noun}.",
                (
                    "What do I need before I can apply for the {state_noun}?",
                    "Please tell me the documents and requirements for the {state_noun}.",
                    "I am trying to understand who qualifies for the {state_noun}.",
                    "What conditions have to be met for a {state_noun}?",
                ),
            ),
            ActionSpec(
                "submit",
                "new application",
                "The applicant wants to submit a new application for the {state_noun}.",
                (
                    "I am ready to apply for a new {state_noun}.",
                    "How do I submit my first application for the {state_noun}?",
                    "I need to start a new {state_noun} application.",
                    "Where can I file an application for the {state_noun}?",
                ),
            ),
            ActionSpec(
                "amend",
                "application amendment",
                "The applicant wants to correct, update, or amend an existing {state_noun} application after submission.",
                (
                    "I already filed for the {state_noun}, but I need to change something.",
                    "How can I correct information in my existing {state_noun} application?",
                    "A detail in the submitted {state_noun} application needs updating.",
                    "I need to amend the {state_noun} paperwork I already sent.",
                ),
            ),
            ActionSpec(
                "status",
                "application status",
                "The applicant is checking the review, approval, rejection, or processing status of an existing {state_noun} application.",
                (
                    "Can you tell me the current status of my {state_noun} application?",
                    "I am waiting for an update on the {state_noun} I applied for.",
                    "Has my {state_noun} application been reviewed yet?",
                    "What stage is my existing {state_noun} application at now?",
                ),
            ),
        ),
    )


def _payroll() -> DomainSpec:
    return DomainSpec(
        domain_id="BE",
        workflow="w9-small-business-payroll",
        question="Which payroll-service intent best matches this request?",
        subjects=(
            SubjectSpec("employee", "employee pay", "employee payroll"),
            SubjectSpec("tax", "payroll tax filing", "payroll tax filing"),
            SubjectSpec("benefit", "benefit deduction", "benefit deduction"),
            SubjectSpec("contractor", "contractor payment", "contractor payment"),
        ),
        actions=(
            ActionSpec(
                "setup",
                "setup request",
                "The business wants to create or configure a new {state_noun} item that is not yet active.",
                (
                    "I need to set up a new {state_noun} item for the business.",
                    "How do I add the {state_noun} for the first time?",
                    "We have not configured this {state_noun} yet and need to create it.",
                    "Please help me start a new {state_noun} setup.",
                ),
            ),
            ActionSpec(
                "correct",
                "correction request",
                "The business wants to correct an error in an existing or completed {state_noun} record.",
                (
                    "There is a mistake in an existing {state_noun} record that needs fixing.",
                    "I entered something incorrectly for the {state_noun}.",
                    "How can I correct an error that already appears in the {state_noun}?",
                    "The current {state_noun} information is wrong and needs a correction.",
                ),
            ),
            ActionSpec(
                "schedule",
                "schedule change",
                "The business wants to change the date, timing, frequency, or processing schedule for the {state_noun}.",
                (
                    "I need to change when the {state_noun} will be processed.",
                    "Can I move the date for this {state_noun}?",
                    "The current schedule for the {state_noun} no longer works for us.",
                    "I want to adjust the timing or frequency of the {state_noun}.",
                ),
            ),
            ActionSpec(
                "status",
                "processing status",
                "The business is checking whether an existing {state_noun} item has been processed, filed, completed, or is still pending.",
                (
                    "What is the current processing status of the {state_noun}?",
                    "I need to know whether the {state_noun} has finished processing.",
                    "Can you check if the {state_noun} is still pending?",
                    "I am waiting for an update about the existing {state_noun}.",
                ),
            ),
        ),
    )


DOMAIN_SPECS = {
    spec.domain_id: spec
    for spec in (
        _travel(),
        _education(),
        _appliance(),
        _insurance(),
        _tickets(),
        _permits(),
        _payroll(),
    )
}


def _intent_specs(domain: DomainSpec) -> tuple[IntentSpec, ...]:
    intents: list[IntentSpec] = []
    for subject in domain.subjects:
        for action in domain.actions:
            label = f"{subject.label} {action.label}"
            definition = action.definition_template.format(
                state_noun=subject.state_noun,
            )
            states = tuple(
                template.format(state_noun=subject.state_noun)
                for template in action.state_templates
            )
            intents.append(
                IntentSpec(
                    intent_id=(
                        f"{domain.domain_id.lower()}-"
                        f"{subject.key}-{action.key}"
                    ),
                    terse_label=label,
                    natural_definition=definition,
                    state_texts=states,  # type: ignore[arg-type]
                )
            )
    if len(intents) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W9 domain must contain exactly 16 intents")
    if len({row.intent_id for row in intents}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W9 intent IDs must be unique")
    if len({row.terse_label for row in intents}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W9 terse labels must be unique")
    if len({row.natural_definition for row in intents}) != INTENTS_PER_DOMAIN:
        raise RuntimeError("W9 definitions must be unique")
    return tuple(intents)


INTENTS = {
    domain_id: _intent_specs(spec)
    for domain_id, spec in DOMAIN_SPECS.items()
}


def _base_candidate_orders(
    domain_id: str,
    intent_index: int,
    state_variant: int,
) -> tuple[list[str], dict[str, int]]:
    intents = INTENTS[domain_id]
    gold = intents[intent_index].intent_id
    negatives = [row.intent_id for row in intents if row.intent_id != gold]
    seed = (
        DOMAIN_SEEDS[domain_id]
        + intent_index * 1009
        + state_variant * 97
    )
    random.Random(seed).shuffle(negatives)
    membership = [gold, *negatives]
    presentation = list(membership)
    random.Random(seed + 31_337).shuffle(presentation)
    rank = {intent_id: index for index, intent_id in enumerate(presentation)}
    return membership, rank


def generate_w9_domain(
    domain_id: str,
    *,
    allow_confirm: bool = False,
) -> list[SemanticAlignmentView]:
    if domain_id not in ALL_DOMAINS:
        raise ValueError("unknown W9 domain")
    if domain_id in CONFIRM_DOMAINS and not allow_confirm:
        raise PermissionError(
            "W9 CONFIRM domain generation requires allow_confirm=True"
        )
    domain = DOMAIN_SPECS[domain_id]
    intents = INTENTS[domain_id]
    by_id = {row.intent_id: row for row in intents}
    rows: list[SemanticAlignmentView] = []

    for intent_index, intent in enumerate(intents):
        for state_variant, state_text in enumerate(intent.state_texts):
            base_id = (
                f"w9-{domain_id.lower()}-{intent_index:02d}-"
                f"{state_variant}"
            )
            membership, rank = _base_candidate_orders(
                domain_id,
                intent_index,
                state_variant,
            )
            for k in K_VALUES:
                selected = membership[:k]
                presented = sorted(
                    selected,
                    key=lambda intent_id: rank[intent_id],
                )
                gold_index = presented.index(intent.intent_id)
                for view_id in VIEW_IDS:
                    if view_id == "label":
                        option_texts = tuple(
                            by_id[intent_id].terse_label
                            for intent_id in presented
                        )
                    else:
                        option_texts = tuple(
                            by_id[intent_id].natural_definition
                            for intent_id in presented
                        )
                    rows.append(
                        SemanticAlignmentView(
                            case_id=(
                                f"{base_id}-{view_id}-k{k}"
                            ),
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
            f"W9 domain expected {expected} views, got {len(rows)}"
        )
    return rows


def generate_w9_split(
    split: str,
    *,
    allow_confirm: bool = False,
) -> list[SemanticAlignmentView]:
    if split == "train":
        domains = TRAIN_DOMAINS
    elif split == "dev":
        domains = DEV_DOMAINS
    elif split == "confirm":
        if not allow_confirm:
            raise PermissionError(
                "W9 CONFIRM generation requires allow_confirm=True"
            )
        domains = CONFIRM_DOMAINS
    else:
        raise ValueError("W9 split must be train/dev/confirm")

    return [
        row
        for domain_id in domains
        for row in generate_w9_domain(
            domain_id,
            allow_confirm=(split == "confirm" and allow_confirm),
        )
    ]


def alignment_training_views() -> list[SemanticAlignmentView]:
    """Return exactly one definition/K16 row per TRAIN base."""
    rows = generate_w9_split("train")
    selected = [
        row
        for row in rows
        if row.view_id == "definition" and row.diagnosis_k == 16
    ]
    expected = len(TRAIN_DOMAINS) * BASES_PER_DOMAIN
    if len(selected) != expected:
        raise RuntimeError("W9 training view count mismatch")
    return selected


def all_w9_text_atoms(*, include_confirm: bool = False) -> set[str]:
    domains: Iterable[str] = (
        ALL_DOMAINS if include_confirm else TRAIN_DOMAINS + DEV_DOMAINS
    )
    atoms: set[str] = set()
    for domain_id in domains:
        domain = DOMAIN_SPECS[domain_id]
        atoms.add(domain.question)
        for intent in INTENTS[domain_id]:
            atoms.add(intent.intent_id)
            atoms.add(intent.terse_label)
            atoms.add(intent.natural_definition)
            atoms.update(intent.state_texts)
    return atoms
