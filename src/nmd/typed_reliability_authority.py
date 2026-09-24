from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .contracts import LogicalOption
from .typed_decisions import TypedDecision, TypedDecisionCase


TRAIN_K_COUNTS = {8: 96, 16: 96, 32: 96, 64: 96}
DEV_K_COUNTS = {8: 48, 16: 48, 32: 48, 64: 48}
CONFIRM_K_COUNTS = {8: 48, 16: 48, 32: 48, 64: 48}

TRAIN_SEED = 171137
DEV_SEED = 172239
CONFIRM_SEED = 173341
CALIBRATION_SEED = 1009

TRAIN_TEMPLATES = (
    "w6c-train-flight-readout",
    "w6c-train-systems-digest",
    "w6c-train-service-matrix",
    "w6c-train-telemetry-note",
)
DEV_TEMPLATES = (
    "w6c-dev-checkout-record",
    "w6c-dev-status-ledger",
)
CONFIRM_TEMPLATES = (
    "w6c-confirm-audit-snapshot",
    "w6c-confirm-validation-brief",
    "w6c-confirm-assurance-readout",
)

TRAIN_COMPONENTS = (
    "azimuth gyroscope", "cryogenic pump", "vector thruster",
    "inertial coupler", "plasma igniter", "guidance resolver",
    "reaction wheel", "fuel metering servo", "star tracker",
    "thermal shutter", "attitude actuator", "bus isolator",
    "optical relay", "power sequencer", "navigation combiner",
    "antenna gimbal", "coolant circulator", "battery contactor",
    "telemetry encoder", "pressure controller", "signal multiplexer",
    "drive amplifier", "valve synchronizer", "sensor concentrator",
)
TRAIN_ZONES = (
    "forward avionics rack", "aft propulsion shelf",
    "port guidance bay", "starboard power cabinet",
    "upper payload tunnel", "lower thermal deck",
    "central service spine", "navigation equipment cage",
    "telemetry processing rack", "propellant control niche",
    "attitude systems bench", "battery distribution panel",
    "sensor interface locker", "flight control enclosure",
    "communications equipment tray", "thermal regulation alcove",
    "auxiliary systems shelf", "power conditioning bay",
    "guidance electronics drawer", "propulsion service panel",
    "payload support frame", "diagnostic access rack",
    "control electronics vault", "instrument routing bay",
)
TRAIN_ANOMALIES = (
    "phase discontinuity", "coolant cavitation", "resolver drift",
    "command echo", "thruster misfire", "voltage droop",
    "encoder desynchronization", "bearing precession",
    "relay chatter", "current asymmetry", "signal aliasing",
    "pressure overshoot", "timing discontinuity", "thermal lag",
    "gain compression", "actuator deadband", "sensor saturation",
    "bus contention", "frequency offset", "valve rebound",
    "telemetry dropout", "torque bias", "optical jitter",
    "feedback delay",
)
TRAIN_CHANNELS = (
    "atlas bus", "boreal bus", "cirrus bus", "deneb bus",
    "equinox bus", "fresco bus", "gemini bus", "helios bus",
    "icarus bus", "janus bus", "lyra bus", "mercury bus",
    "nadir bus", "orion bus", "pegasus bus", "quasar bus",
    "rigel bus", "solstice bus", "titan bus", "umbriel bus",
    "vega bus", "waypoint bus", "xenon bus", "zenith bus",
)

CONFIRM_COMPONENTS = (
    "momentum damper", "propellant heater", "sun sensor",
    "command decoder", "power latch", "thermal mixer",
    "rate gyro", "bus transceiver", "pressure accumulator",
    "optical bench drive", "antenna coupler", "battery equalizer",
    "valve driver", "guidance latch", "signal repeater",
    "coolant diverter",
)
CONFIRM_ZONES = (
    "orbital systems cabinet", "payload interface shelf",
    "guidance checkout bay", "power relay compartment",
    "thermal service tunnel", "communications support rack",
    "propulsion electronics niche", "sensor calibration drawer",
    "navigation support panel", "battery maintenance alcove",
    "telemetry verification frame", "attitude control shelf",
    "flight systems enclosure", "instrument access bay",
    "power conversion locker", "service diagnostics bench",
)
CONFIRM_ANOMALIES = (
    "rate excursion", "heater cycling", "decoder mismatch",
    "latch bounce", "mixer imbalance", "gyro bias",
    "transceiver retry", "accumulator creep",
    "drive oscillation", "coupler detuning", "cell imbalance",
    "driver saturation", "guidance dropout", "repeater clipping",
    "diverter sticking", "sensor intermittency",
)
CONFIRM_CHANNELS = (
    "apogee link", "beacon link", "corona link", "draco link",
    "eclipse link", "flare link", "galaxy link", "horizon link",
    "ionosphere link", "keystone link", "libration link",
    "meteor link", "nebula link", "pulsar link", "radian link",
    "umbra link",
)

SEVERITY_LABELS = ("nominal", "elevated", "serious", "critical")
CONFIDENCE_LEVELS = ("verified", "provisional", "uncertain")
CONFIDENCE_MASS = {
    "verified": 0.90,
    "provisional": 0.75,
    "uncertain": 0.60,
}

RESPONSE_OPTIONS = (
    LogicalOption(
        "observe",
        "Maintain observation for a nominal condition.",
    ),
    LogicalOption(
        "plan-service",
        "Plan service for an elevated condition.",
    ),
    LogicalOption(
        "contain",
        "Contain the affected subsystem for a serious condition.",
    ),
    LogicalOption(
        "safe-mode",
        "Enter safe mode for a critical condition.",
    ),
)
REVIEW_OPTIONS = (
    LogicalOption(
        "clear",
        "Independent review is unnecessary when evidence is sufficiently reliable and the condition is not critical.",
        value=0.0,
    ),
    LogicalOption(
        "review",
        "Independent review is required when evidence is uncertain or the condition is critical.",
        value=1.0,
    ),
)
RISK_OPTIONS = tuple(
    LogicalOption(
        f"hazard-{index}",
        text,
        value=float(index),
    )
    for index, text in enumerate((
        "Hazard band zero matches nominal severity.",
        "Hazard band one matches elevated severity.",
        "Hazard band two matches serious severity.",
        "Hazard band three matches critical severity.",
    ))
)
URGENCY_OPTIONS = tuple(
    LogicalOption(
        f"priority-{index}",
        text,
        value=float(index),
    )
    for index, text in enumerate((
        "Priority zero allows routine observation.",
        "Priority one requests scheduled attention.",
        "Priority two requests rapid intervention.",
        "Priority three requests immediate intervention.",
    ))
)


@dataclass(frozen=True)
class ReliabilityAuthorityCase:
    typed: TypedDecisionCase
    split: str
    template_id: str
    diagnosis_k: int
    severity: int
    confidence: str


def all_w6c_values() -> set[str]:
    groups = (
        TRAIN_COMPONENTS,
        TRAIN_ZONES,
        TRAIN_ANOMALIES,
        TRAIN_CHANNELS,
        CONFIRM_COMPONENTS,
        CONFIRM_ZONES,
        CONFIRM_ANOMALIES,
        CONFIRM_CHANNELS,
    )
    return {value for group in groups for value in group}


def _pools(split: str):
    if split in {"train", "dev"}:
        return (
            TRAIN_COMPONENTS,
            TRAIN_ZONES,
            TRAIN_ANOMALIES,
            TRAIN_CHANNELS,
        )
    if split == "confirm":
        return (
            CONFIRM_COMPONENTS,
            CONFIRM_ZONES,
            CONFIRM_ANOMALIES,
            CONFIRM_CHANNELS,
        )
    raise ValueError(f"unknown W6c split: {split}")


def _signature_text(signature: tuple[str, str, str, str]) -> str:
    component, zone, anomaly, channel = signature
    return (
        f"component {component}; compartment {zone}; fault {anomaly}; "
        f"telemetry {channel}"
    )


def _render_state(
    signature: tuple[str, str, str, str],
    *,
    severity: int,
    confidence: str,
    template_id: str,
) -> str:
    component, zone, anomaly, channel = signature
    severity_label = SEVERITY_LABELS[severity]
    if template_id == "w6c-train-flight-readout":
        return (
            f"Flight readout identifies {component} inside {zone}. "
            f"It reports {anomaly} through {channel}. "
            f"Severity is {severity_label}; evidence is {confidence}."
        )
    if template_id == "w6c-train-systems-digest":
        return (
            f"Systems digest maps compartment={zone}, component={component}, "
            f"telemetry={channel}, fault={anomaly}. "
            f"Condition level is {severity_label} with {confidence} support."
        )
    if template_id == "w6c-train-service-matrix":
        return (
            f"Service matrix associates {component} / {anomaly} / {zone} / {channel}. "
            f"The reported condition is {severity_label}; evidence grade is {confidence}."
        )
    if template_id == "w6c-train-telemetry-note":
        return (
            f"Telemetry note links {channel} to {component} in {zone}, where {anomaly} is observed. "
            f"Severity reads {severity_label} and evidence reads {confidence}."
        )
    if template_id == "w6c-dev-checkout-record":
        return (
            f"Checkout record lists {zone}, {component}, {anomaly}, and {channel}. "
            f"The condition is classified {severity_label}; support is {confidence}."
        )
    if template_id == "w6c-dev-status-ledger":
        return (
            f"Status ledger records fault {anomaly} for {component} at {zone} on {channel}. "
            f"Severity classification: {severity_label}. Evidence quality: {confidence}."
        )
    if template_id == "w6c-confirm-audit-snapshot":
        return (
            f"Audit snapshot reports component {component}, compartment {zone}, "
            f"fault {anomaly}, telemetry {channel}. "
            f"Severity is {severity_label}; evidence status is {confidence}."
        )
    if template_id == "w6c-confirm-validation-brief":
        return (
            f"Validation brief ties {channel} to {component} within {zone}, with {anomaly} present. "
            f"Condition level is {severity_label}, backed by {confidence} evidence."
        )
    if template_id == "w6c-confirm-assurance-readout":
        return (
            f"Assurance readout shows {anomaly} at {component} in {zone}, observed on {channel}. "
            f"Severity: {severity_label}. Reliability: {confidence}."
        )
    raise ValueError(f"unknown W6c template: {template_id}")


def _random_signature(
    pools,
    rng: random.Random,
) -> tuple[str, str, str, str]:
    return tuple(rng.choice(pool) for pool in pools)  # type: ignore[return-value]


def _unique_signature(
    pools,
    rng: random.Random,
    used: set[tuple[str, str, str, str]],
) -> tuple[str, str, str, str]:
    while True:
        signature = _random_signature(pools, rng)
        if signature not in used:
            used.add(signature)
            return signature


def _neighbors(
    target: tuple[str, str, str, str],
    pools,
    changes: int,
) -> list[tuple[str, str, str, str]]:
    if changes not in {1, 2}:
        raise ValueError("changes must be one or two")
    rows: list[tuple[str, str, str, str]] = []
    if changes == 1:
        for index in range(4):
            for value in pools[index]:
                if value == target[index]:
                    continue
                row = list(target)
                row[index] = value
                rows.append(tuple(row))
        return rows
    for first in range(4):
        for second in range(first + 1, 4):
            for left in pools[first]:
                if left == target[first]:
                    continue
                for right in pools[second]:
                    if right == target[second]:
                        continue
                    row = list(target)
                    row[first] = left
                    row[second] = right
                    rows.append(tuple(row))
    return rows


def _diagnosis_options(
    *,
    target: tuple[str, str, str, str],
    pools,
    k: int,
    rng: random.Random,
) -> tuple[tuple[LogicalOption, ...], int]:
    used = {target}
    distractors: list[tuple[str, str, str, str]] = []
    one = _neighbors(target, pools, 1)
    two = _neighbors(target, pools, 2)
    rng.shuffle(one)
    rng.shuffle(two)

    one_target = int(math.floor(0.50 * (k - 1)))
    two_target = int(math.floor(0.30 * (k - 1)))
    for row in one[:one_target]:
        if row not in used:
            used.add(row)
            distractors.append(row)
    desired = min(k - 1, one_target + two_target)
    for row in two:
        if len(distractors) >= desired:
            break
        if row not in used:
            used.add(row)
            distractors.append(row)
    while len(distractors) < k - 1:
        row = _random_signature(pools, rng)
        if row in used:
            continue
        used.add(row)
        distractors.append(row)

    signatures = [target, *distractors[: k - 1]]
    rng.shuffle(signatures)
    gold = signatures.index(target)
    options = tuple(
        LogicalOption(
            option_id=f"signature-{index:03d}",
            criterion_text=_signature_text(signature),
        )
        for index, signature in enumerate(signatures)
    )
    return options, gold


def _categorical_distribution(
    count: int,
    gold_index: int,
    gold_mass: float,
) -> tuple[float, ...]:
    remainder = (1.0 - gold_mass) / (count - 1)
    return tuple(
        gold_mass if index == gold_index else remainder
        for index in range(count)
    )


def _ordinal_distribution(
    count: int,
    gold_index: int,
    gold_mass: float,
) -> tuple[float, ...]:
    result = [0.0] * count
    result[gold_index] = gold_mass
    remainder = 1.0 - gold_mass
    neighbors = [
        index
        for index in (gold_index - 1, gold_index + 1)
        if 0 <= index < count
    ]
    share = remainder / len(neighbors)
    for index in neighbors:
        result[index] = share
    return tuple(result)


def _expected_score(probabilities: tuple[float, ...]) -> float:
    return sum(
        index * probability
        for index, probability in enumerate(probabilities)
    )


def generate_w6c_case(
    *,
    split: str,
    diagnosis_k: int,
    case_index: int,
    severity: int,
    confidence: str,
    rng: random.Random,
    template_id: str,
    used_targets: set[tuple[str, str, str, str]],
) -> ReliabilityAuthorityCase:
    pools = _pools(split)
    target = _unique_signature(pools, rng, used_targets)
    gold_mass = CONFIDENCE_MASS[confidence]

    diagnosis_options, diagnosis_gold = _diagnosis_options(
        target=target,
        pools=pools,
        k=diagnosis_k,
        rng=rng,
    )
    response_gold = severity
    review_gold = int(confidence == "uncertain" or severity == 3)
    risk_gold = severity
    urgency_gold = min(
        3,
        severity + int(confidence == "uncertain"),
    )

    diagnosis_prob = _categorical_distribution(
        diagnosis_k,
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
    risk_prob = _ordinal_distribution(4, risk_gold, gold_mass)
    urgency_prob = _ordinal_distribution(
        4,
        urgency_gold,
        gold_mass,
    )

    state = _render_state(
        target,
        severity=severity,
        confidence=confidence,
        template_id=template_id,
    )
    decisions = (
        TypedDecision(
            question_id="diagnosis",
            primitive="choice",
            question_text=(
                "Which component signature reproduces every reported field?"
            ),
            options=diagnosis_options,
            gold_index=diagnosis_gold,
            gold_probabilities=diagnosis_prob,
        ),
        TypedDecision(
            question_id="response",
            primitive="choice",
            question_text=(
                "Which handling action corresponds to the stated severity?"
            ),
            options=RESPONSE_OPTIONS,
            gold_index=response_gold,
            gold_probabilities=response_prob,
        ),
        TypedDecision(
            question_id="needs_review",
            primitive="noul",
            question_text=(
                "Is independent review required because evidence is uncertain or the condition is critical?"
            ),
            options=REVIEW_OPTIONS,
            gold_index=review_gold,
            gold_probabilities=review_prob,
        ),
        TypedDecision(
            question_id="risk",
            primitive="score",
            question_text=(
                "Which hazard band corresponds to the stated severity?"
            ),
            options=RISK_OPTIONS,
            gold_index=risk_gold,
            gold_probabilities=risk_prob,
            gold_score=_expected_score(risk_prob),
        ),
        TypedDecision(
            question_id="urgency",
            primitive="score",
            question_text=(
                "What intervention priority follows from severity and evidence reliability?"
            ),
            options=URGENCY_OPTIONS,
            gold_index=urgency_gold,
            gold_probabilities=urgency_prob,
            gold_score=_expected_score(urgency_prob),
        ),
    )
    typed = TypedDecisionCase(
        case_id=f"w6c-{split}-{diagnosis_k}-{case_index:04d}",
        workflow="w6c-reliability-calibration",
        state_text=state,
        decisions=decisions,
    )
    return ReliabilityAuthorityCase(
        typed=typed,
        split=split,
        template_id=template_id,
        diagnosis_k=diagnosis_k,
        severity=severity,
        confidence=confidence,
    )


def generate_w6c_authority(
    split: str,
    *,
    allow_confirm: bool = False,
) -> list[ReliabilityAuthorityCase]:
    if split == "train":
        counts, seed, templates = (
            TRAIN_K_COUNTS,
            TRAIN_SEED,
            TRAIN_TEMPLATES,
        )
    elif split == "dev":
        counts, seed, templates = (
            DEV_K_COUNTS,
            DEV_SEED,
            DEV_TEMPLATES,
        )
    elif split == "confirm":
        if not allow_confirm:
            raise RuntimeError(
                "W6c CONFIRM authority is sealed until post-selection evaluation"
            )
        counts, seed, templates = (
            CONFIRM_K_COUNTS,
            CONFIRM_SEED,
            CONFIRM_TEMPLATES,
        )
    else:
        raise ValueError(f"unknown W6c split: {split}")

    rng = random.Random(seed)
    cases: list[ReliabilityAuthorityCase] = []
    used_targets: set[tuple[str, str, str, str]] = set()
    case_index = 0

    for k in sorted(counts):
        count = counts[k]
        if count % 12:
            raise RuntimeError(
                "W6c per-K count must divide the 12 severity/confidence strata exactly"
            )
        repeats = count // 12
        strata = [
            (severity, confidence)
            for _ in range(repeats)
            for severity in range(4)
            for confidence in CONFIDENCE_LEVELS
        ]
        rng.shuffle(strata)
        for severity, confidence in strata:
            cases.append(
                generate_w6c_case(
                    split=split,
                    diagnosis_k=k,
                    case_index=case_index,
                    severity=severity,
                    confidence=confidence,
                    rng=rng,
                    template_id=templates[
                        case_index % len(templates)
                    ],
                    used_targets=used_targets,
                )
            )
            case_index += 1
    return cases
