from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .contracts import LogicalOption
from .typed_decisions import TypedDecision, TypedDecisionCase


TRAIN_K_COUNTS = {8: 80, 16: 80, 32: 80, 64: 80}
DEV_K_COUNTS = {8: 32, 16: 32, 32: 32, 64: 32}
CONFIRM_K_COUNTS = {8: 40, 16: 40, 32: 40, 64: 40}

TRAIN_SEED = 161127
DEV_SEED = 162229
CONFIRM_SEED = 163331
GLOBAL_SEED = 809

TRAIN_TEMPLATES = (
    "w6b-train-maintenance-capsule",
    "w6b-train-operations-folio",
    "w6b-train-diagnostic-slate",
    "w6b-train-control-summary",
)
DEV_TEMPLATES = (
    "w6b-dev-inspection-brief",
    "w6b-dev-service-tableau",
)
CONFIRM_TEMPLATES = (
    "w6b-confirm-verification-sheet",
    "w6b-confirm-assurance-card",
    "w6b-confirm-reliability-panel",
)

TRAIN_EQUIPMENT = (
    "centrifugal compressor", "servo actuator", "optical transducer",
    "hydraulic manifold", "power inverter", "coolant exchanger",
    "vacuum regulator", "bearing assembly", "drive controller",
    "flow modulator", "thermal coupler", "signal conditioner",
    "pressure intensifier", "rotary separator", "voltage converter",
    "pneumatic positioner", "frequency stabilizer", "lubrication module",
    "torque limiter", "filter housing", "impedance monitor",
    "cooling impeller", "linear carriage", "sensor gateway",
)
TRAIN_ZONES = (
    "north service bay", "east process deck", "upper utility gallery",
    "west control alcove", "central machine hall", "lower transfer corridor",
    "auxiliary pump room", "primary relay chamber", "remote test enclosure",
    "secondary valve deck", "instrument mezzanine", "backup power cell",
    "cooling distribution bay", "process sampling room", "drive service lane",
    "thermal exchange vault", "hydraulic service pit", "signal routing loft",
    "mechanical inspection cell", "energy conversion bay", "pressure test booth",
    "automation cabinet row", "maintenance staging area", "monitoring annex",
)
TRAIN_ANOMALIES = (
    "thermal drift", "pressure sag", "phase noise", "flow oscillation",
    "bearing chatter", "signal clipping", "coolant restriction",
    "torque ripple", "voltage flutter", "valve hysteresis",
    "sensor bias", "seal seepage", "frequency wander", "shaft imbalance",
    "control saturation", "filter loading", "position lag", "current leakage",
    "response jitter", "thermal cycling", "pressure pulsing",
    "feedback inversion", "gain instability", "alignment offset",
)
TRAIN_CHANNELS = (
    "alpha telemetry", "bravo telemetry", "charlie telemetry",
    "delta telemetry", "echo telemetry", "foxtrot telemetry",
    "gamma telemetry", "helix telemetry", "indigo telemetry",
    "juno telemetry", "kappa telemetry", "lambda telemetry",
    "matrix telemetry", "nova telemetry", "omega telemetry",
    "prism telemetry", "quartz telemetry", "radial telemetry",
    "sigma telemetry", "tango telemetry", "umbra telemetry",
    "vector telemetry", "whiskey telemetry", "zenith telemetry",
)

CONFIRM_EQUIPMENT = (
    "magnetic clutch", "metering piston", "isolation transformer",
    "recirculation blower", "differential probe", "synchronizing relay",
    "metering valve", "ceramic heater", "logic backplane",
    "purge controller", "vibration pickup", "current shunt",
    "gear reducer", "thermal switch", "air ejector", "load balancer",
)
CONFIRM_ZONES = (
    "calibration workshop", "southern service trench", "relay testing balcony",
    "compressor access tunnel", "metering laboratory", "emergency power room",
    "instrument service bridge", "ventilation equipment loft",
    "distribution switch room", "quality inspection bay",
    "utility isolation deck", "control verification booth",
    "mechanical overhaul room", "process observation cabin",
    "electrical service court", "reliability test cell",
)
CONFIRM_ANOMALIES = (
    "magnetic slip", "pressure creep", "timing skew", "air entrainment",
    "contact bounce", "thermal runaway", "baseline shift", "gear backlash",
    "harmonic distortion", "valve stiction", "probe dropout", "seal hardening",
    "relay flutter", "load hunting", "coolant aeration", "command latency",
)
CONFIRM_CHANNELS = (
    "aurora telemetry", "beacon telemetry", "cobalt telemetry",
    "drift telemetry", "ember telemetry", "focal telemetry",
    "glacier telemetry", "harbor telemetry", "ion telemetry",
    "kepler telemetry", "lumen telemetry", "meridian telemetry",
    "nimbus telemetry", "orbit telemetry", "polar telemetry",
    "ranger telemetry",
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
        "monitor",
        "Continue monitoring when the reported severity is nominal.",
    ),
    LogicalOption(
        "schedule",
        "Schedule maintenance when the reported severity is elevated.",
    ),
    LogicalOption(
        "isolate",
        "Isolate the affected unit when the reported severity is serious.",
    ),
    LogicalOption(
        "shutdown",
        "Trigger an emergency shutdown when the reported severity is critical.",
    ),
)
REVIEW_OPTIONS = (
    LogicalOption(
        "false",
        "Human review is not required when evidence is sufficiently reliable and the condition is not critical.",
        value=0.0,
    ),
    LogicalOption(
        "true",
        "Human review is required when evidence is uncertain or the condition is critical.",
        value=1.0,
    ),
)
RISK_OPTIONS = tuple(
    LogicalOption(
        f"risk-{index}",
        text,
        value=float(index),
    )
    for index, text in enumerate((
        "Risk level zero corresponds to nominal severity.",
        "Risk level one corresponds to elevated severity.",
        "Risk level two corresponds to serious severity.",
        "Risk level three corresponds to critical severity.",
    ))
)
URGENCY_OPTIONS = tuple(
    LogicalOption(
        f"urgency-{index}",
        text,
        value=float(index),
    )
    for index, text in enumerate((
        "Urgency zero permits routine observation.",
        "Urgency one calls for planned attention.",
        "Urgency two calls for prompt intervention.",
        "Urgency three calls for immediate intervention.",
    ))
)


@dataclass(frozen=True)
class AuthorityCase:
    typed: TypedDecisionCase
    split: str
    template_id: str
    diagnosis_k: int
    severity: int
    confidence: str


def all_w6b_values() -> set[str]:
    groups = (
        TRAIN_EQUIPMENT,
        TRAIN_ZONES,
        TRAIN_ANOMALIES,
        TRAIN_CHANNELS,
        CONFIRM_EQUIPMENT,
        CONFIRM_ZONES,
        CONFIRM_ANOMALIES,
        CONFIRM_CHANNELS,
    )
    return {value for group in groups for value in group}


def _pools(split: str):
    if split in {"train", "dev"}:
        return (
            TRAIN_EQUIPMENT,
            TRAIN_ZONES,
            TRAIN_ANOMALIES,
            TRAIN_CHANNELS,
        )
    if split == "confirm":
        return (
            CONFIRM_EQUIPMENT,
            CONFIRM_ZONES,
            CONFIRM_ANOMALIES,
            CONFIRM_CHANNELS,
        )
    raise ValueError(f"unknown W6b split: {split}")


def _signature_text(signature: tuple[str, str, str, str]) -> str:
    equipment, zone, anomaly, channel = signature
    return (
        f"equipment {equipment}; zone {zone}; anomaly {anomaly}; "
        f"channel {channel}"
    )


def _render_state(
    signature: tuple[str, str, str, str],
    *,
    severity: int,
    confidence: str,
    template_id: str,
) -> str:
    equipment, zone, anomaly, channel = signature
    severity_label = SEVERITY_LABELS[severity]
    if template_id == "w6b-train-maintenance-capsule":
        return (
            f"Maintenance capsule reports {equipment} in {zone}. "
            f"The observed anomaly is {anomaly} on {channel}. "
            f"Reported severity is {severity_label}; evidence status is {confidence}."
        )
    if template_id == "w6b-train-operations-folio":
        return (
            f"Operations folio: zone={zone}; equipment={equipment}; "
            f"channel={channel}; anomaly={anomaly}. "
            f"Assessment marks severity {severity_label} with {confidence} evidence."
        )
    if template_id == "w6b-train-diagnostic-slate":
        return (
            f"Diagnostic slate places {anomaly} at the {equipment}, located in {zone}, "
            f"using {channel}. The severity band is {severity_label} and the evidence is {confidence}."
        )
    if template_id == "w6b-train-control-summary":
        return (
            f"Control summary identifies channel {channel} for {equipment} in {zone}; "
            f"its condition is {anomaly}. Severity: {severity_label}. Reliability: {confidence}."
        )
    if template_id == "w6b-dev-inspection-brief":
        return (
            f"Inspection brief records {equipment} at {zone}, with {anomaly} detected through {channel}. "
            f"The condition is rated {severity_label}; confidence is {confidence}."
        )
    if template_id == "w6b-dev-service-tableau":
        return (
            f"Service tableau associates {zone} / {channel} / {equipment} / {anomaly}. "
            f"Severity classification is {severity_label}, and evidence quality is {confidence}."
        )
    if template_id == "w6b-confirm-verification-sheet":
        return (
            f"Verification sheet lists equipment {equipment}, location {zone}, "
            f"observation {anomaly}, and telemetry source {channel}. "
            f"Severity is {severity_label}; evidence is {confidence}."
        )
    if template_id == "w6b-confirm-assurance-card":
        return (
            f"Assurance card: {anomaly} affects {equipment} within {zone} and appears on {channel}. "
            f"The declared severity is {severity_label}, supported by {confidence} evidence."
        )
    if template_id == "w6b-confirm-reliability-panel":
        return (
            f"Reliability panel links {channel} to {equipment} in {zone}, where {anomaly} is present. "
            f"Severity reads {severity_label}; assessment confidence reads {confidence}."
        )
    raise ValueError(f"unknown W6b template: {template_id}")


def _random_signature(
    pools,
    rng: random.Random,
) -> tuple[str, str, str, str]:
    return tuple(rng.choice(pool) for pool in pools)  # type: ignore[return-value]


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
            option_id=f"diagnosis-{index:03d}",
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
    if not 0 <= gold_index < count or count < 2:
        raise ValueError("invalid categorical target")
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
    if not 0 <= gold_index < count or count < 2:
        raise ValueError("invalid ordinal target")
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
    return sum(index * probability for index, probability in enumerate(probabilities))


def generate_w6b_case(
    *,
    split: str,
    diagnosis_k: int,
    case_index: int,
    rng: random.Random,
    template_id: str,
) -> AuthorityCase:
    if diagnosis_k not in {8, 16, 32, 64}:
        raise ValueError("W6b diagnosis K must be 8, 16, 32 or 64")
    pools = _pools(split)
    target = _random_signature(pools, rng)
    severity = rng.randrange(4)
    confidence = CONFIDENCE_LEVELS[rng.randrange(len(CONFIDENCE_LEVELS))]
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
    urgency_gold = min(3, severity + int(confidence == "uncertain"))

    diagnosis_prob = _categorical_distribution(
        diagnosis_k, diagnosis_gold, gold_mass
    )
    response_prob = _categorical_distribution(4, response_gold, gold_mass)
    review_prob = _categorical_distribution(2, review_gold, gold_mass)
    risk_prob = _ordinal_distribution(4, risk_gold, gold_mass)
    urgency_prob = _ordinal_distribution(4, urgency_gold, gold_mass)

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
                "Which equipment signature matches all four reported fields?"
            ),
            options=diagnosis_options,
            gold_index=diagnosis_gold,
            gold_probabilities=diagnosis_prob,
        ),
        TypedDecision(
            question_id="response",
            primitive="choice",
            question_text=(
                "Which operational response matches the reported severity?"
            ),
            options=RESPONSE_OPTIONS,
            gold_index=response_gold,
            gold_probabilities=response_prob,
        ),
        TypedDecision(
            question_id="needs_review",
            primitive="noul",
            question_text=(
                "Does this case require human review because evidence is uncertain or severity is critical?"
            ),
            options=REVIEW_OPTIONS,
            gold_index=review_gold,
            gold_probabilities=review_prob,
        ),
        TypedDecision(
            question_id="risk",
            primitive="score",
            question_text="What risk level matches the reported severity?",
            options=RISK_OPTIONS,
            gold_index=risk_gold,
            gold_probabilities=risk_prob,
            gold_score=_expected_score(risk_prob),
        ),
        TypedDecision(
            question_id="urgency",
            primitive="score",
            question_text=(
                "How urgent is intervention after combining severity with evidence confidence?"
            ),
            options=URGENCY_OPTIONS,
            gold_index=urgency_gold,
            gold_probabilities=urgency_prob,
            gold_score=_expected_score(urgency_prob),
        ),
    )
    case_id = f"w6b-{split}-{diagnosis_k}-{case_index:04d}"
    typed = TypedDecisionCase(
        case_id=case_id,
        workflow="w6b-production-reliability",
        state_text=state,
        decisions=decisions,
    )
    return AuthorityCase(
        typed=typed,
        split=split,
        template_id=template_id,
        diagnosis_k=diagnosis_k,
        severity=severity,
        confidence=confidence,
    )


def generate_w6b_authority(
    split: str,
    *,
    allow_confirm: bool = False,
) -> list[AuthorityCase]:
    if split == "train":
        counts, seed, templates = TRAIN_K_COUNTS, TRAIN_SEED, TRAIN_TEMPLATES
    elif split == "dev":
        counts, seed, templates = DEV_K_COUNTS, DEV_SEED, DEV_TEMPLATES
    elif split == "confirm":
        if not allow_confirm:
            raise RuntimeError(
                "W6b CONFIRM authority is sealed until post-selection evaluation"
            )
        counts, seed, templates = CONFIRM_K_COUNTS, CONFIRM_SEED, CONFIRM_TEMPLATES
    else:
        raise ValueError(f"unknown W6b split: {split}")

    rng = random.Random(seed)
    cases: list[AuthorityCase] = []
    index = 0
    for k in sorted(counts):
        for _ in range(counts[k]):
            cases.append(
                generate_w6b_case(
                    split=split,
                    diagnosis_k=k,
                    case_index=index,
                    rng=rng,
                    template_id=templates[index % len(templates)],
                )
            )
            index += 1
    return cases
