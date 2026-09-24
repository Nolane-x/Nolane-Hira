from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Iterable

from .contracts import LogicalOption
from .typed_decisions import TypedDecision, TypedDecisionCase
from .typed_reliability_authority import (
    CONFIDENCE_LEVELS,
    CONFIDENCE_MASS,
    RESPONSE_OPTIONS,
    REVIEW_OPTIONS,
    RISK_OPTIONS,
    URGENCY_OPTIONS,
    SEVERITY_LABELS,
    _categorical_distribution,
    _ordinal_distribution,
    _expected_score,
)


SOURCE_A_SEED = 181149
SOURCE_B_SEED = 181151
SOURCE_C_SEED = 181153
SOURCE_D_SEED = 181157
DEV_E_SEED = 182251
CONFIRM_F_SEED = 183353
GLOBAL_SEED = 1201

K_VALUES = (8, 16, 32, 64)


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    workflow: str
    roles: tuple[str, str, str, str]
    pools: tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]
    templates: tuple[str, ...]


@dataclass(frozen=True)
class DomainAuthorityCase:
    typed: TypedDecisionCase
    split: str
    domain_id: str
    template_id: str
    diagnosis_k: int
    severity: int
    confidence: str


def _expand(
    bases: tuple[str, ...],
    qualifiers: tuple[str, str, str],
) -> tuple[str, ...]:
    rows = tuple(
        f"{qualifier} {base}"
        for base in bases
        for qualifier in qualifiers
    )
    if len(rows) != 24 or len(set(rows)) != 24:
        raise RuntimeError("W6d pool must contain 24 unique values")
    return rows


DOMAIN_A = DomainSpec(
    domain_id="A",
    workflow="w6d-maritime-source",
    roles=("vessel subsystem", "deck sector", "operational fault", "sensor stream"),
    pools=(
        _expand(
            ("bilge pump", "rudder actuator", "ballast valve", "shaft bearing",
             "cooling loop", "navigation relay", "fuel governor", "winch controller"),
            ("portside", "midship", "starboard"),
        ),
        _expand(
            ("machinery bay", "service deck", "control alcove", "aft gallery",
             "forward vault", "inspection lane", "utility room", "access trunk"),
            ("lower", "central", "upper"),
        ),
        _expand(
            ("cavitation", "seal leakage", "bearing vibration", "flow reversal",
             "thermal rise", "signal dropout", "torque surge", "pressure decay"),
            ("intermittent", "persistent", "cyclic"),
        ),
        _expand(
            ("sonar feed", "engine bus", "hydraulic trace", "thermal channel",
             "navigation link", "power monitor", "vibration line", "pressure feed"),
            ("redline", "silverline", "blueline"),
        ),
    ),
    templates=(
        "w6d-a-watch-log",
        "w6d-a-engineering-sheet",
        "w6d-a-vessel-brief",
        "w6d-a-deck-report",
    ),
)

DOMAIN_B = DomainSpec(
    domain_id="B",
    workflow="w6d-agriculture-source",
    roles=("field machine", "farm sector", "crop-system issue", "monitor feed"),
    pools=(
        _expand(
            ("seed drill", "irrigation pump", "grain auger", "harvest drive",
             "fertilizer mixer", "orchard sprayer", "tractor clutch", "dryer fan"),
            ("northrow", "centerrow", "southrow"),
        ),
        _expand(
            ("orchard block", "storage barn", "irrigation lane", "sorting shed",
             "greenhouse aisle", "grain pad", "field station", "service yard"),
            ("inner", "middle", "outer"),
        ),
        _expand(
            ("moisture drift", "belt slippage", "nozzle blockage", "seed misfeed",
             "temperature excess", "pressure loss", "motor chatter", "flow shortage"),
            ("seasonal", "steady", "repeating"),
        ),
        _expand(
            ("soil probe", "yield stream", "motor telemetry", "moisture feed",
             "pump trace", "weather link", "drive monitor", "pressure channel"),
            ("amberpath", "greenpath", "ochrepath"),
        ),
    ),
    templates=(
        "w6d-b-field-card",
        "w6d-b-farm-digest",
    ),
)

DOMAIN_C = DomainSpec(
    domain_id="C",
    workflow="w6d-telecom-source",
    roles=("network node", "service region", "link anomaly", "diagnostic channel"),
    pools=(
        _expand(
            ("edge router", "optical switch", "radio controller", "gateway cluster",
             "packet broker", "clock module", "fiber repeater", "power shelf"),
            ("coregrid", "metrogrid", "edgegrid"),
        ),
        _expand(
            ("backbone sector", "metro ring", "access zone", "peering hall",
             "radio district", "fiber corridor", "gateway room", "relay site"),
            ("eastmesh", "centermesh", "westmesh"),
        ),
        _expand(
            ("packet loss", "clock drift", "optical fade", "route flap",
             "latency burst", "power sag", "frame corruption", "link jitter"),
            ("sporadic", "sustained", "periodic"),
        ),
        _expand(
            ("snmp stream", "optical trace", "packet counter", "timing feed",
             "radio telemetry", "route monitor", "power channel", "error stream"),
            ("cyanbus", "violetbus", "goldbus"),
        ),
    ),
    templates=(
        "w6d-c-network-snapshot",
        "w6d-c-link-ledger",
    ),
)

DOMAIN_D = DomainSpec(
    domain_id="D",
    workflow="w6d-laboratory-source",
    roles=("lab instrument", "test station", "measurement anomaly", "acquisition stream"),
    pools=(
        _expand(
            ("mass spectrometer", "centrifuge rotor", "thermal cycler", "vacuum pump",
             "sample carousel", "laser driver", "fluid dispenser", "sensor bridge"),
            ("benchone", "benchtwo", "benchthree"),
        ),
        _expand(
            ("analysis bay", "prep room", "instrument suite", "sample vault",
             "calibration cell", "optics room", "fluidics booth", "control desk"),
            ("nearfield", "midfield", "farfield"),
        ),
        _expand(
            ("baseline drift", "sample carryover", "vacuum loss", "thermal overshoot",
             "alignment error", "flow pulsation", "detector noise", "timing offset"),
            ("transient", "continuous", "recurrent"),
        ),
        _expand(
            ("detector trace", "vacuum readout", "thermal record", "fluid stream",
             "laser monitor", "sample counter", "timing channel", "control feed"),
            ("indigotrace", "coppertrace", "pearltrace"),
        ),
    ),
    templates=(
        "w6d-d-lab-notebook",
        "w6d-d-instrument-record",
    ),
)

DOMAIN_E = DomainSpec(
    domain_id="E",
    workflow="w6d-rail-heldout-dev",
    roles=("rail assembly", "track section", "service condition", "wayside signal"),
    pools=(
        _expand(
            ("traction inverter", "brake actuator", "door controller", "axle bearing",
             "air compressor", "signal relay", "pantograph drive", "bogie sensor"),
            ("uptrack", "centertrack", "downtrack"),
        ),
        _expand(
            ("platform throat", "depot lane", "tunnel section", "switch yard",
             "viaduct span", "service siding", "station approach", "maintenance road"),
            ("inbound", "centralrun", "outbound"),
        ),
        _expand(
            ("brake lag", "wheel vibration", "voltage dip", "air leakage",
             "relay chatter", "door stall", "thermal rise", "signal dropout"),
            ("brief", "lasting", "recurring"),
        ),
        _expand(
            ("axle feed", "traction bus", "brake trace", "signal channel",
             "door monitor", "air readout", "thermal stream", "position feed"),
            ("scarletwire", "ivorywire", "navywire"),
        ),
    ),
    templates=(
        "w6d-e-rail-inspection",
        "w6d-e-transit-status",
        "w6d-e-depot-check",
    ),
)

DOMAIN_F = DomainSpec(
    domain_id="F",
    workflow="w6d-water-heldout-confirm",
    roles=("treatment unit", "process basin", "water-system deviation", "plant telemetry"),
    pools=(
        _expand(
            ("lift pump", "clarifier drive", "ozone injector", "filter valve",
             "aeration blower", "chlorine feeder", "sludge mixer", "flow meter"),
            ("intakebank", "processbank", "outfallbank"),
        ),
        _expand(
            ("raw water bay", "settling basin", "filter gallery", "chemical room",
             "pump station", "aeration lane", "sludge hall", "outlet chamber"),
            ("upstream", "midprocess", "downstream"),
        ),
        _expand(
            ("turbidity rise", "flow imbalance", "pressure drop", "dose drift",
             "motor vibration", "valve sticking", "oxygen deficit", "level oscillation"),
            ("episodic", "prolonged", "rhythmic"),
        ),
        _expand(
            ("turbidity feed", "flow trace", "pressure stream", "dose monitor",
             "motor channel", "valve telemetry", "oxygen readout", "level signal"),
            ("aquabus", "riverbus", "deltaflow"),
        ),
    ),
    templates=(
        "w6d-f-plant-audit",
        "w6d-f-water-verification",
        "w6d-f-process-assurance",
    ),
)

DOMAINS = {
    spec.domain_id: spec
    for spec in (DOMAIN_A, DOMAIN_B, DOMAIN_C, DOMAIN_D, DOMAIN_E, DOMAIN_F)
}


def all_w6d_values() -> set[str]:
    return {
        value
        for spec in DOMAINS.values()
        for pool in spec.pools
        for value in pool
    }


def domain_value_sets() -> dict[str, set[str]]:
    return {
        domain_id: {value for pool in spec.pools for value in pool}
        for domain_id, spec in DOMAINS.items()
    }


def _signature_text(
    spec: DomainSpec,
    signature: tuple[str, str, str, str],
) -> str:
    return "; ".join(
        f"{role} {value}"
        for role, value in zip(spec.roles, signature)
    )


def _render_state(
    spec: DomainSpec,
    signature: tuple[str, str, str, str],
    *,
    severity: int,
    confidence: str,
    template_id: str,
) -> str:
    a, b, c, d = signature
    r1, r2, r3, r4 = spec.roles
    sev = SEVERITY_LABELS[severity]
    slot = spec.templates.index(template_id)
    if slot % 4 == 0:
        return (
            f"{template_id} reports {r1} {a} in {r2} {b}. "
            f"The {r3} is {c}, observed through {r4} {d}. "
            f"Severity is {sev}; evidence is {confidence}."
        )
    if slot % 4 == 1:
        return (
            f"{template_id}: {r2}={b}; {r1}={a}; {r4}={d}; {r3}={c}. "
            f"Condition level {sev}; evidence grade {confidence}."
        )
    if slot % 4 == 2:
        return (
            f"{template_id} links {r4} {d} with {r1} {a} at {r2} {b}, "
            f"where {r3} {c} is present. Severity {sev}; support {confidence}."
        )
    return (
        f"{template_id} records {r3} {c} for {r1} {a}; "
        f"location {r2} {b}; source {r4} {d}. "
        f"Severity {sev}; reliability {confidence}."
    )


def _random_signature(
    pools: tuple[tuple[str, ...], ...],
    rng: random.Random,
) -> tuple[str, str, str, str]:
    return tuple(rng.choice(pool) for pool in pools)  # type: ignore[return-value]


def _unique_signature(
    pools: tuple[tuple[str, ...], ...],
    rng: random.Random,
    used: set[tuple[str, str, str, str]],
) -> tuple[str, str, str, str]:
    while True:
        row = _random_signature(pools, rng)
        if row not in used:
            used.add(row)
            return row


def _neighbors(
    target: tuple[str, str, str, str],
    pools: tuple[tuple[str, ...], ...],
    changes: int,
) -> list[tuple[str, str, str, str]]:
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
    if changes != 2:
        raise ValueError("changes must be one or two")
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
    spec: DomainSpec,
    *,
    target: tuple[str, str, str, str],
    k: int,
    rng: random.Random,
) -> tuple[tuple[LogicalOption, ...], int]:
    used = {target}
    distractors: list[tuple[str, str, str, str]] = []
    one = _neighbors(target, spec.pools, 1)
    two = _neighbors(target, spec.pools, 2)
    rng.shuffle(one)
    rng.shuffle(two)
    one_target = int(math.floor(0.50 * (k - 1)))
    two_target = int(math.floor(0.30 * (k - 1)))
    for row in one[:one_target]:
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
        row = _random_signature(spec.pools, rng)
        if row in used:
            continue
        used.add(row)
        distractors.append(row)
    signatures = [target, *distractors[: k - 1]]
    rng.shuffle(signatures)
    gold = signatures.index(target)
    return (
        tuple(
            LogicalOption(
                option_id=f"{spec.domain_id.lower()}-signature-{index:03d}",
                criterion_text=_signature_text(spec, signature),
            )
            for index, signature in enumerate(signatures)
        ),
        gold,
    )


def _make_case(
    spec: DomainSpec,
    *,
    split: str,
    diagnosis_k: int,
    case_index: int,
    severity: int,
    confidence: str,
    rng: random.Random,
    used_targets: set[tuple[str, str, str, str]],
) -> DomainAuthorityCase:
    target = _unique_signature(spec.pools, rng, used_targets)
    options, diagnosis_gold = _diagnosis_options(
        spec,
        target=target,
        k=diagnosis_k,
        rng=rng,
    )
    gold_mass = CONFIDENCE_MASS[confidence]
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

    template_id = spec.templates[case_index % len(spec.templates)]
    state_text = _render_state(
        spec,
        target,
        severity=severity,
        confidence=confidence,
        template_id=template_id,
    )
    role_phrase = ", ".join(spec.roles)
    decisions = (
        TypedDecision(
            question_id="diagnosis",
            primitive="choice",
            question_text=(
                f"Which candidate matches all reported {role_phrase} fields?"
            ),
            options=options,
            gold_index=diagnosis_gold,
            gold_probabilities=diagnosis_prob,
        ),
        TypedDecision(
            question_id="response",
            primitive="choice",
            question_text="Which response corresponds to the stated severity?",
            options=RESPONSE_OPTIONS,
            gold_index=response_gold,
            gold_probabilities=response_prob,
        ),
        TypedDecision(
            question_id="needs_review",
            primitive="noul",
            question_text=(
                "Is review required because evidence is uncertain or severity is critical?"
            ),
            options=REVIEW_OPTIONS,
            gold_index=review_gold,
            gold_probabilities=review_prob,
        ),
        TypedDecision(
            question_id="risk",
            primitive="score",
            question_text="Which risk level corresponds to the stated severity?",
            options=RISK_OPTIONS,
            gold_index=risk_gold,
            gold_probabilities=risk_prob,
            gold_score=_expected_score(risk_prob),
        ),
        TypedDecision(
            question_id="urgency",
            primitive="score",
            question_text=(
                "What urgency follows from severity and evidence confidence?"
            ),
            options=URGENCY_OPTIONS,
            gold_index=urgency_gold,
            gold_probabilities=urgency_prob,
            gold_score=_expected_score(urgency_prob),
        ),
    )
    typed = TypedDecisionCase(
        case_id=(
            f"w6d-{split}-{spec.domain_id.lower()}-"
            f"{diagnosis_k}-{case_index:04d}"
        ),
        workflow=spec.workflow,
        state_text=state_text,
        decisions=decisions,
    )
    return DomainAuthorityCase(
        typed=typed,
        split=split,
        domain_id=spec.domain_id,
        template_id=template_id,
        diagnosis_k=diagnosis_k,
        severity=severity,
        confidence=confidence,
    )


def _generate_domain(
    spec: DomainSpec,
    *,
    split: str,
    per_k: int,
    seed: int,
) -> list[DomainAuthorityCase]:
    if per_k % 12:
        raise ValueError("per-K count must divide 12 joint strata")
    rng = random.Random(seed)
    used_targets: set[tuple[str, str, str, str]] = set()
    cases: list[DomainAuthorityCase] = []
    index = 0
    repeats = per_k // 12
    for k in K_VALUES:
        strata = [
            (severity, confidence)
            for _ in range(repeats)
            for severity in range(4)
            for confidence in CONFIDENCE_LEVELS
        ]
        rng.shuffle(strata)
        for severity, confidence in strata:
            cases.append(
                _make_case(
                    spec,
                    split=split,
                    diagnosis_k=k,
                    case_index=index,
                    severity=severity,
                    confidence=confidence,
                    rng=rng,
                    used_targets=used_targets,
                )
            )
            index += 1
    return cases


def generate_w6d_single_train() -> list[DomainAuthorityCase]:
    return _generate_domain(
        DOMAIN_A,
        split="train-single",
        per_k=96,
        seed=SOURCE_A_SEED,
    )


def _stratified_take(
    cases: Iterable[DomainAuthorityCase],
    per_stratum: int,
) -> list[DomainAuthorityCase]:
    buckets: dict[tuple[int, int, str], list[DomainAuthorityCase]] = {}
    for case in cases:
        key = (case.diagnosis_k, case.severity, case.confidence)
        buckets.setdefault(key, []).append(case)
    selected: list[DomainAuthorityCase] = []
    for key in sorted(buckets):
        rows = buckets[key]
        if len(rows) < per_stratum:
            raise RuntimeError(f"insufficient W6d stratum {key}")
        selected.extend(rows[:per_stratum])
    return selected


def _as_multi_source_a(
    case: DomainAuthorityCase,
    index: int,
) -> DomainAuthorityCase:
    typed = replace(
        case.typed,
        case_id=f"w6d-train-multi-a-{case.diagnosis_k}-{index:04d}",
    )
    return replace(case, typed=typed, split="train-multi")


def generate_w6d_multi_train() -> list[DomainAuthorityCase]:
    source_a_raw = _stratified_take(generate_w6d_single_train(), 2)
    source_a = [
        _as_multi_source_a(case, index)
        for index, case in enumerate(source_a_raw)
    ]
    source_b = _generate_domain(
        DOMAIN_B, split="train-multi", per_k=24, seed=SOURCE_B_SEED
    )
    source_c = _generate_domain(
        DOMAIN_C, split="train-multi", per_k=24, seed=SOURCE_C_SEED
    )
    source_d = _generate_domain(
        DOMAIN_D, split="train-multi", per_k=24, seed=SOURCE_D_SEED
    )
    rows = [*source_a, *source_b, *source_c, *source_d]
    if len(rows) != 384:
        raise RuntimeError("W6d multi-source TRAIN must contain 384 cases")
    return rows


def generate_w6d_dev() -> list[DomainAuthorityCase]:
    return _generate_domain(
        DOMAIN_E,
        split="dev",
        per_k=48,
        seed=DEV_E_SEED,
    )


def generate_w6d_confirm(
    *,
    allow_confirm: bool = False,
) -> list[DomainAuthorityCase]:
    if not allow_confirm:
        raise RuntimeError(
            "W6d CONFIRM domain F is sealed until all DEV checkpoints freeze"
        )
    return _generate_domain(
        DOMAIN_F,
        split="confirm",
        per_k=48,
        seed=CONFIRM_F_SEED,
    )
