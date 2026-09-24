from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

from .contracts import LogicalOption
from .typed_decisions import TypedDecision, TypedDecisionCase
from .typed_domain_generalization_authority import (
    DomainSpec,
    _expand,
    _neighbors,
    _render_state,
    _signature_text,
    _unique_signature,
)
from .typed_reliability_authority import (
    CONFIDENCE_LEVELS,
    CONFIDENCE_MASS,
    _categorical_distribution,
)


DOMAIN_N_SEED = 201173
DOMAIN_O_SEED = 201181
DOMAIN_P_SEED = 201187

K_VALUES = (8, 16, 32, 64)
BASES_PER_DOMAIN = 96

# Exact cumulative distractor-distance composition at each nested K.
DISTANCE_TARGETS = {
    8: {1: 5, 2: 2, 3: 0},
    16: {1: 8, 2: 5, 3: 2},
    32: {1: 12, 2: 12, 3: 7},
    64: {1: 20, 2: 25, 3: 18},
}


DOMAIN_N = DomainSpec(
    domain_id="N",
    workflow="w6f-maritime-cargo-diagnostic",
    roles=(
        "cargo handling unit",
        "terminal sector",
        "freight-system event",
        "operations signal",
    ),
    pools=(
        _expand(
            (
                "gantry trolley",
                "twistlock actuator",
                "reefer manifold",
                "yard tractor drive",
                "container spreader",
                "berth capstan",
                "stack crane motor",
                "chassis brake module",
            ),
            ("quayside", "stackside", "landside"),
        ),
        _expand(
            (
                "container apron",
                "reefer row",
                "interchange lane",
                "stack block",
                "berth pocket",
                "transfer corridor",
                "gate complex",
                "equipment bay",
            ),
            ("harborfront", "harbormid", "harborrear"),
        ),
        _expand(
            (
                "hoist desync",
                "lock hesitation",
                "coolant imbalance",
                "traction slip",
                "spreader skew",
                "line tension rise",
                "motor current surge",
                "brake pressure drift",
            ),
            ("cargoquick", "cargosteady", "cargocycle"),
        ),
        _expand(
            (
                "gantry encoder",
                "lock feedback",
                "reefer telemetry",
                "tractor monitor",
                "spreader sensor",
                "capstan load feed",
                "crane current trace",
                "brake pressure link",
            ),
            ("dockbus", "freightbus", "terminalink"),
        ),
    ),
    templates=(
        "w6f-n-terminal-inspection-card",
        "w6f-n-cargo-equipment-note",
        "w6f-n-port-operations-sheet",
    ),
)

DOMAIN_O = DomainSpec(
    domain_id="O",
    workflow="w6f-water-treatment-diagnostic",
    roles=(
        "purification assembly",
        "treatment stage",
        "process-quality deviation",
        "instrumentation feed",
    ),
    pools=(
        _expand(
            (
                "membrane rack",
                "coagulant skid",
                "backwash pump",
                "uv reactor",
                "carbon contactor",
                "lime feeder",
                "flocculator drive",
                "residual analyzer",
            ),
            ("purityline", "processline", "finishline"),
        ),
        _expand(
            (
                "membrane hall",
                "coagulation gallery",
                "backwash cell",
                "uv chamber",
                "carbon basin",
                "chemical mezzanine",
                "flocculation lane",
                "quality station",
            ),
            ("treatfront", "treatmid", "treatrear"),
        ),
        _expand(
            (
                "permeate decline",
                "dose oscillation",
                "wash pressure loss",
                "irradiance drop",
                "adsorption breakthrough",
                "alkalinity drift",
                "floc torque rise",
                "residual offset",
            ),
            ("waterquick", "watersteady", "watercycle"),
        ),
        _expand(
            (
                "permeate trace",
                "dose telemetry",
                "wash pressure feed",
                "uv intensity stream",
                "carbon monitor",
                "alkalinity channel",
                "floc drive signal",
                "residual readout",
            ),
            ("purifybus", "treatmentbus", "qualitylink"),
        ),
    ),
    templates=(
        "w6f-o-purification-audit",
        "w6f-o-treatment-process-card",
        "w6f-o-water-quality-note",
    ),
)

DOMAIN_P = DomainSpec(
    domain_id="P",
    workflow="w6f-robotics-manufacturing-diagnostic",
    roles=(
        "robotic cell unit",
        "automation sector",
        "motion-process anomaly",
        "control telemetry",
    ),
    pools=(
        _expand(
            (
                "servo wrist",
                "linear slide",
                "vision gantry",
                "gripper manifold",
                "turntable drive",
                "weld positioner",
                "safety scanner",
                "tool changer",
            ),
            ("robotlane", "assemblylane", "motionlane"),
        ),
        _expand(
            (
                "assembly cell",
                "vision station",
                "weld booth",
                "material handoff",
                "inspection nest",
                "tooling bay",
                "safety perimeter",
                "transfer zone",
            ),
            ("autofront", "automid", "autorear"),
        ),
        _expand(
            (
                "position overshoot",
                "slide backlash",
                "vision offset",
                "grip pressure loss",
                "indexing jitter",
                "weld path drift",
                "scanner dropout",
                "tool latch delay",
            ),
            ("motionquick", "motionsteady", "motioncycle"),
        ),
        _expand(
            (
                "servo trace",
                "slide encoder",
                "vision stream",
                "gripper feedback",
                "turntable monitor",
                "weld telemetry",
                "scanner channel",
                "tool-change signal",
            ),
            ("robotbus", "cellbus", "motionlink"),
        ),
    ),
    templates=(
        "w6f-p-robot-cell-report",
        "w6f-p-automation-diagnostic-card",
        "w6f-p-motion-control-note",
    ),
)

DOMAINS = {
    spec.domain_id: spec
    for spec in (DOMAIN_N, DOMAIN_O, DOMAIN_P)
}


@dataclass(frozen=True)
class HighKDiagnosticView:
    typed: TypedDecisionCase
    base_id: str
    domain_id: str
    template_id: str
    diagnosis_k: int
    severity: int
    confidence: str
    gold_signature: tuple[str, str, str, str]
    option_signatures: tuple[tuple[str, str, str, str], ...]
    option_distances: tuple[int, ...]


def all_w6f_values() -> set[str]:
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


def domain_template_sets() -> dict[str, set[str]]:
    return {
        domain_id: set(spec.templates)
        for domain_id, spec in DOMAINS.items()
    }


def domain_role_sets() -> dict[str, set[str]]:
    return {
        domain_id: set(spec.roles)
        for domain_id, spec in DOMAINS.items()
    }


def signature_distance(
    left: tuple[str, str, str, str],
    right: tuple[str, str, str, str],
) -> int:
    return sum(a != b for a, b in zip(left, right))


def _three_plus_neighbors(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    rng: random.Random,
    *,
    count: int,
    forbidden: set[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    while len(rows) < count:
        row = tuple(rng.choice(pool) for pool in spec.pools)
        if row in forbidden or signature_distance(row, target) < 3:
            continue
        forbidden.add(row)
        rows.append(row)
    return rows


def _take_unique(
    source: list[tuple[str, str, str, str]],
    count: int,
    forbidden: set[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for row in source:
        if row in forbidden:
            continue
        forbidden.add(row)
        rows.append(row)
        if len(rows) == count:
            return rows
    raise RuntimeError("insufficient W6f diagnostic neighbors")


def _master_distractors(
    spec: DomainSpec,
    target: tuple[str, str, str, str],
    rng: random.Random,
) -> list[tuple[str, str, str, str]]:
    one = _neighbors(target, spec.pools, 1)
    two = _neighbors(target, spec.pools, 2)
    rng.shuffle(one)
    rng.shuffle(two)

    forbidden = {target}
    one_rows = _take_unique(one, 20, forbidden)
    two_rows = _take_unique(two, 25, forbidden)
    three_rows = _three_plus_neighbors(
        target,
        spec,
        rng,
        count=18,
        forbidden=forbidden,
    )

    # Layered order makes every prefix satisfy the preregistered cumulative
    # hard-negative composition.
    layers = (
        (one_rows[0:5], two_rows[0:2], three_rows[0:0]),
        (one_rows[5:8], two_rows[2:5], three_rows[0:2]),
        (one_rows[8:12], two_rows[5:12], three_rows[2:7]),
        (one_rows[12:20], two_rows[12:25], three_rows[7:18]),
    )
    ordered: list[tuple[str, str, str, str]] = []
    for one_layer, two_layer, three_layer in layers:
        merged = [*one_layer, *two_layer, *three_layer]
        rng.shuffle(merged)
        ordered.extend(merged)

    if len(ordered) != 63 or len(set(ordered)) != 63:
        raise RuntimeError("W6f master distractor set must contain 63 unique rows")
    return ordered


def _view_options(
    spec: DomainSpec,
    *,
    base_id: str,
    target: tuple[str, str, str, str],
    master_distractors: list[tuple[str, str, str, str]],
    k: int,
    seed: int,
) -> tuple[
    tuple[LogicalOption, ...],
    int,
    tuple[tuple[str, str, str, str], ...],
    tuple[int, ...],
]:
    signatures = [target, *master_distractors[: k - 1]]
    master_index = {
        signature: index
        for index, signature in enumerate([target, *master_distractors])
    }
    order = list(range(len(signatures)))
    random.Random(seed).shuffle(order)
    shuffled = [signatures[index] for index in order]
    gold_index = shuffled.index(target)

    options = tuple(
        LogicalOption(
            option_id=(
                f"{base_id}-candidate-{master_index[signature]:02d}"
            ),
            criterion_text=_signature_text(spec, signature),
        )
        for signature in shuffled
    )
    distances = tuple(signature_distance(target, row) for row in shuffled)
    return options, gold_index, tuple(shuffled), distances


def _make_base_views(
    spec: DomainSpec,
    *,
    base_index: int,
    severity: int,
    confidence: str,
    rng: random.Random,
    used_targets: set[tuple[str, str, str, str]],
    domain_seed: int,
) -> list[HighKDiagnosticView]:
    target = _unique_signature(spec.pools, rng, used_targets)
    master = _master_distractors(spec, target, rng)
    template_id = spec.templates[base_index % len(spec.templates)]
    state_text = _render_state(
        spec,
        target,
        severity=severity,
        confidence=confidence,
        template_id=template_id,
    )
    base_id = f"w6f-diag-{spec.domain_id.lower()}-{base_index:04d}"
    role_phrase = ", ".join(spec.roles)

    views: list[HighKDiagnosticView] = []
    for k in K_VALUES:
        options, gold_index, signatures, distances = _view_options(
            spec,
            base_id=base_id,
            target=target,
            master_distractors=master,
            k=k,
            seed=domain_seed + base_index * 131 + k,
        )
        decision = TypedDecision(
            question_id="diagnosis",
            primitive="choice",
            question_text=(
                f"Which candidate matches all reported {role_phrase} fields?"
            ),
            options=options,
            gold_index=gold_index,
            gold_probabilities=_categorical_distribution(
                k,
                gold_index,
                CONFIDENCE_MASS[confidence],
            ),
        )
        typed = TypedDecisionCase(
            case_id=f"{base_id}-k{k}",
            workflow=spec.workflow,
            state_text=state_text,
            decisions=(decision,),
        )
        views.append(
            HighKDiagnosticView(
                typed=typed,
                base_id=base_id,
                domain_id=spec.domain_id,
                template_id=template_id,
                diagnosis_k=k,
                severity=severity,
                confidence=confidence,
                gold_signature=target,
                option_signatures=signatures,
                option_distances=distances,
            )
        )
    return views


def generate_w6f_domain(domain_id: str) -> list[HighKDiagnosticView]:
    if domain_id not in DOMAINS:
        raise ValueError("W6f diagnostic domain must be N, O or P")
    spec = DOMAINS[domain_id]
    seed = {
        "N": DOMAIN_N_SEED,
        "O": DOMAIN_O_SEED,
        "P": DOMAIN_P_SEED,
    }[domain_id]
    rng = random.Random(seed)
    used_targets: set[tuple[str, str, str, str]] = set()
    strata = [
        (severity, confidence)
        for _ in range(BASES_PER_DOMAIN // 12)
        for severity in range(4)
        for confidence in CONFIDENCE_LEVELS
    ]
    rng.shuffle(strata)

    rows: list[HighKDiagnosticView] = []
    for base_index, (severity, confidence) in enumerate(strata):
        rows.extend(
            _make_base_views(
                spec,
                base_index=base_index,
                severity=severity,
                confidence=confidence,
                rng=rng,
                used_targets=used_targets,
                domain_seed=seed,
            )
        )
    if len(rows) != BASES_PER_DOMAIN * len(K_VALUES):
        raise RuntimeError("W6f diagnostic domain count mismatch")
    return rows


def generate_w6f_diagnostics() -> list[HighKDiagnosticView]:
    return [
        *generate_w6f_domain("N"),
        *generate_w6f_domain("O"),
        *generate_w6f_domain("P"),
    ]


def distance_histogram(view: HighKDiagnosticView) -> dict[int, int]:
    counter = Counter(distance for distance in view.option_distances if distance)
    return {
        1: counter.get(1, 0),
        2: counter.get(2, 0),
        3: sum(count for distance, count in counter.items() if distance >= 3),
    }
