from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import hashlib
import random

from .contracts import LogicalOption
from .typed_decisions import TypedDecision, TypedDecisionCase
from .typed_domain_generalization_authority import (
    DomainSpec,
    _render_state,
    _signature_text,
    _unique_signature,
)
from .typed_reliability_authority import (
    CONFIDENCE_LEVELS,
    CONFIDENCE_MASS,
    _categorical_distribution,
)


DOMAIN_AA_SEED = 231269
DOMAIN_AB_SEED = 231271
DOMAIN_AC_SEED = 231277

BASES_PER_DOMAIN = 64
ROLE_KEYS = ("entity", "location", "anomaly", "channel")
PAIR_VIEW_IDS = tuple(f"pair-{role}" for role in ROLE_KEYS)
VIEW_IDS = (*PAIR_VIEW_IDS, "core-k8", "master-k64")


def _expand64(
    bases: tuple[str, ...],
    qualifiers: tuple[str, ...],
) -> tuple[str, ...]:
    rows = tuple(
        f"{qualifier} {base}"
        for base in bases
        for qualifier in qualifiers
    )
    if len(rows) != 64 or len(set(rows)) != 64:
        raise RuntimeError("W6i pool must contain exactly 64 unique values")
    return rows


_AA_QUALIFIERS = (
    "orbitalpha",
    "orbitbeta",
    "orbitgamma",
    "orbitdelta",
    "orbitepsilon",
    "orbitkappa",
    "orbitlambda",
    "orbitomega",
)
_AB_QUALIFIERS = (
    "aquaprime",
    "aquasecure",
    "aquasterile",
    "aquaclear",
    "aquapure",
    "aquaguard",
    "aquaflow",
    "aquatrace",
)
_AC_QUALIFIERS = (
    "wareprime",
    "waresafe",
    "waregrid",
    "waremotion",
    "wareguard",
    "waretrack",
    "warecell",
    "wareline",
)


DOMAIN_AA = DomainSpec(
    domain_id="AA",
    workflow="w6i-satellite-battery-audit",
    roles=(
        "servicing assembly",
        "orbital service sector",
        "battery condition",
        "maintenance telemetry",
    ),
    pools=(
        _expand64(
            (
                "battery handling cradle",
                "cell balancing cart",
                "thermal service head",
                "connector inspection rig",
                "pack rotation fixture",
                "charge conditioning unit",
                "isolation test module",
                "battery transfer arm",
            ),
            _AA_QUALIFIERS,
        ),
        _expand64(
            (
                "service airlock",
                "battery work bay",
                "orbital maintenance rack",
                "inspection alcove",
                "transfer corridor",
                "conditioning chamber",
                "isolation station",
                "robotics service deck",
            ),
            _AA_QUALIFIERS,
        ),
        _expand64(
            (
                "cell imbalance drift",
                "connector resistance rise",
                "thermal spread increase",
                "charge acceptance loss",
                "isolation leakage onset",
                "pack voltage skew",
                "balancing delay",
                "capacity estimate divergence",
            ),
            _AA_QUALIFIERS,
        ),
        _expand64(
            (
                "cell voltage stream",
                "connector impedance trace",
                "pack thermal feed",
                "charge response channel",
                "isolation monitor",
                "balancing telemetry",
                "capacity estimator link",
                "service controller log",
            ),
            _AA_QUALIFIERS,
        ),
    ),
    templates=(
        "w6i-aa-orbital-battery-service-card",
        "w6i-aa-pack-maintenance-ledger",
        "w6i-aa-cell-condition-brief",
        "w6i-aa-service-bay-checksheet",
    ),
)


DOMAIN_AB = DomainSpec(
    domain_id="AB",
    workflow="w6i-sterile-water-audit",
    roles=(
        "water distribution asset",
        "sterile loop sector",
        "water-quality deviation",
        "sanitary instrumentation",
    ),
    pools=(
        _expand64(
            (
                "sanitary circulation pump",
                "ultrapure valve cluster",
                "uv treatment chamber",
                "membrane polishing skid",
                "sterile heat exchanger",
                "return loop regulator",
                "point use manifold",
                "conductivity trim unit",
            ),
            _AB_QUALIFIERS,
        ),
        _expand64(
            (
                "generation outlet",
                "distribution spine",
                "sterile return leg",
                "point use branch",
                "polishing gallery",
                "thermal sanitization zone",
                "sampling corridor",
                "storage recirculation bay",
            ),
            _AB_QUALIFIERS,
        ),
        _expand64(
            (
                "conductivity rise",
                "bioburden indicator drift",
                "return temperature sag",
                "flow stagnation onset",
                "uv dose shortfall",
                "pressure oscillation",
                "organic carbon increase",
                "sanitization timing slip",
            ),
            _AB_QUALIFIERS,
        ),
        _expand64(
            (
                "conductivity analyzer",
                "toc measurement feed",
                "loop temperature trace",
                "flow verification stream",
                "uv dose monitor",
                "pressure recorder",
                "sanitary sample channel",
                "sanitization event log",
            ),
            _AB_QUALIFIERS,
        ),
    ),
    templates=(
        "w6i-ab-sterile-water-loop-card",
        "w6i-ab-sanitary-distribution-log",
        "w6i-ab-water-quality-brief",
        "w6i-ab-loop-integrity-checksheet",
    ),
)


DOMAIN_AC = DomainSpec(
    domain_id="AC",
    workflow="w6i-warehouse-safety-audit",
    roles=(
        "warehouse automation asset",
        "safety operating zone",
        "motion-safety deviation",
        "safety observation feed",
    ),
    pools=(
        _expand64(
            (
                "autonomous pallet carrier",
                "shuttle retrieval crane",
                "sortation diverter",
                "robot picking cell",
                "vertical lift module",
                "conveyor transfer unit",
                "dock positioning robot",
                "aisle inspection rover",
            ),
            _AC_QUALIFIERS,
        ),
        _expand64(
            (
                "highbay aisle",
                "sortation mezzanine",
                "robot picking zone",
                "dock transfer lane",
                "vertical storage shaft",
                "conveyor merge area",
                "charging enclosure",
                "inspection crossing",
            ),
            _AC_QUALIFIERS,
        ),
        _expand64(
            (
                "unexpected stop",
                "clearance margin loss",
                "position tracking drift",
                "obstacle detection delay",
                "speed envelope breach",
                "load stability warning",
                "route reservation conflict",
                "guard interlock latency",
            ),
            _AC_QUALIFIERS,
        ),
        _expand64(
            (
                "safety lidar stream",
                "position encoder trace",
                "route controller log",
                "load stability feed",
                "guard interlock channel",
                "speed supervision record",
                "obstacle camera bus",
                "fleet coordination telemetry",
            ),
            _AC_QUALIFIERS,
        ),
    ),
    templates=(
        "w6i-ac-warehouse-safety-card",
        "w6i-ac-automation-risk-ledger",
        "w6i-ac-motion-integrity-brief",
        "w6i-ac-safety-zone-checksheet",
    ),
)


DOMAINS = {
    spec.domain_id: spec
    for spec in (DOMAIN_AA, DOMAIN_AB, DOMAIN_AC)
}


@dataclass(frozen=True)
class RepresentationBridgeView:
    typed: TypedDecisionCase
    base_id: str
    domain_id: str
    template_id: str
    view_id: str
    diagnosis_k: int
    severity: int
    confidence: str
    target_role: str | None
    target_role_index: int | None
    gold_signature: tuple[str, str, str, str]
    target_negative_signature: tuple[str, str, str, str] | None
    one_field_signatures: tuple[tuple[str, str, str, str], ...]
    option_signatures: tuple[tuple[str, str, str, str], ...]
    gold_option_id: str
    target_negative_option_id: str | None


def all_w6i_values() -> set[str]:
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


def canonical_tagged_text(
    signature: tuple[str, str, str, str],
) -> str:
    return " ".join(
        f"[F{index}] {value}"
        for index, value in enumerate(signature)
    )


def factorized_role_value_phrases(
    spec: DomainSpec,
    signature: tuple[str, str, str, str],
) -> tuple[str, str, str, str]:
    return tuple(
        f"{role} {value}"
        for role, value in zip(spec.roles, signature)
    )  # type: ignore[return-value]


def _one_field_candidate(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    role_index: int,
    rng: random.Random,
    forbidden: set[tuple[str, str, str, str]],
) -> tuple[str, str, str, str]:
    values = [
        value
        for value in spec.pools[role_index]
        if value != target[role_index]
    ]
    rng.shuffle(values)
    for value in values:
        row = list(target)
        row[role_index] = value
        candidate = tuple(row)
        if candidate not in forbidden:
            forbidden.add(candidate)
            return candidate  # type: ignore[return-value]
    raise RuntimeError("W6i could not construct one-field negative")


def _two_field_candidate(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    rng: random.Random,
    forbidden: set[tuple[str, str, str, str]],
    pair: tuple[int, int],
) -> tuple[str, str, str, str]:
    left, right = pair
    left_values = [v for v in spec.pools[left] if v != target[left]]
    right_values = [v for v in spec.pools[right] if v != target[right]]
    rng.shuffle(left_values)
    rng.shuffle(right_values)
    for first in left_values:
        for second in right_values:
            row = list(target)
            row[left] = first
            row[right] = second
            candidate = tuple(row)
            if candidate not in forbidden:
                forbidden.add(candidate)
                return candidate  # type: ignore[return-value]
    raise RuntimeError("W6i could not construct two-field negative")


def _cross_combinations(
    target: tuple[str, str, str, str],
    one_field: tuple[tuple[str, str, str, str], ...],
    forbidden: set[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    alternate = tuple(
        one_field[index][index]
        for index in range(4)
    )
    rows: list[tuple[str, str, str, str]] = []
    for width in (2, 3, 4):
        for selected in combinations(range(4), width):
            row = list(target)
            for role_index in selected:
                row[role_index] = alternate[role_index]
            candidate = tuple(row)
            if candidate in forbidden:
                continue
            forbidden.add(candidate)
            rows.append(candidate)  # type: ignore[arg-type]
    if len(rows) != 11:
        raise RuntimeError("W6i cross-combination contract must yield 11 rows")
    return rows


def _far_candidates(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    rng: random.Random,
    forbidden: set[tuple[str, str, str, str]],
    *,
    count: int,
) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("W6i exhausted far-negative search")
        candidate = tuple(rng.choice(pool) for pool in spec.pools)
        if candidate in forbidden:
            continue
        if signature_distance(target, candidate) < 3:
            continue
        forbidden.add(candidate)
        rows.append(candidate)  # type: ignore[arg-type]
    return rows


def _stable_option_id(
    base_id: str,
    target: tuple[str, str, str, str],
    signature: tuple[str, str, str, str],
    one_field: tuple[tuple[str, str, str, str], ...],
) -> str:
    if signature == target:
        return f"{base_id}-gold"
    for index, row in enumerate(one_field):
        if signature == row:
            return f"{base_id}-one-{ROLE_KEYS[index]}"
    payload = "\x1f".join(signature).encode("utf-8")
    code = hashlib.sha256(payload).hexdigest()[:20]
    return f"{base_id}-candidate-{code}"


def _make_view(
    spec: DomainSpec,
    *,
    base_id: str,
    template_id: str,
    state_text: str,
    severity: int,
    confidence: str,
    target: tuple[str, str, str, str],
    one_field: tuple[tuple[str, str, str, str], ...],
    signatures: list[tuple[str, str, str, str]],
    view_id: str,
    seed: int,
    target_role_index: int | None,
) -> RepresentationBridgeView:
    order = list(range(len(signatures)))
    random.Random(seed).shuffle(order)
    shuffled = [signatures[index] for index in order]
    gold_index = shuffled.index(target)
    options = tuple(
        LogicalOption(
            option_id=_stable_option_id(
                base_id,
                target,
                signature,
                one_field,
            ),
            criterion_text=_signature_text(spec, signature),
        )
        for signature in shuffled
    )
    decision = TypedDecision(
        question_id="diagnosis",
        primitive="choice",
        question_text=(
            "Which candidate matches all four reported operational fields?"
        ),
        options=options,
        gold_index=gold_index,
        gold_probabilities=_categorical_distribution(
            len(options),
            gold_index,
            CONFIDENCE_MASS[confidence],
        ),
    )
    typed = TypedDecisionCase(
        case_id=f"{base_id}-{view_id}",
        workflow=spec.workflow,
        state_text=state_text,
        decisions=(decision,),
    )
    target_negative = (
        one_field[target_role_index]
        if target_role_index is not None
        else None
    )
    return RepresentationBridgeView(
        typed=typed,
        base_id=base_id,
        domain_id=spec.domain_id,
        template_id=template_id,
        view_id=view_id,
        diagnosis_k=len(options),
        severity=severity,
        confidence=confidence,
        target_role=(
            ROLE_KEYS[target_role_index]
            if target_role_index is not None
            else None
        ),
        target_role_index=target_role_index,
        gold_signature=target,
        target_negative_signature=target_negative,
        one_field_signatures=one_field,
        option_signatures=tuple(shuffled),
        gold_option_id=_stable_option_id(
            base_id,
            target,
            target,
            one_field,
        ),
        target_negative_option_id=(
            _stable_option_id(
                base_id,
                target,
                target_negative,
                one_field,
            )
            if target_negative is not None
            else None
        ),
    )


def _make_base_views(
    spec: DomainSpec,
    *,
    base_index: int,
    severity: int,
    confidence: str,
    rng: random.Random,
    used_targets: set[tuple[str, str, str, str]],
    domain_seed: int,
) -> list[RepresentationBridgeView]:
    target = _unique_signature(spec.pools, rng, used_targets)
    template_id = spec.templates[base_index % len(spec.templates)]
    state_text = _render_state(
        spec,
        target,
        severity=severity,
        confidence=confidence,
        template_id=template_id,
    )
    base_id = f"w6i-diag-{spec.domain_id.lower()}-{base_index:04d}"

    forbidden: set[tuple[str, str, str, str]] = {target}
    one_field = tuple(
        _one_field_candidate(
            target,
            spec,
            role_index,
            rng,
            forbidden,
        )
        for role_index in range(4)
    )

    pair_cycle = (
        (0, 1),
        (2, 3),
        (0, 2),
        (1, 3),
        (0, 3),
        (1, 2),
    )
    two_field = [
        _two_field_candidate(
            target,
            spec,
            rng,
            forbidden,
            pair_cycle[index % len(pair_cycle)],
        )
        for index in range(12)
    ]

    cross = _cross_combinations(target, one_field, forbidden)
    far = _far_candidates(
        target,
        spec,
        random.Random(domain_seed + base_index * 4099 + 9901),
        forbidden,
        count=36,
    )

    core = [target, *one_field, *two_field[:3]]
    master = [target, *one_field, *two_field, *cross, *far]
    if len(core) != 8 or len(set(core)) != 8:
        raise RuntimeError("W6i K8 core must contain eight unique signatures")
    if len(master) != 64 or len(set(master)) != 64:
        raise RuntimeError("W6i K64 master must contain 64 unique signatures")
    if not set(core).issubset(master):
        raise RuntimeError("W6i K8 core must be nested in K64 master")

    rows: list[RepresentationBridgeView] = []
    for role_index, role in enumerate(ROLE_KEYS):
        rows.append(
            _make_view(
                spec,
                base_id=base_id,
                template_id=template_id,
                state_text=state_text,
                severity=severity,
                confidence=confidence,
                target=target,
                one_field=one_field,
                signatures=[target, one_field[role_index]],
                view_id=f"pair-{role}",
                seed=domain_seed + base_index * 4099 + 101 + role_index,
                target_role_index=role_index,
            )
        )

    rows.append(
        _make_view(
            spec,
            base_id=base_id,
            template_id=template_id,
            state_text=state_text,
            severity=severity,
            confidence=confidence,
            target=target,
            one_field=one_field,
            signatures=list(core),
            view_id="core-k8",
            seed=domain_seed + base_index * 4099 + 808,
            target_role_index=None,
        )
    )
    rows.append(
        _make_view(
            spec,
            base_id=base_id,
            template_id=template_id,
            state_text=state_text,
            severity=severity,
            confidence=confidence,
            target=target,
            one_field=one_field,
            signatures=list(master),
            view_id="master-k64",
            seed=domain_seed + base_index * 4099 + 6464,
            target_role_index=None,
        )
    )
    return rows


def generate_w6i_domain(domain_id: str) -> list[RepresentationBridgeView]:
    try:
        spec = DOMAINS[domain_id]
    except KeyError as exc:
        raise ValueError("W6i domain must be AA, AB or AC") from exc
    seed = {
        "AA": DOMAIN_AA_SEED,
        "AB": DOMAIN_AB_SEED,
        "AC": DOMAIN_AC_SEED,
    }[domain_id]
    rng = random.Random(seed)
    used_targets: set[tuple[str, str, str, str]] = set()
    strata = [
        (severity, confidence)
        for severity in range(4)
        for confidence in CONFIDENCE_LEVELS
    ]
    rows: list[RepresentationBridgeView] = []
    for base_index in range(BASES_PER_DOMAIN):
        severity, confidence = strata[base_index % len(strata)]
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
    if len(rows) != BASES_PER_DOMAIN * len(VIEW_IDS):
        raise RuntimeError("W6i domain view count mismatch")
    return rows


def generate_w6i_diagnostics() -> list[RepresentationBridgeView]:
    rows: list[RepresentationBridgeView] = []
    for domain_id in ("AA", "AB", "AC"):
        rows.extend(generate_w6i_domain(domain_id))
    expected = 3 * BASES_PER_DOMAIN * len(VIEW_IDS)
    if len(rows) != expected:
        raise RuntimeError("W6i diagnostic view count mismatch")
    return rows


def view_histogram(
    rows: list[RepresentationBridgeView],
) -> Counter[str]:
    return Counter(row.view_id for row in rows)
