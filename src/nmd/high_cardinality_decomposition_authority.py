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


DOMAIN_AD_SEED = 241301
DOMAIN_AE_SEED = 241307
DOMAIN_AF_SEED = 241319

BASES_PER_DOMAIN = 64
ROLE_KEYS = ("entity", "location", "anomaly", "channel")
VIEW_KS = (8, 16, 32, 64)
VIEW_IDS = tuple(f"k{k}" for k in VIEW_KS)


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
        raise RuntimeError("W6j pool must contain exactly 64 unique values")
    return rows


_AD_Q = (
    "cryoalpha",
    "cryobeta",
    "cryogamma",
    "cryodelta",
    "cryoepsilon",
    "cryokappa",
    "cryolambda",
    "cryoomega",
)
_AE_Q = (
    "ventprime",
    "ventsecure",
    "ventmetro",
    "ventdeep",
    "ventguard",
    "ventflow",
    "venttrace",
    "ventdelta",
)
_AF_Q = (
    "coastprime",
    "coastblue",
    "coasttide",
    "coastguard",
    "coastwave",
    "coastgrid",
    "coasttrack",
    "coastsignal",
)


DOMAIN_AD = DomainSpec(
    domain_id="AD",
    workflow="w6j-cryogenic-telescope-audit",
    roles=(
        "cryogenic telescope assembly",
        "observatory service zone",
        "cryogenic condition",
        "instrumentation channel",
    ),
    pools=(
        _expand64(
            (
                "mirror cooling manifold",
                "detector cryostat",
                "vacuum transfer line",
                "cold head compressor",
                "thermal intercept stage",
                "filter wheel bearing",
                "helium circulation unit",
                "focal plane mount",
            ),
            _AD_Q,
        ),
        _expand64(
            (
                "primary mirror bay",
                "detector service deck",
                "vacuum equipment room",
                "cold optics chamber",
                "thermal shield gallery",
                "instrument platform",
                "compressor alcove",
                "focal plane enclosure",
            ),
            _AD_Q,
        ),
        _expand64(
            (
                "temperature drift",
                "vacuum pressure rise",
                "cooling flow loss",
                "thermal gradient increase",
                "compressor vibration",
                "helium inventory drop",
                "shield heat leak",
                "detector warmup onset",
            ),
            _AD_Q,
        ),
        _expand64(
            (
                "cryogenic temperature trace",
                "vacuum gauge stream",
                "helium flow monitor",
                "compressor vibration feed",
                "thermal shield telemetry",
                "detector temperature bus",
                "pressure recorder",
                "instrument service log",
            ),
            _AD_Q,
        ),
    ),
    templates=(
        "w6j-ad-cryo-service-card",
        "w6j-ad-telescope-maintenance-ledger",
        "w6j-ad-cold-system-brief",
        "w6j-ad-observatory-checksheet",
    ),
)


DOMAIN_AE = DomainSpec(
    domain_id="AE",
    workflow="w6j-transit-ventilation-audit",
    roles=(
        "ventilation asset",
        "underground transit sector",
        "airflow condition",
        "environmental sensing feed",
    ),
    pools=(
        _expand64(
            (
                "supply fan array",
                "exhaust damper bank",
                "smoke extraction fan",
                "shaft pressure regulator",
                "platform air handler",
                "tunnel jet fan",
                "filter bypass gate",
                "station ventilation controller",
            ),
            _AE_Q,
        ),
        _expand64(
            (
                "platform cavern",
                "running tunnel",
                "ventilation shaft",
                "cross passage",
                "station concourse",
                "equipment gallery",
                "emergency egress route",
                "service crossover",
            ),
            _AE_Q,
        ),
        _expand64(
            (
                "airflow deficit",
                "pressure imbalance",
                "smoke extraction delay",
                "temperature rise",
                "damper response lag",
                "filter restriction",
                "fan speed oscillation",
                "air quality deterioration",
            ),
            _AE_Q,
        ),
        _expand64(
            (
                "air velocity stream",
                "differential pressure trace",
                "smoke detector bus",
                "temperature monitor",
                "damper position feed",
                "fan current channel",
                "air quality telemetry",
                "ventilation controller log",
            ),
            _AE_Q,
        ),
    ),
    templates=(
        "w6j-ae-transit-ventilation-card",
        "w6j-ae-underground-airflow-ledger",
        "w6j-ae-station-environment-brief",
        "w6j-ae-tunnel-ventilation-checksheet",
    ),
)


DOMAIN_AF = DomainSpec(
    domain_id="AF",
    workflow="w6j-coastal-monitoring-audit",
    roles=(
        "coastal monitoring asset",
        "marine observation sector",
        "environmental event",
        "observation channel",
    ),
    pools=(
        _expand64(
            (
                "autonomous surface buoy",
                "shore radar node",
                "wave sensing platform",
                "tidal observation mast",
                "coastal camera station",
                "salinity profiler",
                "current meter pod",
                "weather telemetry beacon",
            ),
            _AF_Q,
        ),
        _expand64(
            (
                "harbor approach",
                "outer reef sector",
                "estuary mouth",
                "nearshore transect",
                "breakwater corridor",
                "tidal inlet",
                "coastal shelf",
                "storm observation line",
            ),
            _AF_Q,
        ),
        _expand64(
            (
                "wave height surge",
                "current direction shift",
                "salinity excursion",
                "visibility reduction",
                "tidal level anomaly",
                "wind speed burst",
                "surface temperature change",
                "sensor drift onset",
            ),
            _AF_Q,
        ),
        _expand64(
            (
                "wave spectrum feed",
                "surface radar stream",
                "salinity telemetry",
                "optical camera bus",
                "tidal gauge trace",
                "weather station channel",
                "current profiler record",
                "coastal operations log",
            ),
            _AF_Q,
        ),
    ),
    templates=(
        "w6j-af-coastal-monitor-card",
        "w6j-af-marine-observation-ledger",
        "w6j-af-shoreline-event-brief",
        "w6j-af-autonomous-survey-checksheet",
    ),
)


DOMAINS = {
    spec.domain_id: spec
    for spec in (DOMAIN_AD, DOMAIN_AE, DOMAIN_AF)
}


@dataclass(frozen=True)
class DecompositionView:
    typed: TypedDecisionCase
    base_id: str
    domain_id: str
    template_id: str
    view_id: str
    diagnosis_k: int
    severity: int
    confidence: str
    gold_signature: tuple[str, str, str, str]
    option_signatures: tuple[tuple[str, str, str, str], ...]
    option_distances: tuple[int, ...]
    gold_option_id: str


def all_w6j_values() -> set[str]:
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


def changed_roles(
    left: tuple[str, str, str, str],
    right: tuple[str, str, str, str],
) -> tuple[str, ...]:
    return tuple(
        ROLE_KEYS[index]
        for index, (a, b) in enumerate(zip(left, right))
        if a != b
    )


def _candidate_with_roles(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    roles: tuple[int, ...],
    rng: random.Random,
    forbidden: set[tuple[str, str, str, str]],
) -> tuple[str, str, str, str]:
    pools = []
    for role in roles:
        values = [v for v in spec.pools[role] if v != target[role]]
        rng.shuffle(values)
        pools.append(values)

    # Randomized deterministic cartesian search. The pools are large (63 each)
    # and only a small number of candidates is requested per base.
    attempts = 0
    while attempts < 100000:
        attempts += 1
        row = list(target)
        for role, values in zip(roles, pools):
            row[role] = values[rng.randrange(len(values))]
        candidate = tuple(row)
        if candidate in forbidden:
            continue
        forbidden.add(candidate)
        return candidate  # type: ignore[return-value]
    raise RuntimeError("W6j exhausted controlled-negative search")


def _distance_candidates(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    *,
    distance: int,
    count: int,
    rng: random.Random,
    forbidden: set[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    if distance not in {1, 2, 3, 4}:
        raise ValueError("W6j distance must be 1..4")
    role_sets = list(combinations(range(4), distance))
    rows: list[tuple[str, str, str, str]] = []
    cursor = 0
    while len(rows) < count:
        roles = role_sets[cursor % len(role_sets)]
        rows.append(
            _candidate_with_roles(
                target,
                spec,
                roles,
                rng,
                forbidden,
            )
        )
        cursor += 1
    return rows


def _stable_option_id(
    base_id: str,
    target: tuple[str, str, str, str],
    signature: tuple[str, str, str, str],
) -> str:
    if signature == target:
        return f"{base_id}-gold"
    payload = "\x1f".join(signature).encode("utf-8")
    return (
        f"{base_id}-d{signature_distance(target, signature)}-"
        f"{hashlib.sha256(payload).hexdigest()[:20]}"
    )


def _make_view(
    spec: DomainSpec,
    *,
    base_id: str,
    template_id: str,
    state_text: str,
    severity: int,
    confidence: str,
    target: tuple[str, str, str, str],
    signatures: list[tuple[str, str, str, str]],
    view_id: str,
    seed: int,
) -> DecompositionView:
    order = list(range(len(signatures)))
    random.Random(seed).shuffle(order)
    shuffled = [signatures[index] for index in order]
    gold_index = shuffled.index(target)
    options = tuple(
        LogicalOption(
            option_id=_stable_option_id(base_id, target, signature),
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
    return DecompositionView(
        typed=typed,
        base_id=base_id,
        domain_id=spec.domain_id,
        template_id=template_id,
        view_id=view_id,
        diagnosis_k=len(options),
        severity=severity,
        confidence=confidence,
        gold_signature=target,
        option_signatures=tuple(shuffled),
        option_distances=tuple(
            signature_distance(target, signature)
            for signature in shuffled
        ),
        gold_option_id=_stable_option_id(base_id, target, target),
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
) -> list[DecompositionView]:
    target = _unique_signature(spec.pools, rng, used_targets)
    template_id = spec.templates[base_index % len(spec.templates)]
    state_text = _render_state(
        spec,
        target,
        severity=severity,
        confidence=confidence,
        template_id=template_id,
    )
    base_id = f"w6j-diag-{spec.domain_id.lower()}-{base_index:04d}"

    forbidden: set[tuple[str, str, str, str]] = {target}
    one = _distance_candidates(
        target,
        spec,
        distance=1,
        count=12,
        rng=rng,
        forbidden=forbidden,
    )
    two = _distance_candidates(
        target,
        spec,
        distance=2,
        count=20,
        rng=rng,
        forbidden=forbidden,
    )
    three = _distance_candidates(
        target,
        spec,
        distance=3,
        count=15,
        rng=rng,
        forbidden=forbidden,
    )
    four = _distance_candidates(
        target,
        spec,
        distance=4,
        count=16,
        rng=rng,
        forbidden=forbidden,
    )

    master = [target, *one, *two, *three, *four]
    k8 = [target, *one[:4], *two[:3]]
    k16 = [target, *one[:8], *two[:5], *three[:2]]
    k32 = [target, *one, *two[:12], *three[:7]]

    expected = {
        8: k8,
        16: k16,
        32: k32,
        64: master,
    }
    for k, signatures in expected.items():
        if len(signatures) != k or len(set(signatures)) != k:
            raise RuntimeError(f"W6j K{k} view must contain {k} unique signatures")
        if target not in signatures:
            raise RuntimeError("W6j nested view lost gold")

    if not set(k8).issubset(k16):
        raise RuntimeError("W6j K8 must be nested in K16")
    if not set(k16).issubset(k32):
        raise RuntimeError("W6j K16 must be nested in K32")
    if not set(k32).issubset(master):
        raise RuntimeError("W6j K32 must be nested in K64")

    rows = []
    for k in VIEW_KS:
        rows.append(
            _make_view(
                spec,
                base_id=base_id,
                template_id=template_id,
                state_text=state_text,
                severity=severity,
                confidence=confidence,
                target=target,
                signatures=list(expected[k]),
                view_id=f"k{k}",
                seed=domain_seed + base_index * 4099 + k * 101,
            )
        )
    return rows


def generate_w6j_domain(domain_id: str) -> list[DecompositionView]:
    try:
        spec = DOMAINS[domain_id]
    except KeyError as exc:
        raise ValueError("W6j domain must be AD, AE or AF") from exc
    seed = {
        "AD": DOMAIN_AD_SEED,
        "AE": DOMAIN_AE_SEED,
        "AF": DOMAIN_AF_SEED,
    }[domain_id]
    rng = random.Random(seed)
    used_targets: set[tuple[str, str, str, str]] = set()
    strata = [
        (severity, confidence)
        for severity in range(4)
        for confidence in CONFIDENCE_LEVELS
    ]
    rows: list[DecompositionView] = []
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
        raise RuntimeError("W6j domain view count mismatch")
    return rows


def generate_w6j_diagnostics() -> list[DecompositionView]:
    rows: list[DecompositionView] = []
    for domain_id in ("AD", "AE", "AF"):
        rows.extend(generate_w6j_domain(domain_id))
    expected = 3 * BASES_PER_DOMAIN * len(VIEW_IDS)
    if len(rows) != expected:
        raise RuntimeError("W6j diagnostic view count mismatch")
    return rows


def view_histogram(rows: list[DecompositionView]) -> dict[str, int]:
    return dict(Counter(row.view_id for row in rows))
