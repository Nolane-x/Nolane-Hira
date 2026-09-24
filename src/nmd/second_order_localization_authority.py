from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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


DOMAIN_Q_SEED = 211193
DOMAIN_R_SEED = 211199
DOMAIN_S_SEED = 211213

BASES_PER_DOMAIN = 64
ROLE_KEYS = ("entity", "location", "anomaly", "channel")
PAIR_VIEW_IDS = tuple(f"pair-{role}" for role in ROLE_KEYS)
DENSE_VIEW_IDS = tuple(f"dense-{role}64" for role in ROLE_KEYS)
VIEW_IDS = (*PAIR_VIEW_IDS, "core-k8", "far64", *DENSE_VIEW_IDS)


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
        raise RuntimeError("W6g pool must contain exactly 64 unique values")
    return rows


_Q_QUALIFIERS = (
    "apogee",
    "perigee",
    "helios",
    "umbra",
    "zenith",
    "nadir",
    "aurora",
    "ecliptic",
)
_R_QUALIFIERS = (
    "aseptic",
    "cryolane",
    "bioguard",
    "serumline",
    "vialpath",
    "medicold",
    "steriflow",
    "assaychain",
)
_S_QUALIFIERS = (
    "caldera",
    "fumarole",
    "silicic",
    "basaltflow",
    "hydrothermal",
    "magmatic",
    "fractureline",
    "steamfield",
)


DOMAIN_Q = DomainSpec(
    domain_id="Q",
    workflow="w6g-orbital-greenhouse-diagnostic",
    roles=(
        "cultivation module",
        "orbital bay",
        "biosystem deviation",
        "telemetry conduit",
    ),
    pools=(
        _expand64(
            (
                "nutrient manifold",
                "root aerator",
                "light array",
                "humidity exchanger",
                "seed cassette",
                "canopy drive",
                "water recycler",
                "growth rack",
            ),
            _Q_QUALIFIERS,
        ),
        _expand64(
            (
                "photosynthesis deck",
                "germination vault",
                "root chamber",
                "canopy corridor",
                "nutrient alcove",
                "recycle gallery",
                "light compartment",
                "harvest bay",
            ),
            _Q_QUALIFIERS,
        ),
        _expand64(
            (
                "root oxygen sag",
                "nutrient ratio drift",
                "leaf temperature rise",
                "humidity cycling",
                "light timing skew",
                "water uptake loss",
                "seed pressure offset",
                "canopy growth lag",
            ),
            _Q_QUALIFIERS,
        ),
        _expand64(
            (
                "root sensor stream",
                "nutrient spectrometer feed",
                "leaf thermal trace",
                "humidity telemetry",
                "photon counter link",
                "water balance channel",
                "seed pressure readout",
                "canopy imaging bus",
            ),
            _Q_QUALIFIERS,
        ),
    ),
    templates=(
        "w6g-q-orbital-crop-ledger",
        "w6g-q-biosphere-check-card",
        "w6g-q-greenhouse-orbit-note",
        "w6g-q-cultivation-status-sheet",
    ),
)


DOMAIN_R = DomainSpec(
    domain_id="R",
    workflow="w6g-biomedical-coldchain-diagnostic",
    roles=(
        "cryogenic carrier",
        "sterile waypoint",
        "specimen condition",
        "assay telemetry",
    ),
    pools=(
        _expand64(
            (
                "plasma vial shuttle",
                "tissue capsule rack",
                "reagent pod",
                "serum cassette",
                "biopsy carrier",
                "culture tray",
                "antibody cartridge",
                "genomic tube cradle",
            ),
            _R_QUALIFIERS,
        ),
        _expand64(
            (
                "intake freezer",
                "sterile transfer cell",
                "assay vestibule",
                "sample handoff zone",
                "cryostorage aisle",
                "reagent staging room",
                "specimen checkpoint",
                "analysis antechamber",
            ),
            _R_QUALIFIERS,
        ),
        _expand64(
            (
                "temperature excursion",
                "seal integrity loss",
                "thaw onset",
                "barcode association drift",
                "vial pressure rise",
                "sterility indicator shift",
                "sample volume mismatch",
                "carrier timing lag",
            ),
            _R_QUALIFIERS,
        ),
        _expand64(
            (
                "temperature logger",
                "seal monitor",
                "thaw sensor trace",
                "barcode audit stream",
                "vial pressure feed",
                "sterility telemetry",
                "volume verification link",
                "carrier timing channel",
            ),
            _R_QUALIFIERS,
        ),
    ),
    templates=(
        "w6g-r-specimen-coldchain-record",
        "w6g-r-sterile-transfer-brief",
        "w6g-r-biomedical-assay-note",
        "w6g-r-cryostorage-verification-card",
    ),
)


DOMAIN_S = DomainSpec(
    domain_id="S",
    workflow="w6g-geothermal-control-diagnostic",
    roles=(
        "steam-cycle assembly",
        "reservoir cell",
        "thermofluid deviation",
        "plant observation link",
    ),
    pools=(
        _expand64(
            (
                "flash-stage valve",
                "brine recirculator",
                "steam moisture separator",
                "binary-loop pump",
                "reinjection regulator",
                "turbine inlet guide",
                "condensate exchanger",
                "wellhead throttle",
            ),
            _S_QUALIFIERS,
        ),
        _expand64(
            (
                "production manifold",
                "flash-stage platform",
                "binary-loop gallery",
                "reinjection terrace",
                "steam header bay",
                "condensate corridor",
                "wellhead service cell",
                "turbine approach chamber",
            ),
            _S_QUALIFIERS,
        ),
        _expand64(
            (
                "enthalpy drop",
                "brine density shift",
                "steam quality loss",
                "reinjection pressure rise",
                "condensate subcooling drift",
                "flow partition imbalance",
                "wellhead temperature sag",
                "turbine inlet pulsation",
            ),
            _S_QUALIFIERS,
        ),
        _expand64(
            (
                "enthalpy telemetry",
                "brine densitometer feed",
                "steam quality trace",
                "reinjection pressure channel",
                "condensate thermal link",
                "flow partition monitor",
                "wellhead temperature stream",
                "turbine inlet readout",
            ),
            _S_QUALIFIERS,
        ),
    ),
    templates=(
        "w6g-s-thermofluid-operation-ledger",
        "w6g-s-steam-cycle-inspection-card",
        "w6g-s-reservoir-control-note",
        "w6g-s-geothermal-observation-sheet",
    ),
)


DOMAINS = {
    spec.domain_id: spec
    for spec in (DOMAIN_Q, DOMAIN_R, DOMAIN_S)
}


@dataclass(frozen=True)
class SecondOrderDiagnosticView:
    typed: TypedDecisionCase
    split: str
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
    one_field_option_ids: tuple[str, str, str, str]
    option_signatures: tuple[tuple[str, str, str, str], ...]
    option_distances: tuple[int, ...]
    option_changed_roles: tuple[tuple[str, ...], ...]
    gold_option_id: str
    target_negative_option_id: str | None


def all_w6g_values() -> set[str]:
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
    gold: tuple[str, str, str, str],
    candidate: tuple[str, str, str, str],
) -> tuple[str, ...]:
    return tuple(
        ROLE_KEYS[index]
        for index, (left, right) in enumerate(zip(gold, candidate))
        if left != right
    )


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
    raise RuntimeError("W6g could not build one-field hard negative")


def _two_field_candidate(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    pair: tuple[int, int],
    rng: random.Random,
    forbidden: set[tuple[str, str, str, str]],
) -> tuple[str, str, str, str]:
    first, second = pair
    left_values = [
        value for value in spec.pools[first]
        if value != target[first]
    ]
    right_values = [
        value for value in spec.pools[second]
        if value != target[second]
    ]
    rng.shuffle(left_values)
    rng.shuffle(right_values)
    for left in left_values:
        for right in right_values:
            row = list(target)
            row[first] = left
            row[second] = right
            candidate = tuple(row)
            if candidate not in forbidden:
                forbidden.add(candidate)
                return candidate  # type: ignore[return-value]
    raise RuntimeError("W6g could not build two-field negative")


def _far_spectators(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    rng: random.Random,
    *,
    count: int,
    forbidden: set[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    while len(rows) < count:
        candidate = tuple(rng.choice(pool) for pool in spec.pools)
        if candidate in forbidden:
            continue
        if signature_distance(target, candidate) < 3:
            continue
        forbidden.add(candidate)
        rows.append(candidate)  # type: ignore[arg-type]
    return rows


def _dense_role_spectators(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    role_index: int,
    rng: random.Random,
    *,
    count: int,
    forbidden: set[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    values = [
        value
        for value in spec.pools[role_index]
        if value != target[role_index]
    ]
    rng.shuffle(values)
    rows: list[tuple[str, str, str, str]] = []
    for value in values:
        row = list(target)
        row[role_index] = value
        candidate = tuple(row)
        if candidate in forbidden:
            continue
        forbidden.add(candidate)
        rows.append(candidate)  # type: ignore[arg-type]
        if len(rows) == count:
            return rows
    raise RuntimeError("W6g dense-role pool is too small")


def _stable_option_id(
    base_id: str,
    target: tuple[str, str, str, str],
    signature: tuple[str, str, str, str],
    one_field: tuple[tuple[str, str, str, str], ...],
    two_field: tuple[tuple[str, str, str, str], ...],
) -> str:
    if signature == target:
        return f"{base_id}-gold"
    for index, row in enumerate(one_field):
        if signature == row:
            return f"{base_id}-one-{ROLE_KEYS[index]}"
    for index, row in enumerate(two_field):
        if signature == row:
            return f"{base_id}-two-{index}"
    # Spectators remain deterministic across processes/runners.
    payload = "\x1f".join(signature).encode("utf-8")
    code = hashlib.sha256(payload).hexdigest()[:20]
    return f"{base_id}-spectator-{code}"


def _view(
    spec: DomainSpec,
    *,
    base_id: str,
    template_id: str,
    state_text: str,
    severity: int,
    confidence: str,
    target: tuple[str, str, str, str],
    one_field: tuple[tuple[str, str, str, str], ...],
    two_field: tuple[tuple[str, str, str, str], ...],
    signatures: list[tuple[str, str, str, str]],
    view_id: str,
    seed: int,
    target_role_index: int | None,
) -> SecondOrderDiagnosticView:
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
                two_field,
            ),
            criterion_text=_signature_text(spec, signature),
        )
        for signature in shuffled
    )
    gold_option_id = _stable_option_id(
        base_id, target, target, one_field, two_field
    )

    target_negative = (
        one_field[target_role_index]
        if target_role_index is not None
        else None
    )
    target_negative_id = (
        _stable_option_id(
            base_id,
            target,
            target_negative,
            one_field,
            two_field,
        )
        if target_negative is not None
        else None
    )

    role_phrase = ", ".join(spec.roles)
    decision = TypedDecision(
        question_id="diagnosis",
        primitive="choice",
        question_text=(
            f"Which candidate matches all reported {role_phrase} fields?"
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
    one_field_option_ids = tuple(
        _stable_option_id(
            base_id,
            target,
            signature,
            one_field,
            two_field,
        )
        for signature in one_field
    )
    return SecondOrderDiagnosticView(
        typed=typed,
        split="diagnostic",
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
        one_field_option_ids=one_field_option_ids,
        option_signatures=tuple(shuffled),
        option_distances=tuple(
            signature_distance(target, row)
            for row in shuffled
        ),
        option_changed_roles=tuple(
            changed_roles(target, row)
            for row in shuffled
        ),
        gold_option_id=gold_option_id,
        target_negative_option_id=target_negative_id,
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
) -> list[SecondOrderDiagnosticView]:
    target = _unique_signature(spec.pools, rng, used_targets)
    template_id = spec.templates[base_index % len(spec.templates)]
    state_text = _render_state(
        spec,
        target,
        severity=severity,
        confidence=confidence,
        template_id=template_id,
    )
    base_id = f"w6g-diag-{spec.domain_id.lower()}-{base_index:04d}"

    forbidden = {target}
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
    two_field = tuple(
        _two_field_candidate(
            target,
            spec,
            pair,
            rng,
            forbidden,
        )
        for pair in ((0, 1), (2, 3), (0, 2))
    )
    core = [target, *one_field, *two_field]
    if len(core) != 8 or len(set(core)) != 8:
        raise RuntimeError("W6g core must contain eight unique signatures")

    rows: list[SecondOrderDiagnosticView] = []

    for role_index, role in enumerate(ROLE_KEYS):
        rows.append(
            _view(
                spec,
                base_id=base_id,
                template_id=template_id,
                state_text=state_text,
                severity=severity,
                confidence=confidence,
                target=target,
                one_field=one_field,
                two_field=two_field,
                signatures=[target, one_field[role_index]],
                view_id=f"pair-{role}",
                seed=domain_seed + base_index * 1009 + role_index * 17 + 2,
                target_role_index=role_index,
            )
        )

    rows.append(
        _view(
            spec,
            base_id=base_id,
            template_id=template_id,
            state_text=state_text,
            severity=severity,
            confidence=confidence,
            target=target,
            one_field=one_field,
            two_field=two_field,
            signatures=list(core),
            view_id="core-k8",
            seed=domain_seed + base_index * 1009 + 8,
            target_role_index=None,
        )
    )

    far_forbidden = set(core)
    far = _far_spectators(
        target,
        spec,
        random.Random(domain_seed + base_index * 1009 + 6401),
        count=56,
        forbidden=far_forbidden,
    )
    rows.append(
        _view(
            spec,
            base_id=base_id,
            template_id=template_id,
            state_text=state_text,
            severity=severity,
            confidence=confidence,
            target=target,
            one_field=one_field,
            two_field=two_field,
            signatures=[*core, *far],
            view_id="far64",
            seed=domain_seed + base_index * 1009 + 64,
            target_role_index=None,
        )
    )

    for role_index, role in enumerate(ROLE_KEYS):
        dense_forbidden = set(core)
        dense = _dense_role_spectators(
            target,
            spec,
            role_index,
            random.Random(
                domain_seed
                + base_index * 1009
                + 7001
                + role_index * 101
            ),
            count=56,
            forbidden=dense_forbidden,
        )
        rows.append(
            _view(
                spec,
                base_id=base_id,
                template_id=template_id,
                state_text=state_text,
                severity=severity,
                confidence=confidence,
                target=target,
                one_field=one_field,
                two_field=two_field,
                signatures=[*core, *dense],
                view_id=f"dense-{role}64",
                seed=(
                    domain_seed
                    + base_index * 1009
                    + 8001
                    + role_index * 103
                ),
                target_role_index=role_index,
            )
        )

    if len(rows) != 10:
        raise RuntimeError("W6g base must materialize exactly ten views")
    return rows


def generate_w6g_domain(domain_id: str) -> list[SecondOrderDiagnosticView]:
    if domain_id not in DOMAINS:
        raise ValueError("W6g diagnostic domain must be Q, R or S")
    spec = DOMAINS[domain_id]
    seed = {
        "Q": DOMAIN_Q_SEED,
        "R": DOMAIN_R_SEED,
        "S": DOMAIN_S_SEED,
    }[domain_id]
    rng = random.Random(seed)
    used_targets: set[tuple[str, str, str, str]] = set()

    strata = [
        (severity, confidence)
        for severity in range(4)
        for confidence in CONFIDENCE_LEVELS
    ]
    repeated = [
        strata[index % len(strata)]
        for index in range(BASES_PER_DOMAIN)
    ]
    rng.shuffle(repeated)

    rows: list[SecondOrderDiagnosticView] = []
    for base_index, (severity, confidence) in enumerate(repeated):
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

    if len(rows) != BASES_PER_DOMAIN * 10:
        raise RuntimeError("W6g diagnostic domain count mismatch")
    return rows


def generate_w6g_diagnostics() -> list[SecondOrderDiagnosticView]:
    return [
        *generate_w6g_domain("Q"),
        *generate_w6g_domain("R"),
        *generate_w6g_domain("S"),
    ]


def view_histogram(
    rows: list[SecondOrderDiagnosticView],
) -> Counter[str]:
    return Counter(row.view_id for row in rows)
