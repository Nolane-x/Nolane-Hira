from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import random

from .contracts import LogicalOption
from .typed_decisions import TypedDecisionCase
from .typed_domain_generalization_authority import (
    DomainAuthorityCase,
    DomainSpec,
    _expand,
    _generate_domain,
    _signature_text,
)
from .typed_reliability_authority import (
    CONFIDENCE_MASS,
    _categorical_distribution,
)


SOURCE_AG_SEED = 251347
SOURCE_AH_SEED = 251353
SOURCE_AI_SEED = 251359
SOURCE_AJ_SEED = 251363
DEV_AK_SEED = 252461
CONFIRM_AL_SEED = 253567
CONFIRM_AM_SEED = 254671

FREEFORM_SEED = 1901
CONJUNCTIVE_PRIMARY_SEED = 1907
CONJUNCTIVE_REPLICA_SEED = 1913

K_VALUES = (8, 16, 32, 64)
ROLE_COUNT = 4


DOMAIN_AG = DomainSpec(
    domain_id="AG",
    workflow="w7-orbital-power-source",
    roles=(
        "orbital power assembly",
        "spacecraft distribution sector",
        "electrical service deviation",
        "power diagnostic feed",
    ),
    pools=(
        _expand(
            (
                "bus tie module",
                "array shunt regulator",
                "battery interface unit",
                "load switch matrix",
                "power conditioning stage",
                "distribution contactor",
                "charge routing module",
                "converter isolation unit",
            ),
            ("orbitprime", "orbitvector", "orbitzenith"),
        ),
        _expand(
            (
                "service truss",
                "avionics shelf",
                "battery compartment",
                "payload power bay",
                "distribution rack",
                "converter enclosure",
                "array interface zone",
                "utility deck",
            ),
            ("sunward", "crossorbital", "nadirward"),
        ),
        _expand(
            (
                "current sharing drift",
                "bus voltage sag",
                "contactor timing slip",
                "charge routing loss",
                "converter ripple growth",
                "isolation resistance drop",
                "load transfer delay",
                "regulator oscillation",
            ),
            ("cyclebrief", "cyclesteady", "cyclerepeat"),
        ),
        _expand(
            (
                "bus current recorder",
                "distribution voltage trace",
                "contactor status stream",
                "converter ripple monitor",
                "battery interface telemetry",
                "isolation test channel",
                "load transfer log",
                "array regulation feed",
            ),
            ("orbitbus", "powervector", "zenithwire"),
        ),
    ),
    templates=(
        "w7-ag-orbital-power-card",
        "w7-ag-distribution-service-log",
        "w7-ag-spacecraft-electrical-brief",
        "w7-ag-power-routing-checksheet",
    ),
)


DOMAIN_AH = DomainSpec(
    domain_id="AH",
    workflow="w7-cleanroom-control-source",
    roles=(
        "cleanroom process asset",
        "sterile control sector",
        "environmental process deviation",
        "cleanroom verification feed",
    ),
    pools=(
        _expand(
            (
                "laminar flow canopy",
                "airlock pressure module",
                "sterile transfer isolator",
                "hepa fan terminal",
                "humidity conditioning skid",
                "particle sampling manifold",
                "gowning air handler",
                "aseptic passbox actuator",
            ),
            ("steriprime", "sterivector", "sterizenith"),
        ),
        _expand(
            (
                "aseptic filling cell",
                "material airlock",
                "gowning corridor",
                "sterile compounding room",
                "isolator service bay",
                "clean utility alcove",
                "sampling vestibule",
                "transfer suite",
            ),
            ("cleanfront", "cleanmid", "cleanrear"),
        ),
        _expand(
            (
                "pressure cascade drift",
                "particle count rise",
                "humidity control lag",
                "airflow uniformity loss",
                "filter loading increase",
                "door interlock delay",
                "temperature band excursion",
                "sampling flow instability",
            ),
            ("asepticbrief", "asepticsteady", "asepticrepeat"),
        ),
        _expand(
            (
                "differential pressure record",
                "particle counter stream",
                "humidity verification trace",
                "air velocity monitor",
                "filter loading telemetry",
                "interlock status channel",
                "temperature qualification feed",
                "sampling flow log",
            ),
            ("cleanbus", "sterilelink", "asepticwire"),
        ),
    ),
    templates=(
        "w7-ah-cleanroom-control-card",
        "w7-ah-aseptic-environment-ledger",
        "w7-ah-sterile-process-brief",
        "w7-ah-clean-zone-checksheet",
    ),
)


DOMAIN_AI = DomainSpec(
    domain_id="AI",
    workflow="w7-railyard-inspection-source",
    roles=(
        "rail-yard inspection asset",
        "yard movement sector",
        "rolling-stock inspection deviation",
        "yard diagnostic feed",
    ),
    pools=(
        _expand(
            (
                "wheel profile scanner",
                "brake temperature portal",
                "coupler vision rig",
                "axle acoustic station",
                "wagon identity reader",
                "clearance laser frame",
                "bearing inspection node",
                "switch approach camera",
            ),
            ("yardprime", "yardvector", "yardzenith"),
        ),
        _expand(
            (
                "arrival lead",
                "classification track",
                "departure ladder",
                "inspection bypass",
                "maintenance siding",
                "transfer throat",
                "wagon staging line",
                "switching corridor",
            ),
            ("yardnorth", "yardcenter", "yardsouth"),
        ),
        _expand(
            (
                "wheel flange deviation",
                "brake heat excess",
                "coupler alignment shift",
                "axle acoustic spike",
                "identity read mismatch",
                "clearance envelope breach",
                "bearing signature change",
                "switch approach obstruction",
            ),
            ("inspectbrief", "inspectsteady", "inspectrepeat"),
        ),
        _expand(
            (
                "profile measurement stream",
                "thermal portal trace",
                "coupler vision feed",
                "acoustic bearing channel",
                "wagon identity log",
                "clearance lidar record",
                "bearing spectrum telemetry",
                "approach camera bus",
            ),
            ("yardbus", "inspectlink", "raildata"),
        ),
    ),
    templates=(
        "w7-ai-railyard-inspection-card",
        "w7-ai-wagon-condition-ledger",
        "w7-ai-yard-safety-brief",
        "w7-ai-rolling-stock-checksheet",
    ),
)


DOMAIN_AJ = DomainSpec(
    domain_id="AJ",
    workflow="w7-deepocean-service-source",
    roles=(
        "deep-ocean sensing asset",
        "subsea observation sector",
        "oceanographic instrument deviation",
        "subsea diagnostic feed",
    ),
    pools=(
        _expand(
            (
                "pressure observatory pod",
                "acoustic modem node",
                "chemical sensing package",
                "current profiler frame",
                "seafloor camera unit",
                "oxygen sensor rack",
                "temperature chain module",
                "sediment sampling controller",
            ),
            ("abyssprime", "abyssvector", "abysszenith"),
        ),
        _expand(
            (
                "seafloor junction",
                "hydrothermal transect",
                "canyon observation line",
                "mooring service zone",
                "benthic survey grid",
                "subsea cable spur",
                "instrument landing site",
                "deep current station",
            ),
            ("deepnorth", "deepcenter", "deepsouth"),
        ),
        _expand(
            (
                "pressure offset drift",
                "acoustic link fading",
                "chemical baseline shift",
                "current velocity dropout",
                "camera illumination loss",
                "oxygen response lag",
                "temperature chain break",
                "sampler actuation delay",
            ),
            ("divebrief", "divesteady", "diverepeat"),
        ),
        _expand(
            (
                "pressure telemetry stream",
                "acoustic modem trace",
                "chemical sensor record",
                "current profiler feed",
                "camera health channel",
                "oxygen measurement log",
                "temperature chain telemetry",
                "sampler controller bus",
            ),
            ("abyssbus", "subsealink", "deepwire"),
        ),
    ),
    templates=(
        "w7-aj-deepocean-service-card",
        "w7-aj-subsea-observation-ledger",
        "w7-aj-ocean-instrument-brief",
        "w7-aj-abyssal-maintenance-checksheet",
    ),
)


DOMAIN_AK = DomainSpec(
    domain_id="AK",
    workflow="w7-highaltitude-weather-dev",
    roles=(
        "upper-air instrument asset",
        "high-altitude observation sector",
        "atmospheric instrument deviation",
        "upper-air diagnostic feed",
    ),
    pools=(
        _expand(
            (
                "radiosonde launch controller",
                "ozone sounding module",
                "wind lidar terminal",
                "ceilometer optics unit",
                "balloon inflation manifold",
                "upper-air receiver",
                "humidity calibration rack",
                "pressure reference unit",
            ),
            ("stratoprime", "stratovector", "stratozenith"),
        ),
        _expand(
            (
                "launch shelter",
                "upper-air platform",
                "optics enclosure",
                "receiver gallery",
                "calibration chamber",
                "balloon preparation bay",
                "meteorological mast zone",
                "reference instrument room",
            ),
            ("skyfront", "skymid", "skyrear"),
        ),
        _expand(
            (
                "pressure reference drift",
                "humidity response delay",
                "wind retrieval dropout",
                "optical contamination rise",
                "inflation flow instability",
                "receiver frequency offset",
                "temperature calibration shift",
                "ozone pump slowdown",
            ),
            ("weatherbrief", "weathersteady", "weatherrepeat"),
        ),
        _expand(
            (
                "radiosonde telemetry bus",
                "ozone sounding trace",
                "wind lidar stream",
                "ceilometer diagnostic feed",
                "inflation flow record",
                "receiver spectrum channel",
                "humidity calibration log",
                "pressure reference telemetry",
            ),
            ("skybus", "upperairlink", "stratowire"),
        ),
    ),
    templates=(
        "w7-ak-upperair-calibration-card",
        "w7-ak-weather-instrument-ledger",
        "w7-ak-stratosphere-observation-brief",
        "w7-ak-sounding-system-checksheet",
    ),
)


DOMAIN_AL = DomainSpec(
    domain_id="AL",
    workflow="w7-microgrid-thermal-confirm",
    roles=(
        "distributed energy thermal asset",
        "microgrid cooling sector",
        "energy-thermal deviation",
        "microgrid thermal diagnostic feed",
    ),
    pools=(
        _expand(
            (
                "battery cooling manifold",
                "inverter cold plate",
                "transformer ventilation unit",
                "storage loop pump",
                "converter heat exchanger",
                "relay cabinet fan bank",
                "thermal buffer vessel",
                "coolant distribution valve",
            ),
            ("gridprime", "gridvector", "gridzenith"),
        ),
        _expand(
            (
                "storage enclosure",
                "inverter gallery",
                "transformer pad",
                "converter shelter",
                "relay equipment room",
                "coolant service bay",
                "thermal buffer zone",
                "distribution skid area",
            ),
            ("gridnorth", "gridcenter", "gridsouth"),
        ),
        _expand(
            (
                "coolant temperature rise",
                "flow balance drift",
                "cold plate restriction",
                "fan delivery loss",
                "heat exchanger fouling",
                "loop pressure decay",
                "buffer stratification shift",
                "valve response lag",
            ),
            ("thermalbrief", "thermalsteady", "thermalrepeat"),
        ),
        _expand(
            (
                "coolant temperature record",
                "loop flow telemetry",
                "cold plate pressure trace",
                "fan delivery monitor",
                "heat exchanger delta feed",
                "loop pressure channel",
                "buffer temperature profile",
                "valve position log",
            ),
            ("gridthermalbus", "coolingvector", "energylink"),
        ),
    ),
    templates=(
        "w7-al-microgrid-thermal-card",
        "w7-al-energy-cooling-ledger",
        "w7-al-storage-thermal-brief",
        "w7-al-distributed-energy-checksheet",
    ),
)


DOMAIN_AM = DomainSpec(
    domain_id="AM",
    workflow="w7-foodrobot-safety-confirm",
    roles=(
        "food-packaging robotic asset",
        "automated packaging safety sector",
        "robotic process deviation",
        "packaging safety diagnostic feed",
    ),
    pools=(
        _expand(
            (
                "pick-and-place arm",
                "case packing robot",
                "tray loading gantry",
                "palletizing manipulator",
                "seal inspection cobot",
                "carton transfer robot",
                "vision-guided picker",
                "reject handling actuator",
            ),
            ("foodprime", "foodvector", "foodzenith"),
        ),
        _expand(
            (
                "primary packing cell",
                "tray loading zone",
                "carton transfer lane",
                "palletizing enclosure",
                "seal inspection station",
                "robot service aisle",
                "vision picking area",
                "reject handling bay",
            ),
            ("foodfront", "foodmid", "foodrear"),
        ),
        _expand(
            (
                "gripper force drift",
                "guard response delay",
                "trajectory offset growth",
                "vision alignment loss",
                "transfer timing slip",
                "seal handling collision",
                "payload position mismatch",
                "reject routing hesitation",
            ),
            ("safetybrief", "safetysteady", "safetyrepeat"),
        ),
        _expand(
            (
                "robot force telemetry",
                "guard status trace",
                "trajectory controller log",
                "vision alignment feed",
                "transfer timing channel",
                "collision monitor stream",
                "payload position record",
                "reject routing telemetry",
            ),
            ("foodsafetybus", "robotlink", "packwire"),
        ),
    ),
    templates=(
        "w7-am-robotic-packaging-card",
        "w7-am-food-safety-ledger",
        "w7-am-automated-packaging-brief",
        "w7-am-robot-cell-checksheet",
    ),
)


DOMAINS = {
    spec.domain_id: spec
    for spec in (
        DOMAIN_AG,
        DOMAIN_AH,
        DOMAIN_AI,
        DOMAIN_AJ,
        DOMAIN_AK,
        DOMAIN_AL,
        DOMAIN_AM,
    )
}


@dataclass(frozen=True)
class ConjunctiveAuthorityCase:
    typed: TypedDecisionCase
    split: str
    domain_id: str
    template_id: str
    diagnosis_k: int
    severity: int
    confidence: str
    diagnosis_signatures: tuple[
        tuple[str, str, str, str], ...
    ]
    diagnosis_factors: tuple[
        tuple[str, str, str, str], ...
    ]


def all_w7_values() -> set[str]:
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


def _criterion_signature(
    spec: DomainSpec,
    text: str,
) -> tuple[str, str, str, str]:
    fields = tuple(part.strip() for part in text.split(";"))
    if len(fields) != 4:
        raise RuntimeError("W7 diagnosis criterion must contain four fields")
    values: list[str] = []
    for role, field in zip(spec.roles, fields):
        prefix = role + " "
        if not field.startswith(prefix):
            raise RuntimeError("W7 criterion role/value identity mismatch")
        value = field[len(prefix):]
        if value not in spec.pools[len(values)]:
            raise RuntimeError("W7 criterion value outside domain pool")
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def _candidate_with_roles(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    roles: tuple[int, ...],
    rng: random.Random,
    forbidden: set[tuple[str, str, str, str]],
) -> tuple[str, str, str, str]:
    attempts = 0
    while attempts < 100000:
        attempts += 1
        row = list(target)
        for role in roles:
            alternatives = [
                value
                for value in spec.pools[role]
                if value != target[role]
            ]
            row[role] = rng.choice(alternatives)
        candidate = tuple(row)
        if candidate in forbidden:
            continue
        forbidden.add(candidate)
        return candidate  # type: ignore[return-value]
    raise RuntimeError("W7 controlled-negative search exhausted")


def _distance_rows(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    *,
    distance: int,
    count: int,
    rng: random.Random,
    forbidden: set[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    if distance == 1:
        role_sets = ((0,), (1,), (2,), (3,))
    elif distance == 2:
        role_sets = (
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3),
        )
    elif distance == 3:
        role_sets = (
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        )
    elif distance == 4:
        role_sets = ((0, 1, 2, 3),)
    else:
        raise ValueError("W7 semantic distance must be 1..4")

    rows = []
    cursor = 0
    while len(rows) < count:
        rows.append(
            _candidate_with_roles(
                target,
                spec,
                role_sets[cursor % len(role_sets)],
                rng,
                forbidden,
            )
        )
        cursor += 1
    return rows


def _controlled_signatures(
    target: tuple[str, str, str, str],
    spec: DomainSpec,
    *,
    k: int,
    rng: random.Random,
) -> list[tuple[str, str, str, str]]:
    compositions = {
        8: (4, 3, 0, 0),
        16: (8, 5, 2, 0),
        32: (12, 12, 7, 0),
        64: (12, 20, 15, 16),
    }
    if k not in compositions:
        raise ValueError("W7 diagnosis K must be 8/16/32/64")
    counts = compositions[k]
    forbidden = {target}
    rows = [target]
    for distance, count in enumerate(counts, start=1):
        rows.extend(
            _distance_rows(
                target,
                spec,
                distance=distance,
                count=count,
                rng=rng,
                forbidden=forbidden,
            )
        )
    if len(rows) != k or len(set(rows)) != k:
        raise RuntimeError("W7 controlled diagnosis set mismatch")
    rng.shuffle(rows)
    return rows


def _rebuild_case(
    row: DomainAuthorityCase,
    spec: DomainSpec,
    *,
    seed: int,
    case_index: int,
) -> ConjunctiveAuthorityCase:
    diagnosis = row.typed.decisions[0]
    original_gold = diagnosis.options[int(diagnosis.gold_index)]
    target = _criterion_signature(spec, original_gold.criterion_text)

    rng = random.Random(
        seed
        + 1_000_003
        + case_index * 1009
        + int(row.diagnosis_k) * 17
    )
    signatures = _controlled_signatures(
        target,
        spec,
        k=row.diagnosis_k,
        rng=rng,
    )
    gold_index = signatures.index(target)
    options = tuple(
        LogicalOption(
            option_id=(
                f"{spec.domain_id.lower()}-w7-"
                f"{case_index:04d}-{index:03d}"
            ),
            criterion_text=_signature_text(spec, signature),
        )
        for index, signature in enumerate(signatures)
    )
    gold_probabilities = _categorical_distribution(
        row.diagnosis_k,
        gold_index,
        CONFIDENCE_MASS[row.confidence],
    )
    diagnosis = replace(
        diagnosis,
        options=options,
        gold_index=gold_index,
        gold_probabilities=gold_probabilities,
    )
    typed = replace(
        row.typed,
        case_id=(
            "w7-"
            + row.typed.case_id[len("w6d-"):]
            if row.typed.case_id.startswith("w6d-")
            else row.typed.case_id
        ),
        decisions=(diagnosis, *row.typed.decisions[1:]),
    )

    factors = tuple(
        tuple(
            f"{role} {value}"
            for role, value in zip(spec.roles, signature)
        )
        for signature in signatures
    )
    for option, factor_row in zip(options, factors):
        if option.criterion_text != "; ".join(factor_row):
            raise RuntimeError("W7 free-form/factor identity mismatch")

    return ConjunctiveAuthorityCase(
        typed=typed,
        split=row.split,
        domain_id=row.domain_id,
        template_id=row.template_id,
        diagnosis_k=row.diagnosis_k,
        severity=row.severity,
        confidence=row.confidence,
        diagnosis_signatures=tuple(signatures),
        diagnosis_factors=factors,  # type: ignore[arg-type]
    )


def _generate(
    spec: DomainSpec,
    *,
    split: str,
    per_k: int,
    seed: int,
) -> list[ConjunctiveAuthorityCase]:
    base = _generate_domain(
        spec,
        split=split,
        per_k=per_k,
        seed=seed,
    )
    result = [
        _rebuild_case(
            row,
            spec,
            seed=seed,
            case_index=index,
        )
        for index, row in enumerate(base)
    ]
    return result


def generate_w7_train() -> list[ConjunctiveAuthorityCase]:
    rows: list[ConjunctiveAuthorityCase] = []
    for spec, seed in (
        (DOMAIN_AG, SOURCE_AG_SEED),
        (DOMAIN_AH, SOURCE_AH_SEED),
        (DOMAIN_AI, SOURCE_AI_SEED),
        (DOMAIN_AJ, SOURCE_AJ_SEED),
    ):
        rows.extend(
            _generate(
                spec,
                split="train",
                per_k=24,
                seed=seed,
            )
        )
    if len(rows) != 384:
        raise RuntimeError("W7 TRAIN must contain 384 states")
    return rows


def generate_w7_dev() -> list[ConjunctiveAuthorityCase]:
    rows = _generate(
        DOMAIN_AK,
        split="dev-ak",
        per_k=48,
        seed=DEV_AK_SEED,
    )
    if len(rows) != 192:
        raise RuntimeError("W7 DEV-AK must contain 192 states")
    return rows


def generate_w7_confirm(
    domain_id: str,
    *,
    allow_confirm: bool = False,
) -> list[ConjunctiveAuthorityCase]:
    if not allow_confirm:
        raise RuntimeError(
            "W7 CONFIRM-AL/AM is sealed until all DEV-AK checkpoints freeze"
        )
    if domain_id == "AL":
        spec, seed = DOMAIN_AL, CONFIRM_AL_SEED
    elif domain_id == "AM":
        spec, seed = DOMAIN_AM, CONFIRM_AM_SEED
    else:
        raise ValueError("W7 confirm domain must be AL or AM")
    rows = _generate(
        spec,
        split=f"confirm-{domain_id.lower()}",
        per_k=48,
        seed=seed,
    )
    if len(rows) != 192:
        raise RuntimeError("W7 CONFIRM domain must contain 192 states")
    return rows


def diagnosis_distance_histogram(
    case: ConjunctiveAuthorityCase,
) -> dict[int, int]:
    gold = case.diagnosis_signatures[
        int(case.typed.decisions[0].gold_index)
    ]
    return dict(
        Counter(
            signature_distance(gold, signature)
            for signature in case.diagnosis_signatures
        )
    )
