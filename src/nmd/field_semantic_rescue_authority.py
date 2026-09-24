from __future__ import annotations

from dataclasses import replace

from .typed_domain_generalization_authority import (
    DomainAuthorityCase,
    DomainSpec,
    _expand,
    _generate_domain,
)

SOURCE_T_SEED = 221227
SOURCE_U_SEED = 221231
SOURCE_V_SEED = 221237
SOURCE_W_SEED = 221239
DEV_X_SEED = 222347
CONFIRM_Y_SEED = 223451
CONFIRM_Z_SEED = 224557
GLOBAL_SEED = 1601

DOMAIN_T = DomainSpec(
    domain_id="T",
    workflow="w6h-aerospace-ground-source",
    roles=("ground support unit", "apron sector", "turnaround deviation", "ramp telemetry"),
    pools=(
        _expand(
            ("tow tractor", "ground power cart", "air start unit", "fuel hydrant cart",
             "belt loader", "cabin air unit", "deicing rig", "cargo lift"),
            ("taxiway", "gateward", "rampward"),
        ),
        _expand(
            ("service stand", "baggage lane", "fuel apron", "equipment bay",
             "boarding zone", "cargo pad", "deicing point", "maintenance strip"),
            ("northapron", "midapron", "southapron"),
        ),
        _expand(
            ("power fluctuation", "hose restriction", "drive hesitation", "lift skew",
             "flow interruption", "temperature overshoot", "steering lag", "sensor dropout"),
            ("turnbrief", "turnsteady", "turnrepeat"),
        ),
        _expand(
            ("ramp bus", "power trace", "fuel stream", "lift encoder",
             "drive monitor", "thermal feed", "steering channel", "service telemetry"),
            ("amberramp", "cobaltramp", "pearlramp"),
        ),
    ),
    templates=(
        "w6h-t-ramp-turnaround-card",
        "w6h-t-ground-support-log",
        "w6h-t-apron-service-brief",
    ),
)

DOMAIN_U = DomainSpec(
    domain_id="U",
    workflow="w6h-coldchain-food-source",
    roles=("coldchain machine", "storage segment", "quality excursion", "inspection feed"),
    pools=(
        _expand(
            ("blast chiller", "spiral freezer", "dock cooler", "crate conveyor",
             "defrost pump", "evaporator fan", "pallet shuttle", "seal checker"),
            ("frostline", "icepath", "chillroute"),
        ),
        _expand(
            ("receiving chamber", "freezer aisle", "staging room", "dispatch dock",
             "inspection lane", "buffer zone", "packout cell", "return corridor"),
            ("coldfront", "coldmid", "coldrear"),
        ),
        _expand(
            ("temperature rebound", "frost buildup", "airflow deficit", "seal leakage",
             "defrost delay", "belt hesitation", "humidity rise", "cooling imbalance"),
            ("batchbrief", "batchsteady", "batchrepeat"),
        ),
        _expand(
            ("temperature logger", "humidity stream", "airflow trace", "seal camera",
             "defrost monitor", "belt encoder", "cooling feed", "dock telemetry"),
            ("frozenbus", "qualityline", "coldlink"),
        ),
    ),
    templates=(
        "w6h-u-coldchain-audit",
        "w6h-u-freezer-quality-sheet",
        "w6h-u-dispatch-temperature-note",
    ),
)

DOMAIN_V = DomainSpec(
    domain_id="V",
    workflow="w6h-flood-pump-source",
    roles=("drainage asset", "flood zone", "hydraulic condition", "storm sensing line"),
    pools=(
        _expand(
            ("storm pump", "sluice actuator", "trash screen drive", "sump mixer",
             "gate hoist", "backup generator", "level regulator", "outfall valve"),
            ("estuary", "harbor", "coastward"),
        ),
        _expand(
            ("intake basin", "pump hall", "outfall channel", "barrier chamber",
             "screen bay", "generator room", "drainage tunnel", "gate platform"),
            ("seaward", "midreach", "landward"),
        ),
        _expand(
            ("level surge", "screen blockage", "pump cavitation", "gate drag",
             "flow reversal", "motor heating", "vibration pulse", "discharge loss"),
            ("stormbrief", "stormsteady", "stormrepeat"),
        ),
        _expand(
            ("level radar", "flow recorder", "motor trace", "gate encoder",
             "vibration feed", "rain gauge", "discharge monitor", "storm telemetry"),
            ("tidalbus", "surgebus", "drainlink"),
        ),
    ),
    templates=(
        "w6h-v-flood-control-log",
        "w6h-v-drainage-status-card",
        "w6h-v-storm-pump-brief",
    ),
)

DOMAIN_W = DomainSpec(
    domain_id="W",
    workflow="w6h-datacenter-thermal-source",
    roles=("compute cooling asset", "thermal zone", "facility deviation", "environment stream"),
    pools=(
        _expand(
            ("chilled water pump", "rack fan wall", "cooling tower cell", "air handler",
             "rear door exchanger", "liquid manifold", "economizer damper", "coolant valve"),
            ("serverline", "clusterline", "fabricline"),
        ),
        _expand(
            ("white space row", "mechanical gallery", "cooling plant", "network aisle",
             "power room", "liquid loop bay", "roof plant", "containment zone"),
            ("thermalnorth", "thermalmid", "thermalsouth"),
        ),
        _expand(
            ("coolant warming", "airflow recirculation", "pressure imbalance", "valve hunting",
             "dewpoint rise", "fan slowdown", "loop restriction", "temperature spread"),
            ("heatbrief", "heatsteady", "heatrepeat"),
        ),
        _expand(
            ("rack inlet feed", "coolant trace", "pressure monitor", "fan telemetry",
             "dewpoint channel", "valve feedback", "loop meter", "thermal stream"),
            ("rackbus", "coolingbus", "facilitylink"),
        ),
    ),
    templates=(
        "w6h-w-thermal-operations-note",
        "w6h-w-facility-cooling-card",
        "w6h-w-datacenter-environment-brief",
    ),
)

DOMAIN_X = DomainSpec(
    domain_id="X",
    workflow="w6h-rail-signal-dev",
    roles=("signalling device", "route section", "interlocking anomaly", "wayside diagnostic"),
    pools=(
        _expand(
            ("point machine", "track circuit", "axle counter", "signal lamp",
             "interlocking relay", "balise reader", "crossing actuator", "route controller"),
            ("mainline", "branchline", "yardline"),
        ),
        _expand(
            ("station throat", "junction approach", "platform route", "depot exit",
             "crossing zone", "signal cabin", "switch ladder", "block section"),
            ("routeeast", "routecenter", "routewest"),
        ),
        _expand(
            ("occupancy mismatch", "point detection loss", "lamp current drift", "relay timing lag",
             "counter discrepancy", "route lock delay", "balise read error", "crossing timeout"),
            ("signalbrief", "signalsteady", "signalrepeat"),
        ),
        _expand(
            ("interlocking trace", "track feed", "axle stream", "lamp monitor",
             "relay recorder", "balise telemetry", "crossing channel", "route log"),
            ("signalbus", "trackbus", "routewire"),
        ),
    ),
    templates=(
        "w6h-x-signalling-maintenance-card",
        "w6h-x-route-integrity-log",
        "w6h-x-interlocking-status-note",
    ),
)

DOMAIN_Y = DomainSpec(
    domain_id="Y",
    workflow="w6h-pharma-packaging-confirm",
    roles=("packaging station", "line segment", "packaging defect", "inspection channel"),
    pools=(
        _expand(
            ("blister former", "carton erector", "label applicator", "vision station",
             "checkweigher", "foil sealer", "serialization printer", "bottle capper"),
            ("dosepack", "steripack", "medpack"),
        ),
        _expand(
            ("forming lane", "cartoning cell", "label zone", "vision booth",
             "weigh station", "sealing bay", "serialization aisle", "capping room"),
            ("packfront", "packmid", "packrear"),
        ),
        _expand(
            ("seal wrinkle", "label offset", "weight deviation", "print blur",
             "missing tablet", "carton skew", "code mismatch", "cap torque drift"),
            ("lotbrief", "lotsteady", "lotrepeat"),
        ),
        _expand(
            ("vision feed", "weight trace", "seal monitor", "print camera",
             "tablet counter", "carton sensor", "code verifier", "torque telemetry"),
            ("pharmabus", "packbus", "qualitywire"),
        ),
    ),
    templates=(
        "w6h-y-packaging-batch-record",
        "w6h-y-line-release-check",
        "w6h-y-serialization-quality-note",
    ),
)

DOMAIN_Z = DomainSpec(
    domain_id="Z",
    workflow="w6h-solar-storage-confirm",
    roles=("microgrid asset", "energy sector", "power condition", "control measurement"),
    pools=(
        _expand(
            ("battery inverter", "solar combiner", "dc converter", "storage rack",
             "grid relay", "charge controller", "transformer module", "meter gateway"),
            ("sunfield", "storagefield", "gridfield"),
        ),
        _expand(
            ("array block", "battery room", "converter pad", "relay cabinet",
             "transformer bay", "metering point", "control shelter", "dc corridor"),
            ("energynorth", "energymid", "energysouth"),
        ),
        _expand(
            ("state drift", "voltage ripple", "charge imbalance", "relay chatter",
             "current clipping", "thermal rise", "conversion loss", "frequency wobble"),
            ("powerbrief", "powersteady", "powerrepeat"),
        ),
        _expand(
            ("dc voltage feed", "battery stream", "relay trace", "current monitor",
             "thermal channel", "charge telemetry", "frequency record", "meter link"),
            ("solarbus", "storagebus", "microgridwire"),
        ),
    ),
    templates=(
        "w6h-z-microgrid-operations-card",
        "w6h-z-storage-dispatch-log",
        "w6h-z-solar-control-note",
    ),
)

DOMAINS = {
    spec.domain_id: spec
    for spec in (
        DOMAIN_T,
        DOMAIN_U,
        DOMAIN_V,
        DOMAIN_W,
        DOMAIN_X,
        DOMAIN_Y,
        DOMAIN_Z,
    )
}


def all_w6h_values() -> set[str]:
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


def _criterion_fields(text: str) -> tuple[str, str, str, str]:
    fields = tuple(part.strip() for part in text.split(";"))
    if len(fields) != 4:
        raise RuntimeError("W6h diagnosis criterion must contain four fields")
    return fields  # type: ignore[return-value]


def _ensure_four_one_field_pairs(
    row: DomainAuthorityCase,
    spec: DomainSpec,
) -> DomainAuthorityCase:
    diagnosis = row.typed.decisions[0]
    options = list(diagnosis.options)
    gold_index = int(diagnosis.gold_index)
    gold_fields = _criterion_fields(options[gold_index].criterion_text)

    present: set[int] = set()
    replaceable: list[int] = []
    used_texts = {option.criterion_text for option in options}
    for index, option in enumerate(options):
        if index == gold_index:
            continue
        fields = _criterion_fields(option.criterion_text)
        changed = [
            role
            for role, (left, right) in enumerate(zip(gold_fields, fields))
            if left != right
        ]
        if len(changed) == 1:
            present.add(changed[0])
        else:
            replaceable.append(index)

    missing = [role for role in range(4) if role not in present]
    if len(replaceable) < len(missing):
        raise RuntimeError("W6h lacks replacement budget for semantic pairs")

    for role in missing:
        replacement_text = None
        for value in spec.pools[role]:
            field = f"{spec.roles[role]} {value}"
            if field == gold_fields[role]:
                continue
            candidate_fields = list(gold_fields)
            candidate_fields[role] = field
            candidate = "; ".join(candidate_fields)
            if candidate not in used_texts:
                replacement_text = candidate
                break
        if replacement_text is None:
            raise RuntimeError("W6h could not construct fresh one-field pair")
        option_index = replaceable.pop()
        options[option_index] = replace(
            options[option_index],
            criterion_text=replacement_text,
        )
        used_texts.add(replacement_text)

    diagnosis = replace(diagnosis, options=tuple(options))
    typed = replace(
        row.typed,
        decisions=(diagnosis, *row.typed.decisions[1:]),
    )
    return replace(row, typed=typed)


def _relabel(
    rows: list[DomainAuthorityCase],
    spec: DomainSpec,
) -> list[DomainAuthorityCase]:
    result: list[DomainAuthorityCase] = []
    for row in rows:
        if not row.typed.case_id.startswith("w6d-"):
            raise RuntimeError("unexpected shared-generator case ID")
        typed = replace(
            row.typed,
            case_id="w6h-" + row.typed.case_id[len("w6d-"):],
        )
        relabeled = replace(row, typed=typed)
        result.append(_ensure_four_one_field_pairs(relabeled, spec))
    return result


def _generate(
    spec: DomainSpec,
    *,
    split: str,
    per_k: int,
    seed: int,
) -> list[DomainAuthorityCase]:
    return _relabel(
        _generate_domain(
            spec,
            split=split,
            per_k=per_k,
            seed=seed,
        ),
        spec,
    )


def generate_w6h_train() -> list[DomainAuthorityCase]:
    rows: list[DomainAuthorityCase] = []
    for spec, seed in (
        (DOMAIN_T, SOURCE_T_SEED),
        (DOMAIN_U, SOURCE_U_SEED),
        (DOMAIN_V, SOURCE_V_SEED),
        (DOMAIN_W, SOURCE_W_SEED),
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
        raise RuntimeError("W6h TRAIN must contain 384 states")
    return rows


def generate_w6h_dev() -> list[DomainAuthorityCase]:
    return _generate(
        DOMAIN_X,
        split="dev-x",
        per_k=48,
        seed=DEV_X_SEED,
    )


def generate_w6h_confirm(
    domain_id: str,
    *,
    allow_confirm: bool = False,
) -> list[DomainAuthorityCase]:
    if not allow_confirm:
        raise RuntimeError(
            "W6h CONFIRM-Y/Z is sealed until DEV-X checkpoints freeze"
        )
    if domain_id == "Y":
        spec, seed = DOMAIN_Y, CONFIRM_Y_SEED
    elif domain_id == "Z":
        spec, seed = DOMAIN_Z, CONFIRM_Z_SEED
    else:
        raise ValueError("W6h confirm domain must be Y or Z")
    return _generate(
        spec,
        split=f"confirm-{domain_id.lower()}",
        per_k=48,
        seed=seed,
    )
