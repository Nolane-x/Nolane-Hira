from __future__ import annotations

from dataclasses import replace

from .typed_domain_generalization_authority import (
    DomainAuthorityCase,
    DomainSpec,
    _expand,
    _generate_domain,
)


SOURCE_G_SEED = 191161
SOURCE_H_SEED = 191167
SOURCE_I_SEED = 191173
SOURCE_J_SEED = 191179
DEV_K_SEED = 192283
CONFIRM_L_SEED = 193387
CONFIRM_M_SEED = 194489
PRIMARY_TRAIN_SEED = 1409
REPLICA_TRAIN_SEED = 2411


DOMAIN_G = DomainSpec(
    domain_id="G",
    workflow="w6e-aviation-source",
    roles=("flight actuator", "airframe sector", "airborne condition", "avionics feed"),
    pools=(
        _expand(
            ("flap servo", "cabin compressor", "fuel transfer pump", "landing gear motor",
             "deice valve", "trim actuator", "hydraulic reservoir", "inertial sensor"),
            ("skyward", "cruiseband", "runwayline"),
        ),
        _expand(
            ("wing bay", "avionics rack", "nose compartment", "tail service zone",
             "pressure deck", "gear well", "cabin plant", "flight control bay"),
            ("forwardarc", "midarc", "aftarc"),
        ),
        _expand(
            ("servo lag", "pressure bleed", "fuel pulsation", "sensor bias",
             "valve chatter", "thermal spike", "vibration burst", "timing drift"),
            ("fleetingair", "steadyair", "repeatingair"),
        ),
        _expand(
            ("flight bus", "hydraulic trace", "fuel monitor", "inertial stream",
             "pressure link", "thermal readout", "control telemetry", "vibration feed"),
            ("azurelink", "cloudlink", "sunlink"),
        ),
    ),
    templates=(
        "w6e-g-flight-maintenance-note",
        "w6e-g-airworthiness-brief",
        "w6e-g-avionics-status-card",
    ),
)

DOMAIN_H = DomainSpec(
    domain_id="H",
    workflow="w6e-geothermal-source",
    roles=("geothermal module", "wellfield sector", "reservoir deviation", "plant sensing line"),
    pools=(
        _expand(
            ("brine pump", "steam separator", "injection valve", "turbine governor",
             "condensate drive", "flash vessel", "wellhead choke", "cooling fan"),
            ("basaltic", "igneous", "mantleward"),
        ),
        _expand(
            ("production pad", "separator deck", "reinjection lane", "turbine hall",
             "condenser bay", "well cellar", "steam corridor", "cooling terrace"),
            ("hotfield", "warmfield", "coolfield"),
        ),
        _expand(
            ("brine scaling", "steam carryover", "pressure drawdown", "valve erosion",
             "silica buildup", "temperature swing", "flow choking", "vapor instability"),
            ("geopulse", "geosteady", "geocycle"),
        ),
        _expand(
            ("wellhead trace", "steam meter", "brine monitor", "pressure recorder",
             "temperature string", "flow telemetry", "valve feedback", "turbine feed"),
            ("obsidianbus", "pumicebus", "quartzbus"),
        ),
    ),
    templates=(
        "w6e-h-wellfield-log",
        "w6e-h-steam-cycle-sheet",
        "w6e-h-reservoir-operations-note",
    ),
)

DOMAIN_I = DomainSpec(
    domain_id="I",
    workflow="w6e-semiconductor-source",
    roles=("fab tool", "cleanroom sector", "process excursion", "metrology stream"),
    pools=(
        _expand(
            ("etch chamber", "wafer handler", "deposition source", "vacuum stage",
             "photo track", "plasma generator", "gas manifold", "inspection scanner"),
            ("siliconlane", "oxideplane", "copperline"),
        ),
        _expand(
            ("lithography bay", "etch cell", "deposition aisle", "metrology room",
             "vacuum service zone", "chemical cabinet", "wafer transfer lane", "utility chase"),
            ("cleanfront", "cleanmid", "cleanrear"),
        ),
        _expand(
            ("plasma nonuniformity", "particle excursion", "vacuum creep", "dose offset",
             "wafer slip", "gas imbalance", "temperature gradient", "alignment shift"),
            ("microburst", "microsteady", "microrepeat"),
        ),
        _expand(
            ("endpoint trace", "vacuum gauge", "particle counter", "dose monitor",
             "wafer telemetry", "gas readout", "thermal channel", "alignment feed"),
            ("silicabus", "maskbus", "fablink"),
        ),
    ),
    templates=(
        "w6e-i-fab-excursion-record",
        "w6e-i-cleanroom-process-brief",
        "w6e-i-metrology-status-sheet",
    ),
)

DOMAIN_J = DomainSpec(
    domain_id="J",
    workflow="w6e-food-process-source",
    roles=("processing machine", "production zone", "line quality issue", "hygiene monitor"),
    pools=(
        _expand(
            ("pasteurizer pump", "filling carousel", "mixing agitator", "conveyor drive",
             "cooling tunnel", "dosing valve", "packaging sealer", "wash skid"),
            ("batchalpha", "batchbeta", "batchgamma"),
        ),
        _expand(
            ("mixing room", "filling lane", "packaging cell", "cold corridor",
             "wash station", "ingredient deck", "inspection booth", "dispatch line"),
            ("sanitaryfront", "sanitarymid", "sanitaryrear"),
        ),
        _expand(
            ("fill variation", "seal weakness", "temperature drift", "flow shortage",
             "mix inconsistency", "rinse residue", "belt slip", "dose deviation"),
            ("lotbrief", "lotsteady", "lotcycle"),
        ),
        _expand(
            ("fill sensor", "seal camera", "temperature probe", "flow meter",
             "mix monitor", "rinse analyzer", "belt encoder", "dose telemetry"),
            ("hygienebus", "batchbus", "qualitybus"),
        ),
    ),
    templates=(
        "w6e-j-production-quality-card",
        "w6e-j-sanitation-process-note",
        "w6e-j-line-assurance-brief",
    ),
)

DOMAIN_K = DomainSpec(
    domain_id="K",
    workflow="w6e-grid-heldout-dev",
    roles=("grid asset", "distribution sector", "electrical disturbance", "protection channel"),
    pools=(
        _expand(
            ("feeder breaker", "tap changer", "bus coupler", "reactive bank",
             "recloser drive", "meter transformer", "substation relay", "cable monitor"),
            ("voltarc", "amperearc", "phaseline"),
        ),
        _expand(
            ("substation yard", "feeder corridor", "switch room", "transformer bay",
             "metering deck", "relay house", "cable tunnel", "distribution node"),
            ("gridnorth", "gridcenter", "gridsouth"),
        ),
        _expand(
            ("voltage notch", "phase imbalance", "relay delay", "harmonic surge",
             "current drift", "contact heating", "frequency dip", "insulation warning"),
            ("gridflash", "gridsteady", "gridrepeat"),
        ),
        _expand(
            ("phasor stream", "breaker trace", "meter feed", "relay telemetry",
             "current channel", "voltage monitor", "frequency record", "thermal link"),
            ("circuitbus", "relaybus", "powerlink"),
        ),
    ),
    templates=(
        "w6e-k-grid-inspection-panel",
        "w6e-k-distribution-event-sheet",
        "w6e-k-protection-status-note",
    ),
)

DOMAIN_L = DomainSpec(
    domain_id="L",
    workflow="w6e-mining-heldout-confirm-l",
    roles=("mine machine", "extraction sector", "underground condition", "safety telemetry"),
    pools=(
        _expand(
            ("haul drive", "roof bolter", "ventilation fan", "crusher motor",
             "dewatering pump", "conveyor gearbox", "drill head", "skip hoist"),
            ("oreline", "rockline", "shaftline"),
        ),
        _expand(
            ("haulage drift", "crusher chamber", "ventilation raise", "pump sump",
             "loading pocket", "drill heading", "service crosscut", "shaft station"),
            ("deepfront", "deepmid", "deeprear"),
        ),
        _expand(
            ("roof movement", "motor overload", "airflow loss", "water ingress",
             "gear vibration", "belt misalignment", "drill chatter", "hoist slip"),
            ("minebrief", "minesteady", "minecycle"),
        ),
        _expand(
            ("airflow sensor", "load trace", "water monitor", "vibration feed",
             "belt telemetry", "roof gauge", "hoist encoder", "motor channel"),
            ("orebus", "shaftbus", "safetybus"),
        ),
    ),
    templates=(
        "w6e-l-underground-safety-card",
        "w6e-l-extraction-equipment-log",
        "w6e-l-mine-assurance-sheet",
    ),
)

DOMAIN_M = DomainSpec(
    domain_id="M",
    workflow="w6e-datacenter-heldout-confirm-m",
    roles=("compute facility unit", "hall sector", "infrastructure incident", "operations telemetry"),
    pools=(
        _expand(
            ("rack power shelf", "cooling manifold", "ups inverter", "network spine",
             "battery string", "air handler", "busway tap", "water loop pump"),
            ("clusterlane", "fabriclane", "powerlane"),
        ),
        _expand(
            ("compute hall", "power room", "cooling gallery", "network row",
             "battery suite", "mechanical mezzanine", "busway corridor", "plant enclosure"),
            ("datakeepfront", "datakeepmid", "datakeeprear"),
        ),
        _expand(
            ("thermal hotspot", "power ripple", "coolant loss", "link congestion",
             "battery sag", "airflow imbalance", "busway heating", "pump oscillation"),
            ("facilityburst", "facilitysteady", "facilitycycle"),
        ),
        _expand(
            ("rack telemetry", "cooling trace", "ups monitor", "fabric counter",
             "battery feed", "airflow channel", "busway sensor", "pump readout"),
            ("databus", "facilitybus", "opslink"),
        ),
    ),
    templates=(
        "w6e-m-facility-operations-card",
        "w6e-m-datacenter-assurance-note",
        "w6e-m-infrastructure-status-sheet",
    ),
)


DOMAINS = {
    spec.domain_id: spec
    for spec in (
        DOMAIN_G,
        DOMAIN_H,
        DOMAIN_I,
        DOMAIN_J,
        DOMAIN_K,
        DOMAIN_L,
        DOMAIN_M,
    )
}


def all_w6e_values() -> set[str]:
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


def _relabel_w6e(
    rows: list[DomainAuthorityCase],
) -> list[DomainAuthorityCase]:
    result: list[DomainAuthorityCase] = []
    for row in rows:
        case_id = row.typed.case_id
        if not case_id.startswith("w6d-"):
            raise RuntimeError("unexpected shared-generator case id")
        typed = replace(
            row.typed,
            case_id="w6e-" + case_id[len("w6d-"):],
        )
        result.append(replace(row, typed=typed))
    return result


def _generate_w6e_domain(
    spec: DomainSpec,
    *,
    split: str,
    per_k: int,
    seed: int,
) -> list[DomainAuthorityCase]:
    return _relabel_w6e(
        _generate_domain(
            spec,
            split=split,
            per_k=per_k,
            seed=seed,
        )
    )


def generate_w6e_multi_train() -> list[DomainAuthorityCase]:
    rows: list[DomainAuthorityCase] = []
    for spec, seed in (
        (DOMAIN_G, SOURCE_G_SEED),
        (DOMAIN_H, SOURCE_H_SEED),
        (DOMAIN_I, SOURCE_I_SEED),
        (DOMAIN_J, SOURCE_J_SEED),
    ):
        rows.extend(
            _generate_w6e_domain(
                spec,
                split="train-multi",
                per_k=24,
                seed=seed,
            )
        )
    return rows


def generate_w6e_dev() -> list[DomainAuthorityCase]:
    return _generate_w6e_domain(
        DOMAIN_K,
        split="dev-k",
        per_k=48,
        seed=DEV_K_SEED,
    )


def generate_w6e_confirm(
    domain_id: str,
    *,
    allow_confirm: bool = False,
) -> list[DomainAuthorityCase]:
    if not allow_confirm:
        raise RuntimeError(
            "W6e CONFIRM-L/M authority is sealed until post-freeze evaluation"
        )
    if domain_id == "L":
        spec, seed = DOMAIN_L, CONFIRM_L_SEED
    elif domain_id == "M":
        spec, seed = DOMAIN_M, CONFIRM_M_SEED
    else:
        raise ValueError("W6e confirm domain must be L or M")
    return _generate_w6e_domain(
        spec,
        split=f"confirm-{domain_id.lower()}",
        per_k=48,
        seed=seed,
    )
