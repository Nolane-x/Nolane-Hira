from __future__ import annotations

from dataclasses import replace

from .conjunctive_authority import (
    ConjunctiveAuthorityCase,
    _expand,
    _generate,
    diagnosis_distance_histogram,
)
from .contracts import LogicalOption
from .typed_domain_generalization_authority import DomainSpec


SOURCE_AN_SEED = 261401
SOURCE_AO_SEED = 261407
SOURCE_AP_SEED = 261419
SOURCE_AQ_SEED = 261427
DEV_AR_SEED = 262531
CONFIRM_AS_SEED = 263641
CONFIRM_AT_SEED = 264749

TYPED_ONLY_SEED = 2003
PAIR_ONLY_SEED = 2011
TYPED_PLUS_PAIR_PRIMARY_SEED = 2017
TYPED_PLUS_PAIR_REPLICA_SEED = 2027


def _domain(domain_id: str, workflow: str, topic: str, roles, roots, suffixes, templates):
    pools = tuple(
        _expand(base, suffix)
        for base, suffix in zip(roots, suffixes)
    )
    return DomainSpec(
        domain_id=domain_id,
        workflow=workflow,
        roles=tuple(roles),
        pools=pools,
        templates=tuple(templates),
    )


DOMAIN_AN = _domain(
    "AN",
    "w7b-orbital-coolant-source",
    "orbital coolant",
    (
        "orbital coolant asset",
        "thermal routing sector",
        "coolant service deviation",
        "thermal diagnostic feed",
    ),
    (
        ("loop accumulator", "radiator bypass valve", "coolant pump module", "coldplate manifold", "thermal buffer tank", "flow balancing unit", "heat rejection valve", "coolant filter stage"),
        ("service truss", "thermal rack", "radiator bay", "payload deck", "pump enclosure", "coolant utility zone", "thermal interface shelf", "loop service alcove"),
        ("flow imbalance", "temperature overshoot", "pump cavitation", "valve timing slip", "pressure decay", "filter restriction", "thermal lag", "bypass instability"),
        ("loop flow telemetry", "coolant pressure trace", "valve position stream", "pump vibration feed", "coldplate temperature log", "radiator delta channel", "buffer level record", "filter pressure bus"),
    ),
    (
        ("anprime", "anvector", "anzenith"),
        ("anfront", "anmid", "anrear"),
        ("anbrief", "ansteady", "anrepeat"),
        ("anbus", "anlink", "anwire"),
    ),
    (
        "w7b-an-orbital-coolant-card",
        "w7b-an-thermal-routing-ledger",
        "w7b-an-space-service-brief",
        "w7b-an-coolant-checksheet",
    ),
)

DOMAIN_AO = _domain(
    "AO",
    "w7b-sterile-packaging-source",
    "sterile packaging",
    (
        "sterile packaging asset",
        "aseptic logistics sector",
        "packaging process deviation",
        "sterile diagnostic feed",
    ),
    (
        ("pouch sealing station", "tray denest module", "sterile transfer gate", "label verification unit", "barrier film feeder", "carton loading cell", "aseptic conveyor drive", "package inspection head"),
        ("sterile staging lane", "seal room", "transfer airlock", "labeling zone", "film handling bay", "carton cell", "inspection alcove", "clean logistics corridor"),
        ("seal temperature drift", "tray feed hesitation", "transfer delay", "label read mismatch", "film tension rise", "carton timing slip", "conveyor speed ripple", "inspection contrast loss"),
        ("seal temperature record", "tray feed trace", "transfer status channel", "label vision stream", "film tension telemetry", "carton timing log", "conveyor drive bus", "inspection image feed"),
    ),
    (
        ("aoprime", "aovector", "aozenith"),
        ("aofront", "aomid", "aorear"),
        ("aobrief", "aosteady", "aorepeat"),
        ("aobus", "aolink", "aowire"),
    ),
    (
        "w7b-ao-sterile-package-card",
        "w7b-ao-aseptic-logistics-ledger",
        "w7b-ao-packaging-process-brief",
        "w7b-ao-sterile-line-checksheet",
    ),
)

DOMAIN_AP = _domain(
    "AP",
    "w7b-bridge-inspection-source",
    "bridge inspection",
    (
        "bridge inspection asset",
        "structural survey sector",
        "bridge condition deviation",
        "inspection diagnostic feed",
    ),
    (
        ("deck imaging rover", "bearing acoustic node", "cable vision mast", "pier lidar unit", "joint displacement sensor", "surface crack scanner", "drainage survey camera", "girder vibration logger"),
        ("north approach", "main span", "south approach", "pier gallery", "bearing seat", "deck service lane", "cable anchorage", "underside inspection zone"),
        ("crack width growth", "bearing noise spike", "cable surface defect", "pier alignment shift", "joint movement excess", "drainage blockage", "girder vibration change", "deck delamination signal"),
        ("deck image stream", "bearing acoustic trace", "cable vision feed", "pier lidar record", "joint displacement log", "crack scan channel", "drainage image bus", "girder spectrum telemetry"),
    ),
    (
        ("apprime", "apvector", "apzenith"),
        ("apnorth", "apcenter", "apsouth"),
        ("apbrief", "apsteady", "aprepeat"),
        ("apbus", "aplink", "apwire"),
    ),
    (
        "w7b-ap-bridge-inspection-card",
        "w7b-ap-structural-survey-ledger",
        "w7b-ap-bridge-condition-brief",
        "w7b-ap-span-checksheet",
    ),
)

DOMAIN_AQ = _domain(
    "AQ",
    "w7b-subsea-comms-source",
    "subsea communications",
    (
        "subsea communications asset",
        "underwater network sector",
        "communications service deviation",
        "subsea communications feed",
    ),
    (
        ("acoustic modem node", "fiber repeater pod", "junction controller", "antenna interface unit", "power-line modem", "telemetry gateway", "subsea router", "signal conditioning module"),
        ("cable spur", "junction field", "mooring link", "seafloor hub", "repeater corridor", "instrument cluster", "shoreward segment", "deepwater relay zone"),
        ("packet loss burst", "timing offset", "signal attenuation", "handshake delay", "carrier drift", "routing instability", "power coupling noise", "telemetry dropout"),
        ("modem packet trace", "repeater spectrum feed", "junction event log", "antenna signal channel", "power-line telemetry", "gateway routing record", "router status stream", "conditioning monitor bus"),
    ),
    (
        ("aqprime", "aqvector", "aqzenith"),
        ("aqnorth", "aqcenter", "aqsouth"),
        ("aqbrief", "aqsteady", "aqrepeat"),
        ("aqbus", "aqlink", "aqwire"),
    ),
    (
        "w7b-aq-subsea-comms-card",
        "w7b-aq-network-service-ledger",
        "w7b-aq-underwater-link-brief",
        "w7b-aq-communications-checksheet",
    ),
)

DOMAIN_AR = _domain(
    "AR",
    "w7b-desert-atmospheric-dev",
    "desert atmospheric sensing",
    (
        "desert atmospheric asset",
        "arid observation sector",
        "atmospheric sensing deviation",
        "desert diagnostic feed",
    ),
    (
        ("aerosol lidar head", "radiation sensor mast", "dust sampler unit", "humidity reference rack", "surface flux station", "wind profiler node", "solar tracker module", "pressure reference pod"),
        ("dune margin", "dry lake station", "rocky plateau", "instrument shelter", "flux tower zone", "remote sensor pad", "calibration bay", "weather mast sector"),
        ("optical attenuation rise", "radiation bias drift", "dust flow restriction", "humidity response lag", "flux baseline shift", "wind retrieval dropout", "tracker pointing error", "pressure offset drift"),
        ("lidar return stream", "radiation trace", "dust sampler log", "humidity calibration feed", "flux telemetry bus", "wind profiler channel", "tracker status record", "pressure reference stream"),
    ),
    (
        ("arprime", "arvector", "arzenith"),
        ("arfront", "armid", "arrear"),
        ("arbrief", "arsteady", "arrepeat"),
        ("arbus", "arlink", "arwire"),
    ),
    (
        "w7b-ar-desert-atmosphere-card",
        "w7b-ar-arid-sensing-ledger",
        "w7b-ar-weather-observation-brief",
        "w7b-ar-desert-station-checksheet",
    ),
)

DOMAIN_AS = _domain(
    "AS",
    "w7b-battery-fire-confirm",
    "distributed battery fire safety",
    (
        "battery fire-safety asset",
        "storage safety sector",
        "battery hazard deviation",
        "fire-safety diagnostic feed",
    ),
    (
        ("rack thermal sensor", "ventilation damper", "gas detection manifold", "suppression controller", "module isolation contactor", "smoke aspiration unit", "cooling branch valve", "alarm gateway"),
        ("battery aisle", "inverter room", "storage container", "service corridor", "suppression bay", "ventilation plenum", "thermal utility rack", "safety control alcove"),
        ("temperature runaway cue", "gas concentration rise", "smoke transport delay", "damper response lag", "isolation timing slip", "cooling flow loss", "alarm propagation delay", "suppression pressure decay"),
        ("thermal alarm stream", "gas sensor trace", "smoke aspiration feed", "damper position log", "isolation event channel", "cooling flow telemetry", "alarm gateway record", "suppression pressure bus"),
    ),
    (
        ("asprime", "asvector", "aszenith"),
        ("asfront", "asmid", "asrear"),
        ("asbrief", "assteady", "asrepeat"),
        ("asbus", "aslink", "aswire"),
    ),
    (
        "w7b-as-battery-fire-card",
        "w7b-as-storage-safety-ledger",
        "w7b-as-hazard-response-brief",
        "w7b-as-fire-safety-checksheet",
    ),
)

DOMAIN_AT = _domain(
    "AT",
    "w7b-coldchain-robot-confirm",
    "robotic cold-chain handling",
    (
        "cold-chain robotic asset",
        "refrigerated handling sector",
        "robotic logistics deviation",
        "cold-chain diagnostic feed",
    ),
    (
        ("freezer picking arm", "pallet shuttle robot", "case transfer gantry", "cold-room vision unit", "dock loading cobot", "refrigerated sorter", "crate handling actuator", "inventory scan rover"),
        ("freezer aisle", "cold dock", "staging chamber", "sortation lane", "loading cell", "refrigerated buffer", "inventory corridor", "case transfer bay"),
        ("gripper force drift", "trajectory timing slip", "vision frost occlusion", "shuttle alignment error", "sorter speed ripple", "dock handoff delay", "crate position mismatch", "scan localization loss"),
        ("robot force telemetry", "shuttle motion trace", "vision image feed", "gantry controller log", "sorter speed channel", "dock handoff record", "crate position stream", "scan localization bus"),
    ),
    (
        ("atprime", "atvector", "atzenith"),
        ("atfront", "atmid", "atrear"),
        ("atbrief", "atsteady", "atrepeat"),
        ("atbus", "atlink", "atwire"),
    ),
    (
        "w7b-at-coldchain-robot-card",
        "w7b-at-refrigerated-logistics-ledger",
        "w7b-at-robotic-handling-brief",
        "w7b-at-cold-room-checksheet",
    ),
)

DOMAINS = {
    spec.domain_id: spec
    for spec in (DOMAIN_AN, DOMAIN_AO, DOMAIN_AP, DOMAIN_AQ, DOMAIN_AR, DOMAIN_AS, DOMAIN_AT)
}


def all_w7b_values() -> set[str]:
    return {value for spec in DOMAINS.values() for pool in spec.pools for value in pool}


def domain_value_sets() -> dict[str, set[str]]:
    return {key: {value for pool in spec.pools for value in pool} for key, spec in DOMAINS.items()}


def domain_template_sets() -> dict[str, set[str]]:
    return {key: set(spec.templates) for key, spec in DOMAINS.items()}


def domain_role_sets() -> dict[str, set[str]]:
    return {key: set(spec.roles) for key, spec in DOMAINS.items()}


def _retag(case: ConjunctiveAuthorityCase) -> ConjunctiveAuthorityCase:
    typed = case.typed
    diagnosis = typed.decisions[0]
    options = tuple(
        replace(
            option,
            option_id=option.option_id.replace("-w7-", "-w7b-"),
        )
        for option in diagnosis.options
    )
    diagnosis = replace(diagnosis, options=options)
    typed = replace(
        typed,
        case_id=typed.case_id.replace("w7-", "w7b-", 1),
        decisions=(diagnosis, *typed.decisions[1:]),
    )
    return replace(case, typed=typed)


def _rows(spec: DomainSpec, *, split: str, per_k: int, seed: int):
    return [_retag(row) for row in _generate(spec, split=split, per_k=per_k, seed=seed)]


def generate_w7b_train() -> list[ConjunctiveAuthorityCase]:
    rows = []
    for spec, seed in (
        (DOMAIN_AN, SOURCE_AN_SEED),
        (DOMAIN_AO, SOURCE_AO_SEED),
        (DOMAIN_AP, SOURCE_AP_SEED),
        (DOMAIN_AQ, SOURCE_AQ_SEED),
    ):
        rows.extend(_rows(spec, split="train", per_k=24, seed=seed))
    if len(rows) != 384:
        raise RuntimeError("W7b TRAIN must contain 384 states")
    return rows


def generate_w7b_dev() -> list[ConjunctiveAuthorityCase]:
    rows = _rows(DOMAIN_AR, split="dev-ar", per_k=48, seed=DEV_AR_SEED)
    if len(rows) != 192:
        raise RuntimeError("W7b DEV-AR must contain 192 states")
    return rows


def generate_w7b_confirm(domain_id: str, *, allow_confirm: bool = False):
    if not allow_confirm:
        raise RuntimeError("W7b CONFIRM-AS/AT sealed until all DEV-AR checkpoints freeze")
    if domain_id == "AS":
        spec, seed = DOMAIN_AS, CONFIRM_AS_SEED
    elif domain_id == "AT":
        spec, seed = DOMAIN_AT, CONFIRM_AT_SEED
    else:
        raise ValueError("W7b confirm domain must be AS or AT")
    rows = _rows(spec, split=f"confirm-{domain_id.lower()}", per_k=48, seed=seed)
    if len(rows) != 192:
        raise RuntimeError("W7b CONFIRM domain must contain 192 states")
    return rows
