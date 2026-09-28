from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S7PairCase:
    case_id: str
    split: Split
    domain: str
    language: str
    state: str
    question_a: str
    question_b: str
    option_texts: tuple[str, ...]
    option_aliases: tuple[str, ...]
    option_ids: tuple[str, ...]
    gold_a: int
    gold_b: int

    def to_dict(self) -> dict:
        row = asdict(self)
        row["option_texts"] = list(self.option_texts)
        row["option_aliases"] = list(self.option_aliases)
        row["option_ids"] = list(self.option_ids)
        return row


@dataclass(frozen=True)
class _Domain:
    name: str
    prefix: str
    train_first: tuple[str, ...]
    dev_first: tuple[str, ...]
    train_second: tuple[str, ...]
    dev_second: tuple[str, ...]
    state_train: str
    state_dev: str
    qa_train: str
    qb_train: str
    qa_dev: str
    qb_dev: str
    seed: int
    dev_offset: int


_DOMAINS = (
    _Domain(
        "geothermal_well", "GW",
        ("andesite", "basalt", "rhyolite", "dacite", "tuff", "granodiorite", "obsidian", "pumice"),
        ("serpentinite", "gneiss", "migmatite", "schist", "marble", "phyllite", "pegmatite", "amphibolite"),
        ("142 C", "156 C", "171 C", "186 C", "199 C", "214 C", "229 C", "243 C"),
        ("149 C", "163 C", "178 C", "192 C", "207 C", "222 C", "236 C", "251 C"),
        "Geothermal well {code} intersects rock {first} at reservoir temperature {second}.",
        "Subsurface log {code} identifies formation {first}; measured thermal value is {second}.",
        "Which rock is intersected by this geothermal well?",
        "What reservoir temperature is recorded for this geothermal well?",
        "What formation does this subsurface log identify?",
        "What thermal value is reported in this subsurface log?",
        12701, 42000,
    ),
    _Domain(
        "orchestra_rehearsal", "ORC",
        ("violin", "viola", "cello", "bassoon", "oboe", "clarinet", "horn", "trombone"),
        ("harp", "contrabassoon", "piccolo", "flugelhorn", "euphonium", "celesta", "mandolin", "saxophone"),
        ("bar 18", "bar 27", "bar 36", "bar 45", "bar 54", "bar 63", "bar 72", "bar 81"),
        ("bar 22", "bar 31", "bar 40", "bar 49", "bar 58", "bar 67", "bar 76", "bar 85"),
        "Orchestra rehearsal note {code} assigns instrument {first} and restart point {second}.",
        "Ensemble sheet {code} names section {first}; rehearsal resumes at {second}.",
        "Which instrument is assigned in this rehearsal note?",
        "What restart point is listed in this rehearsal note?",
        "What section does this ensemble sheet name?",
        "Where does rehearsal resume according to this ensemble sheet?",
        12801, 43000,
    ),
    _Domain(
        "cold_chain", "CC",
        ("vial rack", "ampoule tray", "syringe case", "cartridge box", "sample pouch", "cryobox", "tube carrier", "dose cassette"),
        ("insulated pod", "dry-ice shipper", "PCM crate", "nitrogen dewar", "thermal sleeve", "vacuum flask", "coolant case", "phase-change tote"),
        ("2 C", "3 C", "4 C", "5 C", "6 C", "7 C", "8 C", "9 C"),
        ("-20 C", "-40 C", "-60 C", "-80 C", "-100 C", "-120 C", "-140 C", "-160 C"),
        "Cold-chain ticket {code} uses container {first} with setpoint {second}.",
        "Temperature-control manifest {code} specifies carrier {first}; target condition is {second}.",
        "Which container is used by this cold-chain ticket?",
        "What setpoint is listed on this cold-chain ticket?",
        "What carrier does this temperature-control manifest specify?",
        "What target condition is listed on this manifest?",
        12901, 44000,
    ),
    _Domain(
        "quarry_batch", "QB",
        ("limestone", "granite", "dolomite", "sandstone", "slate", "quartzite", "gabbro", "travertine"),
        ("porphyry", "diabase", "arkose", "breccia", "soapstone", "hornfels", "syenite", "conglomerate"),
        ("grade Q1", "grade Q2", "grade Q3", "grade Q4", "grade Q5", "grade Q6", "grade Q7", "grade Q8"),
        ("grade R1", "grade R2", "grade R3", "grade R4", "grade R5", "grade R6", "grade R7", "grade R8"),
        "Quarry batch {code} contains stone {first} and quality class {second}.",
        "Aggregate dossier {code} identifies material {first}; certification band is {second}.",
        "Which stone is contained in this quarry batch?",
        "What quality class is assigned to this quarry batch?",
        "What material does this aggregate dossier identify?",
        "What certification band is recorded in this aggregate dossier?",
        13001, 45000,
    ),
    _Domain(
        "fiber_network", "FN",
        ("SMF-28", "OM1", "OM2", "OM3", "OM4", "NZDSF", "bend-insensitive", "dispersion-shifted"),
        ("hollow-core", "few-mode", "plastic optical", "polarization-maintaining", "photonic-crystal", "erbium-doped", "fluoride", "chalcogenide"),
        ("1310 nm", "1490 nm", "1550 nm", "1577 nm", "1625 nm", "850 nm", "980 nm", "1270 nm"),
        ("1290 nm", "1330 nm", "1470 nm", "1510 nm", "1590 nm", "1610 nm", "1060 nm", "850/1300 nm"),
        "Fiber network record {code} uses cable {first} and wavelength {second}.",
        "Optical route ledger {code} names fiber medium {first}; carrier wavelength is {second}.",
        "Which cable is used in this fiber network record?",
        "What wavelength is used in this fiber network record?",
        "What fiber medium does this optical route ledger name?",
        "What carrier wavelength is listed in this route ledger?",
        13101, 46000,
    ),
    _Domain(
        "coffee_roast", "CRS",
        ("Bourbon", "Typica", "Caturra", "Gesha", "Pacamara", "SL28", "Maragogype", "Mundo Novo"),
        ("Catimor", "Ruiru 11", "Castillo", "Villa Sarchi", "Laurina", "Java", "Batian", "Pink Bourbon"),
        ("196 C", "201 C", "206 C", "211 C", "216 C", "221 C", "226 C", "231 C"),
        ("198 C", "203 C", "208 C", "213 C", "219 C", "223 C", "228 C", "233 C"),
        "Coffee roast batch {code} uses cultivar {first} and drop temperature {second}.",
        "Roastery profile {code} identifies bean variety {first}; final heat point is {second}.",
        "Which cultivar is used in this coffee roast batch?",
        "What drop temperature is recorded for this coffee roast batch?",
        "What bean variety does this roastery profile identify?",
        "What final heat point is listed in this roastery profile?",
        13201, 47000,
    ),
    _Domain(
        "wind_turbine", "WT",
        ("blade A", "blade B", "blade C", "blade D", "blade E", "blade F", "blade G", "blade H"),
        ("blade J", "blade K", "blade L", "blade M", "blade N", "blade P", "blade Q", "blade R"),
        ("pitch 2 deg", "pitch 4 deg", "pitch 6 deg", "pitch 8 deg", "pitch 10 deg", "pitch 12 deg", "pitch 14 deg", "pitch 16 deg"),
        ("pitch 3 deg", "pitch 5 deg", "pitch 7 deg", "pitch 9 deg", "pitch 11 deg", "pitch 13 deg", "pitch 15 deg", "pitch 17 deg"),
        "Wind turbine service {code} checks {first} at {second}.",
        "Rotor maintenance card {code} identifies component {first}; commanded setting is {second}.",
        "Which blade is checked in this wind turbine service?",
        "What pitch is used in this wind turbine service?",
        "What component does this rotor maintenance card identify?",
        "What commanded setting is listed in this maintenance card?",
        13301, 48000,
    ),
    _Domain(
        "forensic_sample", "FSX",
        ("hair", "fiber", "paint", "glass", "soil", "ink", "residue", "polymer"),
        ("pollen", "wax", "adhesive", "ceramic", "metal flake", "paper", "dust", "lubricant"),
        ("locker A4", "locker B7", "locker C2", "locker D9", "locker E5", "locker F1", "locker G8", "locker H3"),
        ("locker J6", "locker K2", "locker L9", "locker M4", "locker N7", "locker P3", "locker Q5", "locker R1"),
        "Forensic sample {code} is material {first} stored in {second}.",
        "Evidence register {code} classifies trace {first}; custody location is {second}.",
        "What material is recorded for this forensic sample?",
        "Where is this forensic sample stored?",
        "What trace class does this evidence register record?",
        "What custody location is listed in this evidence register?",
        13401, 49000,
    ),
    _Domain(
        "rail_signal", "RS",
        ("aspect green", "aspect yellow", "aspect red", "aspect double-yellow", "aspect flashing-yellow", "aspect lunar", "aspect white", "aspect blue"),
        ("aspect violet", "aspect amber", "aspect cyan", "aspect magenta", "aspect flashing-red", "aspect flashing-green", "aspect diagonal-white", "aspect steady-lunar"),
        ("block 14", "block 22", "block 31", "block 47", "block 53", "block 68", "block 74", "block 89"),
        ("block 17", "block 26", "block 35", "block 44", "block 59", "block 63", "block 78", "block 86"),
        "Rail signal record {code} shows {first} for {second}.",
        "Interlocking log {code} records indication {first}; protected section is {second}.",
        "Which aspect is shown in this rail signal record?",
        "Which block is controlled by this rail signal record?",
        "What indication does this interlocking log record?",
        "What protected section is listed in this interlocking log?",
        13501, 50000,
    ),
    _Domain(
        "desalination_unit", "DU",
        ("reverse osmosis", "multi-stage flash", "electrodialysis", "nanofiltration", "forward osmosis", "membrane distillation", "MED", "ion exchange"),
        ("capacitive deionization", "solar still", "freeze desalination", "humidification-dehumidification", "graphene membrane", "vapor compression", "pervaporation", "electrodeionization"),
        ("42 bar", "48 bar", "54 bar", "60 bar", "66 bar", "72 bar", "78 bar", "84 bar"),
        ("45 bar", "51 bar", "57 bar", "63 bar", "69 bar", "75 bar", "81 bar", "87 bar"),
        "Desalination unit {code} uses process {first} at pressure {second}.",
        "Water-treatment dossier {code} specifies method {first}; operating pressure is {second}.",
        "Which process is used by this desalination unit?",
        "What pressure is used by this desalination unit?",
        "What method does this water-treatment dossier specify?",
        "What operating pressure is listed in this dossier?",
        13601, 51000,
    ),
    _Domain(
        "satellite_image", "SI",
        ("panchromatic", "multispectral", "hyperspectral", "thermal", "SAR", "lidar", "nighttime-light", "stereo"),
        ("polarimetric SAR", "shortwave infrared", "microwave", "altimetry", "scatterometry", "radiometry", "ocean color", "atmospheric sounding"),
        ("scene 11", "scene 23", "scene 35", "scene 47", "scene 59", "scene 61", "scene 73", "scene 85"),
        ("scene 14", "scene 26", "scene 38", "scene 42", "scene 54", "scene 66", "scene 78", "scene 90"),
        "Satellite image job {code} uses sensor mode {first} for {second}.",
        "Earth-observation task {code} specifies acquisition mode {first}; target scene is {second}.",
        "Which sensor mode is used in this satellite image job?",
        "Which scene is assigned to this satellite image job?",
        "What acquisition mode does this Earth-observation task specify?",
        "What target scene is listed in this Earth-observation task?",
        13701, 52000,
    ),
    _Domain(
        "bakery_oven", "BO",
        ("sourdough", "brioche", "rye loaf", "baguette", "ciabatta", "focaccia", "pretzel", "croissant"),
        ("panettone", "pumpernickel", "challah", "naan", "lavash", "bagel", "kouign-amann", "pain de mie"),
        ("182 C", "188 C", "194 C", "200 C", "206 C", "212 C", "218 C", "224 C"),
        ("185 C", "191 C", "197 C", "203 C", "209 C", "215 C", "221 C", "227 C"),
        "Bakery oven batch {code} bakes product {first} at {second}.",
        "Bakehouse schedule {code} names item {first}; oven target is {second}.",
        "Which product is baked in this bakery oven batch?",
        "What temperature is used in this bakery oven batch?",
        "What item does this bakehouse schedule name?",
        "What oven target is listed in this bakehouse schedule?",
        13801, 53000,
    ),
)


def _shuffle(case_id: str, first: str, second: str, x: str, y: str, seed: int):
    rows = [("a", first), ("b", second), ("x", x), ("y", y)]
    random.Random(seed).shuffle(rows)
    texts = tuple(f"the requested answer is {value}" for _, value in rows)
    aliases = tuple(f"{value} is the equivalent answer value" for _, value in rows)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _build(spec: _Domain, index: int, split: Split) -> S7PairCase:
    firsts = spec.train_first if split == "train" else spec.dev_first
    seconds = spec.train_second if split == "train" else spec.dev_second
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    x = firsts[(index + 3) % len(firsts)]
    y = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    case_id = f"{split}-s7-{spec.name}-{index:03d}"
    state_template = spec.state_train if split == "train" else spec.state_dev
    qa = spec.qa_train if split == "train" else spec.qa_dev
    qb = spec.qb_train if split == "train" else spec.qb_dev
    state = state_template.format(code=f"{spec.prefix}{offset:05d}", first=first, second=second)
    texts, aliases, ids, ga, gb = _shuffle(
        case_id, first, second, x, y, spec.seed + offset
    )
    return S7PairCase(
        case_id, split, spec.name, "en", state, qa, qb,
        texts, aliases, ids, ga, gb,
    )


def generate_s7_pairs(split: Split) -> tuple[S7PairCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, i, split)
        for spec in _DOMAINS
        for i in range(count)
    )


def validate_s7_partitions(
    train: tuple[S7PairCase, ...],
    dev: tuple[S7PairCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S7 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S7 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S7 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S7 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S7 duplicate case IDs")
        for row in rows:
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("Hira v1 S7 requires K=4 with aliases")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("Hira v1 S7 opaque option IDs changed")
            if row.gold_a == row.gold_b:
                raise RuntimeError("Hira v1 S7 paired golds must differ")

    if {row.state for row in train} & {row.state for row in dev}:
        raise RuntimeError("Hira v1 S7 TRAIN/DEV state overlap")
    train_q = {q for row in train for q in (row.question_a, row.question_b)}
    dev_q = {q for row in dev for q in (row.question_a, row.question_b)}
    if train_q & dev_q:
        raise RuntimeError("Hira v1 S7 TRAIN/DEV question overlap")


__all__ = ["S7PairCase", "generate_s7_pairs", "validate_s7_partitions"]
