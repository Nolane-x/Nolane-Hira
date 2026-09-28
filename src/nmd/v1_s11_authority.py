from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S11BindingCase:
    case_id: str
    split: Split
    domain: str
    language: str
    state_a: str
    state_b: str
    question_a1: str
    question_a2: str
    question_b1: str
    question_b2: str
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
    noun: str
    prefix: str
    field_a: str
    field_b: str
    train_a: tuple[str, ...]
    dev_a: tuple[str, ...]
    train_b: tuple[str, ...]
    dev_b: tuple[str, ...]
    seed: int
    dev_offset: int


_DOMAINS = (
    _Domain(
        "solar_inverter", "solar inverter", "SI", "converter topology", "switching frequency",
        ("two-level", "three-level NPC", "T-type", "flying-capacitor", "H-bridge", "ANPC", "CHB", "Vienna"),
        ("MMC", "quasi-Z-source", "dual-active-bridge", "matrix", "current-source", "LLC", "buck-boost", "resonant"),
        ("8 kHz", "12 kHz", "16 kHz", "20 kHz", "24 kHz", "28 kHz", "32 kHz", "36 kHz"),
        ("10 kHz", "14 kHz", "18 kHz", "22 kHz", "26 kHz", "30 kHz", "34 kHz", "38 kHz"),
        17001, 101000,
    ),
    _Domain(
        "cold_storage", "cold-storage rack", "CS", "refrigerant", "suction pressure",
        ("R744", "R717", "R290", "R600a", "R1234yf", "R1234ze", "R449A", "R452A"),
        ("R513A", "R450A", "R454C", "R455A", "R515B", "R1270", "R170", "R1150"),
        ("1.2 bar", "1.6 bar", "2.0 bar", "2.4 bar", "2.8 bar", "3.2 bar", "3.6 bar", "4.0 bar"),
        ("1.4 bar", "1.8 bar", "2.2 bar", "2.6 bar", "3.0 bar", "3.4 bar", "3.8 bar", "4.2 bar"),
        17101, 102000,
    ),
    _Domain(
        "fiber_link", "fiber link", "FL", "transceiver class", "wavelength",
        ("SR", "LR", "ER", "ZR", "BiDi-A", "BiDi-B", "CWDM4", "PSM4"),
        ("DR4", "FR4", "LR4", "ER4", "ZR4", "SWDM4", "LR8", "FR8"),
        ("850 nm", "980 nm", "1064 nm", "1270 nm", "1310 nm", "1490 nm", "1550 nm", "1625 nm"),
        ("880 nm", "1030 nm", "1260 nm", "1290 nm", "1330 nm", "1470 nm", "1570 nm", "1610 nm"),
        17201, 103000,
    ),
    _Domain(
        "water_treatment", "water-treatment train", "WT", "coagulant", "dose rate",
        ("alum", "ferric chloride", "PACl", "ferric sulfate", "lime", "chitosan", "PAC", "polydadmac"),
        ("ACH", "sodium aluminate", "ferrous sulfate", "magnesium hydroxide", "bentonite", "starch", "tannin", "silica sol"),
        ("8 mg/L", "12 mg/L", "16 mg/L", "20 mg/L", "24 mg/L", "28 mg/L", "32 mg/L", "36 mg/L"),
        ("10 mg/L", "14 mg/L", "18 mg/L", "22 mg/L", "26 mg/L", "30 mg/L", "34 mg/L", "38 mg/L"),
        17301, 104000,
    ),
    _Domain(
        "robot_cell", "robot workcell", "RC", "gripper type", "cycle time",
        ("parallel jaw", "vacuum", "magnetic", "soft pneumatic", "needle", "three-finger", "Bernoulli", "collet"),
        ("electroadhesive", "gecko", "expanding mandrel", "internal", "fork", "suction array", "adaptive finger", "microgripper"),
        ("6 s", "8 s", "10 s", "12 s", "14 s", "16 s", "18 s", "20 s"),
        ("7 s", "9 s", "11 s", "13 s", "15 s", "17 s", "19 s", "21 s"),
        17401, 105000,
    ),
    _Domain(
        "coffee_roaster", "coffee roaster", "CRX", "bean origin", "drum temperature",
        ("Yirgacheffe", "Sidamo", "Huehuetenango", "Tarrazú", "Kona", "Sumatra", "Cerrado", "Antioquia"),
        ("Nyeri", "Kirinyaga", "Boquete", "Nariño", "Minas Gerais", "Sul de Minas", "Kintamani", "Mogiana"),
        ("185 C", "190 C", "195 C", "200 C", "205 C", "210 C", "215 C", "220 C"),
        ("187 C", "192 C", "197 C", "202 C", "207 C", "212 C", "217 C", "222 C"),
        17501, 106000,
    ),
    _Domain(
        "bridge_sensor", "bridge sensor node", "BS", "sensor type", "sample rate",
        ("strain gauge", "accelerometer", "tiltmeter", "LVDT", "fiber Bragg", "load cell", "thermistor", "GNSS"),
        ("MEMS gyro", "laser vibrometer", "radar", "piezometer", "crackmeter", "inclinometer", "acoustic emission", "displacement encoder"),
        ("10 Hz", "20 Hz", "30 Hz", "40 Hz", "50 Hz", "60 Hz", "70 Hz", "80 Hz"),
        ("15 Hz", "25 Hz", "35 Hz", "45 Hz", "55 Hz", "65 Hz", "75 Hz", "85 Hz"),
        17601, 107000,
    ),
    _Domain(
        "electric_ferry", "electric ferry", "EF", "propulsion mode", "bus voltage",
        ("shaft motor", "azimuth pod", "waterjet", "rim drive", "twin screw", "contra-rotating", "Voith", "cycloidal"),
        ("ducted propulsor", "pumpjet", "surface drive", "linear jet", "hubless", "thruster array", "hybrid pod", "tunnel thruster"),
        ("400 V", "500 V", "600 V", "700 V", "800 V", "900 V", "1000 V", "1100 V"),
        ("450 V", "550 V", "650 V", "750 V", "850 V", "950 V", "1050 V", "1150 V"),
        17701, 108000,
    ),
    _Domain(
        "seed_lab", "seed laboratory", "SL", "cultivar", "germination temperature",
        ("Atlas", "Beacon", "Canyon", "Delta", "Ember", "Flint", "Harbor", "Ivory"),
        ("Juniper", "Kestrel", "Lagoon", "Mesa", "Nimbus", "Orchid", "Prairie", "Quartz"),
        ("16 C", "18 C", "20 C", "22 C", "24 C", "26 C", "28 C", "30 C"),
        ("17 C", "19 C", "21 C", "23 C", "25 C", "27 C", "29 C", "31 C"),
        17801, 109000,
    ),
    _Domain(
        "telescope_mount", "telescope mount", "TM", "drive type", "slew rate",
        ("worm gear", "direct drive", "harmonic", "belt", "friction", "roller", "planetary", "capstan"),
        ("linear motor", "torque motor", "cycloidal", "strain-wave", "cable drive", "hydrostatic", "magnetic", "rack-pinion"),
        ("1 deg/s", "2 deg/s", "3 deg/s", "4 deg/s", "5 deg/s", "6 deg/s", "7 deg/s", "8 deg/s"),
        ("1.5 deg/s", "2.5 deg/s", "3.5 deg/s", "4.5 deg/s", "5.5 deg/s", "6.5 deg/s", "7.5 deg/s", "8.5 deg/s"),
        17901, 110000,
    ),
    _Domain(
        "vaccine_freezer", "vaccine freezer", "VF", "coolant", "storage setpoint",
        ("R290", "R600a", "R170", "R744", "R23", "R508B", "ethane", "propane"),
        ("R14", "R116", "R218", "argon", "nitrogen", "neon", "helium", "methane"),
        ("-20 C", "-30 C", "-40 C", "-50 C", "-60 C", "-70 C", "-80 C", "-90 C"),
        ("-25 C", "-35 C", "-45 C", "-55 C", "-65 C", "-75 C", "-85 C", "-95 C"),
        18001, 111000,
    ),
    _Domain(
        "printing_press", "printing press", "PP", "ink type", "line speed",
        ("UV offset", "soy offset", "waterless", "latex", "solvent", "aqueous", "sublimation", "toner"),
        ("EB-cured", "hybrid UV", "ceramic", "conductive", "thermochromic", "fluorescent", "metallic", "bio-based"),
        ("20 m/min", "30 m/min", "40 m/min", "50 m/min", "60 m/min", "70 m/min", "80 m/min", "90 m/min"),
        ("25 m/min", "35 m/min", "45 m/min", "55 m/min", "65 m/min", "75 m/min", "85 m/min", "95 m/min"),
        18101, 112000,
    ),
)


def _shuffle(
    *,
    case_id: str,
    noun: str,
    field_a: str,
    field_b: str,
    first: str,
    second: str,
    distractor_first: str,
    distractor_second: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int, int]:
    rows = [
        ("a", field_a, first),
        ("b", field_b, second),
        ("x", field_a, distractor_first),
        ("y", field_b, distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts = tuple(
        f"for the {noun}, {field} is {value}"
        for _kind, field, value in rows
    )
    aliases = tuple(
        f"{value} is the recorded {field} entry in this {noun}"
        for _kind, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _field, _value) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _field, _value) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(
    spec: _Domain,
    *,
    split: Split,
    code: str,
    first: str,
    second: str,
) -> tuple[str, str, str, str, str, str]:
    if split == "train":
        state_a = (
            f"S11 {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"Record {code} lists {second} beside {spec.field_b}. "
            f"In the same S11 {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S11 {spec.noun} {code}, what is the {spec.field_a}?"
        qa2 = f"Which value fills {spec.field_a} in record {code}?"
        qb1 = f"For S11 {spec.noun} {code}, what is the {spec.field_b}?"
        qb2 = f"Which value fills {spec.field_b} in record {code}?"
    else:
        state_a = (
            f"Inspection dossier {code} for the {spec.noun} places {first} "
            f"under {spec.field_a}; the {spec.field_b} entry reads {second}."
        )
        state_b = (
            f"Within dossier {code}, {spec.field_b} carries {second}. "
            f"Elsewhere in that {spec.noun} dossier, {first} is tagged as {spec.field_a}."
        )
        qa1 = f"From inspection dossier {code}, identify {spec.field_a}."
        qa2 = f"What entry is tagged as {spec.field_a} for dossier {code}?"
        qb1 = f"From inspection dossier {code}, identify {spec.field_b}."
        qb2 = f"What entry is tagged as {spec.field_b} for dossier {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S11BindingCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s11-{spec.name}-{index:03d}"

    state_a, state_b, qa1, qa2, qb1, qb2 = _views(
        spec,
        split=split,
        code=code,
        first=first,
        second=second,
    )
    texts, aliases, ids, ga, gb = _shuffle(
        case_id=case_id,
        noun=spec.noun,
        field_a=spec.field_a,
        field_b=spec.field_b,
        first=first,
        second=second,
        distractor_first=distractor_first,
        distractor_second=distractor_second,
        seed=spec.seed + offset,
    )
    return S11BindingCase(
        case_id=case_id,
        split=split,
        domain=spec.name,
        language="en",
        state_a=state_a,
        state_b=state_b,
        question_a1=qa1,
        question_a2=qa2,
        question_b1=qb1,
        question_b2=qb2,
        option_texts=texts,
        option_aliases=aliases,
        option_ids=ids,
        gold_a=ga,
        gold_b=gb,
    )


def generate_s11_cases(split: Split) -> tuple[S11BindingCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s11_partitions(
    train: tuple[S11BindingCase, ...],
    dev: tuple[S11BindingCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S11 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S11 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S11 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S11 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("S11 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S11 state views must differ")
            if row.question_a1 == row.question_a2 or row.question_b1 == row.question_b2:
                raise RuntimeError("S11 question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S11 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S11 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S11 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S11 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S11 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S11 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S11 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S11 TRAIN/DEV option-alias overlap")


__all__ = [
    "S11BindingCase",
    "generate_s11_cases",
    "validate_s11_partitions",
]
