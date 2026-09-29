from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S14FusionCase:
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
        "accelerator_magnet", "particle-accelerator magnet", "AM", "conductor type", "coil current",
        ("NbTi", "Nb3Sn", "REBCO", "Bi-2212", "copper", "MgB2", "YBCO", "aluminum"),
        ("Bi-2223", "NbAl", "BSCCO tape", "silver-alloy", "CuAg", "HTS stack", "iron-cored copper", "LTS cable"),
        ("2.0 kA", "2.5 kA", "3.0 kA", "3.5 kA", "4.0 kA", "4.5 kA", "5.0 kA", "5.5 kA"),
        ("2.2 kA", "2.7 kA", "3.2 kA", "3.7 kA", "4.2 kA", "4.7 kA", "5.2 kA", "5.7 kA"),
        21001, 161000,
    ),
    _Domain(
        "vertical_farm", "vertical-farm rack", "VF", "crop variety", "photoperiod",
        ("Genovese basil", "butterhead", "arugula", "kale", "mizuna", "pak choi", "red oakleaf", "cilantro"),
        ("Thai basil", "romaine", "watercress", "mustard green", "tatsoi", "Swiss chard", "frisee", "parsley"),
        ("12 h", "13 h", "14 h", "15 h", "16 h", "17 h", "18 h", "19 h"),
        ("12.5 h", "13.5 h", "14.5 h", "15.5 h", "16.5 h", "17.5 h", "18.5 h", "19.5 h"),
        21101, 162000,
    ),
    _Domain(
        "thermal_panel", "satellite thermal panel", "ST", "surface coating", "heater power",
        ("AZ-93", "Z-93", "black Kapton", "white paint", "silver Teflon", "gold foil", "MLI outer", "bare aluminum"),
        ("black anodize", "white anodize", "ITO Kapton", "OSR tile", "aluminized Mylar", "nickel foil", "ceramic white", "graphite black"),
        ("8 W", "12 W", "16 W", "20 W", "24 W", "28 W", "32 W", "36 W"),
        ("10 W", "14 W", "18 W", "22 W", "26 W", "30 W", "34 W", "38 W"),
        21201, 163000,
    ),
    _Domain(
        "aeration_basin", "wastewater aeration basin", "AB", "diffuser type", "DO setpoint",
        ("fine bubble", "coarse bubble", "disc", "tube", "panel", "jet", "surface rotor", "aspirating"),
        ("membrane grid", "ceramic dome", "strip", "plate", "microbubble", "brush aerator", "cascade", "venturi"),
        ("1.2 mg/L", "1.6 mg/L", "2.0 mg/L", "2.4 mg/L", "2.8 mg/L", "3.2 mg/L", "3.6 mg/L", "4.0 mg/L"),
        ("1.4 mg/L", "1.8 mg/L", "2.2 mg/L", "2.6 mg/L", "3.0 mg/L", "3.4 mg/L", "3.8 mg/L", "4.2 mg/L"),
        21301, 164000,
    ),
    _Domain(
        "interferometer", "laser interferometer", "LI", "mirror coating", "cavity length",
        ("Ta2O5/SiO2", "TiO2/SiO2", "HfO2/SiO2", "Al2O3/SiO2", "silver", "gold", "dielectric HR", "protected aluminum"),
        ("Nb2O5/SiO2", "ZrO2/SiO2", "Si3N4/SiO2", "MgF2 stack", "protected silver", "enhanced gold", "chirped HR", "crystalline GaAs"),
        ("0.5 m", "0.8 m", "1.1 m", "1.4 m", "1.7 m", "2.0 m", "2.3 m", "2.6 m"),
        ("0.65 m", "0.95 m", "1.25 m", "1.55 m", "1.85 m", "2.15 m", "2.45 m", "2.75 m"),
        21401, 165000,
    ),
    _Domain(
        "geothermal_well", "geothermal well loop", "GW", "working fluid", "circulation rate",
        ("water", "isobutane", "isopentane", "ammonia", "CO2", "R245fa", "R1233zd", "brine"),
        ("R1234ze", "propane", "pentane", "butane", "organic oil", "glycol", "supercritical CO2", "potassium formate"),
        ("18 kg/s", "24 kg/s", "30 kg/s", "36 kg/s", "42 kg/s", "48 kg/s", "54 kg/s", "60 kg/s"),
        ("21 kg/s", "27 kg/s", "33 kg/s", "39 kg/s", "45 kg/s", "51 kg/s", "57 kg/s", "63 kg/s"),
        21501, 166000,
    ),
    _Domain(
        "robot_arm", "industrial robot arm", "IR", "gripper type", "payload limit",
        ("parallel jaw", "vacuum cup", "soft gripper", "three-finger", "magnetic", "needle", "Bernoulli", "collet"),
        ("angular jaw", "adaptive finger", "electroadhesive", "gecko", "internal expanding", "fork", "suction array", "compliant clamp"),
        ("4 kg", "6 kg", "8 kg", "10 kg", "12 kg", "14 kg", "16 kg", "18 kg"),
        ("5 kg", "7 kg", "9 kg", "11 kg", "13 kg", "15 kg", "17 kg", "19 kg"),
        21601, 167000,
    ),
    _Domain(
        "rf_amplifier", "RF power amplifier", "RF", "transistor technology", "bias current",
        ("GaN HEMT", "LDMOS", "GaAs pHEMT", "SiGe HBT", "CMOS", "InP HEMT", "MESFET", "GaN MMIC"),
        ("GaAs HBT", "SOI CMOS", "GaN-on-diamond", "SiC MESFET", "InGaP HBT", "BiCMOS", "AlGaN HEMT", "diamond FET"),
        ("80 mA", "120 mA", "160 mA", "200 mA", "240 mA", "280 mA", "320 mA", "360 mA"),
        ("100 mA", "140 mA", "180 mA", "220 mA", "260 mA", "300 mA", "340 mA", "380 mA"),
        21701, 168000,
    ),
    _Domain(
        "fermentation_line", "fermentation line", "FL", "microbe strain", "broth temperature",
        ("S. cerevisiae", "K. lactis", "E. coli", "B. subtilis", "Pichia", "Lactobacillus", "Corynebacterium", "Aspergillus"),
        ("Yarrowia", "Kluyveromyces marxianus", "Bacillus licheniformis", "Streptomyces", "Komagataella", "Zymomonas", "Rhizopus", "Penicillium"),
        ("24 C", "26 C", "28 C", "30 C", "32 C", "34 C", "36 C", "38 C"),
        ("25 C", "27 C", "29 C", "31 C", "33 C", "35 C", "37 C", "39 C"),
        21801, 169000,
    ),
    _Domain(
        "autonomous_boat", "autonomous survey boat", "AS", "navigation sensor", "cruise speed",
        ("RTK GNSS", "marine radar", "stereo camera", "LiDAR", "IMU", "DVL", "sonar", "GNSS compass"),
        ("visual odometry", "FMCW radar", "event camera", "solid-state LiDAR", "fiber gyro", "USBL", "multibeam", "star tracker"),
        ("1.2 m/s", "1.6 m/s", "2.0 m/s", "2.4 m/s", "2.8 m/s", "3.2 m/s", "3.6 m/s", "4.0 m/s"),
        ("1.4 m/s", "1.8 m/s", "2.2 m/s", "2.6 m/s", "3.0 m/s", "3.4 m/s", "3.8 m/s", "4.2 m/s"),
        21901, 170000,
    ),
    _Domain(
        "deposition_chamber", "vacuum deposition chamber", "VD", "target material", "source power",
        ("titanium", "aluminum", "chromium", "copper", "tantalum", "silicon", "carbon", "molybdenum"),
        ("tungsten", "nickel", "cobalt", "hafnium", "zirconium", "germanium", "boron", "niobium"),
        ("200 W", "300 W", "400 W", "500 W", "600 W", "700 W", "800 W", "900 W"),
        ("250 W", "350 W", "450 W", "550 W", "650 W", "750 W", "850 W", "950 W"),
        22001, 171000,
    ),
    _Domain(
        "seismic_station", "seismic monitoring station", "SS", "sensor type", "sample rate",
        ("broadband", "short-period", "strong-motion", "geophone", "MEMS", "fiber DAS", "borehole", "triaxial"),
        ("very-broadband", "accelerometer", "rotational", "nodal", "hydrophone", "laser strainmeter", "tiltmeter", "gravimeter"),
        ("50 Hz", "100 Hz", "150 Hz", "200 Hz", "250 Hz", "300 Hz", "350 Hz", "400 Hz"),
        ("75 Hz", "125 Hz", "175 Hz", "225 Hz", "275 Hz", "325 Hz", "375 Hz", "425 Hz"),
        22101, 172000,
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
            f"S14 {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"Record {code} lists {second} beside {spec.field_b}. "
            f"In the same S14 {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S14 {spec.noun} {code}, what is the {spec.field_a}?"
        qa2 = f"Which value fills {spec.field_a} in record {code}?"
        qb1 = f"For S14 {spec.noun} {code}, what is the {spec.field_b}?"
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


def _build(spec: _Domain, index: int, split: Split) -> S14FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s14-{spec.name}-{index:03d}"

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
    return S14FusionCase(
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


def generate_s14_cases(split: Split) -> tuple[S14FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s14_partitions(
    train: tuple[S14FusionCase, ...],
    dev: tuple[S14FusionCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S14 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S14 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S14 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S14 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("S14 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S14 state views must differ")
            if row.question_a1 == row.question_a2 or row.question_b1 == row.question_b2:
                raise RuntimeError("S14 question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S14 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S14 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S14 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S14 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S14 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S14 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S14 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S14 TRAIN/DEV option-alias overlap")


__all__ = [
    "S14FusionCase",
    "generate_s14_cases",
    "validate_s14_partitions",
]
