from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S8InvariantCase:
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
        "hydroponic_batch", "hydroponic batch", "HY",
        "nutrient blend", "conductivity",
        ("Blend Alder", "Blend Birch", "Blend Cedar", "Blend Delta", "Blend Elm", "Blend Fir", "Blend Grove", "Blend Hazel"),
        ("Blend Iris", "Blend Juniper", "Blend Kestrel", "Blend Lotus", "Blend Maple", "Blend Nova", "Blend Orchid", "Blend Pine"),
        ("1.2 mS/cm", "1.5 mS/cm", "1.8 mS/cm", "2.1 mS/cm", "2.4 mS/cm", "2.7 mS/cm", "3.0 mS/cm", "3.3 mS/cm"),
        ("1.35 mS/cm", "1.65 mS/cm", "1.95 mS/cm", "2.25 mS/cm", "2.55 mS/cm", "2.85 mS/cm", "3.15 mS/cm", "3.45 mS/cm"),
        13801, 61000,
    ),
    _Domain(
        "radar_scan", "radar scan", "RD",
        "waveform", "pulse interval",
        ("chirp linear", "chirp stepped", "pulse Doppler", "FMCW", "phase-coded", "burst", "staggered", "monopulse"),
        ("barker-coded", "polyphase", "noise waveform", "frequency-hop", "dual-chirp", "interleaved", "compressed pulse", "pseudo-random"),
        ("0.8 ms", "1.1 ms", "1.4 ms", "1.7 ms", "2.0 ms", "2.3 ms", "2.6 ms", "2.9 ms"),
        ("0.95 ms", "1.25 ms", "1.55 ms", "1.85 ms", "2.15 ms", "2.45 ms", "2.75 ms", "3.05 ms"),
        13901, 62000,
    ),
    _Domain(
        "perfumery_formula", "perfumery formula", "PF",
        "heart note", "fixative level",
        ("orris", "jasmine", "rose absolute", "ylang-ylang", "neroli", "tuberose", "osmanthus", "violet leaf"),
        ("immortelle", "magnolia", "mimosa", "gardenia", "frangipani", "lotus", "champaca", "orange blossom"),
        ("4 percent", "6 percent", "8 percent", "10 percent", "12 percent", "14 percent", "16 percent", "18 percent"),
        ("5 percent", "7 percent", "9 percent", "11 percent", "13 percent", "15 percent", "17 percent", "19 percent"),
        14001, 63000,
    ),
    _Domain(
        "bridge_sensor", "bridge sensor package", "BSX",
        "sensor type", "sampling rate",
        ("strain gauge", "accelerometer", "tiltmeter", "fiber Bragg", "load cell", "LVDT", "thermistor", "displacement laser"),
        ("MEMS gyro", "acoustic emission", "GNSS receiver", "inclinometer", "magnetometer", "radar vibrometer", "piezo film", "corrosion probe"),
        ("25 Hz", "50 Hz", "75 Hz", "100 Hz", "125 Hz", "150 Hz", "175 Hz", "200 Hz"),
        ("30 Hz", "60 Hz", "90 Hz", "120 Hz", "140 Hz", "160 Hz", "180 Hz", "220 Hz"),
        14101, 64000,
    ),
    _Domain(
        "glacier_station", "glacier station", "GL",
        "marker type", "ice velocity",
        ("aluminum stake", "steel stake", "GPS beacon", "reflector prism", "radar tag", "dye marker", "snow pole", "radio beacon"),
        ("GNSS puck", "corner reflector", "fiber marker", "sonic beacon", "magnetic stake", "optical target", "RF tag", "pressure marker"),
        ("18 m/year", "24 m/year", "30 m/year", "36 m/year", "42 m/year", "48 m/year", "54 m/year", "60 m/year"),
        ("21 m/year", "27 m/year", "33 m/year", "39 m/year", "45 m/year", "51 m/year", "57 m/year", "63 m/year"),
        14201, 65000,
    ),
    _Domain(
        "brewery_filter", "brewery filter run", "BF",
        "filter medium", "flow rate",
        ("diatomaceous earth", "cellulose", "perlite", "PVPP", "silica gel", "crossflow membrane", "ceramic membrane", "depth pad"),
        ("activated carbon", "polyethersulfone", "nylon mesh", "sintered glass", "polypropylene", "microfiber", "zeolite", "stainless mesh"),
        ("12 L/min", "16 L/min", "20 L/min", "24 L/min", "28 L/min", "32 L/min", "36 L/min", "40 L/min"),
        ("14 L/min", "18 L/min", "22 L/min", "26 L/min", "30 L/min", "34 L/min", "38 L/min", "42 L/min"),
        14301, 66000,
    ),
    _Domain(
        "microgrid_dispatch", "microgrid dispatch", "MGD",
        "primary source", "reserve target",
        ("solar bank", "wind cluster", "battery rack", "microturbine", "fuel cell", "hydro unit", "biogas set", "diesel set"),
        ("thermal store", "flywheel bank", "tidal unit", "geothermal unit", "hydrogen stack", "supercapacitor bank", "wave unit", "compressed-air store"),
        ("8 percent", "12 percent", "16 percent", "20 percent", "24 percent", "28 percent", "32 percent", "36 percent"),
        ("10 percent", "14 percent", "18 percent", "22 percent", "26 percent", "30 percent", "34 percent", "38 percent"),
        14401, 67000,
    ),
    _Domain(
        "robotics_cell", "robotics cell", "RBX",
        "end effector", "cycle time",
        ("parallel gripper", "vacuum cup", "magnetic gripper", "soft gripper", "welding torch", "dispense nozzle", "screwdriver", "polishing head"),
        ("needle gripper", "Bernoulli gripper", "electrostatic pad", "compliant hand", "laser head", "riveter", "deburring spindle", "inspection camera"),
        ("18 s", "22 s", "26 s", "30 s", "34 s", "38 s", "42 s", "46 s"),
        ("20 s", "24 s", "28 s", "32 s", "36 s", "40 s", "44 s", "48 s"),
        14501, 68000,
    ),
    _Domain(
        "seed_vault", "seed-vault accession", "SV",
        "crop species", "storage humidity",
        ("millet", "chickpea", "lentil", "sorghum", "quinoa", "buckwheat", "flax", "sesame"),
        ("teff", "amaranth", "cowpea", "fonio", "camelina", "lupin", "pigeon pea", "chia"),
        ("8 percent RH", "10 percent RH", "12 percent RH", "14 percent RH", "16 percent RH", "18 percent RH", "20 percent RH", "22 percent RH"),
        ("9 percent RH", "11 percent RH", "13 percent RH", "15 percent RH", "17 percent RH", "19 percent RH", "21 percent RH", "23 percent RH"),
        14601, 69000,
    ),
    _Domain(
        "acoustic_test", "acoustic test", "AC",
        "excitation signal", "level",
        ("pink noise", "white noise", "sine sweep", "MLS sequence", "tone burst", "warble tone", "impulse", "chirp"),
        ("brown noise", "blue noise", "log sweep", "Golay code", "multitone", "band noise", "click train", "pseudo-random burst"),
        ("68 dB", "72 dB", "76 dB", "80 dB", "84 dB", "88 dB", "92 dB", "96 dB"),
        ("70 dB", "74 dB", "78 dB", "82 dB", "86 dB", "90 dB", "94 dB", "98 dB"),
        14701, 70000,
    ),
    _Domain(
        "kiln_glaze", "kiln glaze trial", "KG",
        "glaze family", "hold time",
        ("celadon", "shino", "tenmoku", "ash glaze", "salt glaze", "crystalline", "copper red", "matte white"),
        ("satin black", "rutile blue", "lichen", "nuka", "oil spot", "hare's fur", "chun blue", "floating blue"),
        ("8 min", "12 min", "16 min", "20 min", "24 min", "28 min", "32 min", "36 min"),
        ("10 min", "14 min", "18 min", "22 min", "26 min", "30 min", "34 min", "38 min"),
        14801, 71000,
    ),
    _Domain(
        "courier_route", "courier route", "CQ",
        "vehicle class", "delivery window",
        ("cargo bike", "electric van", "box truck", "scooter", "walking courier", "microvan", "trike", "compact EV"),
        ("autonomous cart", "rail pod", "drone van", "hydrogen van", "robot rover", "electric pickup", "mini bus", "parcel shuttle"),
        ("08:00-09:00", "09:00-10:00", "10:00-11:00", "11:00-12:00", "13:00-14:00", "14:00-15:00", "15:00-16:00", "16:00-17:00"),
        ("08:30-09:30", "09:30-10:30", "10:30-11:30", "11:30-12:30", "13:30-14:30", "14:30-15:30", "15:30-16:30", "16:30-17:30"),
        14901, 72000,
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
    # Domain+field context is semantic content, not routing metadata.  It keeps
    # split-specific option banks globally disjoint even when unrelated domains
    # happen to reuse the same bare numeric/string value.
    rows = [
        ("a", field_a, first),
        ("b", field_b, second),
        ("x", field_a, distractor_first),
        ("y", field_b, distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts = tuple(
        f"for the {noun}, the {field} answer is {value}"
        for _, field, value in rows
    )
    aliases = tuple(
        f"{value} is the semantically equivalent {field} option for the {noun}"
        for _, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _, _) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _, _) in enumerate(rows) if kind == "b")
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
            f"S8 {spec.noun} record {code} lists {spec.field_a} {first} "
            f"and {spec.field_b} {second}."
        )
        state_b = (
            f"For {code}, the documented {spec.field_b} is {second}; "
            f"its {spec.field_a} entry is {first} in the {spec.noun} ledger."
        )
        qa1 = f"In S8 {spec.noun} {code}, what {spec.field_a} is recorded?"
        qa2 = f"Which {spec.field_a} belongs to {code} in the {spec.noun} ledger?"
        qb1 = f"In S8 {spec.noun} {code}, what {spec.field_b} is recorded?"
        qb2 = f"Which {spec.field_b} belongs to {code} in the {spec.noun} ledger?"
    else:
        state_a = (
            f"The S8 dossier for {spec.noun} {code} identifies {first} as "
            f"the {spec.field_a}, while reporting {second} for {spec.field_b}."
        )
        state_b = (
            f"{code} carries a {spec.field_a} value of {first}. "
            f"Separately, the {spec.noun} file gives {spec.field_b} as {second}."
        )
        qa1 = f"From the S8 dossier {code}, identify the {spec.field_a}."
        qa2 = f"What value fills the {spec.field_a} field for {code} in this {spec.noun} file?"
        qb1 = f"From the S8 dossier {code}, identify the {spec.field_b}."
        qb2 = f"What value fills the {spec.field_b} field for {code} in this {spec.noun} file?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S8InvariantCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s8-{spec.name}-{index:03d}"

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
    return S8InvariantCase(
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


def generate_s8_cases(split: Split) -> tuple[S8InvariantCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s8_partitions(
    train: tuple[S8InvariantCase, ...],
    dev: tuple[S8InvariantCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S8 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S8 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S8 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S8 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S8 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S8 state views must differ")
            if row.question_a1 == row.question_a2:
                raise RuntimeError("S8 A question views must differ")
            if row.question_b1 == row.question_b2:
                raise RuntimeError("S8 B question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S8 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S8 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S8 paired golds must differ")

    train_states = {
        text
        for row in train
        for text in (row.state_a, row.state_b)
    }
    dev_states = {
        text
        for row in dev
        for text in (row.state_a, row.state_b)
    }
    if train_states & dev_states:
        raise RuntimeError("S8 TRAIN/DEV state-view overlap")

    train_questions = {
        text
        for row in train
        for text in (
            row.question_a1,
            row.question_a2,
            row.question_b1,
            row.question_b2,
        )
    }
    dev_questions = {
        text
        for row in dev
        for text in (
            row.question_a1,
            row.question_a2,
            row.question_b1,
            row.question_b2,
        )
    }
    if train_questions & dev_questions:
        raise RuntimeError("S8 TRAIN/DEV question-view overlap")

    train_option_texts = {
        text for row in train for text in row.option_texts
    }
    dev_option_texts = {
        text for row in dev for text in row.option_texts
    }
    if train_option_texts & dev_option_texts:
        raise RuntimeError("S8 TRAIN/DEV option-text overlap")

    train_aliases = {
        text for row in train for text in row.option_aliases
    }
    dev_aliases = {
        text for row in dev for text in row.option_aliases
    }
    if train_aliases & dev_aliases:
        raise RuntimeError("S8 TRAIN/DEV option-alias overlap")


__all__ = [
    "S8InvariantCase",
    "generate_s8_cases",
    "validate_s8_partitions",
]
