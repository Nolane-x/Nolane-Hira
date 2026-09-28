from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S9MarginCase:
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
        "battery_pack", "battery pack", "BP",
        "cell chemistry", "nominal voltage",
        ("LFP", "NMC811", "NMC622", "LTO", "LMO", "NCA", "sodium-ion", "zinc-ion"),
        ("solid-state sulfide", "solid-state oxide", "lithium-sulfur", "silicon-anode NMC", "Prussian-blue sodium", "iron-air", "zinc-bromine", "aluminum-ion"),
        ("48 V", "72 V", "96 V", "120 V", "144 V", "168 V", "192 V", "216 V"),
        ("54 V", "78 V", "102 V", "126 V", "150 V", "174 V", "198 V", "222 V"),
        14901, 73000,
    ),
    _Domain(
        "telescope_calibration", "telescope calibration", "TC",
        "reference star", "exposure time",
        ("Vega", "Sirius", "Altair", "Deneb", "Arcturus", "Capella", "Rigel", "Betelgeuse"),
        ("Procyon", "Spica", "Fomalhaut", "Aldebaran", "Pollux", "Regulus", "Antares", "Canopus"),
        ("12 s", "18 s", "24 s", "30 s", "36 s", "42 s", "48 s", "54 s"),
        ("15 s", "21 s", "27 s", "33 s", "39 s", "45 s", "51 s", "57 s"),
        15001, 74000,
    ),
    _Domain(
        "aquaculture_tank", "aquaculture tank", "AQ",
        "species", "salinity",
        ("tilapia", "trout", "carp", "barramundi", "catfish", "perch", "salmon", "sturgeon"),
        ("cobia", "grouper", "snapper", "pompano", "milkfish", "mullet", "seabass", "amberjack"),
        ("2 ppt", "4 ppt", "6 ppt", "8 ppt", "10 ppt", "12 ppt", "14 ppt", "16 ppt"),
        ("3 ppt", "5 ppt", "7 ppt", "9 ppt", "11 ppt", "13 ppt", "15 ppt", "17 ppt"),
        15101, 75000,
    ),
    _Domain(
        "rail_signal", "rail signal block", "RS",
        "aspect", "approach speed",
        ("green", "double yellow", "yellow", "red", "flashing yellow", "lunar", "white", "violet"),
        ("blue", "amber", "flashing green", "flashing red", "cyan", "magenta", "orange", "teal"),
        ("20 km/h", "30 km/h", "40 km/h", "50 km/h", "60 km/h", "70 km/h", "80 km/h", "90 km/h"),
        ("25 km/h", "35 km/h", "45 km/h", "55 km/h", "65 km/h", "75 km/h", "85 km/h", "95 km/h"),
        15201, 76000,
    ),
    _Domain(
        "textile_dye", "textile dye lot", "TD",
        "dye family", "bath temperature",
        ("reactive blue", "acid red", "disperse orange", "vat indigo", "direct black", "sulfur brown", "basic violet", "pigment green"),
        ("mordant crimson", "azo yellow", "anthraquinone blue", "phthalocyanine teal", "natural madder", "logwood purple", "cochineal scarlet", "turmeric gold"),
        ("42 C", "48 C", "54 C", "60 C", "66 C", "72 C", "78 C", "84 C"),
        ("45 C", "51 C", "57 C", "63 C", "69 C", "75 C", "81 C", "87 C"),
        15301, 77000,
    ),
    _Domain(
        "drone_mission", "drone mission", "DM",
        "sensor payload", "cruise altitude",
        ("RGB camera", "thermal camera", "LiDAR", "multispectral camera", "hyperspectral camera", "SAR module", "gas sensor", "magnetometer"),
        ("event camera", "UV camera", "millimeter radar", "acoustic array", "radiation sensor", "soil radar", "fluorescence imager", "polarimetric camera"),
        ("60 m", "80 m", "100 m", "120 m", "140 m", "160 m", "180 m", "200 m"),
        ("70 m", "90 m", "110 m", "130 m", "150 m", "170 m", "190 m", "210 m"),
        15401, 78000,
    ),
    _Domain(
        "cold_storage", "cold-storage chamber", "CS",
        "commodity", "setpoint",
        ("apples", "pears", "berries", "potatoes", "cheese", "yogurt", "vaccines", "flowers"),
        ("mangoes", "kiwifruit", "avocados", "cherries", "butter", "cream", "plasma", "seedlings"),
        ("-2 C", "0 C", "2 C", "4 C", "6 C", "8 C", "10 C", "12 C"),
        ("-1 C", "1 C", "3 C", "5 C", "7 C", "9 C", "11 C", "13 C"),
        15501, 79000,
    ),
    _Domain(
        "ceramics_press", "ceramics press run", "CP",
        "powder blend", "press pressure",
        ("alumina", "zirconia", "silicon carbide", "silicon nitride", "mullite", "cordierite", "steatite", "porcelain"),
        ("boron carbide", "aluminum nitride", "magnesia", "spinel", "titania", "ferrite", "hydroxyapatite", "glass-ceramic"),
        ("40 MPa", "60 MPa", "80 MPa", "100 MPa", "120 MPa", "140 MPa", "160 MPa", "180 MPa"),
        ("50 MPa", "70 MPa", "90 MPa", "110 MPa", "130 MPa", "150 MPa", "170 MPa", "190 MPa"),
        15601, 80000,
    ),
    _Domain(
        "irrigation_zone", "irrigation zone", "IZ",
        "emitter type", "daily volume",
        ("drip tape", "pressure dripper", "micro-sprinkler", "bubbler", "soaker hose", "rotor sprinkler", "mist nozzle", "subsurface drip"),
        ("porous tube", "pulse dripper", "spinner", "impact sprinkler", "fan spray", "capillary mat", "fogger", "wick line"),
        ("12 L", "18 L", "24 L", "30 L", "36 L", "42 L", "48 L", "54 L"),
        ("15 L", "21 L", "27 L", "33 L", "39 L", "45 L", "51 L", "57 L"),
        15701, 81000,
    ),
    _Domain(
        "wind_turbine", "wind turbine service", "WT",
        "lubricant grade", "inspection interval",
        ("ISO VG 32", "ISO VG 46", "ISO VG 68", "ISO VG 100", "PAO 150", "ester 220", "synthetic 320", "mineral 460"),
        ("PAO 32", "PAO 46", "ester 68", "ester 100", "synthetic 150", "bio-oil 220", "gear oil 320", "gear oil 460"),
        ("250 h", "500 h", "750 h", "1000 h", "1250 h", "1500 h", "1750 h", "2000 h"),
        ("300 h", "550 h", "800 h", "1050 h", "1300 h", "1550 h", "1800 h", "2050 h"),
        15801, 82000,
    ),
    _Domain(
        "vaccine_batch", "vaccine batch", "VB",
        "adjuvant", "fill volume",
        ("alum", "MF59", "AS03", "CpG", "MPLA", "squalene emulsion", "calcium phosphate", "liposome"),
        ("Matrix-M", "virosome", "chitosan", "poly-ICLC", "QS-21", "delta-inulin", "nanoemulsion", "PLGA particle"),
        ("0.25 mL", "0.30 mL", "0.35 mL", "0.40 mL", "0.45 mL", "0.50 mL", "0.55 mL", "0.60 mL"),
        ("0.27 mL", "0.32 mL", "0.37 mL", "0.42 mL", "0.47 mL", "0.52 mL", "0.57 mL", "0.62 mL"),
        15901, 83000,
    ),
    _Domain(
        "data_center_rack", "data-center rack", "DRK",
        "cooling mode", "power cap",
        ("rear-door exchanger", "cold aisle", "hot aisle", "direct liquid", "immersion", "in-row cooling", "air economizer", "chilled water"),
        ("two-phase immersion", "warm-water loop", "evaporative cooling", "heat-pipe rack", "liquid-to-air", "dielectric spray", "free-air cooling", "adiabatic cooling"),
        ("8 kW", "12 kW", "16 kW", "20 kW", "24 kW", "28 kW", "32 kW", "36 kW"),
        ("10 kW", "14 kW", "18 kW", "22 kW", "26 kW", "30 kW", "34 kW", "38 kW"),
        16001, 84000,
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
        f"for the {noun}, the {field} value is {value}"
        for _, field, value in rows
    )
    aliases = tuple(
        f"{value} denotes the matching {field} entry for this {noun}"
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
            f"S9 {spec.noun} sheet {code} records {spec.field_a} {first} "
            f"and {spec.field_b} {second}."
        )
        state_b = (
            f"The {spec.field_b} for {code} is documented as {second}. "
            f"In the same {spec.noun} entry, {spec.field_a} is {first}."
        )
        qa1 = f"For S9 {spec.noun} {code}, what {spec.field_a} is recorded?"
        qa2 = f"Which {spec.field_a} value belongs to {code} in this {spec.noun} entry?"
        qb1 = f"For S9 {spec.noun} {code}, what {spec.field_b} is recorded?"
        qb2 = f"Which {spec.field_b} value belongs to {code} in this {spec.noun} entry?"
    else:
        state_a = (
            f"Inspection dossier {code} for the S9 {spec.noun} identifies "
            f"{first} under {spec.field_a}; the listed {spec.field_b} is {second}."
        )
        state_b = (
            f"According to the S9 file for {code}, {spec.field_a} equals {first}. "
            f"The separate {spec.field_b} field reports {second}."
        )
        qa1 = f"Read dossier {code}: which {spec.field_a} is specified?"
        qa2 = f"What does the {spec.field_a} field contain for {code} in the S9 {spec.noun} file?"
        qb1 = f"Read dossier {code}: which {spec.field_b} is specified?"
        qb2 = f"What does the {spec.field_b} field contain for {code} in the S9 {spec.noun} file?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S9MarginCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s9-{spec.name}-{index:03d}"

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
    return S9MarginCase(
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


def generate_s9_cases(split: Split) -> tuple[S9MarginCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s9_partitions(
    train: tuple[S9MarginCase, ...],
    dev: tuple[S9MarginCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S9 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S9 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S9 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S9 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S9 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S9 state views must differ")
            if row.question_a1 == row.question_a2:
                raise RuntimeError("S9 A question views must differ")
            if row.question_b1 == row.question_b2:
                raise RuntimeError("S9 B question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S9 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S9 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S9 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S9 gold index out of range")

    train_states = {
        text for row in train for text in (row.state_a, row.state_b)
    }
    dev_states = {
        text for row in dev for text in (row.state_a, row.state_b)
    }
    if train_states & dev_states:
        raise RuntimeError("S9 TRAIN/DEV state-view overlap")

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
        raise RuntimeError("S9 TRAIN/DEV question-view overlap")

    train_option_texts = {
        text for row in train for text in row.option_texts
    }
    dev_option_texts = {
        text for row in dev for text in row.option_texts
    }
    if train_option_texts & dev_option_texts:
        raise RuntimeError("S9 TRAIN/DEV option-text overlap")

    train_aliases = {
        text for row in train for text in row.option_aliases
    }
    dev_aliases = {
        text for row in dev for text in row.option_aliases
    }
    if train_aliases & dev_aliases:
        raise RuntimeError("S9 TRAIN/DEV option-alias overlap")


__all__ = [
    "S9MarginCase",
    "generate_s9_cases",
    "validate_s9_partitions",
]
