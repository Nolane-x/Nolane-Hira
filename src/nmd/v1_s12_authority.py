from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S12RelationCase:
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
        "tidal_turbine", "tidal turbine", "TT", "blade profile", "rotor speed",
        ("NACA 4412", "NACA 0018", "S814", "S822", "DU 91-W2-250", "FFA-W3-241", "FX 63-137", "SG6043"),
        ("NACA 63415", "NACA 0021", "S809", "S825", "DU 97-W-300", "FFA-W3-301", "RG15", "E387"),
        ("8 rpm", "10 rpm", "12 rpm", "14 rpm", "16 rpm", "18 rpm", "20 rpm", "22 rpm"),
        ("9 rpm", "11 rpm", "13 rpm", "15 rpm", "17 rpm", "19 rpm", "21 rpm", "23 rpm"),
        18001, 121000,
    ),
    _Domain(
        "plasma_etcher", "plasma etcher", "PE", "process gas", "chamber pressure",
        ("SF6", "CF4", "CHF3", "Cl2", "BCl3", "HBr", "O2", "Ar"),
        ("C4F8", "NF3", "SiCl4", "N2", "He", "Xe", "CO", "H2"),
        ("12 mTorr", "18 mTorr", "24 mTorr", "30 mTorr", "36 mTorr", "42 mTorr", "48 mTorr", "54 mTorr"),
        ("15 mTorr", "21 mTorr", "27 mTorr", "33 mTorr", "39 mTorr", "45 mTorr", "51 mTorr", "57 mTorr"),
        18101, 122000,
    ),
    _Domain(
        "grain_silo", "grain silo", "GS", "aeration mode", "target moisture",
        ("continuous", "night-only", "pulse", "reverse-flow", "low-static", "adaptive", "top-down", "bottom-up"),
        ("dewpoint-control", "intermittent", "solar-assisted", "pressure-balanced", "zoned", "humidity-triggered", "variable-speed", "recirculating"),
        ("11 percent", "12 percent", "13 percent", "14 percent", "15 percent", "16 percent", "17 percent", "18 percent"),
        ("11.5 percent", "12.5 percent", "13.5 percent", "14.5 percent", "15.5 percent", "16.5 percent", "17.5 percent", "18.5 percent"),
        18201, 123000,
    ),
    _Domain(
        "datacenter_loop", "data-center cooling loop", "DL", "coolant", "supply temperature",
        ("water", "propylene glycol", "ethylene glycol", "Novec 649", "PAO", "mineral oil", "deionized water", "HFE-7100"),
        ("HFE-7500", "FC-3283", "silicone oil", "bio-glycol", "R1234ze", "R515B", "water-glycol 30%", "water-glycol 50%"),
        ("14 C", "16 C", "18 C", "20 C", "22 C", "24 C", "26 C", "28 C"),
        ("15 C", "17 C", "19 C", "21 C", "23 C", "25 C", "27 C", "29 C"),
        18301, 124000,
    ),
    _Domain(
        "marine_sonar", "marine sonar", "MS", "pulse mode", "center frequency",
        ("CW", "LFM", "HFM", "Barker", "Costas", "BPSK", "QPSK", "stepped-frequency"),
        ("NLFM", "Frank", "P1", "P2", "P3", "P4", "Golay", "polyphase"),
        ("12 kHz", "18 kHz", "24 kHz", "30 kHz", "36 kHz", "42 kHz", "48 kHz", "54 kHz"),
        ("15 kHz", "21 kHz", "27 kHz", "33 kHz", "39 kHz", "45 kHz", "51 kHz", "57 kHz"),
        18401, 125000,
    ),
    _Domain(
        "nutrient_line", "greenhouse nutrient line", "NL", "fertilizer blend", "EC target",
        ("A/B tomato", "A/B lettuce", "calcium-rich", "potassium-rich", "low-N", "high-P", "micro mix", "fruiting mix"),
        ("seedling mix", "leafy mix", "flower mix", "silica mix", "low-salt", "organic hydro", "iron-rich", "magnesium-rich"),
        ("1.2 mS/cm", "1.4 mS/cm", "1.6 mS/cm", "1.8 mS/cm", "2.0 mS/cm", "2.2 mS/cm", "2.4 mS/cm", "2.6 mS/cm"),
        ("1.3 mS/cm", "1.5 mS/cm", "1.7 mS/cm", "1.9 mS/cm", "2.1 mS/cm", "2.3 mS/cm", "2.5 mS/cm", "2.7 mS/cm"),
        18501, 126000,
    ),
    _Domain(
        "traction_converter", "rail traction converter", "TC", "motor type", "DC bus voltage",
        ("PMSM", "induction", "SRM", "synchronous reluctance", "BLDC", "wound-field synchronous", "axial-flux PMSM", "linear induction"),
        ("IPMSM", "line-start synchronous", "dual-stator induction", "flux-switching", "transverse-flux", "vernier", "linear synchronous", "doubly-fed"),
        ("600 V", "750 V", "900 V", "1050 V", "1200 V", "1350 V", "1500 V", "1650 V"),
        ("675 V", "825 V", "975 V", "1125 V", "1275 V", "1425 V", "1575 V", "1725 V"),
        18601, 127000,
    ),
    _Domain(
        "powder_printer", "metal powder-bed printer", "PB", "powder alloy", "layer height",
        ("Ti-6Al-4V", "316L", "AlSi10Mg", "Inconel 718", "CoCr", "maraging 300", "17-4PH", "CuCrZr"),
        ("Inconel 625", "Ti-6242", "Scalmalloy", "Hastelloy X", "tool steel H13", "M2 steel", "pure copper", "nickel 200"),
        ("20 um", "30 um", "40 um", "50 um", "60 um", "70 um", "80 um", "90 um"),
        ("25 um", "35 um", "45 um", "55 um", "65 um", "75 um", "85 um", "95 um"),
        18701, 128000,
    ),
    _Domain(
        "earth_payload", "Earth-observation payload", "EO", "sensor band", "integration period",
        ("visible", "NIR", "SWIR", "MWIR", "LWIR", "UV", "C-band SAR", "X-band SAR"),
        ("L-band SAR", "Ku-band", "Ka-band", "hyperspectral VNIR", "hyperspectral SWIR", "thermal dual-band", "multispectral red-edge", "microwave radiometer"),
        ("2 ms", "4 ms", "6 ms", "8 ms", "10 ms", "12 ms", "14 ms", "16 ms"),
        ("3 ms", "5 ms", "7 ms", "9 ms", "11 ms", "13 ms", "15 ms", "17 ms"),
        18801, 129000,
    ),
    _Domain(
        "battery_test", "battery test channel", "BT", "cathode chemistry", "charge rate",
        ("LFP", "NMC811", "NMC622", "NCA", "LCO", "LMO", "LNMO", "sulfur"),
        ("NMC532", "LFP-Mn", "LMFP", "LNO", "high-Mn", "Prussian blue", "organic cathode", "conversion cathode"),
        ("0.2 C", "0.4 C", "0.6 C", "0.8 C", "1.0 C", "1.2 C", "1.4 C", "1.6 C"),
        ("0.3 C", "0.5 C", "0.7 C", "0.9 C", "1.1 C", "1.3 C", "1.5 C", "1.7 C"),
        18901, 130000,
    ),
    _Domain(
        "optical_coater", "optical coating chamber", "OC", "coating material", "deposition rate",
        ("SiO2", "TiO2", "Ta2O5", "HfO2", "Al2O3", "MgF2", "ZnS", "Y2O3"),
        ("Nb2O5", "ZrO2", "LaF3", "CeF3", "Si3N4", "ITO", "AZO", "Cr2O3"),
        ("0.2 nm/s", "0.4 nm/s", "0.6 nm/s", "0.8 nm/s", "1.0 nm/s", "1.2 nm/s", "1.4 nm/s", "1.6 nm/s"),
        ("0.3 nm/s", "0.5 nm/s", "0.7 nm/s", "0.9 nm/s", "1.1 nm/s", "1.3 nm/s", "1.5 nm/s", "1.7 nm/s"),
        19001, 131000,
    ),
    _Domain(
        "weather_buoy", "weather buoy", "WB", "sensor package", "sampling interval",
        ("wind", "wave", "barometer", "thermistor", "salinity", "radiation", "visibility", "precipitation"),
        ("ceilometer", "current meter", "fluorometer", "oxygen", "pyranometer", "lightning", "aerosol", "camera"),
        ("10 s", "20 s", "30 s", "40 s", "50 s", "60 s", "70 s", "80 s"),
        ("15 s", "25 s", "35 s", "45 s", "55 s", "65 s", "75 s", "85 s"),
        19101, 132000,
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
            f"S12 {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"Record {code} lists {second} beside {spec.field_b}. "
            f"In the same S12 {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S12 {spec.noun} {code}, what is the {spec.field_a}?"
        qa2 = f"Which value fills {spec.field_a} in record {code}?"
        qb1 = f"For S12 {spec.noun} {code}, what is the {spec.field_b}?"
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


def _build(spec: _Domain, index: int, split: Split) -> S12RelationCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s12-{spec.name}-{index:03d}"

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
    return S12RelationCase(
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


def generate_s12_cases(split: Split) -> tuple[S12RelationCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s12_partitions(
    train: tuple[S12RelationCase, ...],
    dev: tuple[S12RelationCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S12 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S12 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S12 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S12 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("S12 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S12 state views must differ")
            if row.question_a1 == row.question_a2 or row.question_b1 == row.question_b2:
                raise RuntimeError("S12 question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S12 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S12 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S12 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S12 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S12 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S12 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S12 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S12 TRAIN/DEV option-alias overlap")


__all__ = [
    "S12RelationCase",
    "generate_s12_cases",
    "validate_s12_partitions",
]
