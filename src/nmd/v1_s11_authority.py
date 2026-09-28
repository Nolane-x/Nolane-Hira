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
        "wind_substation", "wind farm substation", "WS",
        "transformer coolant", "tap position",
        ("mineral oil", "natural ester", "synthetic ester", "silicone fluid", "FR3 ester", "Midel 7131", "Envirotemp 200", "bio-based ester"),
        ("naphthenic oil", "paraffinic oil", "Midel eN", "siloxane fluid", "vegetable ester", "high-oleic ester", "ester blend X", "ester blend Y"),
        ("tap -4", "tap -3", "tap -2", "tap -1", "tap 0", "tap +1", "tap +2", "tap +3"),
        ("tap -5", "tap -2.5", "tap -1.5", "tap +0.5", "tap +1.5", "tap +2.5", "tap +3.5", "tap +4"),
        17001, 101000,
    ),
    _Domain(
        "coastal_pump", "coastal pump station", "CP",
        "pump type", "discharge head",
        ("axial flow", "mixed flow", "split case", "vertical turbine", "end suction", "multistage", "propeller", "volute"),
        ("double suction", "diagonal flow", "submersible axial", "vertical mixed", "barrel multistage", "ring section", "screw centrifugal", "pit turbine"),
        ("18 m", "24 m", "30 m", "36 m", "42 m", "48 m", "54 m", "60 m"),
        ("21 m", "27 m", "33 m", "39 m", "45 m", "51 m", "57 m", "63 m"),
        17101, 102000,
    ),
    _Domain(
        "ceramic_kiln", "continuous ceramic kiln", "CK",
        "firing atmosphere", "belt speed",
        ("oxidizing", "reducing", "neutral", "nitrogen-rich", "argon-rich", "steam-rich", "low-oxygen", "air-fired"),
        ("CO-rich", "hydrogen-rich", "vacuum-assist", "ammonia-cracked", "oxygen-enriched", "inert mix", "wet nitrogen", "dry nitrogen"),
        ("0.8 m/min", "1.0 m/min", "1.2 m/min", "1.4 m/min", "1.6 m/min", "1.8 m/min", "2.0 m/min", "2.2 m/min"),
        ("0.9 m/min", "1.1 m/min", "1.3 m/min", "1.5 m/min", "1.7 m/min", "1.9 m/min", "2.1 m/min", "2.3 m/min"),
        17201, 103000,
    ),
    _Domain(
        "orbital_payload", "orbital payload", "OP",
        "sensor mode", "downlink rate",
        ("panchromatic", "multispectral", "hyperspectral", "SAR stripmap", "SAR spotlight", "thermal IR", "UV imaging", "lidar"),
        ("SAR scanSAR", "shortwave IR", "microwave radiometer", "GNSS reflectometry", "ocean color", "stereo optical", "altimeter", "magnetometer"),
        ("50 Mbps", "75 Mbps", "100 Mbps", "125 Mbps", "150 Mbps", "175 Mbps", "200 Mbps", "225 Mbps"),
        ("60 Mbps", "85 Mbps", "110 Mbps", "135 Mbps", "160 Mbps", "185 Mbps", "210 Mbps", "235 Mbps"),
        17301, 104000,
    ),
    _Domain(
        "vertical_farm", "vertical farm zone", "VF",
        "nutrient formula", "light cycle",
        ("NF-A", "NF-B", "NF-C", "NF-D", "NF-E", "NF-F", "NF-G", "NF-H"),
        ("NF-J", "NF-K", "NF-L", "NF-M", "NF-N", "NF-P", "NF-Q", "NF-R"),
        ("14 h", "15 h", "16 h", "17 h", "18 h", "19 h", "20 h", "21 h"),
        ("13.5 h", "14.5 h", "15.5 h", "16.5 h", "17.5 h", "18.5 h", "19.5 h", "20.5 h"),
        17401, 105000,
    ),
    _Domain(
        "tunnel_ventilation", "road tunnel ventilation section", "TV",
        "fan blade profile", "airflow",
        ("NACA-0012", "NACA-4412", "Clark-Y", "S809", "FX-63", "LS-0417", "RAE-2822", "custom-A"),
        ("NACA-23012", "NACA-6409", "Eppler-387", "S814", "DU-96", "GAW-1", "custom-B", "custom-C"),
        ("42 m3/s", "48 m3/s", "54 m3/s", "60 m3/s", "66 m3/s", "72 m3/s", "78 m3/s", "84 m3/s"),
        ("45 m3/s", "51 m3/s", "57 m3/s", "63 m3/s", "69 m3/s", "75 m3/s", "81 m3/s", "87 m3/s"),
        17501, 106000,
    ),
    _Domain(
        "additive_cell", "metal additive manufacturing cell", "AM",
        "powder alloy", "layer height",
        ("Ti-6Al-4V", "Inconel 718", "316L", "AlSi10Mg", "CoCrMo", "17-4PH", "maraging 300", "Hastelloy X"),
        ("Inconel 625", "Ti-6242", "Scalmalloy", "CuCrZr", "Haynes 282", "duplex 2205", "M300 variant", "NiTi"),
        ("20 um", "30 um", "40 um", "50 um", "60 um", "70 um", "80 um", "90 um"),
        ("25 um", "35 um", "45 um", "55 um", "65 um", "75 um", "85 um", "95 um"),
        17601, 107000,
    ),
    _Domain(
        "hydrology_buoy", "hydrology buoy", "HB",
        "sensor package", "sampling interval",
        ("CTD", "ADCP", "turbidity", "fluorometer", "nitrate probe", "pH-ORP", "wave radar", "weather pack"),
        ("optical DO", "CDOM", "PAR sensor", "ammonium probe", "hydrocarbon", "silicate", "pressure array", "acoustic modem"),
        ("1 min", "2 min", "3 min", "4 min", "5 min", "6 min", "7 min", "8 min"),
        ("90 s", "150 s", "210 s", "270 s", "330 s", "390 s", "450 s", "510 s"),
        17701, 108000,
    ),
    _Domain(
        "railway_switch", "railway switch machine", "RS",
        "actuator type", "heater power",
        ("electro-mechanical", "hydraulic", "pneumatic", "linear motor", "rotary servo", "rack drive", "screw drive", "cam drive"),
        ("brushless linear", "electro-hydraulic", "servo hydraulic", "direct torque", "ball-screw", "harmonic drive", "chain drive", "magnetic latch"),
        ("120 W", "180 W", "240 W", "300 W", "360 W", "420 W", "480 W", "540 W"),
        ("150 W", "210 W", "270 W", "330 W", "390 W", "450 W", "510 W", "570 W"),
        17801, 109000,
    ),
    _Domain(
        "cold_chain_pallet", "cold-chain pallet", "CC",
        "coolant pack", "target temperature",
        ("PCM-5", "PCM-10", "gel pack", "dry ice", "eutectic plate", "salt hydrate", "paraffin PCM", "water ice"),
        ("PCM-2", "PCM-8", "vacuum gel", "CO2 snow", "bio-gel", "fatty-acid PCM", "hydrate-B", "microPCM"),
        ("2 C", "4 C", "6 C", "8 C", "10 C", "12 C", "14 C", "16 C"),
        ("1 C", "3 C", "5 C", "7 C", "9 C", "11 C", "13 C", "15 C"),
        17901, 110000,
    ),
    _Domain(
        "semiconductor_stepper", "semiconductor lithography stepper", "SS",
        "illumination wavelength", "stage speed",
        ("365 nm", "248 nm", "193 nm", "157 nm", "436 nm", "405 nm", "355 nm", "266 nm"),
        ("213 nm", "308 nm", "351 nm", "442 nm", "532 nm", "224 nm", "257 nm", "289 nm"),
        ("80 mm/s", "100 mm/s", "120 mm/s", "140 mm/s", "160 mm/s", "180 mm/s", "200 mm/s", "220 mm/s"),
        ("90 mm/s", "110 mm/s", "130 mm/s", "150 mm/s", "170 mm/s", "190 mm/s", "210 mm/s", "230 mm/s"),
        18001, 111000,
    ),
    _Domain(
        "bioreactor_line", "bioreactor production line", "BL",
        "host strain", "feed rate",
        ("E. coli BL21", "CHO-K1", "HEK293", "Pichia X33", "Bacillus 168", "Sf9", "Vero", "Corynebacterium"),
        ("E. coli C41", "CHO-S", "HEK293F", "Pichia GS115", "Bacillus WB800", "Sf21", "MDCK", "Komagataella"),
        ("0.8 L/h", "1.2 L/h", "1.6 L/h", "2.0 L/h", "2.4 L/h", "2.8 L/h", "3.2 L/h", "3.6 L/h"),
        ("1.0 L/h", "1.4 L/h", "1.8 L/h", "2.2 L/h", "2.6 L/h", "3.0 L/h", "3.4 L/h", "3.8 L/h"),
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
        for _, field, value in rows
    )
    aliases = tuple(
        f"{value} is the recorded value for {field} in this {noun}"
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
            f"S11 {spec.noun} record {code} stores {spec.field_a} {first} "
            f"and {spec.field_b} {second}."
        )
        state_b = (
            f"Record {code} gives {second} for {spec.field_b}. "
            f"The same S11 {spec.noun} entry gives {first} for {spec.field_a}."
        )
        qa1 = f"For S11 {spec.noun} {code}, what is the {spec.field_a}?"
        qa2 = f"Which value is assigned to {spec.field_a} in record {code}?"
        qb1 = f"For S11 {spec.noun} {code}, what is the {spec.field_b}?"
        qb2 = f"Which value is assigned to {spec.field_b} in record {code}?"
    else:
        state_a = (
            f"Audit sheet {code} describes the {spec.noun}: "
            f"{spec.field_a} appears as {first}; meanwhile the "
            f"{spec.field_b} entry reads {second}."
        )
        state_b = (
            f"On audit sheet {code}, {second} is filed beneath {spec.field_b}. "
            f"A separate line for the same {spec.noun} lists {first} beneath "
            f"{spec.field_a}."
        )
        qa1 = f"Read audit sheet {code} and identify its {spec.field_a}."
        qa2 = f"What does sheet {code} list under the {spec.field_a} heading?"
        qb1 = f"Read audit sheet {code} and identify its {spec.field_b}."
        qb2 = f"What does sheet {code} list under the {spec.field_b} heading?"
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
