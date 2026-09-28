from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S10GroundingCase:
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
        "geothermal_well", "geothermal well", "GW",
        "working fluid", "flow rate",
        ("isobutane", "isopentane", "R245fa", "water", "ammonia", "propane", "CO2", "pentane"),
        ("R1233zd", "R1234ze", "butane", "ethanol", "toluene", "cyclopentane", "methanol", "hexane"),
        ("18 kg/s", "24 kg/s", "30 kg/s", "36 kg/s", "42 kg/s", "48 kg/s", "54 kg/s", "60 kg/s"),
        ("21 kg/s", "27 kg/s", "33 kg/s", "39 kg/s", "45 kg/s", "51 kg/s", "57 kg/s", "63 kg/s"),
        16001, 85000,
    ),
    _Domain(
        "satellite_downlink", "satellite downlink", "SD",
        "modulation", "symbol rate",
        ("QPSK", "8PSK", "16QAM", "32APSK", "64QAM", "BPSK", "OQPSK", "16APSK"),
        ("128QAM", "256QAM", "GMSK", "MSK", "pi/4-DQPSK", "4FSK", "8FSK", "64APSK"),
        ("2 Msps", "4 Msps", "6 Msps", "8 Msps", "10 Msps", "12 Msps", "14 Msps", "16 Msps"),
        ("3 Msps", "5 Msps", "7 Msps", "9 Msps", "11 Msps", "13 Msps", "15 Msps", "17 Msps"),
        16101, 86000,
    ),
    _Domain(
        "fermentation_tank", "fermentation tank", "FT",
        "culture strain", "agitation speed",
        ("strain A17", "strain B24", "strain C31", "strain D46", "strain E52", "strain F68", "strain G74", "strain H89"),
        ("strain J14", "strain K27", "strain L35", "strain M41", "strain N59", "strain P63", "strain Q78", "strain R86"),
        ("120 rpm", "160 rpm", "200 rpm", "240 rpm", "280 rpm", "320 rpm", "360 rpm", "400 rpm"),
        ("140 rpm", "180 rpm", "220 rpm", "260 rpm", "300 rpm", "340 rpm", "380 rpm", "420 rpm"),
        16201, 87000,
    ),
    _Domain(
        "metro_platform", "metro platform", "MP",
        "line color", "headway",
        ("scarlet", "indigo", "jade", "amber", "violet", "silver", "cobalt", "coral"),
        ("ochre", "turquoise", "maroon", "lime", "navy", "bronze", "crimson", "pearl"),
        ("3 min", "4 min", "5 min", "6 min", "7 min", "8 min", "9 min", "10 min"),
        ("3.5 min", "4.5 min", "5.5 min", "6.5 min", "7.5 min", "8.5 min", "9.5 min", "10.5 min"),
        16301, 88000,
    ),
    _Domain(
        "lab_incubator", "laboratory incubator", "LI",
        "gas mix", "temperature",
        ("5% CO2", "10% CO2", "5% O2", "10% O2", "nitrogen-rich", "air", "argon-rich", "helium-rich"),
        ("3% CO2", "7% CO2", "12% CO2", "2% O2", "8% O2", "microaerobic", "anaerobic", "hyperoxic"),
        ("30 C", "32 C", "34 C", "36 C", "38 C", "40 C", "42 C", "44 C"),
        ("31 C", "33 C", "35 C", "37 C", "39 C", "41 C", "43 C", "45 C"),
        16401, 89000,
    ),
    _Domain(
        "shipping_container", "shipping container", "SH",
        "cargo class", "seal number",
        ("machinery", "electronics", "coffee", "textiles", "ceramics", "glassware", "paper", "rubber"),
        ("medical devices", "optics", "spices", "timber", "resins", "instruments", "cables", "filters"),
        ("seal A104", "seal B208", "seal C312", "seal D416", "seal E520", "seal F624", "seal G728", "seal H832"),
        ("seal J115", "seal K219", "seal L323", "seal M427", "seal N531", "seal P635", "seal Q739", "seal R843"),
        16501, 90000,
    ),
    _Domain(
        "laser_cutter", "laser cutter", "LC",
        "assist gas", "feed speed",
        ("oxygen", "nitrogen", "argon", "air", "helium", "CO2", "neon", "hydrogen"),
        ("xenon", "krypton", "forming gas", "steam", "ammonia", "methane", "propane", "butane"),
        ("8 mm/s", "12 mm/s", "16 mm/s", "20 mm/s", "24 mm/s", "28 mm/s", "32 mm/s", "36 mm/s"),
        ("10 mm/s", "14 mm/s", "18 mm/s", "22 mm/s", "26 mm/s", "30 mm/s", "34 mm/s", "38 mm/s"),
        16601, 91000,
    ),
    _Domain(
        "river_station", "river monitoring station", "RV",
        "probe type", "sampling depth",
        ("conductivity", "pH", "turbidity", "oxygen", "chlorophyll", "nitrate", "phosphate", "temperature"),
        ("ammonium", "ORP", "salinity", "fluorescence", "hydrocarbon", "silicate", "CDOM", "pressure"),
        ("0.5 m", "1.0 m", "1.5 m", "2.0 m", "2.5 m", "3.0 m", "3.5 m", "4.0 m"),
        ("0.75 m", "1.25 m", "1.75 m", "2.25 m", "2.75 m", "3.25 m", "3.75 m", "4.25 m"),
        16701, 92000,
    ),
    _Domain(
        "camera_rig", "camera rig", "CR",
        "lens mount", "frame rate",
        ("EF", "RF", "E-mount", "L-mount", "Z-mount", "MFT", "PL", "F-mount"),
        ("X-mount", "K-mount", "A-mount", "C-mount", "B4", "Leica M", "Hasselblad X", "Phase One XF"),
        ("24 fps", "30 fps", "48 fps", "60 fps", "72 fps", "90 fps", "120 fps", "144 fps"),
        ("25 fps", "36 fps", "50 fps", "75 fps", "96 fps", "100 fps", "125 fps", "150 fps"),
        16801, 93000,
    ),
    _Domain(
        "orchard_block", "orchard block", "OB",
        "cultivar", "row spacing",
        ("Gala", "Fuji", "Honeycrisp", "Braeburn", "Jonagold", "Pink Lady", "Granny Smith", "Golden Delicious"),
        ("Jazz", "Envy", "Ambrosia", "Kanzi", "Opal", "Cosmic Crisp", "Mutsu", "Arkansas Black"),
        ("2.5 m", "3.0 m", "3.5 m", "4.0 m", "4.5 m", "5.0 m", "5.5 m", "6.0 m"),
        ("2.75 m", "3.25 m", "3.75 m", "4.25 m", "4.75 m", "5.25 m", "5.75 m", "6.25 m"),
        16901, 94000,
    ),
    _Domain(
        "compressor_stage", "compressor stage", "CM",
        "impeller alloy", "discharge pressure",
        ("7075 aluminum", "Inconel 718", "Ti-6Al-4V", "17-4PH", "316L", "duplex steel", "Monel 400", "Hastelloy C276"),
        ("Inconel 625", "Ti-6242", "maraging 300", "Nitronic 60", "2205 duplex", "2507 duplex", "Monel K500", "Hastelloy X"),
        ("2 bar", "3 bar", "4 bar", "5 bar", "6 bar", "7 bar", "8 bar", "9 bar"),
        ("2.5 bar", "3.5 bar", "4.5 bar", "5.5 bar", "6.5 bar", "7.5 bar", "8.5 bar", "9.5 bar"),
        17001, 95000,
    ),
    _Domain(
        "museum_case", "museum display case", "MC",
        "lighting mode", "humidity setpoint",
        ("warm LED", "cool LED", "fiber optic", "halogen", "UV-filtered LED", "tunable white", "amber LED", "daylight LED"),
        ("RGBW LED", "violet-safe LED", "remote phosphor", "laser-phosphor", "OLED panel", "edge light", "micro-LED", "filtered fluorescent"),
        ("38 percent", "42 percent", "46 percent", "50 percent", "54 percent", "58 percent", "62 percent", "66 percent"),
        ("40 percent", "44 percent", "48 percent", "52 percent", "56 percent", "60 percent", "64 percent", "68 percent"),
        17101, 96000,
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
        f"for the {noun}, {field} is recorded as {value}"
        for _, field, value in rows
    )
    aliases = tuple(
        f"{value} is the semantic entry for {field} in this {noun}"
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
            f"S10 {spec.noun} record {code} lists {spec.field_a} {first} "
            f"together with {spec.field_b} {second}."
        )
        state_b = (
            f"For {code}, the {spec.field_b} entry is {second}. "
            f"The same S10 {spec.noun} record states {spec.field_a} {first}."
        )
        qa1 = f"In S10 {spec.noun} {code}, which {spec.field_a} is listed?"
        qa2 = f"What value is attached to {spec.field_a} for {code}?"
        qb1 = f"In S10 {spec.noun} {code}, which {spec.field_b} is listed?"
        qb2 = f"What value is attached to {spec.field_b} for {code}?"
    else:
        state_a = (
            f"Review file {code} for the S10 {spec.noun}: the {spec.field_a} "
            f"field contains {first}, while {spec.field_b} is {second}."
        )
        state_b = (
            f"The S10 dossier for {code} reports {second} under {spec.field_b}. "
            f"Elsewhere in the same {spec.noun} dossier, {spec.field_a} is {first}."
        )
        qa1 = f"From review file {code}, identify the {spec.field_a}."
        qa2 = f"Which entry fills the {spec.field_a} field in dossier {code}?"
        qb1 = f"From review file {code}, identify the {spec.field_b}."
        qb2 = f"Which entry fills the {spec.field_b} field in dossier {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S10GroundingCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s10-{spec.name}-{index:03d}"

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
    return S10GroundingCase(
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


def generate_s10_cases(split: Split) -> tuple[S10GroundingCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s10_partitions(
    train: tuple[S10GroundingCase, ...],
    dev: tuple[S10GroundingCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S10 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S10 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S10 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S10 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S10 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S10 state views must differ")
            if row.question_a1 == row.question_a2:
                raise RuntimeError("S10 A question views must differ")
            if row.question_b1 == row.question_b2:
                raise RuntimeError("S10 B question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S10 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S10 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S10 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S10 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S10 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S10 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S10 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S10 TRAIN/DEV option-alias overlap")


__all__ = [
    "S10GroundingCase",
    "generate_s10_cases",
    "validate_s10_partitions",
]
