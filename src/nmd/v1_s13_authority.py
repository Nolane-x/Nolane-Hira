from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S13CanonicalCase:
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
        "fusion_loop", "fusion coolant loop", "FU", "coolant chemistry", "primary flow",
        ("FLiBe", "FLiNaK", "helium", "water", "lead-lithium", "sodium", "supercritical CO2", "nitrogen"),
        ("lithium", "molten chloride", "argon", "heavy water", "tin-lithium", "potassium", "neon", "hydrogen"),
        ("18 kg/s", "22 kg/s", "26 kg/s", "30 kg/s", "34 kg/s", "38 kg/s", "42 kg/s", "46 kg/s"),
        ("20 kg/s", "24 kg/s", "28 kg/s", "32 kg/s", "36 kg/s", "40 kg/s", "44 kg/s", "48 kg/s"),
        19001, 141000,
    ),
    _Domain(
        "lidar_station", "atmospheric lidar station", "LS", "laser wavelength", "pulse rate",
        ("355 nm", "532 nm", "1064 nm", "1570 nm", "905 nm", "1550 nm", "266 nm", "808 nm"),
        ("349 nm", "515 nm", "1030 nm", "1545 nm", "940 nm", "1310 nm", "213 nm", "980 nm"),
        ("10 Hz", "20 Hz", "30 Hz", "40 Hz", "50 Hz", "60 Hz", "70 Hz", "80 Hz"),
        ("15 Hz", "25 Hz", "35 Hz", "45 Hz", "55 Hz", "65 Hz", "75 Hz", "85 Hz"),
        19101, 142000,
    ),
    _Domain(
        "digester", "anaerobic digester", "AD", "feedstock", "retention time",
        ("food waste", "sludge", "manure", "corn silage", "brewery waste", "glycerol", "whey", "straw"),
        ("algae", "bagasse", "molasses", "poultry litter", "rice husk", "spent grain", "paper sludge", "sewage scum"),
        ("12 d", "16 d", "20 d", "24 d", "28 d", "32 d", "36 d", "40 d"),
        ("14 d", "18 d", "22 d", "26 d", "30 d", "34 d", "38 d", "42 d"),
        19201, 143000,
    ),
    _Domain(
        "beamline", "synchrotron beamline", "SB", "monochromator crystal", "photon energy",
        ("Si111", "Si311", "diamond111", "Ge111", "Si220", "quartz", "sapphire", "graphite"),
        ("diamond220", "Ge220", "Si400", "Si511", "LiF200", "CaF2", "MgO", "KTP"),
        ("6 keV", "8 keV", "10 keV", "12 keV", "14 keV", "16 keV", "18 keV", "20 keV"),
        ("7 keV", "9 keV", "11 keV", "13 keV", "15 keV", "17 keV", "19 keV", "21 keV"),
        19301, 144000,
    ),
    _Domain(
        "electrolyzer", "hydrogen electrolyzer", "HE", "membrane type", "stack current",
        ("PEM", "AEM", "alkaline", "SOEC", "PBI", "Nafion117", "Nafion212", "SPEEK"),
        ("PFSA-short", "FAA3", "Zirfon", "ceramic proton", "ETFE-graft", "Sustainion", "Aquivion", "PES"),
        ("120 A", "160 A", "200 A", "240 A", "280 A", "320 A", "360 A", "400 A"),
        ("140 A", "180 A", "220 A", "260 A", "300 A", "340 A", "380 A", "420 A"),
        19401, 145000,
    ),
    _Domain(
        "wafer_metrology", "wafer metrology station", "WM", "measurement mode", "scan pitch",
        ("ellipsometry", "scatterometry", "AFM", "profilometry", "interferometry", "Raman", "XRR", "reflectometry"),
        ("CD-SEM", "XPS", "SIMS", "FTIR", "photoluminescence", "OCT", "confocal", "dark-field"),
        ("2 um", "4 um", "6 um", "8 um", "10 um", "12 um", "14 um", "16 um"),
        ("3 um", "5 um", "7 um", "9 um", "11 um", "13 um", "15 um", "17 um"),
        19501, 146000,
    ),
    _Domain(
        "reef_sensor", "reef monitoring node", "RN", "probe package", "sample depth",
        ("pH", "oxygen", "salinity", "PAR", "chlorophyll", "turbidity", "nitrate", "temperature"),
        ("ORP", "CDOM", "ammonium", "phosphate", "fluorescence", "pressure", "current", "acoustic"),
        ("1 m", "2 m", "3 m", "4 m", "5 m", "6 m", "7 m", "8 m"),
        ("1.5 m", "2.5 m", "3.5 m", "4.5 m", "5.5 m", "6.5 m", "7.5 m", "8.5 m"),
        19601, 147000,
    ),
    _Domain(
        "cargo_drone", "cargo drone", "CD", "propeller type", "cruise speed",
        ("two-blade", "three-blade", "folding", "ducted", "coaxial", "variable-pitch", "scimitar", "contra-rotating"),
        ("ring-wing", "five-blade", "six-blade", "toroidal", "shrouded", "tip-driven", "cyclorotor", "fan-array"),
        ("12 m/s", "16 m/s", "20 m/s", "24 m/s", "28 m/s", "32 m/s", "36 m/s", "40 m/s"),
        ("14 m/s", "18 m/s", "22 m/s", "26 m/s", "30 m/s", "34 m/s", "38 m/s", "42 m/s"),
        19701, 148000,
    ),
    _Domain(
        "thermal_store", "thermal energy store", "TS", "storage medium", "charge temperature",
        ("molten salt", "concrete", "graphite", "sand", "phase-change wax", "ceramic", "oil", "rock bed"),
        ("liquid metal", "alumina", "magnesia", "silica", "sodium nitrate", "chloride salt", "steel shot", "glass"),
        ("250 C", "300 C", "350 C", "400 C", "450 C", "500 C", "550 C", "600 C"),
        ("275 C", "325 C", "375 C", "425 C", "475 C", "525 C", "575 C", "625 C"),
        19801, 149000,
    ),
    _Domain(
        "precision_lathe", "precision lathe", "PL", "tool insert", "feed rate",
        ("CBN", "PCD", "carbide", "ceramic", "cermet", "HSS", "diamond", "SiAlON"),
        ("whisker ceramic", "coated carbide", "TiB2", "WC-Co", "TiCN", "Al2O3", "binderless CBN", "MCD"),
        ("0.02 mm/rev", "0.04 mm/rev", "0.06 mm/rev", "0.08 mm/rev", "0.10 mm/rev", "0.12 mm/rev", "0.14 mm/rev", "0.16 mm/rev"),
        ("0.03 mm/rev", "0.05 mm/rev", "0.07 mm/rev", "0.09 mm/rev", "0.11 mm/rev", "0.13 mm/rev", "0.15 mm/rev", "0.17 mm/rev"),
        19901, 150000,
    ),
    _Domain(
        "radio_array", "radio astronomy array", "RA", "receiver band", "channel width",
        ("L band", "S band", "C band", "X band", "Ku band", "K band", "Ka band", "Q band"),
        ("P band", "VHF", "UHF", "W band", "E band", "D band", "V band", "G band"),
        ("1 kHz", "2 kHz", "4 kHz", "8 kHz", "16 kHz", "32 kHz", "64 kHz", "128 kHz"),
        ("1.5 kHz", "3 kHz", "6 kHz", "12 kHz", "24 kHz", "48 kHz", "96 kHz", "192 kHz"),
        20001, 151000,
    ),
    _Domain(
        "photobioreactor", "photobioreactor", "PR", "algae strain", "light intensity",
        ("Chlorella", "Spirulina", "Nannochloropsis", "Dunaliella", "Haematococcus", "Scenedesmus", "Tetraselmis", "Isochrysis"),
        ("Phaeodactylum", "Porphyridium", "Botryococcus", "Arthrospira", "Monoraphidium", "Picochlorum", "Desmodesmus", "Skeletonema"),
        ("80 umol/m2/s", "120 umol/m2/s", "160 umol/m2/s", "200 umol/m2/s", "240 umol/m2/s", "280 umol/m2/s", "320 umol/m2/s", "360 umol/m2/s"),
        ("100 umol/m2/s", "140 umol/m2/s", "180 umol/m2/s", "220 umol/m2/s", "260 umol/m2/s", "300 umol/m2/s", "340 umol/m2/s", "380 umol/m2/s"),
        20101, 152000,
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
            f"S13 {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"Record {code} lists {second} beside {spec.field_b}. "
            f"In the same S13 {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S13 {spec.noun} {code}, what is the {spec.field_a}?"
        qa2 = f"Which value fills {spec.field_a} in record {code}?"
        qb1 = f"For S13 {spec.noun} {code}, what is the {spec.field_b}?"
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


def _build(spec: _Domain, index: int, split: Split) -> S13CanonicalCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s13-{spec.name}-{index:03d}"

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
    return S13CanonicalCase(
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


def generate_s13_cases(split: Split) -> tuple[S13CanonicalCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s13_partitions(
    train: tuple[S13CanonicalCase, ...],
    dev: tuple[S13CanonicalCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S13 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S13 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S13 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S13 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("S13 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S13 state views must differ")
            if row.question_a1 == row.question_a2 or row.question_b1 == row.question_b2:
                raise RuntimeError("S13 question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S13 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S13 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S13 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S13 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S13 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S13 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S13 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S13 TRAIN/DEV option-alias overlap")


__all__ = [
    "S13CanonicalCase",
    "generate_s13_cases",
    "validate_s13_partitions",
]
