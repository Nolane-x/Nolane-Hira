from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S15FusionCase:
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
        "cryogenic_compressor", "cryogenic compressor", "CC", "bearing technology", "shaft speed",
        ("magnetic", "foil-gas", "ceramic ball", "hydrodynamic", "hydrostatic", "air journal", "hybrid magnetic", "gas-lubricated"),
        ("tilting-pad", "diamond-coated", "superconducting magnetic", "flexure", "spiral-groove", "carbon bush", "compliant foil", "active magnetic"),
        ("9000 rpm", "12000 rpm", "15000 rpm", "18000 rpm", "21000 rpm", "24000 rpm", "27000 rpm", "30000 rpm"),
        ("10500 rpm", "13500 rpm", "16500 rpm", "19500 rpm", "22500 rpm", "25500 rpm", "28500 rpm", "31500 rpm"),
        23001, 181000,
    ),
    _Domain(
        "photon_counter", "single-photon counter", "PC", "detector material", "gate width",
        ("InGaAs", "silicon", "SNSPD", "HgCdTe", "Ge", "SiPM", "GaAs", "InP"),
        ("NbN", "WSi", "MoSi", "GaN", "AlGaAs", "InAs", "PbS", "black silicon"),
        ("1 ns", "2 ns", "3 ns", "4 ns", "5 ns", "6 ns", "7 ns", "8 ns"),
        ("1.5 ns", "2.5 ns", "3.5 ns", "4.5 ns", "5.5 ns", "6.5 ns", "7.5 ns", "8.5 ns"),
        23101, 182000,
    ),
    _Domain(
        "membrane_reactor", "membrane reactor", "MR", "membrane type", "feed pressure",
        ("palladium", "zeolite", "silica", "perovskite", "polyimide", "MOF", "carbon molecular sieve", "PEM"),
        ("Pd-Ag", "ceramic proton", "graphene oxide", "alumina", "PTFE", "PBI", "Nafion", "mixed-matrix"),
        ("2 bar", "4 bar", "6 bar", "8 bar", "10 bar", "12 bar", "14 bar", "16 bar"),
        ("3 bar", "5 bar", "7 bar", "9 bar", "11 bar", "13 bar", "15 bar", "17 bar"),
        23201, 183000,
    ),
    _Domain(
        "adaptive_optics", "adaptive optics bench", "AO", "wavefront sensor", "loop frequency",
        ("Shack-Hartmann", "pyramid", "curvature", "phase-diversity", "Zernike", "interferometric", "plenoptic", "quad-cell"),
        ("modulated pyramid", "knife-edge", "lateral shearing", "holographic", "Foucault", "Hartmann mask", "fiber bundle", "event-camera"),
        ("200 Hz", "400 Hz", "600 Hz", "800 Hz", "1000 Hz", "1200 Hz", "1400 Hz", "1600 Hz"),
        ("300 Hz", "500 Hz", "700 Hz", "900 Hz", "1100 Hz", "1300 Hz", "1500 Hz", "1700 Hz"),
        23301, 184000,
    ),
    _Domain(
        "carbon_capture", "carbon-capture absorber", "CA", "solvent system", "lean loading",
        ("MEA", "MDEA", "AMP", "PZ", "DEEA", "K2CO3", "chilled ammonia", "ionic liquid"),
        ("AEEA", "DGA", "DEA", "TEA", "Piperazine blend", "amino-acid salt", "water-lean amine", "carbonate slurry"),
        ("0.10 mol/mol", "0.15 mol/mol", "0.20 mol/mol", "0.25 mol/mol", "0.30 mol/mol", "0.35 mol/mol", "0.40 mol/mol", "0.45 mol/mol"),
        ("0.12 mol/mol", "0.17 mol/mol", "0.22 mol/mol", "0.27 mol/mol", "0.32 mol/mol", "0.37 mol/mol", "0.42 mol/mol", "0.47 mol/mol"),
        23401, 185000,
    ),
    _Domain(
        "ion_thruster", "ion thruster", "IT", "propellant", "beam voltage",
        ("xenon", "krypton", "argon", "iodine", "bismuth", "mercury", "cesium", "neon"),
        ("magnesium", "zinc", "indium", "tin", "helium", "nitrogen", "sodium", "potassium"),
        ("600 V", "800 V", "1000 V", "1200 V", "1400 V", "1600 V", "1800 V", "2000 V"),
        ("700 V", "900 V", "1100 V", "1300 V", "1500 V", "1700 V", "1900 V", "2100 V"),
        23501, 186000,
    ),
    _Domain(
        "quantum_magnetometer", "quantum magnetometer", "QM", "sensing medium", "bias field",
        ("NV diamond", "rubidium", "cesium", "helium-3", "SQUID", "SERF vapor", "ytterbium", "potassium vapor"),
        ("silicon vacancy", "sodium vapor", "metastable helium", "BEC sodium", "fluxgate hybrid", "cold rubidium", "erbium crystal", "cesium ensemble"),
        ("5 uT", "10 uT", "15 uT", "20 uT", "25 uT", "30 uT", "35 uT", "40 uT"),
        ("7.5 uT", "12.5 uT", "17.5 uT", "22.5 uT", "27.5 uT", "32.5 uT", "37.5 uT", "42.5 uT"),
        23601, 187000,
    ),
    _Domain(
        "metal_forming", "servo metal-forming press", "MF", "die material", "stroke rate",
        ("D2 steel", "H13 steel", "carbide", "maraging steel", "tool steel A2", "M2 steel", "tungsten carbide", "ceramic"),
        ("DC53", "S7 steel", "powder HSS", "cermet", "PCBN", "M42 steel", "Stellite", "Si3N4"),
        ("20 spm", "30 spm", "40 spm", "50 spm", "60 spm", "70 spm", "80 spm", "90 spm"),
        ("25 spm", "35 spm", "45 spm", "55 spm", "65 spm", "75 spm", "85 spm", "95 spm"),
        23701, 188000,
    ),
    _Domain(
        "micropropagation", "plant micropropagation line", "MP", "growth medium", "light cycle",
        ("MS basal", "B5", "WPM", "DKW", "N6", "LS", "SH", "NN"),
        ("Gamborg modified", "QL", "Anderson", "Rugini", "Driver-Kuniyuki", "Chu N6 modified", "Knop", "White medium"),
        ("8 h", "10 h", "12 h", "14 h", "16 h", "18 h", "20 h", "22 h"),
        ("9 h", "11 h", "13 h", "15 h", "17 h", "19 h", "21 h", "23 h"),
        23801, 189000,
    ),
    _Domain(
        "mmwave_radar", "millimeter-wave radar", "MW", "chirp bandwidth", "frame period",
        ("0.5 GHz", "1.0 GHz", "1.5 GHz", "2.0 GHz", "2.5 GHz", "3.0 GHz", "3.5 GHz", "4.0 GHz"),
        ("0.75 GHz", "1.25 GHz", "1.75 GHz", "2.25 GHz", "2.75 GHz", "3.25 GHz", "3.75 GHz", "4.25 GHz"),
        ("10 ms", "20 ms", "30 ms", "40 ms", "50 ms", "60 ms", "70 ms", "80 ms"),
        ("15 ms", "25 ms", "35 ms", "45 ms", "55 ms", "65 ms", "75 ms", "85 ms"),
        23901, 190000,
    ),
    _Domain(
        "thermal_imager", "cooled thermal imager", "TI", "focal-plane material", "integration time",
        ("InSb", "MCT", "QWIP", "T2SL", "VOx", "a-Si", "PbSe", "InGaAs"),
        ("InAsSb", "HgZnTe", "GaAs QWIP", "superlattice MWIR", "graphene", "BST", "PbS", "InGaAsSb"),
        ("0.5 ms", "1 ms", "1.5 ms", "2 ms", "2.5 ms", "3 ms", "3.5 ms", "4 ms"),
        ("0.75 ms", "1.25 ms", "1.75 ms", "2.25 ms", "2.75 ms", "3.25 ms", "3.75 ms", "4.25 ms"),
        24001, 191000,
    ),
    _Domain(
        "fuel_cell_stack", "fuel-cell stack", "FC", "electrolyte type", "stack temperature",
        ("PEM", "SOFC", "AFC", "PAFC", "MCFC", "AEM", "PBI", "protonic ceramic"),
        ("anion membrane", "solid acid", "YSZ", "GDC", "ScSZ", "carbonate matrix", "phosphoric composite", "ceria composite"),
        ("60 C", "90 C", "120 C", "150 C", "180 C", "210 C", "240 C", "270 C"),
        ("75 C", "105 C", "135 C", "165 C", "195 C", "225 C", "255 C", "285 C"),
        24101, 192000,
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
            f"S15 {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"Record {code} lists {second} beside {spec.field_b}. "
            f"In the same S15 {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S15 {spec.noun} {code}, what is the {spec.field_a}?"
        qa2 = f"Which value fills {spec.field_a} in record {code}?"
        qb1 = f"For S15 {spec.noun} {code}, what is the {spec.field_b}?"
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


def _build(spec: _Domain, index: int, split: Split) -> S15FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s15-{spec.name}-{index:03d}"

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
    return S15FusionCase(
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


def generate_s15_cases(split: Split) -> tuple[S15FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s15_partitions(
    train: tuple[S15FusionCase, ...],
    dev: tuple[S15FusionCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S15 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S15 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S15 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S15 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("S15 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S15 state views must differ")
            if row.question_a1 == row.question_a2 or row.question_b1 == row.question_b2:
                raise RuntimeError("S15 question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S15 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S15 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S15 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S15 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S15 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S15 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S15 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S15 TRAIN/DEV option-alias overlap")


__all__ = [
    "S15FusionCase",
    "generate_s15_cases",
    "validate_s15_partitions",
]
