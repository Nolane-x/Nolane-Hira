from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S16FusionCase:
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
        "supercritical_loop", "supercritical CO2 test loop", "SL", "heat-exchanger alloy", "circulation rate",
        ("Inconel 625", "Haynes 230", "316L", "347H", "Incoloy 800H", "Alloy 617", "SS310", "Hastelloy X"),
        ("Alloy 740H", "HR6W", "Sanicro 25", "NF709", "Alloy 282", "Inconel 718", "253MA", "C22"),
        ("2 kg/s", "3 kg/s", "4 kg/s", "5 kg/s", "6 kg/s", "7 kg/s", "8 kg/s", "9 kg/s"),
        ("2.5 kg/s", "3.5 kg/s", "4.5 kg/s", "5.5 kg/s", "6.5 kg/s", "7.5 kg/s", "8.5 kg/s", "9.5 kg/s"),
        25001, 201000,
    ),
    _Domain(
        "lidar_receiver", "coherent lidar receiver", "LR", "local oscillator", "sample rate",
        ("1550 nm DFB", "1064 nm NPRO", "780 nm ECDL", "1310 nm DFB", "2 um Tm fiber", "850 nm VCSEL", "532 nm DPSS", "1535 nm DBR"),
        ("1625 nm DFB", "1030 nm fiber", "795 nm ECDL", "1490 nm DFB", "2050 nm Tm", "980 nm DBR", "515 nm DPSS", "1560 nm ECL"),
        ("50 MS/s", "100 MS/s", "150 MS/s", "200 MS/s", "250 MS/s", "300 MS/s", "350 MS/s", "400 MS/s"),
        ("75 MS/s", "125 MS/s", "175 MS/s", "225 MS/s", "275 MS/s", "325 MS/s", "375 MS/s", "425 MS/s"),
        25101, 202000,
    ),
    _Domain(
        "protein_crystallizer", "protein crystallization robot", "PR", "precipitant", "drop volume",
        ("PEG 3350", "PEG 8000", "ammonium sulfate", "MPD", "NaCl", "sodium citrate", "ethanol", "Jeffamine"),
        ("PEG 400", "PEG 6000", "lithium sulfate", "isopropanol", "magnesium chloride", "sodium malonate", "acetone", "Tacsimate"),
        ("100 nL", "150 nL", "200 nL", "250 nL", "300 nL", "350 nL", "400 nL", "450 nL"),
        ("125 nL", "175 nL", "225 nL", "275 nL", "325 nL", "375 nL", "425 nL", "475 nL"),
        25201, 203000,
    ),
    _Domain(
        "plasma_diagnostic", "tokamak plasma diagnostic", "PD", "diagnostic channel", "sample window",
        ("Thomson scattering", "ECE", "interferometer", "bolometer", "MSE", "soft X-ray", "reflectometry", "charge exchange"),
        ("ECE imaging", "polarimetry", "neutron camera", "gamma spectrometer", "Langmuir probe", "D-alpha", "FIR laser", "visible bremsstrahlung"),
        ("2 us", "4 us", "6 us", "8 us", "10 us", "12 us", "14 us", "16 us"),
        ("3 us", "5 us", "7 us", "9 us", "11 us", "13 us", "15 us", "17 us"),
        25301, 204000,
    ),
    _Domain(
        "battery_cycler", "battery cycler rack", "BC", "cell chemistry", "charge current",
        ("LFP", "NMC811", "NCA", "LCO", "LMO", "LTO", "sodium-ion", "Li-S"),
        ("NMC622", "NMC532", "LNMO", "LMFP", "Prussian white", "hard-carbon Na", "solid-state Li", "Li-air"),
        ("1 A", "2 A", "3 A", "4 A", "5 A", "6 A", "7 A", "8 A"),
        ("1.5 A", "2.5 A", "3.5 A", "4.5 A", "5.5 A", "6.5 A", "7.5 A", "8.5 A"),
        25401, 205000,
    ),
    _Domain(
        "microwave_source", "high-power microwave source", "MS", "source topology", "pulse width",
        ("magnetron", "klystron", "TWT", "gyrotron", "BWO", "vircator", "solid-state array", "clinotron"),
        ("gyro-klystron", "CFA", "extended-interaction klystron", "FEL", "IMPATT array", "SWS TWT", "radial-line oscillator", "orotron"),
        ("50 ns", "100 ns", "150 ns", "200 ns", "250 ns", "300 ns", "350 ns", "400 ns"),
        ("75 ns", "125 ns", "175 ns", "225 ns", "275 ns", "325 ns", "375 ns", "425 ns"),
        25501, 206000,
    ),
    _Domain(
        "algae_harvester", "microalgae harvester", "AH", "separation method", "throughput",
        ("disc centrifuge", "membrane filtration", "flocculation", "DAF", "belt filter", "hydrocyclone", "electrocoagulation", "sedimentation"),
        ("spiral separator", "ultrafiltration", "magnetic harvesting", "acoustic focusing", "rotary vacuum", "crossflow", "foam fractionation", "lamella clarifier"),
        ("10 L/min", "20 L/min", "30 L/min", "40 L/min", "50 L/min", "60 L/min", "70 L/min", "80 L/min"),
        ("15 L/min", "25 L/min", "35 L/min", "45 L/min", "55 L/min", "65 L/min", "75 L/min", "85 L/min"),
        25601, 207000,
    ),
    _Domain(
        "laser_cooling", "laser-cooling apparatus", "LC", "atomic species", "detuning",
        ("rubidium-87", "cesium-133", "strontium-88", "ytterbium-174", "potassium-40", "lithium-6", "sodium-23", "calcium-40"),
        ("rubidium-85", "cesium-135", "strontium-87", "ytterbium-171", "potassium-39", "lithium-7", "magnesium-24", "barium-138"),
        ("-1 MHz", "-2 MHz", "-3 MHz", "-4 MHz", "-5 MHz", "-6 MHz", "-7 MHz", "-8 MHz"),
        ("-1.5 MHz", "-2.5 MHz", "-3.5 MHz", "-4.5 MHz", "-5.5 MHz", "-6.5 MHz", "-7.5 MHz", "-8.5 MHz"),
        25701, 208000,
    ),
    _Domain(
        "composite_autoclave", "composite curing autoclave", "CA", "bagging film", "cure pressure",
        ("nylon 6", "nylon 66", "FEP", "PTFE", "PEEK film", "polyimide", "ETFE", "PVF"),
        ("PFA", "PMP", "PET", "PPS", "PEI", "PES", "ECTFE", "PVDF"),
        ("2 bar", "3 bar", "4 bar", "5 bar", "6 bar", "7 bar", "8 bar", "9 bar"),
        ("2.5 bar", "3.5 bar", "4.5 bar", "5.5 bar", "6.5 bar", "7.5 bar", "8.5 bar", "9.5 bar"),
        25801, 209000,
    ),
    _Domain(
        "acoustic_modem", "underwater acoustic modem", "UM", "modulation", "symbol rate",
        ("BPSK", "QPSK", "OFDM", "FSK", "DSSS", "QAM16", "MSK", "chirp spread"),
        ("8PSK", "OQPSK", "FBMC", "MFSK", "CSS", "QAM64", "GMSK", "wavelet-OFDM"),
        ("100 sym/s", "200 sym/s", "300 sym/s", "400 sym/s", "500 sym/s", "600 sym/s", "700 sym/s", "800 sym/s"),
        ("150 sym/s", "250 sym/s", "350 sym/s", "450 sym/s", "550 sym/s", "650 sym/s", "750 sym/s", "850 sym/s"),
        25901, 210000,
    ),
    _Domain(
        "vacuum_furnace", "vacuum heat-treatment furnace", "VF", "heating element", "soak temperature",
        ("graphite", "molybdenum", "tungsten", "SiC", "Kanthal", "nichrome", "MoSi2", "carbon composite"),
        ("tantalum", "zirconium diboride", "lanthanum chromite", "Inconel ribbon", "FeCrAl", "platinum", "TiB2", "hafnium carbide"),
        ("600 C", "700 C", "800 C", "900 C", "1000 C", "1100 C", "1200 C", "1300 C"),
        ("650 C", "750 C", "850 C", "950 C", "1050 C", "1150 C", "1250 C", "1350 C"),
        26001, 211000,
    ),
    _Domain(
        "spectral_flow", "spectral flow cytometer", "SF", "fluorophore", "event rate",
        ("FITC", "PE", "APC", "PerCP", "BV421", "Alexa488", "Cy5", "DAPI"),
        ("PE-Cy7", "APC-Cy7", "BV510", "BV605", "Alexa647", "mCherry", "eFluor450", "7-AAD"),
        ("5000/s", "10000/s", "15000/s", "20000/s", "25000/s", "30000/s", "35000/s", "40000/s"),
        ("7500/s", "12500/s", "17500/s", "22500/s", "27500/s", "32500/s", "37500/s", "42500/s"),
        26101, 212000,
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
            f"S16 {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"Record {code} lists {second} beside {spec.field_b}. "
            f"In the same S16 {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S16 {spec.noun} {code}, what is the {spec.field_a}?"
        qa2 = f"Which value fills {spec.field_a} in record {code}?"
        qb1 = f"For S16 {spec.noun} {code}, what is the {spec.field_b}?"
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


def _build(spec: _Domain, index: int, split: Split) -> S16FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s16-{spec.name}-{index:03d}"

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
    return S16FusionCase(
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


def generate_s16_cases(split: Split) -> tuple[S16FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s16_partitions(
    train: tuple[S16FusionCase, ...],
    dev: tuple[S16FusionCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S16 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S16 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S16 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S16 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("S16 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S16 state views must differ")
            if row.question_a1 == row.question_a2 or row.question_b1 == row.question_b2:
                raise RuntimeError("S16 question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S16 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S16 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S16 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S16 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S16 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S16 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S16 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S16 TRAIN/DEV option-alias overlap")


__all__ = [
    "S16FusionCase",
    "generate_s16_cases",
    "validate_s16_partitions",
]
