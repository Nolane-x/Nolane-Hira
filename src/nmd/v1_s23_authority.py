from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]

@dataclass(frozen=True)
class S23FusionCase:
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
    _Domain("magnetic_tweezers","magnetic tweezers controller","MTW","bead coating","field gradient",
        ("streptavidin","protein G","anti-digoxigenin","carboxyl","amine","silica","PEG","biotin",),
        ("avidin","Fc-binding","dig-antibody","COOH","NH2","glass","PEGylated","biotinylated",),
        ("2 T/m","4 T/m","6 T/m","8 T/m","10 T/m","12 T/m","14 T/m","16 T/m",),
        ("3 T/m","5 T/m","7 T/m","9 T/m","11 T/m","13 T/m","15 T/m","17 T/m",),41001,301000),
    _Domain("picosecond_laser","picosecond laser controller","PL","pulse picker","repetition rate",
        ("AOM","EOM","Pockels cell","acousto-optic gate","electro-optic gate","fiber switch","cavity dumper","pulse divider",),
        ("TeO2 AOM","LiNbO3 EOM","KD*P cell","RF acoustic gate","high-voltage EO gate","MEMS fiber switch","intracavity dumper","digital divider",),
        ("1 MHz","2 MHz","4 MHz","8 MHz","10 MHz","20 MHz","40 MHz","80 MHz",),
        ("1.5 MHz","3 MHz","5 MHz","9 MHz","12 MHz","24 MHz","48 MHz","96 MHz",),41101,302000),
    _Domain("ftir_stage","FTIR sample stage","FT","window material","scan velocity",
        ("KBr","CaF2","ZnSe","BaF2","diamond","silicon","quartz","sapphire",),
        ("pressed KBr","IR-grade CaF2","CVD ZnSe","optical BaF2","type-IIa diamond","float-zone silicon","fused silica","Al2O3",),
        ("0.1 cm/s","0.2 cm/s","0.3 cm/s","0.4 cm/s","0.5 cm/s","0.6 cm/s","0.7 cm/s","0.8 cm/s",),
        ("0.15 cm/s","0.25 cm/s","0.35 cm/s","0.45 cm/s","0.55 cm/s","0.65 cm/s","0.75 cm/s","0.85 cm/s",),41201,303000),
    _Domain("ion_mobility","ion mobility spectrometer","IMS","drift gas","electric field",
        ("nitrogen","helium","argon","air","carbon dioxide","neon","hydrogen","nitrous oxide",),
        ("dry N2","ultrapure He","research Ar","zero air","CO2 mix","high-purity Ne","H2 mix","N2O mix",),
        ("5 V/cm","10 V/cm","15 V/cm","20 V/cm","25 V/cm","30 V/cm","35 V/cm","40 V/cm",),
        ("7 V/cm","12 V/cm","17 V/cm","22 V/cm","27 V/cm","32 V/cm","37 V/cm","42 V/cm",),41301,304000),
    _Domain("lockin_amplifier","digital lock-in amplifier","LA","demodulation mode","time constant",
        ("single phase","dual phase","harmonic","heterodyne","PLL","multi-frequency","quadrature","vector",),
        ("in-phase only","X-Y dual","second harmonic","offset heterodyne","phase locked","multi-tone","I-Q","complex vector",),
        ("1 ms","3 ms","10 ms","30 ms","100 ms","300 ms","1 s","3 s",),
        ("2 ms","5 ms","20 ms","50 ms","200 ms","500 ms","2 s","5 s",),41401,305000),
    _Domain("plasma_etch","plasma etch reactor","PE","chemistry","bias power",
        ("SF6","CF4","CHF3","Cl2","BCl3","HBr","O2","Ar",),
        ("SF6-O2","CF4-H2","CHF3-Ar","Cl2-BCl3","BCl3-Cl2","HBr-O2","oxygen ash","argon sputter",),
        ("20 W","40 W","60 W","80 W","100 W","120 W","140 W","160 W",),
        ("30 W","50 W","70 W","90 W","110 W","130 W","150 W","170 W",),41501,306000),
    _Domain("uv_vis","UV-visible spectrophotometer","UV","slit width","scan interval",
        ("0.5 nm","1.0 nm","1.5 nm","2.0 nm","2.5 nm","3.0 nm","3.5 nm","4.0 nm",),
        ("0.6 nm","1.1 nm","1.6 nm","2.1 nm","2.6 nm","3.1 nm","3.6 nm","4.1 nm",),
        ("0.1 nm","0.2 nm","0.3 nm","0.4 nm","0.5 nm","0.6 nm","0.7 nm","0.8 nm",),
        ("0.15 nm","0.25 nm","0.35 nm","0.45 nm","0.55 nm","0.65 nm","0.75 nm","0.85 nm",),41601,307000),
    _Domain("microprobe","electron microprobe","MP","crystal analyzer","beam current",
        ("TAP","PET","LIF","LDE1","LDE2","PC1","PC2","multilayer",),
        ("TAP-100","PET-200","LiF-220","LDE1H","LDE2H","PC1-60","PC2-45","synthetic multilayer",),
        ("5 nA","10 nA","15 nA","20 nA","25 nA","30 nA","35 nA","40 nA",),
        ("7 nA","12 nA","17 nA","22 nA","27 nA","32 nA","37 nA","42 nA",),41701,308000),
    _Domain("optical_profiler","optical surface profiler","OP","objective mode","vertical range",
        ("white-light","phase-shift","confocal","focus variation","differential","Mirau","Michelson","Linnik",),
        ("broadband VSI","PSI","chromatic confocal","focus stack","DIC","Mirau VSI","Michelson VSI","Linnik interferometry",),
        ("5 um","10 um","20 um","40 um","80 um","160 um","320 um","640 um",),
        ("7 um","15 um","30 um","60 um","120 um","240 um","480 um","960 um",),41801,309000),
    _Domain("rf_probe","RF network probe fixture","RF","calibration method","IF bandwidth",
        ("SOLT","TRL","LRRM","unknown-thru","ECal","LRL","multiline TRL","response",),
        ("electronic SOLT","on-wafer TRL","LRM","UOSM","automatic ECal","line-reflect-line","multi-line TRL","response-through",),
        ("10 Hz","30 Hz","100 Hz","300 Hz","1 kHz","3 kHz","10 kHz","30 kHz",),
        ("20 Hz","50 Hz","200 Hz","500 Hz","2 kHz","5 kHz","20 kHz","50 kHz",),41901,310000),
    _Domain("cryo_microscope","cryogenic microscope stage","CM","cooling mode","sample temperature",
        ("liquid helium","closed-cycle","liquid nitrogen","dilution","pulse tube","Joule-Thomson","sorption","adiabatic",),
        ("LHe bath","GM cryocooler","LN2 bath","dilution fridge","pulse-tube cryostat","JT cooler","He3 sorption","ADR",),
        ("4 K","6 K","8 K","10 K","20 K","40 K","60 K","80 K",),
        ("5 K","7 K","9 K","12 K","25 K","45 K","65 K","85 K",),42001,311000),
    _Domain("particle_sizer","dynamic light scattering analyzer","DLS","correlator mode","scattering angle",
        ("multi-tau","linear","cross-correlation","photon counting","heterodyne","two-color","backscatter","side scatter",),
        ("adaptive multi-tau","uniform lag","dual-detector cross","TCSPC","frequency-shifted","dual-wavelength","NIBS backscatter","90-degree side",),
        ("15 deg","30 deg","45 deg","60 deg","75 deg","90 deg","120 deg","150 deg",),
        ("20 deg","35 deg","50 deg","65 deg","80 deg","95 deg","125 deg","155 deg",),42101,312000),
)

def _shuffle(*, case_id: str, noun: str, field_a: str, field_b: str,
             first: str, second: str, distractor_first: str,
             distractor_second: str, seed: int):
    rows = [
        ("a", field_a, first),
        ("b", field_b, second),
        ("x", field_a, distractor_first),
        ("y", field_b, distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts = tuple(f"for the {noun}, {field} is {value}" for _k, field, value in rows)
    aliases = tuple(f"{value} is the recorded {field} entry in this {noun}" for _k, field, value in rows)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i,(kind,_f,_v) in enumerate(rows) if kind == "a")
    gb = next(i for i,(kind,_f,_v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb

def _views(spec: _Domain, *, split: Split, code: str, first: str, second: str):
    if split == "train":
        state_a = (
            f"S23-Bisector {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"S23-Bisector record {code} stores {second} beside {spec.field_b}. "
            f"The same {spec.noun} assigns {first} to {spec.field_a}."
        )
        qa1 = f"For S23-Bisector {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S23-Bisector record {code}, which entry is filed as {spec.field_a}?"
        qb1 = f"For S23-Bisector {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S23-Bisector record {code}, which entry is filed as {spec.field_b}?"
    else:
        state_a = (
            f"S23-Bisector audit sheet {code} for the {spec.noun} assigns "
            f"{first} to {spec.field_a}, while {spec.field_b} is documented as {second}."
        )
        state_b = (
            f"On S23-Bisector sheet {code}, {second} is the {spec.field_b} setting. "
            f"The {spec.noun} sheet separately records {first} for {spec.field_a}."
        )
        qa1 = f"Consult S23-Bisector audit {code}: what is the {spec.field_a} entry?"
        qa2 = f"Which value carries the {spec.field_a} label on S23-Bisector sheet {code}?"
        qb1 = f"Consult S23-Bisector audit {code}: what is the {spec.field_b} entry?"
        qb2 = f"Which value carries the {spec.field_b} label on S23-Bisector sheet {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2

def _build(spec: _Domain, index: int, split: Split) -> S23FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s23-{spec.name}-{index:03d}"
    state_a, state_b, qa1, qa2, qb1, qb2 = _views(
        spec, split=split, code=code, first=first, second=second
    )
    texts, aliases, ids, ga, gb = _shuffle(
        case_id=case_id, noun=spec.noun, field_a=spec.field_a, field_b=spec.field_b,
        first=first, second=second, distractor_first=distractor_first,
        distractor_second=distractor_second, seed=spec.seed + offset,
    )
    return S23FusionCase(
        case_id=case_id, split=split, domain=spec.name, language="en",
        state_a=state_a, state_b=state_b, question_a1=qa1, question_a2=qa2,
        question_b1=qb1, question_b2=qb2, option_texts=texts,
        option_aliases=aliases, option_ids=ids, gold_a=ga, gold_b=gb,
    )

def generate_s23_cases(split: Split) -> tuple[S23FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, index, split) for spec in _DOMAINS for index in range(count))

def validate_s23_partitions(train: tuple[S23FusionCase, ...], dev: tuple[S23FusionCase, ...]) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S23 partition size changed")
    if len({row.domain for row in train}) != 12 or len({row.domain for row in dev}) != 12:
        raise RuntimeError("S23 domain count changed")
    if any(row.split != "train" for row in train) or any(row.split != "dev" for row in dev):
        raise RuntimeError("S23 split labels changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("S23 language changed")
    if any(len(row.option_texts) != 4 or len(row.option_aliases) != 4 or len(row.option_ids) != 4 for row in (*train, *dev)):
        raise RuntimeError("S23 option cardinality changed")
    train_states = {x for row in train for x in (row.state_a,row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a,row.state_b)}
    train_q = {x for row in train for x in (row.question_a1,row.question_a2,row.question_b1,row.question_b2)}
    dev_q = {x for row in dev for x in (row.question_a1,row.question_a2,row.question_b1,row.question_b2)}
    train_options = {x for row in train for x in (*row.option_texts,*row.option_aliases)}
    dev_options = {x for row in dev for x in (*row.option_texts,*row.option_aliases)}
    if train_states & dev_states:
        raise RuntimeError("S23 TRAIN/DEV exact state overlap")
    if train_q & dev_q:
        raise RuntimeError("S23 TRAIN/DEV exact question overlap")
    if train_options & dev_options:
        raise RuntimeError("S23 TRAIN/DEV exact option overlap")

__all__ = ["S23FusionCase","generate_s23_cases","validate_s23_partitions"]
