from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S26RelationCase:
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
        "lockin_amplifier", "digital lock-in amplifier", "LI",
        "reference frequency", "time constant",
        ("117 Hz","173 Hz","229 Hz","283 Hz","337 Hz","397 Hz","457 Hz","523 Hz"),
        ("131 Hz","191 Hz","241 Hz","307 Hz","359 Hz","419 Hz","479 Hz","541 Hz"),
        ("3 ms","5 ms","8 ms","12 ms","18 ms","25 ms","35 ms","50 ms"),
        ("4 ms","6 ms","9 ms","14 ms","20 ms","28 ms","40 ms","55 ms"),
        47301, 413000,
    ),
    _Domain(
        "confocal_scanner", "confocal fluorescence scanner", "CF",
        "pinhole diameter", "pixel dwell",
        ("18 um","24 um","30 um","36 um","42 um","48 um","54 um","60 um"),
        ("20 um","26 um","32 um","38 um","44 um","50 um","56 um","62 um"),
        ("1 us","2 us","3 us","4 us","5 us","6 us","7 us","8 us"),
        ("1.5 us","2.5 us","3.5 us","4.5 us","5.5 us","6.5 us","7.5 us","8.5 us"),
        47401, 414000,
    ),
    _Domain(
        "impedance_analyzer", "broadband impedance analyzer", "IZ",
        "drive level", "sweep stop",
        ("5 mV","10 mV","15 mV","20 mV","25 mV","30 mV","35 mV","40 mV"),
        ("7 mV","12 mV","17 mV","22 mV","27 mV","32 mV","37 mV","42 mV"),
        ("10 kHz","30 kHz","50 kHz","70 kHz","90 kHz","110 kHz","130 kHz","150 kHz"),
        ("20 kHz","40 kHz","60 kHz","80 kHz","100 kHz","120 kHz","140 kHz","160 kHz"),
        47501, 415000,
    ),
    _Domain(
        "vacuum_controller", "multi-gauge vacuum controller", "VG",
        "gauge mode", "sample period",
        ("pirani","cold cathode","capacitance","ion gauge","auto range","roughing","high vacuum","combined"),
        ("Pirani auto","cold-cathode fast","capacitance absolute","ion high-range","automatic","rough-vac","UHV ion","hybrid"),
        ("20 ms","40 ms","60 ms","80 ms","100 ms","120 ms","140 ms","160 ms"),
        ("30 ms","50 ms","70 ms","90 ms","110 ms","130 ms","150 ms","170 ms"),
        47601, 416000,
    ),
    _Domain(
        "flow_cytometer", "spectral flow cytometer", "FC",
        "laser line", "sheath pressure",
        ("355 nm","405 nm","445 nm","488 nm","532 nm","561 nm","594 nm","640 nm"),
        ("365 nm","415 nm","455 nm","498 nm","542 nm","571 nm","604 nm","650 nm"),
        ("8 psi","10 psi","12 psi","14 psi","16 psi","18 psi","20 psi","22 psi"),
        ("9 psi","11 psi","13 psi","15 psi","17 psi","19 psi","21 psi","23 psi"),
        47701, 417000,
    ),
    _Domain(
        "nano_positioner", "closed-loop nanopositioner", "NP",
        "step mode", "slew rate",
        ("fine","coarse","adaptive","quiet","burst","linear","creep","feedback"),
        ("fine closed-loop","coarse closed-loop","adaptive step","quiet step","burst step","linear step","creep-compensated","feedback step"),
        ("2 um/s","4 um/s","6 um/s","8 um/s","10 um/s","12 um/s","14 um/s","16 um/s"),
        ("3 um/s","5 um/s","7 um/s","9 um/s","11 um/s","13 um/s","15 um/s","17 um/s"),
        47801, 418000,
    ),
    _Domain(
        "photodiode_array", "linear photodiode-array reader", "PD",
        "gain range", "integration time",
        ("1x","2x","4x","8x","16x","32x","64x","128x"),
        ("1.5x","3x","6x","12x","24x","48x","96x","192x"),
        ("0.2 ms","0.4 ms","0.6 ms","0.8 ms","1.0 ms","1.2 ms","1.4 ms","1.6 ms"),
        ("0.3 ms","0.5 ms","0.7 ms","0.9 ms","1.1 ms","1.3 ms","1.5 ms","1.7 ms"),
        47901, 419000,
    ),
    _Domain(
        "microwave_cavity", "tunable microwave cavity", "MC",
        "coupling mode", "sweep span",
        ("undercoupled","critical","overcoupled","loop-A","loop-B","capacitive","inductive","balanced"),
        ("weak undercoupled","critical match","strong overcoupled","loop port A","loop port B","capacitive tap","inductive loop","balanced ports"),
        ("5 MHz","10 MHz","15 MHz","20 MHz","25 MHz","30 MHz","35 MHz","40 MHz"),
        ("7 MHz","12 MHz","17 MHz","22 MHz","27 MHz","32 MHz","37 MHz","42 MHz"),
        48001, 420000,
    ),
    _Domain(
        "electrochem_station", "electrochemical workstation", "EC",
        "reference electrode", "scan rate",
        ("Ag/AgCl","SCE","Hg/HgO","RHE","SHE","Li/Li+","Cu/Cu2+","pseudo-Ag"),
        ("AgCl micro","calomel sat","mercury oxide","reversible hydrogen","standard hydrogen","lithium reference","copper reference","silver pseudo"),
        ("5 mV/s","10 mV/s","15 mV/s","20 mV/s","25 mV/s","30 mV/s","35 mV/s","40 mV/s"),
        ("7 mV/s","12 mV/s","17 mV/s","22 mV/s","27 mV/s","32 mV/s","37 mV/s","42 mV/s"),
        48101, 421000,
    ),
    _Domain(
        "interferometer", "phase-shifting interferometer", "IF",
        "beam splitter", "path offset",
        ("50:50","60:40","70:30","80:20","90:10","45:55","35:65","25:75"),
        ("52:48","62:38","72:28","82:18","92:8","47:53","37:63","27:73"),
        ("10 um","20 um","30 um","40 um","50 um","60 um","70 um","80 um"),
        ("15 um","25 um","35 um","45 um","55 um","65 um","75 um","85 um"),
        48201, 422000,
    ),
    _Domain(
        "fluorescence_lifetime", "time-resolved fluorescence lifetime rig", "FL",
        "excitation rate", "gate width",
        ("1 MHz","2 MHz","3 MHz","4 MHz","5 MHz","6 MHz","7 MHz","8 MHz"),
        ("1.5 MHz","2.5 MHz","3.5 MHz","4.5 MHz","5.5 MHz","6.5 MHz","7.5 MHz","8.5 MHz"),
        ("0.5 ns","1.0 ns","1.5 ns","2.0 ns","2.5 ns","3.0 ns","3.5 ns","4.0 ns"),
        ("0.7 ns","1.2 ns","1.7 ns","2.2 ns","2.7 ns","3.2 ns","3.7 ns","4.2 ns"),
        48301, 423000,
    ),
    _Domain(
        "ion_trap_controller", "quadrupole ion-trap controller", "IT",
        "RF amplitude", "cooling time",
        ("80 V","100 V","120 V","140 V","160 V","180 V","200 V","220 V"),
        ("90 V","110 V","130 V","150 V","170 V","190 V","210 V","230 V"),
        ("2 ms","4 ms","6 ms","8 ms","10 ms","12 ms","14 ms","16 ms"),
        ("3 ms","5 ms","7 ms","9 ms","11 ms","13 ms","15 ms","17 ms"),
        48401, 424000,
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
):
    rows = [
        ("a", field_a, first),
        ("b", field_b, second),
        ("x", field_a, distractor_first),
        ("y", field_b, distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts = tuple(
        f"for the {noun}, {field} is {value}" for _kind, field, value in rows
    )
    aliases = tuple(
        f"{value} is the logged {field} setting for this {noun}"
        for _kind, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(
    spec: _Domain,
    *,
    split: Split,
    code: str,
    first: str,
    second: str,
):
    if split == "train":
        state_a = (
            f"S26-Factorized {spec.noun} record {code}: "
            f"{spec.field_a} {first}; {spec.field_b} {second}."
        )
        state_b = (
            f"S26-Factorized record {code} lists {second} for {spec.field_b}. "
            f"The same {spec.noun} assigns {first} to {spec.field_a}."
        )
        qa1 = f"For S26-Factorized {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S26-Factorized record {code}, which entry is {spec.field_a}?"
        qb1 = f"For S26-Factorized {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S26-Factorized record {code}, which entry is {spec.field_b}?"
    else:
        state_a = (
            f"S26-Factorized audit {code} for the {spec.noun} documents "
            f"{first} as {spec.field_a} and {second} as {spec.field_b}."
        )
        state_b = (
            f"Audit {code} says the {spec.field_b} setting is {second}. "
            f"Separately, this S26-Factorized {spec.noun} records "
            f"{first} for {spec.field_a}."
        )
        qa1 = f"Read S26-Factorized audit {code}: what is the {spec.field_a} entry?"
        qa2 = f"Which value is tagged {spec.field_a} on S26-Factorized audit {code}?"
        qb1 = f"Read S26-Factorized audit {code}: what is the {spec.field_b} entry?"
        qb2 = f"Which value is tagged {spec.field_b} on S26-Factorized audit {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S26RelationCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s26-{spec.name}-{index:03d}"
    state_a, state_b, qa1, qa2, qb1, qb2 = _views(
        spec, split=split, code=code, first=first, second=second
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
    return S26RelationCase(
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


def generate_s26_cases(split: Split) -> tuple[S26RelationCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s26_partitions(
    train: tuple[S26RelationCase, ...],
    dev: tuple[S26RelationCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S26 partition size changed")
    if len({r.domain for r in train}) != 12 or len({r.domain for r in dev}) != 12:
        raise RuntimeError("S26 domain count changed")
    if any(r.split != "train" for r in train) or any(r.split != "dev" for r in dev):
        raise RuntimeError("S26 split labels changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S26 language changed")
    if any(
        len(r.option_texts) != 4
        or len(r.option_aliases) != 4
        or len(r.option_ids) != 4
        for r in (*train, *dev)
    ):
        raise RuntimeError("S26 option cardinality changed")

    train_states = {x for r in train for x in (r.state_a, r.state_b)}
    dev_states = {x for r in dev for x in (r.state_a, r.state_b)}
    train_q = {
        x
        for r in train
        for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)
    }
    dev_q = {
        x
        for r in dev
        for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)
    }
    train_opts = {
        x for r in train for x in (*r.option_texts, *r.option_aliases)
    }
    dev_opts = {
        x for r in dev for x in (*r.option_texts, *r.option_aliases)
    }

    if train_states & dev_states:
        raise RuntimeError("S26 TRAIN/DEV exact state overlap")
    if train_q & dev_q:
        raise RuntimeError("S26 TRAIN/DEV exact question overlap")
    if train_opts & dev_opts:
        raise RuntimeError("S26 TRAIN/DEV exact option overlap")


__all__ = [
    "S26RelationCase",
    "generate_s26_cases",
    "validate_s26_partitions",
]
