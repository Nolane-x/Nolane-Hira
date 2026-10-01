from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S27BlockwiseCase:
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
        "terahertz_spectrometer","terahertz time-domain spectrometer","TH",
        "emitter bias","scan window",
        ("12 V","18 V","24 V","30 V","36 V","42 V","48 V","54 V"),
        ("15 V","21 V","27 V","33 V","39 V","45 V","51 V","57 V"),
        ("20 ps","30 ps","40 ps","50 ps","60 ps","70 ps","80 ps","90 ps"),
        ("25 ps","35 ps","45 ps","55 ps","65 ps","75 ps","85 ps","95 ps"),
        48501, 431000,
    ),
    _Domain(
        "thermal_bridge","thermal-conductivity bridge","TB",
        "heater current","sample interval",
        ("8 mA","12 mA","16 mA","20 mA","24 mA","28 mA","32 mA","36 mA"),
        ("10 mA","14 mA","18 mA","22 mA","26 mA","30 mA","34 mA","38 mA"),
        ("0.5 s","1.0 s","1.5 s","2.0 s","2.5 s","3.0 s","3.5 s","4.0 s"),
        ("0.75 s","1.25 s","1.75 s","2.25 s","2.75 s","3.25 s","3.75 s","4.25 s"),
        48601, 432000,
    ),
    _Domain(
        "fiber_interrogator","fiber Bragg grating interrogator","FB",
        "sweep band","trigger rate",
        ("1510 nm","1520 nm","1530 nm","1540 nm","1550 nm","1560 nm","1570 nm","1580 nm"),
        ("1515 nm","1525 nm","1535 nm","1545 nm","1555 nm","1565 nm","1575 nm","1585 nm"),
        ("50 Hz","100 Hz","150 Hz","200 Hz","250 Hz","300 Hz","350 Hz","400 Hz"),
        ("75 Hz","125 Hz","175 Hz","225 Hz","275 Hz","325 Hz","375 Hz","425 Hz"),
        48701, 433000,
    ),
    _Domain(
        "droplet_generator","digital microfluidic droplet generator","DG",
        "pulse width","drive voltage",
        ("2 ms","4 ms","6 ms","8 ms","10 ms","12 ms","14 ms","16 ms"),
        ("3 ms","5 ms","7 ms","9 ms","11 ms","13 ms","15 ms","17 ms"),
        ("40 V","50 V","60 V","70 V","80 V","90 V","100 V","110 V"),
        ("45 V","55 V","65 V","75 V","85 V","95 V","105 V","115 V"),
        48801, 434000,
    ),
    _Domain(
        "field_mapper","three-axis magnetic field mapper","FM",
        "probe range","grid spacing",
        ("2 mT","4 mT","6 mT","8 mT","10 mT","12 mT","14 mT","16 mT"),
        ("3 mT","5 mT","7 mT","9 mT","11 mT","13 mT","15 mT","17 mT"),
        ("1 mm","2 mm","3 mm","4 mm","5 mm","6 mm","7 mm","8 mm"),
        ("1.5 mm","2.5 mm","3.5 mm","4.5 mm","5.5 mm","6.5 mm","7.5 mm","8.5 mm"),
        48901, 435000,
    ),
    _Domain(
        "delay_line","motorized optical delay line","OD",
        "travel mode","velocity",
        ("fine","coarse","continuous","step","quiet","burst","linear","feedback"),
        ("fine lock","coarse lock","continuous scan","microstep","quiet scan","burst scan","linear scan","feedback scan"),
        ("0.5 mm/s","1.0 mm/s","1.5 mm/s","2.0 mm/s","2.5 mm/s","3.0 mm/s","3.5 mm/s","4.0 mm/s"),
        ("0.75 mm/s","1.25 mm/s","1.75 mm/s","2.25 mm/s","2.75 mm/s","3.25 mm/s","3.75 mm/s","4.25 mm/s"),
        49001, 436000,
    ),
    _Domain(
        "photon_counter","multichannel photon counter","PC",
        "discriminator level","dead time",
        ("12 mV","18 mV","24 mV","30 mV","36 mV","42 mV","48 mV","54 mV"),
        ("15 mV","21 mV","27 mV","33 mV","39 mV","45 mV","51 mV","57 mV"),
        ("8 ns","12 ns","16 ns","20 ns","24 ns","28 ns","32 ns","36 ns"),
        ("10 ns","14 ns","18 ns","22 ns","26 ns","30 ns","34 ns","38 ns"),
        49101, 437000,
    ),
    _Domain(
        "etch_controller","reactive-ion etch controller","RE",
        "gas ratio","platen power",
        ("1:1","2:1","3:1","4:1","5:1","1:2","1:3","1:4"),
        ("3:2","5:2","7:2","9:2","11:2","2:3","2:5","2:7"),
        ("20 W","30 W","40 W","50 W","60 W","70 W","80 W","90 W"),
        ("25 W","35 W","45 W","55 W","65 W","75 W","85 W","95 W"),
        49201, 438000,
    ),
    _Domain(
        "torsion_controller","torsion-pendulum controller","TP",
        "drive frequency","damping gain",
        ("0.2 Hz","0.4 Hz","0.6 Hz","0.8 Hz","1.0 Hz","1.2 Hz","1.4 Hz","1.6 Hz"),
        ("0.3 Hz","0.5 Hz","0.7 Hz","0.9 Hz","1.1 Hz","1.3 Hz","1.5 Hz","1.7 Hz"),
        ("0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8"),
        ("0.15","0.25","0.35","0.45","0.55","0.65","0.75","0.85"),
        49301, 439000,
    ),
    _Domain(
        "acoustic_tube","acoustic impedance tube","AT",
        "source level","sample depth",
        ("70 dB","74 dB","78 dB","82 dB","86 dB","90 dB","94 dB","98 dB"),
        ("72 dB","76 dB","80 dB","84 dB","88 dB","92 dB","96 dB","100 dB"),
        ("10 mm","20 mm","30 mm","40 mm","50 mm","60 mm","70 mm","80 mm"),
        ("15 mm","25 mm","35 mm","45 mm","55 mm","65 mm","75 mm","85 mm"),
        49401, 440000,
    ),
    _Domain(
        "power_stabilizer","laser power stabilizer","PS",
        "feedback mode","target power",
        ("slow","fast","dual-loop","quiet","integrating","predictive","linear","adaptive"),
        ("slow lock","fast lock","dual feedback","quiet lock","integrator","predictive lock","linear lock","adaptive lock"),
        ("5 mW","10 mW","15 mW","20 mW","25 mW","30 mW","35 mW","40 mW"),
        ("7 mW","12 mW","17 mW","22 mW","27 mW","32 mW","37 mW","42 mW"),
        49501, 441000,
    ),
    _Domain(
        "probe_controller","scanning probe controller","SP",
        "feedback bandwidth","line rate",
        ("50 Hz","100 Hz","150 Hz","200 Hz","250 Hz","300 Hz","350 Hz","400 Hz"),
        ("75 Hz","125 Hz","175 Hz","225 Hz","275 Hz","325 Hz","375 Hz","425 Hz"),
        ("0.5 Hz","1.0 Hz","1.5 Hz","2.0 Hz","2.5 Hz","3.0 Hz","3.5 Hz","4.0 Hz"),
        ("0.75 Hz","1.25 Hz","1.75 Hz","2.25 Hz","2.75 Hz","3.25 Hz","3.75 Hz","4.25 Hz"),
        49601, 442000,
    ),
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
    texts = tuple(
        f"for the {noun}, {field} is {value}"
        for _kind, field, value in rows
    )
    aliases = tuple(
        f"{value} is the recorded {field} setting for this {noun}"
        for _kind, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(spec: _Domain, *, split: Split, code: str,
           first: str, second: str):
    if split == "train":
        state_a = (
            f"S27-Blockwise {spec.noun} record {code}: "
            f"{spec.field_a} {first}; {spec.field_b} {second}."
        )
        state_b = (
            f"S27-Blockwise record {code} reports {second} for {spec.field_b}. "
            f"The same {spec.noun} assigns {first} to {spec.field_a}."
        )
        qa1 = f"For S27-Blockwise {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S27-Blockwise record {code}, which entry is {spec.field_a}?"
        qb1 = f"For S27-Blockwise {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S27-Blockwise record {code}, which entry is {spec.field_b}?"
    else:
        state_a = (
            f"S27-Blockwise audit {code} for the {spec.noun} records "
            f"{first} as {spec.field_a} and {second} as {spec.field_b}."
        )
        state_b = (
            f"Audit {code} places {second} under {spec.field_b}. "
            f"Separately, the S27-Blockwise {spec.noun} lists "
            f"{first} for {spec.field_a}."
        )
        qa1 = f"Read S27-Blockwise audit {code}: what is {spec.field_a}?"
        qa2 = f"Which value is tagged {spec.field_a} on S27-Blockwise audit {code}?"
        qb1 = f"Read S27-Blockwise audit {code}: what is {spec.field_b}?"
        qb2 = f"Which value is tagged {spec.field_b} on S27-Blockwise audit {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S27BlockwiseCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s27-{spec.name}-{index:03d}"
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
    return S27BlockwiseCase(
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


def generate_s27_cases(split: Split) -> tuple[S27BlockwiseCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s27_partitions(
    train: tuple[S27BlockwiseCase, ...],
    dev: tuple[S27BlockwiseCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S27 partition size changed")
    if len({r.domain for r in train}) != 12 or len({r.domain for r in dev}) != 12:
        raise RuntimeError("S27 domain count changed")
    if any(r.split != "train" for r in train) or any(r.split != "dev" for r in dev):
        raise RuntimeError("S27 split labels changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S27 language changed")
    if any(
        len(r.option_texts) != 4
        or len(r.option_aliases) != 4
        or len(r.option_ids) != 4
        for r in (*train, *dev)
    ):
        raise RuntimeError("S27 option cardinality changed")

    train_states = {x for r in train for x in (r.state_a, r.state_b)}
    dev_states = {x for r in dev for x in (r.state_a, r.state_b)}
    train_q = {
        x for r in train
        for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)
    }
    dev_q = {
        x for r in dev
        for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)
    }
    train_opts = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    dev_opts = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}

    if train_states & dev_states:
        raise RuntimeError("S27 TRAIN/DEV exact state overlap")
    if train_q & dev_q:
        raise RuntimeError("S27 TRAIN/DEV exact question overlap")
    if train_opts & dev_opts:
        raise RuntimeError("S27 TRAIN/DEV exact option overlap")


__all__ = [
    "S27BlockwiseCase",
    "generate_s27_cases",
    "validate_s27_partitions",
]
