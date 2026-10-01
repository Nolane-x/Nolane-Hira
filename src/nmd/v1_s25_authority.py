from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S25ProjectionCase:
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
        "terahertz_tds", "terahertz time-domain spectrometer", "TH",
        "emitter bias", "scan window",
        ("18 V","22 V","26 V","30 V","34 V","38 V","42 V","46 V"),
        ("19 V","23 V","27 V","31 V","35 V","39 V","43 V","47 V"),
        ("12 ps","18 ps","24 ps","30 ps","36 ps","42 ps","48 ps","54 ps"),
        ("15 ps","21 ps","27 ps","33 ps","39 ps","45 ps","51 ps","57 ps"),
        46101, 401000,
    ),
    _Domain(
        "cryostat_stage", "closed-cycle cryostat stage", "CR",
        "exchange gas", "settle threshold",
        ("helium","neon","argon","nitrogen","hydrogen","vacuum","dry air","xenon"),
        ("He-4","neon gas","ultrapure argon","dry nitrogen","H2 gas","high vacuum","zero air","Xe gas"),
        ("0.02 K","0.03 K","0.04 K","0.05 K","0.06 K","0.07 K","0.08 K","0.09 K"),
        ("0.025 K","0.035 K","0.045 K","0.055 K","0.065 K","0.075 K","0.085 K","0.095 K"),
        46201, 402000,
    ),
    _Domain(
        "tof_mass", "time-of-flight mass spectrometer", "MS",
        "reflectron voltage", "extraction delay",
        ("1.2 kV","1.4 kV","1.6 kV","1.8 kV","2.0 kV","2.2 kV","2.4 kV","2.6 kV"),
        ("1.3 kV","1.5 kV","1.7 kV","1.9 kV","2.1 kV","2.3 kV","2.5 kV","2.7 kV"),
        ("40 ns","60 ns","80 ns","100 ns","120 ns","140 ns","160 ns","180 ns"),
        ("50 ns","70 ns","90 ns","110 ns","130 ns","150 ns","170 ns","190 ns"),
        46301, 403000,
    ),
    _Domain(
        "optical_tweezers", "dual-beam optical tweezers rig", "OT",
        "trap wavelength", "feedback rate",
        ("980 nm","1030 nm","1064 nm","1080 nm","1120 nm","1150 nm","1180 nm","1200 nm"),
        ("990 nm","1040 nm","1070 nm","1090 nm","1130 nm","1160 nm","1190 nm","1210 nm"),
        ("2 kHz","4 kHz","6 kHz","8 kHz","10 kHz","12 kHz","14 kHz","16 kHz"),
        ("3 kHz","5 kHz","7 kHz","9 kHz","11 kHz","13 kHz","15 kHz","17 kHz"),
        46401, 404000,
    ),
    _Domain(
        "uvvis_flowcell", "UV-visible stopped-flow cell", "UV",
        "path length", "mixing dead time",
        ("0.5 mm","1.0 mm","1.5 mm","2.0 mm","2.5 mm","3.0 mm","3.5 mm","4.0 mm"),
        ("0.7 mm","1.2 mm","1.7 mm","2.2 mm","2.7 mm","3.2 mm","3.7 mm","4.2 mm"),
        ("0.8 ms","1.0 ms","1.2 ms","1.4 ms","1.6 ms","1.8 ms","2.0 ms","2.2 ms"),
        ("0.9 ms","1.1 ms","1.3 ms","1.5 ms","1.7 ms","1.9 ms","2.1 ms","2.3 ms"),
        46501, 405000,
    ),
    _Domain(
        "acoustic_micro", "scanning acoustic microscope", "AM",
        "transducer frequency", "water gate",
        ("25 MHz","35 MHz","45 MHz","55 MHz","65 MHz","75 MHz","85 MHz","95 MHz"),
        ("30 MHz","40 MHz","50 MHz","60 MHz","70 MHz","80 MHz","90 MHz","100 MHz"),
        ("0.4 us","0.6 us","0.8 us","1.0 us","1.2 us","1.4 us","1.6 us","1.8 us"),
        ("0.5 us","0.7 us","0.9 us","1.1 us","1.3 us","1.5 us","1.7 us","1.9 us"),
        46601, 406000,
    ),
    _Domain(
        "rf_probe", "RF wafer probe station", "RF",
        "probe pitch", "source power",
        ("75 um","100 um","125 um","150 um","175 um","200 um","225 um","250 um"),
        ("80 um","105 um","130 um","155 um","180 um","205 um","230 um","255 um"),
        ("-30 dBm","-25 dBm","-20 dBm","-15 dBm","-10 dBm","-5 dBm","0 dBm","5 dBm"),
        ("-28 dBm","-23 dBm","-18 dBm","-13 dBm","-8 dBm","-3 dBm","2 dBm","7 dBm"),
        46701, 407000,
    ),
    _Domain(
        "xrd_goniometer", "high-resolution XRD goniometer", "XR",
        "incident slit", "omega step",
        ("0.05 deg","0.08 deg","0.10 deg","0.12 deg","0.15 deg","0.18 deg","0.20 deg","0.25 deg"),
        ("0.06 deg","0.09 deg","0.11 deg","0.13 deg","0.16 deg","0.19 deg","0.22 deg","0.27 deg"),
        ("0.001 deg","0.002 deg","0.003 deg","0.004 deg","0.005 deg","0.006 deg","0.007 deg","0.008 deg"),
        ("0.0015 deg","0.0025 deg","0.0035 deg","0.0045 deg","0.0055 deg","0.0065 deg","0.0075 deg","0.0085 deg"),
        46801, 408000,
    ),
    _Domain(
        "microfluidic_pump", "pressure-driven microfluidic pump", "MP",
        "control mode", "pressure limit",
        ("constant pressure","constant flow","pulse","ramp","PID","step","burst","adaptive"),
        ("pressure hold","flow hold","pulsed drive","linear ramp","closed-loop PID","stair step","burst mode","adaptive loop"),
        ("20 mbar","40 mbar","60 mbar","80 mbar","100 mbar","120 mbar","140 mbar","160 mbar"),
        ("30 mbar","50 mbar","70 mbar","90 mbar","110 mbar","130 mbar","150 mbar","170 mbar"),
        46901, 409000,
    ),
    _Domain(
        "patch_clamp", "automated patch-clamp rig", "PC",
        "pipette resistance", "holding potential",
        ("2 Mohm","3 Mohm","4 Mohm","5 Mohm","6 Mohm","7 Mohm","8 Mohm","9 Mohm"),
        ("2.5 Mohm","3.5 Mohm","4.5 Mohm","5.5 Mohm","6.5 Mohm","7.5 Mohm","8.5 Mohm","9.5 Mohm"),
        ("-90 mV","-80 mV","-70 mV","-60 mV","-50 mV","-40 mV","-30 mV","-20 mV"),
        ("-85 mV","-75 mV","-65 mV","-55 mV","-45 mV","-35 mV","-25 mV","-15 mV"),
        47001, 410000,
    ),
    _Domain(
        "lidar_receiver", "coherent lidar receiver", "LD",
        "local oscillator offset", "integration gate",
        ("10 MHz","20 MHz","30 MHz","40 MHz","50 MHz","60 MHz","70 MHz","80 MHz"),
        ("15 MHz","25 MHz","35 MHz","45 MHz","55 MHz","65 MHz","75 MHz","85 MHz"),
        ("2 us","4 us","6 us","8 us","10 us","12 us","14 us","16 us"),
        ("3 us","5 us","7 us","9 us","11 us","13 us","15 us","17 us"),
        47101, 411000,
    ),
    _Domain(
        "polarimeter", "imaging polarimeter", "PL",
        "analyzer sequence", "exposure time",
        ("0-45-90-135","0-60-120","six-step","eight-step","continuous","Mueller-16","dual-state","rapid-four"),
        ("four-angle","three-angle","6-state","8-state","rotating analyzer","16-state Mueller","dual analyzer","fast four-state"),
        ("2 ms","4 ms","6 ms","8 ms","10 ms","12 ms","14 ms","16 ms"),
        ("3 ms","5 ms","7 ms","9 ms","11 ms","13 ms","15 ms","17 ms"),
        47201, 412000,
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
            f"S25-Decoupled {spec.noun} record {code}: "
            f"{spec.field_a} {first}; {spec.field_b} {second}."
        )
        state_b = (
            f"S25-Decoupled record {code} lists {second} for {spec.field_b}. "
            f"The same {spec.noun} assigns {first} to {spec.field_a}."
        )
        qa1 = f"For S25-Decoupled {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S25-Decoupled record {code}, which entry is {spec.field_a}?"
        qb1 = f"For S25-Decoupled {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S25-Decoupled record {code}, which entry is {spec.field_b}?"
    else:
        state_a = (
            f"S25-Decoupled audit {code} for the {spec.noun} documents "
            f"{first} as {spec.field_a} and {second} as {spec.field_b}."
        )
        state_b = (
            f"Audit {code} says the {spec.field_b} setting is {second}. "
            f"Separately, this S25-Decoupled {spec.noun} records "
            f"{first} for {spec.field_a}."
        )
        qa1 = f"Read S25-Decoupled audit {code}: what is the {spec.field_a} entry?"
        qa2 = f"Which value is tagged {spec.field_a} on S25-Decoupled audit {code}?"
        qb1 = f"Read S25-Decoupled audit {code}: what is the {spec.field_b} entry?"
        qb2 = f"Which value is tagged {spec.field_b} on S25-Decoupled audit {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S25ProjectionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s25-{spec.name}-{index:03d}"
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
    return S25ProjectionCase(
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


def generate_s25_cases(split: Split) -> tuple[S25ProjectionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s25_partitions(
    train: tuple[S25ProjectionCase, ...],
    dev: tuple[S25ProjectionCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S25 partition size changed")
    if len({r.domain for r in train}) != 12 or len({r.domain for r in dev}) != 12:
        raise RuntimeError("S25 domain count changed")
    if any(r.split != "train" for r in train) or any(r.split != "dev" for r in dev):
        raise RuntimeError("S25 split labels changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S25 language changed")
    if any(
        len(r.option_texts) != 4
        or len(r.option_aliases) != 4
        or len(r.option_ids) != 4
        for r in (*train, *dev)
    ):
        raise RuntimeError("S25 option cardinality changed")

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
        raise RuntimeError("S25 TRAIN/DEV exact state overlap")
    if train_q & dev_q:
        raise RuntimeError("S25 TRAIN/DEV exact question overlap")
    if train_opts & dev_opts:
        raise RuntimeError("S25 TRAIN/DEV exact option overlap")


__all__ = [
    "S25ProjectionCase",
    "generate_s25_cases",
    "validate_s25_partitions",
]
