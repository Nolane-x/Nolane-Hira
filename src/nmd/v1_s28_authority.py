from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S28AnchorCase:
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
        "moke_magnetometer","magneto-optic Kerr magnetometer","MK",
        "field amplitude","modulation rate",
        ("5 mT","10 mT","15 mT","20 mT","25 mT","30 mT","35 mT","40 mT"),
        ("7 mT","12 mT","17 mT","22 mT","27 mT","32 mT","37 mT","42 mT"),
        ("40 Hz","80 Hz","120 Hz","160 Hz","200 Hz","240 Hz","280 Hz","320 Hz"),
        ("60 Hz","100 Hz","140 Hz","180 Hz","220 Hz","260 Hz","300 Hz","340 Hz"),
        50101, 451000,
    ),
    _Domain(
        "micro_ct","micro-CT scanner","CT",
        "tube voltage","voxel size",
        ("40 kV","50 kV","60 kV","70 kV","80 kV","90 kV","100 kV","110 kV"),
        ("45 kV","55 kV","65 kV","75 kV","85 kV","95 kV","105 kV","115 kV"),
        ("5 um","10 um","15 um","20 um","25 um","30 um","35 um","40 um"),
        ("7 um","12 um","17 um","22 um","27 um","32 um","37 um","42 um"),
        50201, 452000,
    ),
    _Domain(
        "laser_doppler","laser Doppler velocimeter","LD",
        "beam spacing","sample rate",
        ("1 mm","2 mm","3 mm","4 mm","5 mm","6 mm","7 mm","8 mm"),
        ("1.5 mm","2.5 mm","3.5 mm","4.5 mm","5.5 mm","6.5 mm","7.5 mm","8.5 mm"),
        ("2 kHz","4 kHz","6 kHz","8 kHz","10 kHz","12 kHz","14 kHz","16 kHz"),
        ("3 kHz","5 kHz","7 kHz","9 kHz","11 kHz","13 kHz","15 kHz","17 kHz"),
        50301, 453000,
    ),
    _Domain(
        "ftir_stage","FTIR sampling stage","FT",
        "aperture","scan speed",
        ("1 mm","2 mm","3 mm","4 mm","5 mm","6 mm","7 mm","8 mm"),
        ("1.5 mm","2.5 mm","3.5 mm","4.5 mm","5.5 mm","6.5 mm","7.5 mm","8.5 mm"),
        ("0.2 cm/s","0.4 cm/s","0.6 cm/s","0.8 cm/s","1.0 cm/s","1.2 cm/s","1.4 cm/s","1.6 cm/s"),
        ("0.3 cm/s","0.5 cm/s","0.7 cm/s","0.9 cm/s","1.1 cm/s","1.3 cm/s","1.5 cm/s","1.7 cm/s"),
        50401, 454000,
    ),
    _Domain(
        "micro_reactor","microreactor controller","MR",
        "jacket temperature","flow ratio",
        ("20 C","30 C","40 C","50 C","60 C","70 C","80 C","90 C"),
        ("25 C","35 C","45 C","55 C","65 C","75 C","85 C","95 C"),
        ("1:1","2:1","3:1","4:1","5:1","1:2","1:3","1:4"),
        ("3:2","5:2","7:2","9:2","11:2","2:3","2:5","2:7"),
        50501, 455000,
    ),
    _Domain(
        "streak_camera","streak camera controller","SC",
        "sweep range","trigger delay",
        ("100 ps","200 ps","300 ps","400 ps","500 ps","600 ps","700 ps","800 ps"),
        ("150 ps","250 ps","350 ps","450 ps","550 ps","650 ps","750 ps","850 ps"),
        ("1 ns","2 ns","3 ns","4 ns","5 ns","6 ns","7 ns","8 ns"),
        ("1.5 ns","2.5 ns","3.5 ns","4.5 ns","5.5 ns","6.5 ns","7.5 ns","8.5 ns"),
        50601, 456000,
    ),
    _Domain(
        "load_frame","materials load frame","LF",
        "load limit","crosshead speed",
        ("1 kN","2 kN","3 kN","4 kN","5 kN","6 kN","7 kN","8 kN"),
        ("1.5 kN","2.5 kN","3.5 kN","4.5 kN","5.5 kN","6.5 kN","7.5 kN","8.5 kN"),
        ("0.1 mm/min","0.2 mm/min","0.3 mm/min","0.4 mm/min","0.5 mm/min","0.6 mm/min","0.7 mm/min","0.8 mm/min"),
        ("0.15 mm/min","0.25 mm/min","0.35 mm/min","0.45 mm/min","0.55 mm/min","0.65 mm/min","0.75 mm/min","0.85 mm/min"),
        50701, 457000,
    ),
    _Domain(
        "laser_ablation","laser ablation controller","LA",
        "pulse energy","spot size",
        ("1 mJ","2 mJ","3 mJ","4 mJ","5 mJ","6 mJ","7 mJ","8 mJ"),
        ("1.5 mJ","2.5 mJ","3.5 mJ","4.5 mJ","5.5 mJ","6.5 mJ","7.5 mJ","8.5 mJ"),
        ("20 um","30 um","40 um","50 um","60 um","70 um","80 um","90 um"),
        ("25 um","35 um","45 um","55 um","65 um","75 um","85 um","95 um"),
        50801, 458000,
    ),
    _Domain(
        "dielectric_analyzer","dielectric analyzer","DA",
        "drive frequency","drive amplitude",
        ("1 kHz","2 kHz","3 kHz","4 kHz","5 kHz","6 kHz","7 kHz","8 kHz"),
        ("1.5 kHz","2.5 kHz","3.5 kHz","4.5 kHz","5.5 kHz","6.5 kHz","7.5 kHz","8.5 kHz"),
        ("50 mV","100 mV","150 mV","200 mV","250 mV","300 mV","350 mV","400 mV"),
        ("75 mV","125 mV","175 mV","225 mV","275 mV","325 mV","375 mV","425 mV"),
        50901, 459000,
    ),
    _Domain(
        "laser_cooling","laser-cooling controller","LC",
        "detuning","repump power",
        ("-5 MHz","-10 MHz","-15 MHz","-20 MHz","-25 MHz","-30 MHz","-35 MHz","-40 MHz"),
        ("-7 MHz","-12 MHz","-17 MHz","-22 MHz","-27 MHz","-32 MHz","-37 MHz","-42 MHz"),
        ("1 mW","2 mW","3 mW","4 mW","5 mW","6 mW","7 mW","8 mW"),
        ("1.5 mW","2.5 mW","3.5 mW","4.5 mW","5.5 mW","6.5 mW","7.5 mW","8.5 mW"),
        51001, 460000,
    ),
    _Domain(
        "mass_flow_array","mass-flow controller array","MF",
        "channel range","update period",
        ("10 sccm","20 sccm","30 sccm","40 sccm","50 sccm","60 sccm","70 sccm","80 sccm"),
        ("15 sccm","25 sccm","35 sccm","45 sccm","55 sccm","65 sccm","75 sccm","85 sccm"),
        ("20 ms","40 ms","60 ms","80 ms","100 ms","120 ms","140 ms","160 ms"),
        ("30 ms","50 ms","70 ms","90 ms","110 ms","130 ms","150 ms","170 ms"),
        51101, 461000,
    ),
    _Domain(
        "optical_correlator","optical correlator","OC",
        "delay span","averaging",
        ("5 ps","10 ps","15 ps","20 ps","25 ps","30 ps","35 ps","40 ps"),
        ("7 ps","12 ps","17 ps","22 ps","27 ps","32 ps","37 ps","42 ps"),
        ("2 scans","4 scans","6 scans","8 scans","10 scans","12 scans","14 scans","16 scans"),
        ("3 scans","5 scans","7 scans","9 scans","11 scans","13 scans","15 scans","17 scans"),
        51201, 462000,
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
    ga = next(i for i,(kind,_f,_v) in enumerate(rows) if kind == "a")
    gb = next(i for i,(kind,_f,_v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(spec: _Domain, *, split: Split, code: str, first: str, second: str):
    if split == "train":
        state_a = (
            f"S28-Anchor {spec.noun} record {code}: "
            f"{spec.field_a} {first}; {spec.field_b} {second}."
        )
        state_b = (
            f"S28-Anchor record {code} gives {second} for {spec.field_b}. "
            f"The same {spec.noun} assigns {first} to {spec.field_a}."
        )
        qa1 = f"For S28-Anchor {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S28-Anchor record {code}, which entry is {spec.field_a}?"
        qb1 = f"For S28-Anchor {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S28-Anchor record {code}, which entry is {spec.field_b}?"
    else:
        state_a = (
            f"S28-Anchor audit {code} for the {spec.noun} records "
            f"{first} as {spec.field_a} and {second} as {spec.field_b}."
        )
        state_b = (
            f"Audit {code} places {second} under {spec.field_b}. "
            f"Separately, this S28-Anchor {spec.noun} lists "
            f"{first} for {spec.field_a}."
        )
        qa1 = f"Read S28-Anchor audit {code}: what is {spec.field_a}?"
        qa2 = f"Which value is tagged {spec.field_a} on S28-Anchor audit {code}?"
        qb1 = f"Read S28-Anchor audit {code}: what is {spec.field_b}?"
        qb2 = f"Which value is tagged {spec.field_b} on S28-Anchor audit {code}?"
    return state_a,state_b,qa1,qa2,qb1,qb2


def _build(spec: _Domain, index: int, split: Split) -> S28AnchorCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index*3+1) % len(seconds)]
    distractor_first = firsts[(index+3) % len(firsts)]
    distractor_second = seconds[(index*5+2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s28-{spec.name}-{index:03d}"
    state_a,state_b,qa1,qa2,qb1,qb2 = _views(
        spec,split=split,code=code,first=first,second=second
    )
    texts,aliases,ids,ga,gb = _shuffle(
        case_id=case_id,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=distractor_first,
        distractor_second=distractor_second,seed=spec.seed+offset,
    )
    return S28AnchorCase(
        case_id=case_id,split=split,domain=spec.name,language="en",
        state_a=state_a,state_b=state_b,
        question_a1=qa1,question_a2=qa2,
        question_b1=qb1,question_b2=qb2,
        option_texts=texts,option_aliases=aliases,option_ids=ids,
        gold_a=ga,gold_b=gb,
    )


def generate_s28_cases(split: Split) -> tuple[S28AnchorCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec,index,split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s28_partitions(
    train: tuple[S28AnchorCase, ...],
    dev: tuple[S28AnchorCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S28 partition size changed")
    if len({r.domain for r in train}) != 12 or len({r.domain for r in dev}) != 12:
        raise RuntimeError("S28 domain count changed")
    if any(r.split!="train" for r in train) or any(r.split!="dev" for r in dev):
        raise RuntimeError("S28 split labels changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S28 language changed")
    if any(
        len(r.option_texts)!=4 or len(r.option_aliases)!=4 or len(r.option_ids)!=4
        for r in (*train,*dev)
    ):
        raise RuntimeError("S28 option cardinality changed")

    train_states={x for r in train for x in (r.state_a,r.state_b)}
    dev_states={x for r in dev for x in (r.state_a,r.state_b)}
    train_q={x for r in train for x in (
        r.question_a1,r.question_a2,r.question_b1,r.question_b2
    )}
    dev_q={x for r in dev for x in (
        r.question_a1,r.question_a2,r.question_b1,r.question_b2
    )}
    train_opts={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    dev_opts={x for r in dev for x in (*r.option_texts,*r.option_aliases)}

    if train_states & dev_states:
        raise RuntimeError("S28 TRAIN/DEV exact state overlap")
    if train_q & dev_q:
        raise RuntimeError("S28 TRAIN/DEV exact question overlap")
    if train_opts & dev_opts:
        raise RuntimeError("S28 TRAIN/DEV exact option overlap")


__all__ = ["S28AnchorCase","generate_s28_cases","validate_s28_partitions"]
