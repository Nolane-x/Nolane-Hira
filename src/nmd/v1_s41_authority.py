from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S41OptimizerStepCase:
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
        for key in ("option_texts", "option_aliases", "option_ids"):
            row[key] = list(row[key])
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
        "cavity_optomechanical_array","cavity optomechanical array","CO","resonator material","mechanical linewidth",
        ("SiN","diamond","SiC","AlN","GaP","silicon","LiNbO3","quartz"),
        ("strained SiN","nanobeam diamond","4H-SiC","epitaxial AlN","suspended GaP","phononic silicon","thin-film LiNbO3","SC-cut quartz"),
        ("12 Hz","18 Hz","24 Hz","30 Hz","36 Hz","42 Hz","48 Hz","54 Hz"),
        ("15 Hz","21 Hz","27 Hz","33 Hz","39 Hz","45 Hz","51 Hz","57 Hz"),81101,861000,
    ),
    _Domain(
        "cryogenic_nonlinear_microscope","cryogenic nonlinear microscope","CN","pump mode","conversion efficiency",
        ("CW","pulsed","dual-tone","chirped","comb","burst","phase-locked","swept"),
        ("narrow-CW","picosecond pulsed","coherent dual-tone","adiabatic chirped","microcomb","gated burst","phase-locked pair","rapid swept"),
        ("11%","17%","23%","29%","35%","41%","47%","53%"),
        ("14%","20%","26%","32%","38%","44%","50%","56%"),81201,862000,
    ),
    _Domain(
        "spinwave_interferometer","spin-wave interferometer","SW","waveguide","phase noise",
        ("YIG strip","CoFeB line","NiFe line","Heusler bar","FeGa film","CoNi stack","hematite strip","ferrite guide"),
        ("low-loss YIG strip","exchange CoFeB line","narrow NiFe line","epitaxial Heusler bar","strained FeGa film","synthetic CoNi stack","canted hematite strip","micro-ferrite guide"),
        ("0.8 deg","1.2 deg","1.6 deg","2.0 deg","2.4 deg","2.8 deg","3.2 deg","3.6 deg"),
        ("1.0 deg","1.4 deg","1.8 deg","2.2 deg","2.6 deg","3.0 deg","3.4 deg","3.8 deg"),81301,863000,
    ),
    _Domain(
        "electron_energy_filter","ultrafast electron energy filter","EF","filter type","energy resolution",
        ("omega","hemispherical","Wien","prism","RF cavity","magnetic sector","electrostatic sector","time lens"),
        ("double-omega","aberration-corrected hemisphere","crossed-field Wien","achromatic prism","phase-locked RF cavity","isochronous magnetic sector","retarding electrostatic sector","temporal electron lens"),
        ("20 meV","30 meV","40 meV","50 meV","60 meV","70 meV","80 meV","90 meV"),
        ("25 meV","35 meV","45 meV","55 meV","65 meV","75 meV","85 meV","95 meV"),81401,864000,
    ),
    _Domain(
        "quantum_magnetometer_lattice","quantum magnetometer lattice","QM","sensor species","field resolution",
        ("NV","SiV","Rb","Cs","Er","Yb","Bi donor","molecular spin"),
        ("ensemble NV","strain-selected SiV","cold Rb","buffered Cs","Er crystal","Yb ensemble","isotopic Bi donor","molecular spin array"),
        ("2 pT","4 pT","6 pT","8 pT","10 pT","12 pT","14 pT","16 pT"),
        ("3 pT","5 pT","7 pT","9 pT","11 pT","13 pT","15 pT","17 pT"),81501,865000,
    ),
    _Domain(
        "photonic_timebin_router","photonic time-bin router","TR","switch fabric","slot isolation",
        ("MZI","ring","MEMS","Pockels","AOM","EOM","Bragg","mesh"),
        ("dual-MZI","coupled ring","fast MEMS","thin-film Pockels","wideband AOM","traveling-wave EOM","apodized Bragg","reconfigurable mesh"),
        ("24 dB","28 dB","32 dB","36 dB","40 dB","44 dB","48 dB","52 dB"),
        ("26 dB","30 dB","34 dB","38 dB","42 dB","46 dB","50 dB","54 dB"),81601,866000,
    ),
    _Domain(
        "qubit_multiplexer","superconducting qubit multiplexer","QX","coupler","channel count",
        ("bus","Purcell","tunable","bridge","hybrid","ring","slot","stub"),
        ("bus resonator","Purcell-filtered","flux-tunable","air-bridge","hybridized","ring-resonator","slotline","quarter-wave stub"),
        ("16","24","32","40","48","56","64","72"),
        ("20","28","36","44","52","60","68","76"),81701,867000,
    ),
    _Domain(
        "pressure_imager","nanoscale pressure imager","PI","membrane","pressure floor",
        ("graphene","MoS2","SiN","diamond","hBN","silicon","AlN","SiC"),
        ("encapsulated graphene","few-layer MoS2","strained SiN","nanocrystal diamond","suspended hBN","porous silicon","piezo AlN","4H-SiC"),
        ("1 Pa","2 Pa","3 Pa","4 Pa","5 Pa","6 Pa","7 Pa","8 Pa"),
        ("1.5 Pa","2.5 Pa","3.5 Pa","4.5 Pa","5.5 Pa","6.5 Pa","7.5 Pa","8.5 Pa"),81801,868000,
    ),
    _Domain(
        "xray_wavefront_sensor","coherent x-ray wavefront sensor","XW","optic","phase resolution",
        ("grating","zone plate","speckle","Hartmann","ptycho mask","crystal","multilayer","coded aperture"),
        ("Talbot grating","Fresnel zone plate","engineered speckle","nano-Hartmann","ptychographic mask","channel-cut crystal","chirped multilayer","phase-coded aperture"),
        ("2 mrad","3 mrad","4 mrad","5 mrad","6 mrad","7 mrad","8 mrad","9 mrad"),
        ("2.5 mrad","3.5 mrad","4.5 mrad","5.5 mrad","6.5 mrad","7.5 mrad","8.5 mrad","9.5 mrad"),81901,869000,
    ),
    _Domain(
        "microwave_photon_counter","microwave photon counter","MC","detector","dark rate",
        ("qubit","JPM","KID","bolometer","SNAIL","parametric","spin","calorimeter"),
        ("transmon qubit","Josephson photomultiplier","microwave KID","graphene bolometer","SNAIL detector","parametric latch","spin ensemble","nanocalorimeter"),
        ("1 Hz","2 Hz","3 Hz","4 Hz","5 Hz","6 Hz","7 Hz","8 Hz"),
        ("1.5 Hz","2.5 Hz","3.5 Hz","4.5 Hz","5.5 Hz","6.5 Hz","7.5 Hz","8.5 Hz"),82001,870000,
    ),
    _Domain(
        "atom_interferometer_array","atom interferometer array","AI","species","phase sensitivity",
        ("Rb87","Rb85","Cs133","Sr87","Yb171","K39","K40","Li7"),
        ("BEC Rb87","laser-cooled Rb85","fountain Cs133","clock Sr87","degenerate Yb171","cold K39","fermionic K40","guided Li7"),
        ("3 urad","6 urad","9 urad","12 urad","15 urad","18 urad","21 urad","24 urad"),
        ("4 urad","7 urad","10 urad","13 urad","16 urad","19 urad","22 urad","25 urad"),82101,871000,
    ),
    _Domain(
        "neutron_detector","integrated neutron detector","ND","converter","timing resolution",
        ("B10","Li6","Gd","He3","BF3","scintillator","diamond","superconductor"),
        ("enriched B10","Li6F layer","patterned Gd","micro-He3","micro-BF3","Li-glass scintillator","CVD diamond","superconducting strip"),
        ("2 ns","4 ns","6 ns","8 ns","10 ns","12 ns","14 ns","16 ns"),
        ("3 ns","5 ns","7 ns","9 ns","11 ns","13 ns","15 ns","17 ns"),82201,872000,
    ),
)


def _shuffle(*, case_id, noun, field_a, field_b, first, second, distractor_first, distractor_second, seed):
    rows = [
        ("a", field_a, first),
        ("b", field_b, second),
        ("x", field_a, distractor_first),
        ("y", field_b, distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts = tuple(f"for the {noun}, {field} is {value}" for _kind, field, value in rows)
    aliases = tuple(f"{value} is the S41 audited {field} value for this {noun}" for _kind, field, value in rows)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(spec, *, split, code, first, second):
    if split == "train":
        return (
            f"S41-Step {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S41-Step record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S41-Step {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S41-Step record {code}?",
            f"For S41-Step {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S41-Step record {code}?",
        )
    return (
        f"S41-Step audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S41-Step audit {code}: what is {spec.field_a}?",
        f"In S41-Step audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S41-Step audit {code}: what is {spec.field_b}?",
        f"In S41-Step audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec, index, split):
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    df = firsts[(index + 3) % len(firsts)]
    ds = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s41-{spec.name}-{index:03d}"
    sa, sb, qa1, qa2, qb1, qb2 = _views(spec, split=split, code=code, first=first, second=second)
    texts, aliases, ids, ga, gb = _shuffle(
        case_id=case_id, noun=spec.noun, field_a=spec.field_a, field_b=spec.field_b,
        first=first, second=second, distractor_first=df, distractor_second=ds,
        seed=spec.seed + offset,
    )
    return S41OptimizerStepCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s41_cases(split: Split) -> tuple[S41OptimizerStepCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s41_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S41 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S41 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S41 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S41 language changed")
    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S41 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S41 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S41 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S41 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S41 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S41 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S41 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S41 TRAIN DEV option overlap")


__all__ = ["S41OptimizerStepCase", "generate_s41_cases", "validate_s41_partitions"]
