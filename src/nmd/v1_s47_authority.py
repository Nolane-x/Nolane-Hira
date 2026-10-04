from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S47OrdinalCase:
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
        "graviton_echo_analyzer","graviton echo analyzer","GE","cavity geometry","echo floor",
        ("ring cavity","folded cavity","bowtie cavity","dual-loop cavity","Fabry cavity","triangular cavity","racetrack cavity","nested cavity"),
        ("cryogenic ring cavity","phase-locked folded cavity","balanced bowtie cavity","counterpropagating dual-loop cavity","ultrastable Fabry cavity","dispersion-tuned triangular cavity","low-loss racetrack cavity","shielded nested cavity"),
        ("2.8 arad","3.6 arad","4.4 arad","5.2 arad","6.0 arad","6.8 arad","7.6 arad","8.4 arad"),
        ("3.2 arad","4.0 arad","4.8 arad","5.6 arad","6.4 arad","7.2 arad","8.0 arad","8.8 arad"),101101,981000,
    ),
    _Domain(
        "anyon_braiding_camera","anyon braiding camera","AB","device","braid error",
        ("Hall bar","Majorana wire","moiré plaquette","Josephson network","graphene island","GaAs antidot","InAs island","WTe2 edge"),
        ("fractional Hall bar","parity-locked Majorana wire","twist-tuned moiré plaquette","flux-biased Josephson network","gated graphene island","interferometric GaAs antidot","epitaxial InAs island","helical WTe2 edge"),
        ("0.8 pct","1.1 pct","1.4 pct","1.7 pct","2.0 pct","2.3 pct","2.6 pct","2.9 pct"),
        ("0.95 pct","1.25 pct","1.55 pct","1.85 pct","2.15 pct","2.45 pct","2.75 pct","3.05 pct"),101201,982000,
    ),
    _Domain(
        "quantum_torque_spectrometer","quantum torque spectrometer","QT","rotor","torque floor",
        ("silica dumbbell","diamond rotor","SiN paddle","graphene disk","alumina rod","quartz disk","SiC vane","GaAs paddle"),
        ("levitated silica dumbbell","NV diamond rotor","torsion SiN paddle","tensioned graphene disk","cryogenic alumina rod","low-loss quartz disk","4H-SiC vane","phononic GaAs paddle"),
        ("4 zNm","6 zNm","8 zNm","10 zNm","12 zNm","14 zNm","16 zNm","18 zNm"),
        ("5 zNm","7 zNm","9 zNm","11 zNm","13 zNm","15 zNm","17 zNm","19 zNm"),101301,983000,
    ),
    _Domain(
        "dark_photon_resonator","dark photon resonator","DP","mode","coupling noise",
        ("TM010 mode","TE011 mode","whispering mode","coaxial mode","reentrant mode","dielectric mode","photonic mode","plasma mode"),
        ("tuned TM010 mode","overcoupled TE011 mode","high-Q whispering mode","cryogenic coaxial mode","nested reentrant mode","sapphire dielectric mode","defect photonic mode","metamaterial plasma mode"),
        ("11 nV","15 nV","19 nV","23 nV","27 nV","31 nV","35 nV","39 nV"),
        ("13 nV","17 nV","21 nV","25 nV","29 nV","33 nV","37 nV","41 nV"),101401,984000,
    ),
    _Domain(
        "neutrino_phase_correlator","neutrino phase correlator","NC","target","phase jitter",
        ("argon cell","xenon cell","germanium crystal","silicon crystal","scintillator tile","water cell","tellurium crystal","indium foil"),
        ("ultrapure argon cell","dual-phase xenon cell","enriched germanium crystal","isotopic silicon crystal","timed scintillator tile","gadolinium water cell","bolometric tellurium crystal","cold indium foil"),
        ("0.31 mrad","0.43 mrad","0.55 mrad","0.67 mrad","0.79 mrad","0.91 mrad","1.03 mrad","1.15 mrad"),
        ("0.37 mrad","0.49 mrad","0.61 mrad","0.73 mrad","0.85 mrad","0.97 mrad","1.09 mrad","1.21 mrad"),101501,985000,
    ),
    _Domain(
        "moiré_spin_holograph","moiré spin holograph","MH","stack","spin blur",
        ("WSe2 stack","MoSe2 stack","WS2 stack","MoS2 stack","graphene stack","hBN stack","CrI3 stack","FePS3 stack"),
        ("angle-locked WSe2 stack","moire MoSe2 stack","gated WS2 stack","encapsulated MoS2 stack","twisted graphene stack","defect hBN stack","layered CrI3 stack","antiferromagnetic FePS3 stack"),
        ("0.14 deg","0.20 deg","0.26 deg","0.32 deg","0.38 deg","0.44 deg","0.50 deg","0.56 deg"),
        ("0.17 deg","0.23 deg","0.29 deg","0.35 deg","0.41 deg","0.47 deg","0.53 deg","0.59 deg"),101601,986000,
    ),
    _Domain(
        "quantum_pressure_mapper","quantum pressure mapper","QP","membrane","pressure floor",
        ("graphene drum","SiN drum","diamond membrane","AlN membrane","SiC membrane","MoS2 drum","quartz membrane","GaAs membrane"),
        ("tensioned graphene drum","soft-clamped SiN drum","nanocrystalline diamond membrane","piezo AlN membrane","4H-SiC membrane","encapsulated MoS2 drum","low-loss quartz membrane","phononic GaAs membrane"),
        ("8 nPa","11 nPa","14 nPa","17 nPa","20 nPa","23 nPa","26 nPa","29 nPa"),
        ("9.5 nPa","12.5 nPa","15.5 nPa","18.5 nPa","21.5 nPa","24.5 nPa","27.5 nPa","30.5 nPa"),101701,987000,
    ),
    _Domain(
        "phonon_entanglement_router","phonon entanglement router","PE","waveguide","entanglement loss",
        ("Si guide","SiN guide","AlN guide","LiNbO3 guide","diamond guide","SiC guide","GaN guide","quartz guide"),
        ("phononic Si guide","soft-boundary SiN guide","piezo AlN guide","periodic LiNbO3 guide","nanobeam diamond guide","4H-SiC guide","suspended GaN guide","cryogenic quartz guide"),
        ("0.7 dB","1.0 dB","1.3 dB","1.6 dB","1.9 dB","2.2 dB","2.5 dB","2.8 dB"),
        ("0.85 dB","1.15 dB","1.45 dB","1.75 dB","2.05 dB","2.35 dB","2.65 dB","2.95 dB"),101801,988000,
    ),
    _Domain(
        "vacuum_birefringence_camera","vacuum birefringence camera","VB","magnet","rotation floor",
        ("dipole magnet","solenoid magnet","quadrupole magnet","Halbach ring","toroid magnet","racetrack coil","split-pair coil","canted coil"),
        ("superconducting dipole magnet","persistent solenoid magnet","gradient quadrupole magnet","nested Halbach ring","shielded toroid magnet","cryogenic racetrack coil","balanced split-pair coil","canted-cosine-theta coil"),
        ("12 nrad","17 nrad","22 nrad","27 nrad","32 nrad","37 nrad","42 nrad","47 nrad"),
        ("14.5 nrad","19.5 nrad","24.5 nrad","29.5 nrad","34.5 nrad","39.5 nrad","44.5 nrad","49.5 nrad"),101901,989000,
    ),
    _Domain(
        "nuclear_clock_interferometer","nuclear clock interferometer","NI","isotope","clock drift",
        ("Th229 crystal","U235 crystal","Yb171 ion","Sr87 atom","Al27 ion","Hg199 ion","Lu176 ion","Ca43 ion"),
        ("isomer-addressed Th229 crystal","shielded U235 crystal","logic-readout Yb171 ion","lattice Sr87 atom","quantum-logic Al27 ion","cryogenic Hg199 ion","hyperfine Lu176 ion","clocked Ca43 ion"),
        ("1.1e-18","1.5e-18","1.9e-18","2.3e-18","2.7e-18","3.1e-18","3.5e-18","3.9e-18"),
        ("1.3e-18","1.7e-18","2.1e-18","2.5e-18","2.9e-18","3.3e-18","3.7e-18","4.1e-18"),102001,990000,
    ),
    _Domain(
        "quantum_thermal_gradiometer","quantum thermal gradiometer","TG","sensor","gradient floor",
        ("NV sensor","SiV sensor","SQUID sensor","graphene sensor","MoS2 sensor","Johnson sensor","bolometer","nanowire sensor"),
        ("vector NV sensor","isotopic SiV sensor","nanoSQUID sensor","suspended graphene sensor","encapsulated MoS2 sensor","cross-correlated Johnson sensor","TES bolometer","proximity nanowire sensor"),
        ("0.8 uK/mm","1.1 uK/mm","1.4 uK/mm","1.7 uK/mm","2.0 uK/mm","2.3 uK/mm","2.6 uK/mm","2.9 uK/mm"),
        ("0.95 uK/mm","1.25 uK/mm","1.55 uK/mm","1.85 uK/mm","2.15 uK/mm","2.45 uK/mm","2.75 uK/mm","3.05 uK/mm"),102101,991000,
    ),
    _Domain(
        "spin_photon_phase_bridge","spin photon phase bridge","SP","interface","phase loss",
        ("NV cavity","SiV cavity","QD cavity","rare-earth cavity","Er fiber","Yb crystal","Rb ensemble","Cs ensemble"),
        ("impedance-matched NV cavity","nanophotonic SiV cavity","charged-QD cavity","rare-earth nanocavity","cryogenic Er fiber","spectral-hole Yb crystal","EIT Rb ensemble","spin-wave Cs ensemble"),
        ("0.09 rad","0.13 rad","0.17 rad","0.21 rad","0.25 rad","0.29 rad","0.33 rad","0.37 rad"),
        ("0.11 rad","0.15 rad","0.19 rad","0.23 rad","0.27 rad","0.31 rad","0.35 rad","0.39 rad"),102201,992000,
    ),
)


def _shuffle(*, case_id, noun, field_a, field_b, first, second, distractor_first, distractor_second, seed):
    rows=[
        ("a",field_a,first),
        ("b",field_b,second),
        ("x",field_a,distractor_first),
        ("y",field_b,distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(
        f"{value} is the S47 ordinal-authority {field} value for this {noun}"
        for _kind,field,value in rows
    )
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S47-Ordinal {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S47-Ordinal record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S47-Ordinal {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S47-Ordinal record {code}?",
            f"For S47-Ordinal {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S47-Ordinal record {code}?",
        )
    return (
        f"S47-Ordinal audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S47-Ordinal audit {code}: what is {spec.field_a}?",
        f"In S47-Ordinal audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S47-Ordinal audit {code}: what is {spec.field_b}?",
        f"In S47-Ordinal audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    firsts=spec.train_a if split=="train" else spec.dev_a
    seconds=spec.train_b if split=="train" else spec.dev_b
    first=firsts[index%len(firsts)]
    second=seconds[(index*3+1)%len(seconds)]
    distractor_first=firsts[(index+3)%len(firsts)]
    distractor_second=seconds[(index*5+2)%len(seconds)]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    case_id=f"{split}-s47-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=case_id,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=distractor_first,
        distractor_second=distractor_second,seed=spec.seed+offset,
    )
    return S47OrdinalCase(
        case_id,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,
        texts,aliases,ids,ga,gb,
    )


def generate_s47_cases(split: Split) -> tuple[S47OrdinalCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s47_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192:
        raise RuntimeError("S47 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12:
        raise RuntimeError("S47 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S47 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S47 language changed")

    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S47 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b:
                raise RuntimeError("S47 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S47 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S47 option IDs changed")
            if r.gold_a==r.gold_b:
                raise RuntimeError("S47 paired golds must differ")

    ts={x for r in train for x in (r.state_a,r.state_b)}
    ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds:
        raise RuntimeError("S47 TRAIN DEV state overlap")
    if tq&dq:
        raise RuntimeError("S47 TRAIN DEV question overlap")
    if to&do:
        raise RuntimeError("S47 TRAIN DEV option overlap")


__all__=["S47OrdinalCase","generate_s47_cases","validate_s47_partitions"]
