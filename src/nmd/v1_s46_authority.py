from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S46ConsensusCase:
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
        "axion_phase_tracker","axion phase tracker","AX","pickup geometry","phase floor",
        ("toroidal pickup","racetrack pickup","meander pickup","spiral pickup","gradiometric pickup","saddle pickup","helical pickup","planar pickup"),
        ("nested toroidal pickup","counter-wound racetrack pickup","fractal meander pickup","dual-spiral pickup","balanced gradiometric pickup","shielded saddle pickup","aperiodic helical pickup","cryogenic planar pickup"),
        ("3.1 urad","4.3 urad","5.5 urad","6.7 urad","7.9 urad","9.1 urad","10.3 urad","11.5 urad"),
        ("3.7 urad","4.9 urad","6.1 urad","7.3 urad","8.5 urad","9.7 urad","10.9 urad","12.1 urad"),98101,961000,
    ),
    _Domain(
        "magnon_chirality_mapper","magnon chirality mapper","MG","waveguide","chirality leakage",
        ("YIG strip","CoFeB strip","NiFe strip","FeGa strip","hematite strip","NiO strip","CoO strip","Cr2O3 strip"),
        ("domain-tuned YIG strip","exchange-biased CoFeB strip","low-loss NiFe strip","magnetostrictive FeGa strip","canted hematite strip","antiferromagnetic NiO strip","epitaxial CoO strip","magnetoelectric Cr2O3 strip"),
        ("-22 dB","-26 dB","-30 dB","-34 dB","-38 dB","-42 dB","-46 dB","-50 dB"),
        ("-24 dB","-28 dB","-32 dB","-36 dB","-40 dB","-44 dB","-48 dB","-52 dB"),98201,962000,
    ),
    _Domain(
        "rydberg_stark_holograph","Rydberg Stark holograph","RY","atomic ensemble","field blur",
        ("Rb-50S cloud","Cs-60S cloud","Rb-70D cloud","Cs-80P cloud","K-45S cloud","Na-40D cloud","Li-35P cloud","Sr-55D cloud"),
        ("magic-field Rb-50S cloud","shielded Cs-60S cloud","circular Rb-70D cloud","microwave-dressed Cs-80P cloud","ultracold K-45S cloud","collimated Na-40D cloud","trapped Li-35P cloud","clocked Sr-55D cloud"),
        ("5.2 uV/cm","6.4 uV/cm","7.6 uV/cm","8.8 uV/cm","10.0 uV/cm","11.2 uV/cm","12.4 uV/cm","13.6 uV/cm"),
        ("5.8 uV/cm","7.0 uV/cm","8.2 uV/cm","9.4 uV/cm","10.6 uV/cm","11.8 uV/cm","13.0 uV/cm","14.2 uV/cm"),98301,963000,
    ),
    _Domain(
        "quantum_acoustic_tomograph","quantum acoustic tomograph","QA","mode family","delay noise",
        ("Rayleigh mode","Lamb A0 mode","Lamb S0 mode","Love mode","bulk shear mode","bulk longitudinal mode","edge mode","whispering mode"),
        ("phononic-crystal Rayleigh mode","slow-wave Lamb A0 mode","symmetric Lamb S0 mode","guided Love mode","cryogenic bulk shear mode","coherent bulk longitudinal mode","topological edge mode","high-Q whispering mode"),
        ("8 ps","11 ps","14 ps","17 ps","20 ps","23 ps","26 ps","29 ps"),
        ("9.5 ps","12.5 ps","15.5 ps","18.5 ps","21.5 ps","24.5 ps","27.5 ps","30.5 ps"),98401,964000,
    ),
    _Domain(
        "spin_orbit_interferometer","spin orbit interferometer","SO","channel","spin phase spread",
        ("InAs wire","InSb wire","Ge hole wire","GaAs channel","graphene ribbon","WTe2 edge","HgTe well","MoTe2 edge"),
        ("gated InAs wire","ballistic InSb wire","strained Ge hole wire","high-mobility GaAs channel","encapsulated graphene ribbon","helical WTe2 edge","inverted HgTe well","protected MoTe2 edge"),
        ("0.08 rad","0.12 rad","0.16 rad","0.20 rad","0.24 rad","0.28 rad","0.32 rad","0.36 rad"),
        ("0.10 rad","0.14 rad","0.18 rad","0.22 rad","0.26 rad","0.30 rad","0.34 rad","0.38 rad"),98501,965000,
    ),
    _Domain(
        "optomechanical_recoil_clock","optomechanical recoil clock","OR","oscillator","recoil jitter",
        ("SiN drum","diamond beam","Si beam","GaAs disk","AlN disk","SiC beam","graphene drum","quartz beam"),
        ("soft-clamped SiN drum","phononic diamond beam","dilution-cooled Si beam","resolved-sideband GaAs disk","piezo AlN disk","4H-SiC beam","tensioned graphene drum","low-loss quartz beam"),
        ("2.4 fm","3.1 fm","3.8 fm","4.5 fm","5.2 fm","5.9 fm","6.6 fm","7.3 fm"),
        ("2.75 fm","3.45 fm","4.15 fm","4.85 fm","5.55 fm","6.25 fm","6.95 fm","7.65 fm"),98601,966000,
    ),
    _Domain(
        "topological_photon_router","topological photon router","TP","lattice","backscatter floor",
        ("honeycomb lattice","square lattice","kagome lattice","ring lattice","valley lattice","Floquet lattice","Chern lattice","synthetic lattice"),
        ("domain-wall honeycomb lattice","defect-free square lattice","breathing kagome lattice","coupled-ring lattice","valley-Hall lattice","driven Floquet lattice","magneto-optic Chern lattice","frequency synthetic lattice"),
        ("-31 dB","-35 dB","-39 dB","-43 dB","-47 dB","-51 dB","-55 dB","-59 dB"),
        ("-33 dB","-37 dB","-41 dB","-45 dB","-49 dB","-53 dB","-57 dB","-61 dB"),98701,967000,
    ),
    _Domain(
        "neutron_phase_microscope","neutron phase microscope","NP","sample","phase noise",
        ("Si grating","quartz grating","Ni film","Ti film","Al film","graphite plate","diamond plate","MgO plate"),
        ("perfect-Si grating","strain-relieved quartz grating","polarized Ni film","isotopic Ti film","oxide-free Al film","mosaic graphite plate","CVD diamond plate","annealed MgO plate"),
        ("0.21 mrad","0.27 mrad","0.33 mrad","0.39 mrad","0.45 mrad","0.51 mrad","0.57 mrad","0.63 mrad"),
        ("0.24 mrad","0.30 mrad","0.36 mrad","0.42 mrad","0.48 mrad","0.54 mrad","0.60 mrad","0.66 mrad"),98801,968000,
    ),
    _Domain(
        "molecular_rotation_mapper","molecular rotation mapper","MR","species","rotation linewidth",
        ("OCS beam","CO beam","N2O beam","HCN beam","CH3F beam","NH3 beam","H2CO beam","SO2 beam"),
        ("state-selected OCS beam","Stark-decelerated CO beam","cold N2O beam","oriented HCN beam","symmetric-top CH3F beam","inversion-cooled NH3 beam","para-H2CO beam","supersonic SO2 beam"),
        ("18 kHz","23 kHz","28 kHz","33 kHz","38 kHz","43 kHz","48 kHz","53 kHz"),
        ("20.5 kHz","25.5 kHz","30.5 kHz","35.5 kHz","40.5 kHz","45.5 kHz","50.5 kHz","55.5 kHz"),98901,969000,
    ),
    _Domain(
        "plasmon_momentum_camera","plasmon momentum camera","PM","surface","momentum blur",
        ("Au film","Ag film","graphene sheet","Al film","Cu film","TiN film","ITO film","AZO film"),
        ("single-crystal Au film","template-stripped Ag film","gated graphene sheet","epitaxial Al film","oxide-free Cu film","low-loss TiN film","ENZ ITO film","doped AZO film"),
        ("0.021 um-1","0.029 um-1","0.037 um-1","0.045 um-1","0.053 um-1","0.061 um-1","0.069 um-1","0.077 um-1"),
        ("0.025 um-1","0.033 um-1","0.041 um-1","0.049 um-1","0.057 um-1","0.065 um-1","0.073 um-1","0.081 um-1"),99001,970000,
    ),
    _Domain(
        "atomic_gravity_gradiometer","atomic gravity gradiometer","AG","species","gradient floor",
        ("Rb87 cloud","Cs133 cloud","Sr88 cloud","Yb174 cloud","K39 cloud","Na23 cloud","Li7 cloud","Ca40 cloud"),
        ("delta-kicked Rb87 cloud","fountain Cs133 cloud","clock-state Sr88 cloud","ultracold Yb174 cloud","dual-state K39 cloud","collimated Na23 cloud","degenerate Li7 cloud","laser-cooled Ca40 cloud"),
        ("3.2 E","4.1 E","5.0 E","5.9 E","6.8 E","7.7 E","8.6 E","9.5 E"),
        ("3.65 E","4.55 E","5.45 E","6.35 E","7.25 E","8.15 E","9.05 E","9.95 E"),99101,971000,
    ),
    _Domain(
        "ferroelectric_domain_radar","ferroelectric domain radar","FD","crystal","domain jitter",
        ("BaTiO3 slab","LiNbO3 slab","PZT film","HfZrO film","BiFeO3 film","KTP crystal","KNbO3 crystal","Rochelle salt"),
        ("strained BaTiO3 slab","periodic LiNbO3 slab","epitaxial PZT film","wake-up HfZrO film","multiferroic BiFeO3 film","poled KTP crystal","domain-engineered KNbO3 crystal","dehydrated Rochelle salt"),
        ("1.2 nm","1.7 nm","2.2 nm","2.7 nm","3.2 nm","3.7 nm","4.2 nm","4.7 nm"),
        ("1.45 nm","1.95 nm","2.45 nm","2.95 nm","3.45 nm","3.95 nm","4.45 nm","4.95 nm"),99201,972000,
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
    aliases = tuple(
        f"{value} is the S46 consensus-authority {field} value for this {noun}"
        for _kind, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(spec, *, split, code, first, second):
    if split == "train":
        return (
            f"S46-Consensus {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S46-Consensus record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S46-Consensus {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S46-Consensus record {code}?",
            f"For S46-Consensus {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S46-Consensus record {code}?",
        )
    return (
        f"S46-Consensus audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S46-Consensus audit {code}: what is {spec.field_a}?",
        f"In S46-Consensus audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S46-Consensus audit {code}: what is {spec.field_b}?",
        f"In S46-Consensus audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec, index, split):
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s46-{spec.name}-{index:03d}"
    sa, sb, qa1, qa2, qb1, qb2 = _views(
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
    return S46ConsensusCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s46_cases(split: Split) -> tuple[S46ConsensusCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s46_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S46 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S46 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S46 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S46 language changed")

    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S46 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S46 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S46 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S46 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S46 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S46 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S46 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S46 TRAIN DEV option overlap")


__all__ = ["S46ConsensusCase", "generate_s46_cases", "validate_s46_partitions"]
