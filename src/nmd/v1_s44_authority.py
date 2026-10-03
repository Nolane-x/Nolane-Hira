from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S44PrivateCorrectionCase:
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
        "quantum_phase_gradiometer","quantum phase gradiometer","QG","interferometer","phase floor",
        ("Cs fountain","Rb lattice","Sr clock","Yb array","Ca beam","K cloud","Na fountain","Mg ion"),
        ("entangled Cs fountain","spin-squeezed Rb lattice","clocked Sr array","dual-axis Yb array","velocity-selected Ca beam","degenerate K cloud","delta-kicked Na fountain","sympathetic Mg ion"),
        ("2.1 urad","2.7 urad","3.3 urad","3.9 urad","4.5 urad","5.1 urad","5.7 urad","6.3 urad"),
        ("2.4 urad","3.0 urad","3.6 urad","4.2 urad","4.8 urad","5.4 urad","6.0 urad","6.6 urad"),87101,921000,
    ),
    _Domain(
        "coherent_neutrino_imager","coherent neutrino imager","CN","target","recoil threshold",
        ("Ge crystal","Si wafer","Ar cell","Xe cell","CaWO4 cube","NaI tile","sapphire disk","diamond chip"),
        ("cryogenic Ge crystal","skipper Si wafer","dual-phase Ar cell","low-background Xe cell","phonon CaWO4 cube","segmented NaI tile","TES sapphire disk","CVD diamond chip"),
        ("7 eV","9 eV","11 eV","13 eV","15 eV","17 eV","19 eV","21 eV"),
        ("8 eV","10 eV","12 eV","14 eV","16 eV","18 eV","20 eV","22 eV"),87201,922000,
    ),
    _Domain(
        "magnon_phase_spectrometer","magnon phase spectrometer","MP","analyzer","phase noise",
        ("YIG strip","CoFeB bar","NiFe disk","hematite plate","FeRh wire","MnF2 slab","CrI3 flake","NiO film"),
        ("low-loss YIG strip","exchange-biased CoFeB bar","vortex NiFe disk","canted hematite plate","phase-transition FeRh wire","antiferromagnetic MnF2 slab","encapsulated CrI3 flake","epitaxial NiO film"),
        ("0.9 mrad","1.3 mrad","1.7 mrad","2.1 mrad","2.5 mrad","2.9 mrad","3.3 mrad","3.7 mrad"),
        ("1.1 mrad","1.5 mrad","1.9 mrad","2.3 mrad","2.7 mrad","3.1 mrad","3.5 mrad","3.9 mrad"),87301,923000,
    ),
    _Domain(
        "attonewton_force_tomograph","attonewton force tomograph","AF","probe","force noise",
        ("Si nanowire","diamond beam","graphene ribbon","CNT cantilever","GaAs paddle","SiC needle","hBN drum","quartz whisker"),
        ("doubly-clamped Si nanowire","color-center diamond beam","tensioned graphene ribbon","ultralong CNT cantilever","optomechanical GaAs paddle","4H-SiC needle","encapsulated hBN drum","cryogenic quartz whisker"),
        ("0.6 aN","0.9 aN","1.2 aN","1.5 aN","1.8 aN","2.1 aN","2.4 aN","2.7 aN"),
        ("0.75 aN","1.05 aN","1.35 aN","1.65 aN","1.95 aN","2.25 aN","2.55 aN","2.85 aN"),87401,924000,
    ),
    _Domain(
        "xray_polarization_router","xray polarization router","XP","optic","leakage",
        ("diamond phase plate","Si channel plate","multilayer stack","Bragg mirror","Laue crystal","zone plate","metasurface foil","grazing grating"),
        ("strained diamond phase plate","channel-cut Si plate","graded multilayer stack","asymmetric Bragg mirror","bent Laue crystal","polarizing zone plate","nanostructured metasurface foil","blazed grazing grating"),
        ("-31 dB","-34 dB","-37 dB","-40 dB","-43 dB","-46 dB","-49 dB","-52 dB"),
        ("-32 dB","-35 dB","-38 dB","-41 dB","-44 dB","-47 dB","-50 dB","-53 dB"),87501,925000,
    ),
    _Domain(
        "molecular_parity_clock","molecular parity clock","MC","species","fractional drift",
        ("ThO","HfF+","CaF","SrF","BaF","YO","NH","OH"),
        ("oriented ThO","trapped HfF+","laser-cooled CaF","state-selected SrF","slowed BaF","trapped YO","spin-polarized NH","lambda-doublet OH"),
        ("2e-16","3e-16","4e-16","5e-16","6e-16","7e-16","8e-16","9e-16"),
        ("2.5e-16","3.5e-16","4.5e-16","5.5e-16","6.5e-16","7.5e-16","8.5e-16","9.5e-16"),87601,926000,
    ),
    _Domain(
        "plasmonic_heat_holograph","plasmonic heat holograph","PH","antenna","temperature noise",
        ("Au bowtie","Ag nanocube","Al disk","Cu slot","TiN ring","graphene patch","Pt rod","Pd dimer"),
        ("gap-tuned Au bowtie","facet-controlled Ag nanocube","anodized Al disk","inverse-designed Cu slot","epitaxial TiN ring","gated graphene patch","thermometric Pt rod","hydrogen-loaded Pd dimer"),
        ("8 uK","11 uK","14 uK","17 uK","20 uK","23 uK","26 uK","29 uK"),
        ("9.5 uK","12.5 uK","15.5 uK","18.5 uK","21.5 uK","24.5 uK","27.5 uK","30.5 uK"),87701,927000,
    ),
    _Domain(
        "spin_texture_radar","spin texture radar","ST","sensor","angle floor",
        ("NV array","MOKE grid","XMCD pixel","spin-SEM","Hall cross","SQUID loop","Faraday film","electron holograph"),
        ("vector NV array","balanced MOKE grid","ptychographic XMCD pixel","low-dose spin-SEM","ballistic Hall cross","nanoSQUID loop","bismuth Faraday film","phase-locked electron holograph"),
        ("0.12 deg","0.18 deg","0.24 deg","0.30 deg","0.36 deg","0.42 deg","0.48 deg","0.54 deg"),
        ("0.15 deg","0.21 deg","0.27 deg","0.33 deg","0.39 deg","0.45 deg","0.51 deg","0.57 deg"),87801,928000,
    ),
    _Domain(
        "ultracold_collision_mapper","ultracold collision mapper","UC","channel","energy width",
        ("s-wave","p-wave","d-wave","Feshbach","spin-exchange","dipolar","ion-atom","molecule-atom"),
        ("threshold s-wave","shape-resonant p-wave","centrifugal d-wave","narrow Feshbach","resolved spin-exchange","oriented dipolar","sympathetic ion-atom","state-selected molecule-atom"),
        ("4 nK","6 nK","8 nK","10 nK","12 nK","14 nK","16 nK","18 nK"),
        ("5 nK","7 nK","9 nK","11 nK","13 nK","15 nK","17 nK","19 nK"),87901,929000,
    ),
    _Domain(
        "terahertz_charge_vortimeter","terahertz charge vortimeter","TV","detector","field noise",
        ("graphene FET","InAs nanowire","GaN HEMT","Josephson junction","Schottky diode","2DEG strip","MoS2 FET","HgTe well"),
        ("ballistic graphene FET","gated InAs nanowire","low-noise GaN HEMT","overdamped Josephson junction","zero-bias Schottky diode","plasmonic 2DEG strip","encapsulated MoS2 FET","quantum HgTe well"),
        ("0.7 uV/cm","1.0 uV/cm","1.3 uV/cm","1.6 uV/cm","1.9 uV/cm","2.2 uV/cm","2.5 uV/cm","2.8 uV/cm"),
        ("0.85 uV/cm","1.15 uV/cm","1.45 uV/cm","1.75 uV/cm","2.05 uV/cm","2.35 uV/cm","2.65 uV/cm","2.95 uV/cm"),88001,930000,
    ),
    _Domain(
        "phononic_topology_scanner","phononic topology scanner","PT","waveguide","backscatter",
        ("Si membrane","AlN ridge","LiNbO3 strip","GaAs slab","diamond beam","SiC plate","quartz strip","sapphire membrane"),
        ("valley-Hall Si membrane","chiral AlN ridge","domain-wall LiNbO3 strip","topological GaAs slab","defect diamond beam","spin-locked SiC plate","synthetic-gauge quartz strip","helical sapphire membrane"),
        ("-24 dB","-28 dB","-32 dB","-36 dB","-40 dB","-44 dB","-48 dB","-52 dB"),
        ("-26 dB","-30 dB","-34 dB","-38 dB","-42 dB","-46 dB","-50 dB","-54 dB"),88101,931000,
    ),
    _Domain(
        "quantum_torque_array","quantum torque array","QT","element","torque floor",
        ("diamond paddle","SiN torsor","graphene rotor","CNT lever","GaAs disk","SiC beam","quartz fork","AlN plate"),
        ("spin-readout diamond paddle","phononic SiN torsor","levitated graphene rotor","ultralong CNT lever","cavity GaAs disk","defect SiC beam","cryogenic quartz fork","piezo AlN plate"),
        ("3 yNm","5 yNm","7 yNm","9 yNm","11 yNm","13 yNm","15 yNm","17 yNm"),
        ("4 yNm","6 yNm","8 yNm","10 yNm","12 yNm","14 yNm","16 yNm","18 yNm"),88201,932000,
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
        f"{value} is the S44 private-audit {field} value for this {noun}"
        for _kind, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(spec, *, split, code, first, second):
    if split == "train":
        return (
            f"S44-Priv {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S44-Priv record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S44-Priv {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S44-Priv record {code}?",
            f"For S44-Priv {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S44-Priv record {code}?",
        )
    return (
        f"S44-Priv audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S44-Priv audit {code}: what is {spec.field_a}?",
        f"In S44-Priv audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S44-Priv audit {code}: what is {spec.field_b}?",
        f"In S44-Priv audit {code}, which entry is tagged {spec.field_b}?",
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
    case_id = f"{split}-s44-{spec.name}-{index:03d}"
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
    return S44PrivateCorrectionCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s44_cases(split: Split) -> tuple[S44PrivateCorrectionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s44_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S44 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S44 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S44 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S44 language changed")

    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S44 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S44 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S44 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S44 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S44 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S44 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S44 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S44 TRAIN DEV option overlap")


__all__ = [
    "S44PrivateCorrectionCase",
    "generate_s44_cases",
    "validate_s44_partitions",
]
