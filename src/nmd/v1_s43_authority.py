from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S43RelativeGapCase:
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
        "quantum_vorticity_camera","quantum vorticity camera","QV","tracer","vortex precision",
        ("He2 excimer","Yb impurity","electron bubble","Rb droplet","Cs impurity","Xe dimer","Ne cluster","metastable Ar"),
        ("spin-tagged He2 excimer","laser-cooled Yb impurity","resolved electron bubble","Rydberg Rb droplet","clocked Cs impurity","state-selected Xe dimer","size-selected Ne cluster","optically pumped metastable Ar"),
        ("5 nm","7 nm","9 nm","11 nm","13 nm","15 nm","17 nm","19 nm"),
        ("6 nm","8 nm","10 nm","12 nm","14 nm","16 nm","18 nm","20 nm"),85101,901000,
    ),
    _Domain(
        "femtosecond_magnetization_mapper","femtosecond magnetization mapper","FM","probe","spin-angle error",
        ("MOKE pulse","XMCD burst","THz gate","spin-ARPES","NV pulse","Faraday crystal","electron packet","plasmonic tip"),
        ("balanced MOKE pulse","seeded XMCD burst","phase-locked THz gate","time-resolved spin-ARPES","synchronized NV pulse","shot-noise Faraday crystal","compressed electron packet","heterodyne plasmonic tip"),
        ("0.4 mrad","0.6 mrad","0.8 mrad","1.0 mrad","1.2 mrad","1.4 mrad","1.6 mrad","1.8 mrad"),
        ("0.5 mrad","0.7 mrad","0.9 mrad","1.1 mrad","1.3 mrad","1.5 mrad","1.7 mrad","1.9 mrad"),85201,902000,
    ),
    _Domain(
        "phonon_phase_radar","phonon phase radar","PR","emitter","phase jitter",
        ("AlN transducer","LiNbO3 comb","GaN ring","SiC cavity","diamond beam","quartz plate","hBN drum","Si nanobeam"),
        ("epitaxial AlN transducer","chirped LiNbO3 comb","suspended GaN ring","4H-SiC cavity","defect diamond beam","cryogenic quartz plate","encapsulated hBN drum","phononic Si nanobeam"),
        ("6 mrad","10 mrad","14 mrad","18 mrad","22 mrad","26 mrad","30 mrad","34 mrad"),
        ("8 mrad","12 mrad","16 mrad","20 mrad","24 mrad","28 mrad","32 mrad","36 mrad"),85301,903000,
    ),
    _Domain(
        "rydberg_charge_tomograph","rydberg charge tomograph","RC","atomic probe","field floor",
        ("Rb70S","Cs81D","Sr triplet","Yb Rydberg","K circular","Na Stark","Ca Rydberg","Li high-l"),
        ("dressed Rb70S","circularized Cs81D","clocked Sr triplet","trapped Yb Rydberg","spin-polarized K circular","microwave Na Stark","ion-coupled Ca Rydberg","adiabatic Li high-l"),
        ("0.3 mV/m","0.5 mV/m","0.7 mV/m","0.9 mV/m","1.1 mV/m","1.3 mV/m","1.5 mV/m","1.7 mV/m"),
        ("0.4 mV/m","0.6 mV/m","0.8 mV/m","1.0 mV/m","1.2 mV/m","1.4 mV/m","1.6 mV/m","1.8 mV/m"),85401,904000,
    ),
    _Domain(
        "xuv_wavefront_router","xuv wavefront router","XW","optic","phase residual",
        ("multilayer mirror","Fresnel zoneplate","nanograting","crystal wedge","phase membrane","metasurface","channel plate","grazing prism"),
        ("adaptive multilayer mirror","achromatic Fresnel zoneplate","blazed nanograting","channel-cut crystal wedge","MEMS phase membrane","resonant xuv metasurface","curved channel plate","graded grazing prism"),
        ("3 mrad","5 mrad","7 mrad","9 mrad","11 mrad","13 mrad","15 mrad","17 mrad"),
        ("4 mrad","6 mrad","8 mrad","10 mrad","12 mrad","14 mrad","16 mrad","18 mrad"),85501,905000,
    ),
    _Domain(
        "molecular_spin_interferometer","molecular spin interferometer","MS","species","coherence time",
        ("CaF","SrF","YO","BaF","NH","OH","ThO","HfF+"),
        ("laser-cooled CaF","state-selected SrF","trapped YO","slowed BaF","spin-polarized NH","lambda-doublet OH","metastable ThO","rotationally cooled HfF+"),
        ("0.8 ms","1.2 ms","1.6 ms","2.0 ms","2.4 ms","2.8 ms","3.2 ms","3.6 ms"),
        ("1.0 ms","1.4 ms","1.8 ms","2.2 ms","2.6 ms","3.0 ms","3.4 ms","3.8 ms"),85601,906000,
    ),
    _Domain(
        "nanophotonic_recoil_meter","nanophotonic recoil meter","NR","resonator","force floor",
        ("SiN ring","GaP disk","diamond cavity","AlGaAs beam","SiC ring","LiNbO3 disk","hBN cavity","silicon slot"),
        ("ultralow-loss SiN ring","bound-state GaP disk","color-center diamond cavity","suspended AlGaAs beam","4H-SiC ring","thin-film LiNbO3 disk","encapsulated hBN cavity","inverse-designed silicon slot"),
        ("4 aN","6 aN","8 aN","10 aN","12 aN","14 aN","16 aN","18 aN"),
        ("5 aN","7 aN","9 aN","11 aN","13 aN","15 aN","17 aN","19 aN"),85701,907000,
    ),
    _Domain(
        "spinwave_frequency_router","spinwave frequency router","SF","channel","isolation",
        ("Damon-Eshbach","backward-volume","exchange mode","edge mode","domain-wall mode","vortex mode","surface mode","hybrid magnon"),
        ("phase-coded Damon-Eshbach","confined backward-volume","high-k exchange mode","topological edge mode","guided domain-wall mode","locked vortex mode","nonreciprocal surface mode","cavity-hybrid magnon"),
        ("28 dB","31 dB","34 dB","37 dB","40 dB","43 dB","46 dB","49 dB"),
        ("29 dB","32 dB","35 dB","38 dB","41 dB","44 dB","47 dB","50 dB"),85801,908000,
    ),
    _Domain(
        "cryogenic_charge_calorimeter","cryogenic charge calorimeter","CC","absorber","energy floor",
        ("Au island","graphene dot","Cu island","Ti pad","Al grain","NbN bridge","Pt strip","MoRe loop"),
        ("mesoscopic Au island","Coulomb graphene dot","suspended Cu island","TES Ti pad","parity-resolved Al grain","kinetic NbN bridge","spin-Hall Pt strip","proximitized MoRe loop"),
        ("9 ueV","12 ueV","15 ueV","18 ueV","21 ueV","24 ueV","27 ueV","30 ueV"),
        ("10 ueV","13 ueV","16 ueV","19 ueV","22 ueV","25 ueV","28 ueV","31 ueV"),85901,909000,
    ),
    _Domain(
        "ultrafast_neutron_phase_camera","ultrafast neutron phase camera","UN","converter","timing spread",
        ("B10 film","Li6 glass","Gd foil","diamond layer","He3 cell","scintillator slab","MCP stack","boron nanowire"),
        ("nanostructured B10 film","segmented Li6 glass","thin Gd foil","CVD diamond layer","polarized He3 cell","fast scintillator slab","resonant MCP stack","aligned boron nanowire"),
        ("18 ps","24 ps","30 ps","36 ps","42 ps","48 ps","54 ps","60 ps"),
        ("21 ps","27 ps","33 ps","39 ps","45 ps","51 ps","57 ps","63 ps"),86001,910000,
    ),
    _Domain(
        "plasma_current_holograph","plasma current holograph","PC","diagnostic","current error",
        ("Faraday array","B-dot mesh","proton radiograph","Zeeman camera","Hall grid","Thomson sheet","polarimetry comb","electron beam"),
        ("balanced Faraday array","calibrated B-dot mesh","monoenergetic proton radiograph","resolved Zeeman camera","cryogenic Hall grid","time-gated Thomson sheet","dual-comb polarimetry","ultrashort electron beam"),
        ("0.7%","1.0%","1.3%","1.6%","1.9%","2.2%","2.5%","2.8%"),
        ("0.8%","1.1%","1.4%","1.7%","2.0%","2.3%","2.6%","2.9%"),86101,911000,
    ),
    _Domain(
        "quantum_pressure_lattice","quantum pressure lattice","QP","sensor","pressure noise",
        ("graphene drum","SiN membrane","diamond beam","quartz disk","GaAs paddle","AlN plate","hBN resonator","SiC cantilever"),
        ("tensioned graphene drum","phononic SiN membrane","color-center diamond beam","high-Q quartz disk","optical GaAs paddle","epitaxial AlN plate","encapsulated hBN resonator","4H-SiC cantilever"),
        ("5 nPa","7 nPa","9 nPa","11 nPa","13 nPa","15 nPa","17 nPa","19 nPa"),
        ("6 nPa","8 nPa","10 nPa","12 nPa","14 nPa","16 nPa","18 nPa","20 nPa"),86201,912000,
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
        f"{value} is the S43 gap-audit {field} value for this {noun}"
        for _kind, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(spec, *, split, code, first, second):
    if split == "train":
        return (
            f"S43-Gap {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S43-Gap record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S43-Gap {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S43-Gap record {code}?",
            f"For S43-Gap {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S43-Gap record {code}?",
        )
    return (
        f"S43-Gap audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S43-Gap audit {code}: what is {spec.field_a}?",
        f"In S43-Gap audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S43-Gap audit {code}: what is {spec.field_b}?",
        f"In S43-Gap audit {code}, which entry is tagged {spec.field_b}?",
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
    case_id = f"{split}-s43-{spec.name}-{index:03d}"
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
    return S43RelativeGapCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s43_cases(split: Split) -> tuple[S43RelativeGapCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s43_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S43 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S43 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S43 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S43 language changed")

    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S43 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S43 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S43 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S43 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S43 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S43 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S43 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S43 TRAIN DEV option overlap")


__all__ = [
    "S43RelativeGapCase",
    "generate_s43_cases",
    "validate_s43_partitions",
]
