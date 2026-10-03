from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S42RelationalGeometryCase:
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
        "topological_microwave_tomograph","topological microwave tomograph","TM","lattice medium","phase floor",
        ("gyrotropic ferrite","Chern resonator","synthetic flux mesh","helical cavity","Floquet ring","bianisotropic sheet","valley lattice","nonreciprocal chain"),
        ("graded ferrite lattice","disorder-robust Chern mesh","phase-programmed flux array","dual-helical cavity","multi-tone Floquet ring","tensor bianisotropic sheet","domain-wall valley lattice","active nonreciprocal chain"),
        ("0.6 deg","0.9 deg","1.2 deg","1.5 deg","1.8 deg","2.1 deg","2.4 deg","2.7 deg"),
        ("0.7 deg","1.0 deg","1.3 deg","1.6 deg","1.9 deg","2.2 deg","2.5 deg","2.8 deg"),83101,881000,
    ),
    _Domain(
        "exciton_interference_camera","exciton interference camera","EI","active layer","fringe visibility",
        ("WSe2 monolayer","MoSe2 bilayer","GaAs well","perovskite sheet","MoS2 heterobilayer","InGaAs well","hBN exciton stack","organic crystal"),
        ("twisted WSe2 monolayer","moire MoSe2 bilayer","coupled GaAs wells","encapsulated perovskite sheet","interlayer MoS2 stack","strained InGaAs well","cavity hBN exciton stack","single-domain organic crystal"),
        ("0.61","0.65","0.69","0.73","0.77","0.81","0.85","0.89"),
        ("0.63","0.67","0.71","0.75","0.79","0.83","0.87","0.91"),83201,882000,
    ),
    _Domain(
        "coherent_electron_pulse_shaper","coherent electron pulse shaper","EP","modulator","temporal spread",
        ("THz cavity","RF streaker","ponderomotive grating","PINEM foil","electrostatic lens","magnetic chicane","optical nearfield","dielectric cavity"),
        ("phase-locked THz cavity","dual-frequency RF streaker","chirped ponderomotive grating","resonant PINEM foil","segmented electrostatic lens","isochronous magnetic chicane","plasmonic optical nearfield","traveling dielectric cavity"),
        ("9 fs","13 fs","17 fs","21 fs","25 fs","29 fs","33 fs","37 fs"),
        ("11 fs","15 fs","19 fs","23 fs","27 fs","31 fs","35 fs","39 fs"),83301,883000,
    ),
    _Domain(
        "nanoscale_thermometry_array","nanoscale thermometry array","NT","thermometer","temperature noise",
        ("Josephson sensor","GeV center","nanowire bridge","Coulomb island","Raman crystal","spin defect","bolometric dot","thermoelectric junction"),
        ("dispersive Josephson sensor","strain-selected GeV center","suspended nanowire bridge","metallic Coulomb island","anti-Stokes Raman crystal","clocked spin defect","graphene bolometric dot","superlattice thermoelectric junction"),
        ("7 uK","11 uK","15 uK","19 uK","23 uK","27 uK","31 uK","35 uK"),
        ("9 uK","13 uK","17 uK","21 uK","25 uK","29 uK","33 uK","37 uK"),83401,884000,
    ),
    _Domain(
        "quantum_acoustic_spectrometer","quantum acoustic spectrometer","QA","resonator","quality factor",
        ("LiNbO3 disk","AlN beam","quartz cavity","diamond phononic cell","GaN ring","SiC membrane","sapphire bar","hBN drum"),
        ("thin-film LiNbO3 disk","epitaxial AlN beam","cryogenic quartz cavity","defect diamond phononic cell","suspended GaN ring","4H-SiC membrane","high-purity sapphire bar","encapsulated hBN drum"),
        ("1.2e5","1.8e5","2.4e5","3.0e5","3.6e5","4.2e5","4.8e5","5.4e5"),
        ("1.5e5","2.1e5","2.7e5","3.3e5","3.9e5","4.5e5","5.1e5","5.7e5"),83501,885000,
    ),
    _Domain(
        "soft_xray_polarization_camera","soft x-ray polarization camera","XP","analyzer","angular error",
        ("multilayer mirror","zoneplate pair","Bragg film","nanograting","crystal wedge","metasurface","channel plate","phase mask"),
        ("chirped multilayer mirror","orthogonal zoneplate pair","strained Bragg film","blazed nanograting","channel-cut crystal wedge","resonant xray metasurface","polarizing channel plate","binary phase mask"),
        ("0.8 mrad","1.2 mrad","1.6 mrad","2.0 mrad","2.4 mrad","2.8 mrad","3.2 mrad","3.6 mrad"),
        ("1.0 mrad","1.4 mrad","1.8 mrad","2.2 mrad","2.6 mrad","3.0 mrad","3.4 mrad","3.8 mrad"),83601,886000,
    ),
    _Domain(
        "molecular_force_interferometer","molecular force interferometer","MF","probe molecule","force floor",
        ("CO rotor","NH3 inversion","H2 dimer","benzene ring","CaF molecule","SrF molecule","Rydberg dimer","molecular ion"),
        ("state-selected CO rotor","trapped NH3 inversion","para-H2 dimer","deuterated benzene ring","laser-cooled CaF","optically pumped SrF","macrodimer Rydberg pair","sympathetic molecular ion"),
        ("2 aN","4 aN","6 aN","8 aN","10 aN","12 aN","14 aN","16 aN"),
        ("3 aN","5 aN","7 aN","9 aN","11 aN","13 aN","15 aN","17 aN"),83701,887000,
    ),
    _Domain(
        "photonic_tensor_router","photonic tensor router","PT","routing fabric","channel crosstalk",
        ("MZI mesh","AWG bank","microring grid","Bragg mesh","Pockels crossbar","MEMS fabric","multimode coupler","topological bus"),
        ("self-configuring MZI mesh","thermally trimmed AWG bank","coupled microring grid","apodized Bragg mesh","thin-film Pockels crossbar","fast MEMS fabric","inverse-designed multimode coupler","valley topological bus"),
        ("-31 dB","-34 dB","-37 dB","-40 dB","-43 dB","-46 dB","-49 dB","-52 dB"),
        ("-32 dB","-35 dB","-38 dB","-41 dB","-44 dB","-47 dB","-50 dB","-53 dB"),83801,888000,
    ),
    _Domain(
        "spin_calorimetry_imager","spin calorimetry imager","SC","absorber","energy resolution",
        ("Au island","Pt strip","graphene flake","NbN bridge","Cu dot","W nanowire","Ti pad","MoRe loop"),
        ("mesoscopic Au island","spin-Hall Pt strip","encapsulated graphene flake","kinetic NbN bridge","Coulomb Cu dot","amorphous W nanowire","suspended Ti pad","proximitized MoRe loop"),
        ("18 ueV","24 ueV","30 ueV","36 ueV","42 ueV","48 ueV","54 ueV","60 ueV"),
        ("21 ueV","27 ueV","33 ueV","39 ueV","45 ueV","51 ueV","57 ueV","63 ueV"),83901,889000,
    ),
    _Domain(
        "ultracold_plasma_mapper","ultracold plasma mapper","UP","atomic species","density precision",
        ("Sr88","Ca40","Yb174","Li6","Na23","K41","Rb85","Cs133"),
        ("Rydberg Sr88","photoionized Ca40","clock-state Yb174","degenerate Li6","magnetized Na23","spin-polarized K41","Rydberg Rb85","fountain Cs133"),
        ("0.7%","1.0%","1.3%","1.6%","1.9%","2.2%","2.5%","2.8%"),
        ("0.8%","1.1%","1.4%","1.7%","2.0%","2.3%","2.6%","2.9%"),84001,890000,
    ),
    _Domain(
        "terahertz_frequency_lattice","terahertz frequency lattice","TF","source","linewidth",
        ("QCL comb","photomixer","Josephson source","difference-frequency chip","plasmonic oscillator","multiplier chain","spintronic emitter","optical rectifier"),
        ("dual-comb QCL","balanced photomixer","phase-locked Josephson source","periodically poled DFG chip","graphene plasmonic oscillator","cryogenic multiplier chain","exchange-biased spintronic emitter","tilted-pulse optical rectifier"),
        ("18 kHz","24 kHz","30 kHz","36 kHz","42 kHz","48 kHz","54 kHz","60 kHz"),
        ("21 kHz","27 kHz","33 kHz","39 kHz","45 kHz","51 kHz","57 kHz","63 kHz"),84101,891000,
    ),
    _Domain(
        "coherent_gamma_timing_tile","coherent gamma timing tile","GT","converter","timing jitter",
        ("LYSO pixel","diamond plate","BaF2 crystal","MCP foil","LGAD layer","Cherenkov bar","SiPM tile","perovskite slab"),
        ("metascintillator LYSO pixel","CVD diamond plate","fast BaF2 crystal","secondary-electron MCP foil","AC-LGAD layer","quartz Cherenkov bar","digital SiPM tile","single-crystal perovskite slab"),
        ("14 ps","20 ps","26 ps","32 ps","38 ps","44 ps","50 ps","56 ps"),
        ("17 ps","23 ps","29 ps","35 ps","41 ps","47 ps","53 ps","59 ps"),84201,892000,
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
        f"{value} is the S42 relational-audit {field} value for this {noun}"
        for _kind, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(spec, *, split, code, first, second):
    if split == "train":
        return (
            f"S42-Rel {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S42-Rel record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S42-Rel {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S42-Rel record {code}?",
            f"For S42-Rel {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S42-Rel record {code}?",
        )
    return (
        f"S42-Rel audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S42-Rel audit {code}: what is {spec.field_a}?",
        f"In S42-Rel audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S42-Rel audit {code}: what is {spec.field_b}?",
        f"In S42-Rel audit {code}, which entry is tagged {spec.field_b}?",
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
    case_id = f"{split}-s42-{spec.name}-{index:03d}"
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
    return S42RelationalGeometryCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s42_cases(split: Split) -> tuple[S42RelationalGeometryCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s42_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S42 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S42 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S42 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S42 language changed")

    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S42 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S42 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S42 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S42 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S42 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S42 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S42 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S42 TRAIN DEV option overlap")


__all__ = [
    "S42RelationalGeometryCase",
    "generate_s42_cases",
    "validate_s42_partitions",
]
