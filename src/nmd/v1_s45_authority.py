from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S45CrossViewCorrectionCase:
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
        "quantum_flux_cartographer","quantum flux cartographer","QF","pickup","flux floor",
        ("Nb loop","Al loop","MoRe loop","NbN loop","granular Al loop","Ta loop","V loop","MgB2 loop"),
        ("gradiometric Nb loop","kinetic-inductance Al loop","nanobridge MoRe loop","disordered NbN loop","high-impedance granular Al loop","epitaxial Ta loop","cryogenic V loop","textured MgB2 loop"),
        ("18 nPhi0","23 nPhi0","28 nPhi0","33 nPhi0","38 nPhi0","43 nPhi0","48 nPhi0","53 nPhi0"),
        ("20 nPhi0","25 nPhi0","30 nPhi0","35 nPhi0","40 nPhi0","45 nPhi0","50 nPhi0","55 nPhi0"),96101,941000,
    ),
    _Domain(
        "exciton_coherence_mapper","exciton coherence mapper","EC","material","coherence width",
        ("WSe2 bilayer","MoSe2 bilayer","GaAs well","InGaAs well","perovskite film","hBN defect","organic cavity","ZnO well"),
        ("twist-tuned WSe2 bilayer","moire MoSe2 bilayer","ultraclean GaAs well","strain-balanced InGaAs well","encapsulated perovskite film","charged hBN defect","strong-coupling organic cavity","polar ZnO well"),
        ("0.42 meV","0.56 meV","0.70 meV","0.84 meV","0.98 meV","1.12 meV","1.26 meV","1.40 meV"),
        ("0.49 meV","0.63 meV","0.77 meV","0.91 meV","1.05 meV","1.19 meV","1.33 meV","1.47 meV"),96201,942000,
    ),
    _Domain(
        "muon_spin_interferometer","muon spin interferometer","MS","target","phase spread",
        ("Cu plate","Ag foil","Si crystal","quartz disk","SrTiO3 slab","FeSe crystal","YBCO film","graphite tile"),
        ("annealed Cu plate","high-purity Ag foil","isotopic Si crystal","low-loss quartz disk","oxygen-tuned SrTiO3 slab","detwinned FeSe crystal","epitaxial YBCO film","mosaic graphite tile"),
        ("0.14 rad","0.19 rad","0.24 rad","0.29 rad","0.34 rad","0.39 rad","0.44 rad","0.49 rad"),
        ("0.16 rad","0.21 rad","0.26 rad","0.31 rad","0.36 rad","0.41 rad","0.46 rad","0.51 rad"),96301,943000,
    ),
    _Domain(
        "phonon_vorticity_analyzer","phonon vorticity analyzer","PV","resonator","circulation noise",
        ("Si ring","AlN ring","LiNbO3 ring","GaN ring","diamond ring","SiC ring","quartz ring","sapphire ring"),
        ("etched Si ring","piezo AlN ring","domain-engineered LiNbO3 ring","suspended GaN ring","color-center diamond ring","4H-SiC ring","cryogenic quartz ring","whispering sapphire ring"),
        ("0.31 krad/s","0.39 krad/s","0.47 krad/s","0.55 krad/s","0.63 krad/s","0.71 krad/s","0.79 krad/s","0.87 krad/s"),
        ("0.35 krad/s","0.43 krad/s","0.51 krad/s","0.59 krad/s","0.67 krad/s","0.75 krad/s","0.83 krad/s","0.91 krad/s"),96401,944000,
    ),
    _Domain(
        "electron_orbital_tomograph","electron orbital tomograph","EO","aperture","angular blur",
        ("graphene mask","SiN mask","Au zone","Pt zone","W needle","LaB6 tip","CNT tip","diamond tip"),
        ("aberration-corrected graphene mask","phase SiN mask","chirped Au zone","binary Pt zone","cold-field W needle","stabilized LaB6 tip","single-wall CNT tip","nitrogen-terminated diamond tip"),
        ("0.18 mrad","0.27 mrad","0.36 mrad","0.45 mrad","0.54 mrad","0.63 mrad","0.72 mrad","0.81 mrad"),
        ("0.22 mrad","0.31 mrad","0.40 mrad","0.49 mrad","0.58 mrad","0.67 mrad","0.76 mrad","0.85 mrad"),96501,945000,
    ),
    _Domain(
        "superconducting_phase_router","superconducting phase router","SP","junction","phase leakage",
        ("Al junction","Nb junction","NbN junction","MoRe junction","Ta junction","V junction","Pb junction","MgB2 junction"),
        ("flux-biased Al junction","ballistic Nb junction","kinetic NbN junction","nanowire MoRe junction","epitaxial Ta junction","oxide-tuned V junction","shielded Pb junction","two-gap MgB2 junction"),
        ("-27 dBc","-31 dBc","-35 dBc","-39 dBc","-43 dBc","-47 dBc","-51 dBc","-55 dBc"),
        ("-29 dBc","-33 dBc","-37 dBc","-41 dBc","-45 dBc","-49 dBc","-53 dBc","-57 dBc"),96601,946000,
    ),
    _Domain(
        "polariton_delay_spectrometer","polariton delay spectrometer","PD","cavity","delay jitter",
        ("GaAs cavity","perovskite cavity","organic cavity","ZnO cavity","WS2 cavity","WSe2 cavity","MoS2 cavity","hBN cavity"),
        ("high-Q GaAs cavity","single-crystal perovskite cavity","vibronic organic cavity","UV ZnO cavity","moire WS2 cavity","strain WSe2 cavity","encapsulated MoS2 cavity","defect hBN cavity"),
        ("11 fs","16 fs","21 fs","26 fs","31 fs","36 fs","41 fs","46 fs"),
        ("13 fs","18 fs","23 fs","28 fs","33 fs","38 fs","43 fs","48 fs"),96701,947000,
    ),
    _Domain(
        "nanoscale_strain_holograph","nanoscale strain holograph","NS","probe","strain floor",
        ("diamond NV","Si vacancy","graphene gauge","MoS2 gauge","GaAs dot","InAs dot","SiC vacancy","hBN emitter"),
        ("vector diamond NV","isotopic Si vacancy","suspended graphene gauge","encapsulated MoS2 gauge","resonant GaAs dot","charged InAs dot","divacancy SiC probe","narrow-line hBN emitter"),
        ("7.2 neps","8.4 neps","9.6 neps","10.8 neps","12.0 neps","13.2 neps","14.4 neps","15.6 neps"),
        ("7.8 neps","9.0 neps","10.2 neps","11.4 neps","12.6 neps","13.8 neps","15.0 neps","16.2 neps"),96801,948000,
    ),
    _Domain(
        "ion_mobility_lattice_clock","ion mobility lattice clock","IM","species","mobility drift",
        ("Ca+ chain","Sr+ chain","Ba+ chain","Yb+ chain","Mg+ chain","Al+ chain","Be+ chain","Lu+ chain"),
        ("sympathetic Ca+ chain","micromotion-nulled Sr+ chain","cryogenic Ba+ chain","clocked Yb+ chain","logic Mg+ chain","quantum-logic Al+ chain","ground-state Be+ chain","hyperfine Lu+ chain"),
        ("1.6e-7","2.1e-7","2.6e-7","3.1e-7","3.6e-7","4.1e-7","4.6e-7","5.1e-7"),
        ("1.8e-7","2.3e-7","2.8e-7","3.3e-7","3.8e-7","4.3e-7","4.8e-7","5.3e-7"),96901,949000,
    ),
    _Domain(
        "skyrmion_drift_imager","skyrmion drift imager","SD","film","drift noise",
        ("PtCo film","WCoFeB film","IrFeCo film","TaCoFeB film","MnSi plate","FeGe plate","CoZnMn film","GdFeCo film"),
        ("DMI PtCo film","sputtered WCoFeB film","multilayer IrFeCo film","annealed TaCoFeB film","B20 MnSi plate","lamellar FeGe plate","chiral CoZnMn film","ferrimagnetic GdFeCo film"),
        ("0.8 um/s","1.1 um/s","1.4 um/s","1.7 um/s","2.0 um/s","2.3 um/s","2.6 um/s","2.9 um/s"),
        ("0.95 um/s","1.25 um/s","1.55 um/s","1.85 um/s","2.15 um/s","2.45 um/s","2.75 um/s","3.05 um/s"),97001,950000,
    ),
    _Domain(
        "vacuum_casimir_gradiometer","vacuum Casimir gradiometer","VC","surface","gradient floor",
        ("Au sphere","Si plate","graphene sheet","diamond plate","sapphire plate","mica sheet","Pt sphere","Al plate"),
        ("annealed Au sphere","hydrogen-terminated Si plate","gated graphene sheet","polished diamond plate","low-loss sapphire plate","cleaved mica sheet","clean Pt sphere","oxide-free Al plate"),
        ("0.42 pN/m","0.57 pN/m","0.72 pN/m","0.87 pN/m","1.02 pN/m","1.17 pN/m","1.32 pN/m","1.47 pN/m"),
        ("0.49 pN/m","0.64 pN/m","0.79 pN/m","0.94 pN/m","1.09 pN/m","1.24 pN/m","1.39 pN/m","1.54 pN/m"),97101,951000,
    ),
    _Domain(
        "coherent_raman_vectormeter","coherent Raman vectormeter","CR","medium","vector noise",
        ("Rb vapor","Cs vapor","Na vapor","K vapor","N2 cell","H2 cell","D2 cell","acetylene cell"),
        ("buffer-gas Rb vapor","spin-exchange Cs vapor","cold Na vapor","coated K vapor","aligned N2 cell","para-H2 cell","cryogenic D2 cell","frequency-locked acetylene cell"),
        ("2.2 nT","2.8 nT","3.4 nT","4.0 nT","4.6 nT","5.2 nT","5.8 nT","6.4 nT"),
        ("2.5 nT","3.1 nT","3.7 nT","4.3 nT","4.9 nT","5.5 nT","6.1 nT","6.7 nT"),97201,952000,
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
        f"{value} is the S45 consistency-audit {field} value for this {noun}"
        for _kind, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(spec, *, split, code, first, second):
    if split == "train":
        return (
            f"S45-Cons {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S45-Cons record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S45-Cons {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S45-Cons record {code}?",
            f"For S45-Cons {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S45-Cons record {code}?",
        )
    return (
        f"S45-Cons audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S45-Cons audit {code}: what is {spec.field_a}?",
        f"In S45-Cons audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S45-Cons audit {code}: what is {spec.field_b}?",
        f"In S45-Cons audit {code}, which entry is tagged {spec.field_b}?",
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
    case_id = f"{split}-s45-{spec.name}-{index:03d}"
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
    return S45CrossViewCorrectionCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s45_cases(split: Split) -> tuple[S45CrossViewCorrectionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s45_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S45 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S45 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S45 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S45 language changed")

    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S45 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S45 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S45 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S45 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S45 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S45 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S45 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S45 TRAIN DEV option overlap")


__all__ = [
    "S45CrossViewCorrectionCase",
    "generate_s45_cases",
    "validate_s45_partitions",
]
