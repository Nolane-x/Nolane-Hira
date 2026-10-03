from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S39IsolatedCase:
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
        "cavity_optomechanical_sensor", "cavity optomechanical sensor", "CO", "resonator material", "cooperativity",
        ("SiN", "silica", "diamond", "GaAs", "AlN", "LiNbO3", "SiC", "silicon"),
        ("strained SiN", "wedge silica", "single-crystal diamond", "AlGaAs", "ScAlN", "thin-film LiNbO3", "4H-SiC", "SOI"),
        ("10", "20", "30", "40", "50", "60", "70", "80"),
        ("15", "25", "35", "45", "55", "65", "75", "85"),
        70701, 737000,
    ),
    _Domain(
        "quantum_gas_microscope", "quantum gas microscope", "QG", "objective", "lattice spacing",
        ("NA0.55", "NA0.60", "NA0.65", "NA0.70", "NA0.75", "NA0.80", "NA0.85", "NA0.90"),
        ("compound NA0.57", "aspheric NA0.62", "relay NA0.67", "vacuum NA0.72", "corrected NA0.77", "custom NA0.82", "multi-element NA0.87", "immersion NA0.92"),
        ("400 nm", "450 nm", "500 nm", "550 nm", "600 nm", "650 nm", "700 nm", "750 nm"),
        ("425 nm", "475 nm", "525 nm", "575 nm", "625 nm", "675 nm", "725 nm", "775 nm"),
        70801, 738000,
    ),
    _Domain(
        "coherent_diffraction_imager", "coherent diffraction imager", "CD", "detector", "oversampling",
        ("hybrid pixel", "CCD", "CMOS", "DEPFET", "MCP", "pnCCD", "Medipix", "Timepix"),
        ("charge-integrating pixel", "back-illuminated CCD", "global-shutter CMOS", "fast DEPFET", "chevron MCP", "split-frame pnCCD", "Medipix4", "Timepix4"),
        ("2x", "3x", "4x", "5x", "6x", "7x", "8x", "9x"),
        ("2.5x", "3.5x", "4.5x", "5.5x", "6.5x", "7.5x", "8.5x", "9.5x"),
        70901, 739000,
    ),
    _Domain(
        "kinetic_inductance_array", "kinetic inductance detector array", "KI", "film", "readout bandwidth",
        ("TiN", "Al", "NbTiN", "PtSi", "WSi", "MoSi", "Nb", "Ta"),
        ("substoich TiN", "epitaxial Al", "multilayer NbTiN", "crystalline PtSi", "amorphous WSi", "amorphous MoSi", "sputtered Nb", "alpha-Ta"),
        ("50 MHz", "100 MHz", "150 MHz", "200 MHz", "250 MHz", "300 MHz", "350 MHz", "400 MHz"),
        ("75 MHz", "125 MHz", "175 MHz", "225 MHz", "275 MHz", "325 MHz", "375 MHz", "425 MHz"),
        71001, 740000,
    ),
    _Domain(
        "atom_chip_interferometer", "atom-chip interferometer", "AC", "guide geometry", "interrogation time",
        ("Z-wire", "U-wire", "ring", "spiral", "coplanar", "meander", "quadrupole", "Ioffe"),
        ("segmented Z-wire", "dual U-wire", "racetrack ring", "adiabatic spiral", "shielded coplanar", "chirped meander", "micro-quadrupole", "mini-Ioffe"),
        ("10 ms", "20 ms", "30 ms", "40 ms", "50 ms", "60 ms", "70 ms", "80 ms"),
        ("15 ms", "25 ms", "35 ms", "45 ms", "55 ms", "65 ms", "75 ms", "85 ms"),
        71101, 741000,
    ),
    _Domain(
        "nanoscale_esr_mapper", "nanoscale ESR mapper", "NE", "resonator", "spatial resolution",
        ("loop-gap", "CPW", "dielectric", "split-ring", "microstrip", "cavity", "slotline", "stripline"),
        ("nano loop-gap", "tapered CPW", "high-k dielectric", "asymmetric split-ring", "slow-wave microstrip", "reentrant cavity", "coplanar slotline", "shielded stripline"),
        ("10 nm", "20 nm", "30 nm", "40 nm", "50 nm", "60 nm", "70 nm", "80 nm"),
        ("15 nm", "25 nm", "35 nm", "45 nm", "55 nm", "65 nm", "75 nm", "85 nm"),
        71201, 742000,
    ),
    _Domain(
        "josephson_frequency_converter", "Josephson frequency converter", "JF", "junction stack", "conversion bandwidth",
        ("AlOx", "NbOx", "SNS", "SIS", "graphene", "InAs", "nanobridge", "Dayem"),
        ("double-angle AlOx", "trilayer NbOx", "short SNS", "high-Jc SIS", "encapsulated graphene", "epitaxial InAs", "constriction nanobridge", "thin-film Dayem"),
        ("5 MHz", "10 MHz", "15 MHz", "20 MHz", "25 MHz", "30 MHz", "35 MHz", "40 MHz"),
        ("7 MHz", "12 MHz", "17 MHz", "22 MHz", "27 MHz", "32 MHz", "37 MHz", "42 MHz"),
        71301, 743000,
    ),
    _Domain(
        "integrated_comb_analyzer", "integrated frequency-comb analyzer", "IC", "platform", "line spacing",
        ("SiN", "silica", "AlN", "LiNbO3", "SiC", "Ta2O5", "GaP", "diamond"),
        ("Damascene SiN", "wedge silica", "ScAlN", "TFLN", "4H-SiC", "low-loss Ta2O5", "OP-GaP", "single-crystal diamond"),
        ("10 GHz", "20 GHz", "30 GHz", "40 GHz", "50 GHz", "60 GHz", "70 GHz", "80 GHz"),
        ("15 GHz", "25 GHz", "35 GHz", "45 GHz", "55 GHz", "65 GHz", "75 GHz", "85 GHz"),
        71401, 744000,
    ),
    _Domain(
        "eels_mapper", "electron energy-loss mapper", "EM", "monochromator", "energy resolution",
        ("Wien", "Omega", "electrostatic", "magnetic", "prism", "sector", "RF", "hybrid"),
        ("double-Wien", "corrected Omega", "multipole electrostatic", "aberration-corrected magnetic", "compound prism", "double-sector", "phase-locked RF", "electrostatic-magnetic hybrid"),
        ("10 meV", "20 meV", "30 meV", "40 meV", "50 meV", "60 meV", "70 meV", "80 meV"),
        ("15 meV", "25 meV", "35 meV", "45 meV", "55 meV", "65 meV", "75 meV", "85 meV"),
        71501, 745000,
    ),
    _Domain(
        "attosecond_streak_camera", "attosecond streak camera", "AS", "streak medium", "time resolution",
        ("neon", "argon", "krypton", "xenon", "helium", "nitrogen", "hydrogen", "solid foil"),
        ("Ne jet", "Ar jet", "Kr jet", "Xe jet", "He nanodroplet", "N2 cell", "H2 cell", "nanomembrane foil"),
        ("20 as", "40 as", "60 as", "80 as", "100 as", "120 as", "140 as", "160 as"),
        ("30 as", "50 as", "70 as", "90 as", "110 as", "130 as", "150 as", "170 as"),
        71601, 746000,
    ),
    _Domain(
        "magnetooptic_current_sensor", "magneto-optic current sensor", "MC", "crystal", "current range",
        ("TGG", "YIG", "Bi:YIG", "Ce:YIG", "BGO", "quartz", "GaAs", "LiNbO3"),
        ("ceramic TGG", "thin-film YIG", "Bi-substituted YIG", "Ce-substituted YIG", "BGO fiber", "spun quartz", "semi-insulating GaAs", "poled LiNbO3"),
        ("10 A", "20 A", "30 A", "40 A", "50 A", "60 A", "70 A", "80 A"),
        ("15 A", "25 A", "35 A", "45 A", "55 A", "65 A", "75 A", "85 A"),
        71701, 747000,
    ),
    _Domain(
        "quantum_thermoelectric_probe", "quantum thermoelectric probe", "QT", "absorber", "noise equivalent power",
        ("graphene", "AuPd", "Ti", "Al", "Cu", "MoRe", "NbN", "Pt"),
        ("hBN graphene", "granular AuPd", "alpha-Ti", "epitaxial Al", "nanowire Cu", "amorphous MoRe", "ultrathin NbN", "Pt nanomesh"),
        ("1 aW", "2 aW", "3 aW", "4 aW", "5 aW", "6 aW", "7 aW", "8 aW"),
        ("1.5 aW", "2.5 aW", "3.5 aW", "4.5 aW", "5.5 aW", "6.5 aW", "7.5 aW", "8.5 aW"),
        71801, 748000,
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
    aliases = tuple(f"{value} is the audited {field} value for this {noun}" for _kind, field, value in rows)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(spec, *, split, code, first, second):
    if split == "train":
        return (
            f"S39-Isolated {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S39-Isolated record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S39-Isolated {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S39-Isolated record {code}?",
            f"For S39-Isolated {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S39-Isolated record {code}?",
        )
    return (
        f"S39-Isolated audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S39-Isolated audit {code}: what is {spec.field_a}?",
        f"In S39-Isolated audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S39-Isolated audit {code}: what is {spec.field_b}?",
        f"In S39-Isolated audit {code}, which entry is tagged {spec.field_b}?",
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
    case_id = f"{split}-s39-{spec.name}-{index:03d}"
    sa, sb, qa1, qa2, qb1, qb2 = _views(spec, split=split, code=code, first=first, second=second)
    texts, aliases, ids, ga, gb = _shuffle(
        case_id=case_id,
        noun=spec.noun,
        field_a=spec.field_a,
        field_b=spec.field_b,
        first=first,
        second=second,
        distractor_first=df,
        distractor_second=ds,
        seed=spec.seed + offset,
    )
    return S39IsolatedCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s39_cases(split: Split) -> tuple[S39IsolatedCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s39_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S39 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S39 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S39 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S39 language changed")
    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S39 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S39 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S39 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S39 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S39 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S39 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S39 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S39 TRAIN DEV option overlap")


__all__ = ["S39IsolatedCase", "generate_s39_cases", "validate_s39_partitions"]
