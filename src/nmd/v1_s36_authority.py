from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S36ReadoutCase:
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
        "coherent_lidar", "coherent lidar", "CL", "laser source", "chirp span",
        ("DFB", "ECDL", "fiber laser", "VCSEL", "DBR", "ring laser", "seeded diode", "microchip laser"),
        ("sampled-grating DBR", "external-cavity diode", "PM fiber laser", "MEMS VCSEL", "tunable DBR", "monolithic ring", "injection-locked diode", "Nd:YVO4 microchip"),
        ("2 GHz", "4 GHz", "6 GHz", "8 GHz", "10 GHz", "12 GHz", "14 GHz", "16 GHz"),
        ("3 GHz", "5 GHz", "7 GHz", "9 GHz", "11 GHz", "13 GHz", "15 GHz", "17 GHz"),
        67101, 701000,
    ),
    _Domain(
        "xray_ptychography", "x-ray ptychography stage", "XP", "probe optic", "scan step",
        ("zone plate", "KB mirror", "multilayer Laue lens", "capillary", "CRL", "Fresnel lens", "waveguide", "aperture"),
        ("stacked zone plate", "adaptive KB", "wedged MLL", "tapered capillary", "kinoform CRL", "achromatic Fresnel", "nanowire guide", "coded aperture"),
        ("10 nm", "20 nm", "30 nm", "40 nm", "50 nm", "60 nm", "70 nm", "80 nm"),
        ("15 nm", "25 nm", "35 nm", "45 nm", "55 nm", "65 nm", "75 nm", "85 nm"),
        67201, 702000,
    ),
    _Domain(
        "diamond_magnetometer", "diamond magnetometer", "DM", "defect ensemble", "bias field",
        ("NV-", "NV0", "SiV", "GeV", "SnV", "VSi", "divacancy", "P1"),
        ("near-surface NV", "delta-doped NV", "SiV-", "GeV-", "SnV-", "isotopic VSi", "neutral divacancy", "engineered P1"),
        ("2 mT", "4 mT", "6 mT", "8 mT", "10 mT", "12 mT", "14 mT", "16 mT"),
        ("3 mT", "5 mT", "7 mT", "9 mT", "11 mT", "13 mT", "15 mT", "17 mT"),
        67301, 703000,
    ),
    _Domain(
        "plasmonic_biosensor", "plasmonic biosensor", "PB", "resonator", "spectral shift",
        ("nanorod", "nanohole", "bowtie", "nanodisk", "grating", "nanostar", "split ring", "dimer"),
        ("core-shell rod", "asymmetric hole", "gap bowtie", "Fano disk", "chirped grating", "hollow nanostar", "complementary ring", "heterodimer"),
        ("4 nm", "8 nm", "12 nm", "16 nm", "20 nm", "24 nm", "28 nm", "32 nm"),
        ("6 nm", "10 nm", "14 nm", "18 nm", "22 nm", "26 nm", "30 nm", "34 nm"),
        67401, 704000,
    ),
    _Domain(
        "cryogenic_amplifier", "cryogenic microwave amplifier", "CA", "gain element", "noise temperature",
        ("HEMT", "JPA", "JTWPA", "KIT", "SQUID array", "parametric cavity", "SIS", "graphene FET"),
        ("InP HEMT", "flux-pumped JPA", "reversed-Kerr JTWPA", "NbTiN KIT", "rf-SQUID chain", "3-wave cavity", "Nb SIS", "hBN graphene FET"),
        ("1 K", "2 K", "3 K", "4 K", "5 K", "6 K", "7 K", "8 K"),
        ("1.5 K", "2.5 K", "3.5 K", "4.5 K", "5.5 K", "6.5 K", "7.5 K", "8.5 K"),
        67501, 705000,
    ),
    _Domain(
        "photoacoustic_cell", "photoacoustic spectroscopy cell", "PA", "window", "modulation rate",
        ("CaF2", "ZnSe", "quartz", "sapphire", "BaF2", "diamond", "MgF2", "silicon"),
        ("wedged CaF2", "AR ZnSe", "fused quartz", "c-plane sapphire", "polycrystalline BaF2", "CVD diamond", "UV MgF2", "float-zone silicon"),
        ("200 Hz", "400 Hz", "600 Hz", "800 Hz", "1 kHz", "1.2 kHz", "1.4 kHz", "1.6 kHz"),
        ("300 Hz", "500 Hz", "700 Hz", "900 Hz", "1.1 kHz", "1.3 kHz", "1.5 kHz", "1.7 kHz"),
        67601, 706000,
    ),
    _Domain(
        "microcomb_synthesizer", "microcomb synthesizer", "MC", "resonator", "repetition rate",
        ("SiN ring", "silica wedge", "AlN ring", "LiNbO3 ring", "MgF2 disk", "diamond ring", "SiC ring", "Ta2O5 ring"),
        ("dispersion-engineered SiN", "etched silica wedge", "Pockels AlN", "thin-film LiNbO3", "crystalline MgF2", "single-crystal diamond", "4H-SiC", "low-loss Ta2O5"),
        ("20 GHz", "40 GHz", "60 GHz", "80 GHz", "100 GHz", "120 GHz", "140 GHz", "160 GHz"),
        ("30 GHz", "50 GHz", "70 GHz", "90 GHz", "110 GHz", "130 GHz", "150 GHz", "170 GHz"),
        67701, 707000,
    ),
    _Domain(
        "eels_spectrometer", "electron energy-loss spectrometer", "ES", "monochromator", "energy spread",
        ("Wien filter", "alpha filter", "omega filter", "electrostatic prism", "magnetic prism", "double Wien", "RF cavity", "retarding lens"),
        ("dual-Wien", "aberration-corrected alpha", "double-omega", "toroidal prism", "sector magnet", "crossed-field pair", "phase-locked RF cavity", "multistage retarder"),
        ("20 meV", "30 meV", "40 meV", "50 meV", "60 meV", "70 meV", "80 meV", "90 meV"),
        ("25 meV", "35 meV", "45 meV", "55 meV", "65 meV", "75 meV", "85 meV", "95 meV"),
        67801, 708000,
    ),
    _Domain(
        "superconducting_resonator", "superconducting resonator", "SR", "film", "resonance",
        ("Nb", "NbN", "NbTiN", "Al", "TiN", "MoRe", "Ta", "granular Al"),
        ("epitaxial Nb", "ALD NbN", "reactive NbTiN", "MBE Al", "stoichiometric TiN", "sputtered MoRe", "alpha-Ta", "high-kinetic granular Al"),
        ("2 GHz", "3 GHz", "4 GHz", "5 GHz", "6 GHz", "7 GHz", "8 GHz", "9 GHz"),
        ("2.5 GHz", "3.5 GHz", "4.5 GHz", "5.5 GHz", "6.5 GHz", "7.5 GHz", "8.5 GHz", "9.5 GHz"),
        67901, 709000,
    ),
    _Domain(
        "optofluidic_cytometer", "optofluidic cytometer", "OC", "focusing method", "flow rate",
        ("sheath", "acoustic", "inertial", "dielectrophoretic", "hydrodynamic", "magnetic", "viscoelastic", "optical"),
        ("3D sheath", "standing-wave acoustic", "Dean inertial", "iDEP", "pinched-flow", "ferrofluidic", "elasto-inertial", "dual-beam optical"),
        ("10 uL/min", "20 uL/min", "30 uL/min", "40 uL/min", "50 uL/min", "60 uL/min", "70 uL/min", "80 uL/min"),
        ("15 uL/min", "25 uL/min", "35 uL/min", "45 uL/min", "55 uL/min", "65 uL/min", "75 uL/min", "85 uL/min"),
        68001, 710000,
    ),
    _Domain(
        "spin_torque_oscillator", "spin-torque oscillator", "ST", "free layer", "frequency",
        ("CoFeB", "NiFe", "CoNi", "Heusler", "FeGa", "CoFe", "CoPt", "CoGd"),
        ("dual-MgO CoFeB", "ultrathin NiFe", "PMA CoNi", "Co2MnSi Heusler", "strained FeGa", "B-doped CoFe", "L10 CoPt", "compensated CoGd"),
        ("4 GHz", "6 GHz", "8 GHz", "10 GHz", "12 GHz", "14 GHz", "16 GHz", "18 GHz"),
        ("5 GHz", "7 GHz", "9 GHz", "11 GHz", "13 GHz", "15 GHz", "17 GHz", "19 GHz"),
        68101, 711000,
    ),
    _Domain(
        "electron_holography", "electron holography setup", "EH", "biprism", "fringe spacing",
        ("W wire", "Pt wire", "Au wire", "CNT", "graphene edge", "SiN edge", "electrostatic blade", "nanofabricated fork"),
        ("quartz-coated W", "FIB Pt", "etched Au", "MWCNT", "suspended graphene", "metallized SiN", "MEMS blade", "split-fork electrode"),
        ("0.5 nm", "1 nm", "1.5 nm", "2 nm", "2.5 nm", "3 nm", "3.5 nm", "4 nm"),
        ("0.75 nm", "1.25 nm", "1.75 nm", "2.25 nm", "2.75 nm", "3.25 nm", "3.75 nm", "4.25 nm"),
        68201, 712000,
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
            f"S36-Readout {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S36-Readout record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S36-Readout {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S36-Readout record {code}?",
            f"For S36-Readout {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S36-Readout record {code}?",
        )
    return (
        f"S36-Readout audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S36-Readout audit {code}: what is {spec.field_a}?",
        f"In S36-Readout audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S36-Readout audit {code}: what is {spec.field_b}?",
        f"In S36-Readout audit {code}, which entry is tagged {spec.field_b}?",
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
    case_id = f"{split}-s36-{spec.name}-{index:03d}"
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
    return S36ReadoutCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s36_cases(split: Split) -> tuple[S36ReadoutCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s36_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S36 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S36 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S36 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S36 language changed")
    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S36 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S36 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S36 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S36 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S36 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S36 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S36 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S36 TRAIN DEV option overlap")


__all__ = ["S36ReadoutCase", "generate_s36_cases", "validate_s36_partitions"]
