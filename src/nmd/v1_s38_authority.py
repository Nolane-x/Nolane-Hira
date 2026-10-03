from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S38BilinearCase:
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
        "quantum_acoustic_resonator", "quantum acoustic resonator", "QR", "piezoelectric stack", "mode frequency",
        ("AlN", "LiNbO3", "GaN", "ZnO", "PZT", "quartz", "GaAs", "SiC"),
        ("scandium AlN", "thin-film LiNbO3", "epitaxial GaN", "c-axis ZnO", "PMN-PT", "langasite", "AlGaAs", "4H-SiC"),
        ("1 GHz", "2 GHz", "3 GHz", "4 GHz", "5 GHz", "6 GHz", "7 GHz", "8 GHz"),
        ("1.5 GHz", "2.5 GHz", "3.5 GHz", "4.5 GHz", "5.5 GHz", "6.5 GHz", "7.5 GHz", "8.5 GHz"),
        69501, 725000,
    ),
    _Domain(
        "raman_lidar_receiver", "Raman lidar receiver", "RL", "filter train", "range gate",
        ("interference filter", "Fabry-Perot", "etalon", "grating", "dichroic", "atomic filter", "VIPA", "AWG"),
        ("dual-cavity filter", "tandem FP", "temperature-tuned etalon", "volume grating", "multi-edge dichroic", "iodine filter", "dual-VIPA", "photonic AWG"),
        ("50 m", "100 m", "150 m", "200 m", "250 m", "300 m", "350 m", "400 m"),
        ("75 m", "125 m", "175 m", "225 m", "275 m", "325 m", "375 m", "425 m"),
        69601, 726000,
    ),
    _Domain(
        "nano_squid_microscope", "nano-SQUID microscope", "NS", "pickup loop", "flux noise",
        ("Nb loop", "Al loop", "Pb loop", "YBCO loop", "MoRe loop", "NbN loop", "graphene junction", "Dayem bridge"),
        ("focused-ion Nb", "shadow Al", "epitaxial Pb", "grain-boundary YBCO", "sputtered MoRe", "nanowire NbN", "encapsulated graphene", "constriction bridge"),
        ("10 nPhi", "20 nPhi", "30 nPhi", "40 nPhi", "50 nPhi", "60 nPhi", "70 nPhi", "80 nPhi"),
        ("15 nPhi", "25 nPhi", "35 nPhi", "45 nPhi", "55 nPhi", "65 nPhi", "75 nPhi", "85 nPhi"),
        69701, 727000,
    ),
    _Domain(
        "xray_nanoprobe", "x-ray nanoprobe", "XN", "focusing optic", "working distance",
        ("zone plate", "KB mirror", "MLL", "CRL", "capillary", "waveguide", "Fresnel lens", "multilayer mirror"),
        ("stacked zone plate", "adaptive KB", "wedged MLL", "kinoform CRL", "tapered capillary", "nanowire guide", "achromatic Fresnel", "graded multilayer"),
        ("1 mm", "2 mm", "3 mm", "4 mm", "5 mm", "6 mm", "7 mm", "8 mm"),
        ("1.5 mm", "2.5 mm", "3.5 mm", "4.5 mm", "5.5 mm", "6.5 mm", "7.5 mm", "8.5 mm"),
        69801, 728000,
    ),
    _Domain(
        "single_photon_spectrometer", "single-photon spectrometer", "SP", "disperser", "timing jitter",
        ("grating", "prism", "VIPA", "AWG", "echelle", "etalon", "ring filter", "Bragg filter"),
        ("holographic grating", "compound prism", "dual-VIPA", "silicon AWG", "cross-dispersed echelle", "tandem etalon", "SiN ring bank", "chirped Bragg"),
        ("10 ps", "20 ps", "30 ps", "40 ps", "50 ps", "60 ps", "70 ps", "80 ps"),
        ("15 ps", "25 ps", "35 ps", "45 ps", "55 ps", "65 ps", "75 ps", "85 ps"),
        69901, 729000,
    ),
    _Domain(
        "optical_tweezer_array", "optical tweezer array", "OT", "beam shaper", "trap spacing",
        ("SLM", "AOD", "DMD", "DOE", "microlens", "metasurface", "phase plate", "fiber bundle"),
        ("ferroelectric SLM", "dual-axis AOD", "binary DMD", "multiorder DOE", "aspheric microlens", "dielectric metasurface", "multi-level phase plate", "coherent fiber bundle"),
        ("2 um", "4 um", "6 um", "8 um", "10 um", "12 um", "14 um", "16 um"),
        ("3 um", "5 um", "7 um", "9 um", "11 um", "13 um", "15 um", "17 um"),
        70001, 730000,
    ),
    _Domain(
        "spin_wave_spectrometer", "spin-wave spectrometer", "SW", "waveguide", "bias field",
        ("CPW", "microstrip", "stripline", "loop", "slotline", "GSG probe", "cavity", "meander"),
        ("tapered CPW", "slow-wave microstrip", "shielded stripline", "nano-loop", "coplanar slot", "air-bridge GSG", "reentrant cavity", "chirped meander"),
        ("5 mT", "10 mT", "15 mT", "20 mT", "25 mT", "30 mT", "35 mT", "40 mT"),
        ("7 mT", "12 mT", "17 mT", "22 mT", "27 mT", "32 mT", "37 mT", "42 mT"),
        70101, 731000,
    ),
    _Domain(
        "cryo_photonic_link", "cryogenic photonic link", "CP", "modulator", "link bandwidth",
        ("MZM", "ring", "EAM", "Pockels", "MEMS", "graphene", "plasmonic", "acousto-optic"),
        ("thin-film MZM", "athermal ring", "quantum-well EAM", "LiNbO3 Pockels", "nano-MEMS", "hBN graphene", "hybrid plasmonic", "SAW acousto-optic"),
        ("2 GHz", "4 GHz", "6 GHz", "8 GHz", "10 GHz", "12 GHz", "14 GHz", "16 GHz"),
        ("3 GHz", "5 GHz", "7 GHz", "9 GHz", "11 GHz", "13 GHz", "15 GHz", "17 GHz"),
        70201, 732000,
    ),
    _Domain(
        "electron_tomography_stage", "electron tomography stage", "ET", "holder", "tilt span",
        ("single-tilt", "dual-axis", "on-axis", "cryo", "MEMS", "piezo", "gas-cell", "liquid-cell"),
        ("low-background single", "orthogonal dual", "eucentric on-axis", "cryo-transfer", "MEMS goniometer", "closed-loop piezo", "sealed gas", "graphene liquid"),
        ("60 deg", "70 deg", "80 deg", "90 deg", "100 deg", "110 deg", "120 deg", "130 deg"),
        ("65 deg", "75 deg", "85 deg", "95 deg", "105 deg", "115 deg", "125 deg", "135 deg"),
        70301, 733000,
    ),
    _Domain(
        "microcomb_lidar", "microcomb lidar", "ML", "comb platform", "update rate",
        ("SiN", "silica", "LiNbO3", "AlN", "SiC", "diamond", "Ta2O5", "GaP"),
        ("dispersion-engineered SiN", "wedge silica", "thin-film LiNbO3", "Pockels AlN", "4H-SiC", "single-crystal diamond", "low-loss Ta2O5", "orientation-patterned GaP"),
        ("10 kHz", "20 kHz", "30 kHz", "40 kHz", "50 kHz", "60 kHz", "70 kHz", "80 kHz"),
        ("15 kHz", "25 kHz", "35 kHz", "45 kHz", "55 kHz", "65 kHz", "75 kHz", "85 kHz"),
        70401, 734000,
    ),
    _Domain(
        "quantum_microwave_transducer", "quantum microwave transducer", "QM", "coupler", "conversion bandwidth",
        ("electro-optic", "piezo-optic", "magnon", "mechanical", "Josephson", "Rydberg", "spin ensemble", "membrane"),
        ("triply-resonant EO", "phononic piezo-optic", "YIG magnon", "nanobeam mechanical", "Josephson mixer", "Rydberg EIT", "rare-earth spin", "SiN membrane"),
        ("1 MHz", "2 MHz", "3 MHz", "4 MHz", "5 MHz", "6 MHz", "7 MHz", "8 MHz"),
        ("1.5 MHz", "2.5 MHz", "3.5 MHz", "4.5 MHz", "5.5 MHz", "6.5 MHz", "7.5 MHz", "8.5 MHz"),
        70501, 735000,
    ),
    _Domain(
        "nanocalorimeter", "nanocalorimeter", "NC", "thermometer", "energy resolution",
        ("TES", "NTD", "SNS", "KID", "Johnson", "SQUID", "graphene", "semiconductor"),
        ("bilayer TES", "implanted NTD", "proximity SNS", "lumped KID", "noise thermometer", "dispersive SQUID", "hBN graphene", "strained semiconductor"),
        ("1 eV", "2 eV", "3 eV", "4 eV", "5 eV", "6 eV", "7 eV", "8 eV"),
        ("1.5 eV", "2.5 eV", "3.5 eV", "4.5 eV", "5.5 eV", "6.5 eV", "7.5 eV", "8.5 eV"),
        70601, 736000,
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
            f"S38-Bilinear {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S38-Bilinear record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S38-Bilinear {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S38-Bilinear record {code}?",
            f"For S38-Bilinear {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S38-Bilinear record {code}?",
        )
    return (
        f"S38-Bilinear audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S38-Bilinear audit {code}: what is {spec.field_a}?",
        f"In S38-Bilinear audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S38-Bilinear audit {code}: what is {spec.field_b}?",
        f"In S38-Bilinear audit {code}, which entry is tagged {spec.field_b}?",
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
    case_id = f"{split}-s38-{spec.name}-{index:03d}"
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
    return S38BilinearCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s38_cases(split: Split) -> tuple[S38BilinearCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s38_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S38 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S38 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S38 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S38 language changed")
    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S38 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S38 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S38 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S38 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S38 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S38 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S38 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S38 TRAIN DEV option overlap")


__all__ = ["S38BilinearCase", "generate_s38_cases", "validate_s38_partitions"]
