from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S40AnchoredCase:
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
        "photon_number_resolver", "photon-number resolver", "PN", "absorber stack", "count ceiling",
        ("W TES", "MoAu TES", "Ti TES", "Nb TES", "SNSPD tree", "KID bank", "graphene island", "nanobolometer"),
        ("multilayer W TES", "proximity MoAu TES", "suspended Ti TES", "Nb micro-TES", "cascaded SNSPD tree", "frequency-mux KID bank", "hBN graphene island", "membrane nanobolometer"),
        ("4 photons", "8 photons", "12 photons", "16 photons", "20 photons", "24 photons", "28 photons", "32 photons"),
        ("6 photons", "10 photons", "14 photons", "18 photons", "22 photons", "26 photons", "30 photons", "34 photons"),
        71901, 749000,
    ),
    _Domain(
        "cryo_electrooptic_modulator", "cryogenic electro-optic modulator", "CE", "electrode", "Vpi",
        ("CPW", "GSG", "microstrip", "slotline", "traveling-wave", "lumped", "ring electrode", "mesh electrode"),
        ("slow-wave CPW", "air-bridge GSG", "shielded microstrip", "coplanar slotline", "velocity-matched traveling-wave", "inductive lumped", "segmented ring electrode", "transparent mesh electrode"),
        ("1 V", "2 V", "3 V", "4 V", "5 V", "6 V", "7 V", "8 V"),
        ("1.5 V", "2.5 V", "3.5 V", "4.5 V", "5.5 V", "6.5 V", "7.5 V", "8.5 V"),
        72001, 750000,
    ),
    _Domain(
        "terahertz_nearfield_microscope", "terahertz near-field microscope", "TN", "probe", "spatial resolution",
        ("metal tip", "AFM tip", "antenna tip", "graphene tip", "bowtie", "slot probe", "nanowire", "aperture"),
        ("plasmonic metal tip", "tuning-fork AFM tip", "resonant antenna tip", "gated graphene tip", "nanogap bowtie", "coaxial slot probe", "InAs nanowire", "subwavelength aperture"),
        ("20 nm", "40 nm", "60 nm", "80 nm", "100 nm", "120 nm", "140 nm", "160 nm"),
        ("30 nm", "50 nm", "70 nm", "90 nm", "110 nm", "130 nm", "150 nm", "170 nm"),
        72101, 751000,
    ),
    _Domain(
        "ion_trap_imager", "trapped-ion fluorescence imager", "II", "collection optic", "readout time",
        ("NA0.4", "NA0.5", "NA0.6", "NA0.7", "NA0.8", "parabolic mirror", "fiber cavity", "solid immersion lens"),
        ("corrected NA0.42", "vacuum NA0.52", "aspheric NA0.62", "compound NA0.72", "custom NA0.82", "micro-parabolic mirror", "integrated fiber cavity", "microfabricated SIL"),
        ("10 us", "20 us", "30 us", "40 us", "50 us", "60 us", "70 us", "80 us"),
        ("15 us", "25 us", "35 us", "45 us", "55 us", "65 us", "75 us", "85 us"),
        72201, 752000,
    ),
    _Domain(
        "nanomechanical_mass_sensor", "nanomechanical mass sensor", "NM", "resonator", "mass resolution",
        ("SiN beam", "graphene drum", "CNT", "diamond beam", "SiC beam", "silicon disk", "MoS2 drum", "quartz beam"),
        ("strained SiN beam", "hBN graphene drum", "suspended CNT", "single-crystal diamond beam", "4H-SiC beam", "phononic silicon disk", "encapsulated MoS2 drum", "high-Q quartz beam"),
        ("1 zg", "2 zg", "3 zg", "4 zg", "5 zg", "6 zg", "7 zg", "8 zg"),
        ("1.5 zg", "2.5 zg", "3.5 zg", "4.5 zg", "5.5 zg", "6.5 zg", "7.5 zg", "8.5 zg"),
        72301, 753000,
    ),
    _Domain(
        "spin_photon_interface", "spin-photon interface", "SI", "emitter", "cooperativity",
        ("NV", "SiV", "SnV", "Er", "Yb", "quantum dot", "rare-earth ion", "defect spin"),
        ("nanocavity NV", "strain-tuned SiV", "SnV center", "Er:YSO", "Yb:YVO", "InAs quantum dot", "Eu rare-earth ion", "SiC defect spin"),
        ("2", "4", "6", "8", "10", "12", "14", "16"),
        ("3", "5", "7", "9", "11", "13", "15", "17"),
        72401, 754000,
    ),
    _Domain(
        "superconducting_parametric_amplifier", "superconducting parametric amplifier", "SA", "nonlinear element", "instantaneous bandwidth",
        ("JPA", "JPC", "TWPA", "SNAIL", "SQUID", "Josephson array", "kinetic inductance", "rf-SQUID"),
        ("flux-pumped JPA", "three-wave JPC", "dispersion-engineered TWPA", "SNAIL array", "asymmetric SQUID", "long Josephson array", "NbTiN kinetic inductance", "rf-SQUID chain"),
        ("10 MHz", "20 MHz", "30 MHz", "40 MHz", "50 MHz", "60 MHz", "70 MHz", "80 MHz"),
        ("15 MHz", "25 MHz", "35 MHz", "45 MHz", "55 MHz", "65 MHz", "75 MHz", "85 MHz"),
        72501, 755000,
    ),
    _Domain(
        "metasurface_spectrometer", "metasurface spectrometer", "MS", "meta-atom", "spectral channels",
        ("nanofin", "nanopillar", "split-ring", "disk", "slot", "cross", "Huygens pair", "grating cell"),
        ("TiO2 nanofin", "Si nanopillar", "asymmetric split-ring", "GaN disk", "inverse slot", "cross resonator", "dual Huygens pair", "chirped grating cell"),
        ("32", "64", "96", "128", "160", "192", "224", "256"),
        ("48", "80", "112", "144", "176", "208", "240", "272"),
        72601, 756000,
    ),
    _Domain(
        "ultrafast_diffraction_detector", "ultrafast diffraction detector", "UD", "sensor", "frame rate",
        ("CMOS", "hybrid pixel", "MCP", "pnCCD", "DEPFET", "Timepix", "Medipix", "streak sensor"),
        ("burst CMOS", "charge-integrating hybrid pixel", "gated MCP", "split-frame pnCCD", "fast DEPFET", "Timepix4", "Medipix4", "compressed streak sensor"),
        ("1 kHz", "2 kHz", "3 kHz", "4 kHz", "5 kHz", "6 kHz", "7 kHz", "8 kHz"),
        ("1.5 kHz", "2.5 kHz", "3.5 kHz", "4.5 kHz", "5.5 kHz", "6.5 kHz", "7.5 kHz", "8.5 kHz"),
        72701, 757000,
    ),
    _Domain(
        "magnetic_force_nanoscope", "magnetic-force nanoscope", "MF", "tip coating", "field sensitivity",
        ("Co", "FePt", "Ni", "CoCr", "SmCo", "FeCo", "Heusler", "ferrite"),
        ("low-moment Co", "L10 FePt", "ultrathin Ni", "granular CoCr", "SmCo5", "FeCoB", "half-metal Heusler", "nanocrystal ferrite"),
        ("1 nT", "2 nT", "3 nT", "4 nT", "5 nT", "6 nT", "7 nT", "8 nT"),
        ("1.5 nT", "2.5 nT", "3.5 nT", "4.5 nT", "5.5 nT", "6.5 nT", "7.5 nT", "8.5 nT"),
        72801, 758000,
    ),
    _Domain(
        "photonic_quantum_memory", "photonic quantum memory", "PQ", "storage medium", "storage time",
        ("Pr:YSO", "Eu:YSO", "Rb vapor", "Cs vapor", "cold Rb", "rare-earth crystal", "diamond", "atomic ensemble"),
        ("Pr:YSO AFC", "Eu:YSO spin-wave", "buffered Rb vapor", "coated Cs vapor", "MOT Rb ensemble", "Er rare-earth crystal", "NV diamond ensemble", "cavity atomic ensemble"),
        ("1 us", "2 us", "3 us", "4 us", "5 us", "6 us", "7 us", "8 us"),
        ("1.5 us", "2.5 us", "3.5 us", "4.5 us", "5.5 us", "6.5 us", "7.5 us", "8.5 us"),
        72901, 759000,
    ),
    _Domain(
        "nanoscale_thermometry_array", "nanoscale thermometry array", "NT", "thermometer", "temperature resolution",
        ("NV", "TES", "Johnson", "KID", "RTD", "thermocouple", "graphene", "quantum dot"),
        ("nanodiamond NV", "bilayer TES", "cross-correlation Johnson", "lumped KID", "Pt nano-RTD", "AuNi thermocouple", "hBN graphene", "Coulomb-blockade quantum dot"),
        ("10 uK", "20 uK", "30 uK", "40 uK", "50 uK", "60 uK", "70 uK", "80 uK"),
        ("15 uK", "25 uK", "35 uK", "45 uK", "55 uK", "65 uK", "75 uK", "85 uK"),
        73001, 760000,
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
            f"S40-Anchor {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S40-Anchor record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S40-Anchor {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S40-Anchor record {code}?",
            f"For S40-Anchor {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S40-Anchor record {code}?",
        )
    return (
        f"S40-Anchor audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S40-Anchor audit {code}: what is {spec.field_a}?",
        f"In S40-Anchor audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S40-Anchor audit {code}: what is {spec.field_b}?",
        f"In S40-Anchor audit {code}, which entry is tagged {spec.field_b}?",
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
    case_id = f"{split}-s40-{spec.name}-{index:03d}"
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
    return S40AnchoredCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s40_cases(split: Split) -> tuple[S40AnchoredCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s40_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S40 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S40 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S40 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S40 language changed")
    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S40 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S40 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S40 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S40 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S40 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S40 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S40 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S40 TRAIN DEV option overlap")


__all__ = ["S40AnchoredCase", "generate_s40_cases", "validate_s40_partitions"]
