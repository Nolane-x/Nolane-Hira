from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S37QueryGatedCase:
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
        "atom_interferometer_array", "atom interferometer array", "AI", "laser topology", "baseline",
        ("retroreflected", "counterpropagating", "Raman pair", "Bragg pair", "cavity-assisted", "fiber-fed", "dual-axis", "common-path"),
        ("phase-locked Raman", "four-pulse Bragg", "resonant cavity", "photonic-fed", "tri-axis", "symmetric common-path", "chirped Raman", "multipass Bragg"),
        ("1 m", "2 m", "3 m", "4 m", "5 m", "6 m", "7 m", "8 m"),
        ("1.5 m", "2.5 m", "3.5 m", "4.5 m", "5.5 m", "6.5 m", "7.5 m", "8.5 m"),
        68301, 713000,
    ),
    _Domain(
        "brillouin_microscope", "Brillouin microscope", "BM", "spectral filter", "frequency shift",
        ("VIPA", "Fabry-Perot", "etalon", "grating", "ring resonator", "interferometer", "AWG", "cavity"),
        ("dual-stage VIPA", "tandem FP", "chirped etalon", "echelle", "SiN ring", "Mach-Zehnder", "arrayed grating", "confocal cavity"),
        ("2 GHz", "3 GHz", "4 GHz", "5 GHz", "6 GHz", "7 GHz", "8 GHz", "9 GHz"),
        ("2.5 GHz", "3.5 GHz", "4.5 GHz", "5.5 GHz", "6.5 GHz", "7.5 GHz", "8.5 GHz", "9.5 GHz"),
        68401, 714000,
    ),
    _Domain(
        "mot_controller", "magneto-optic trap controller", "MT", "coil topology", "field gradient",
        ("anti-Helmholtz", "quadrupole", "cloverleaf", "baseball", "planar", "saddle", "octupole", "chip wire"),
        ("segmented anti-Helmholtz", "printed quadrupole", "mini-cloverleaf", "micro-baseball", "MEMS planar", "dual-saddle", "hybrid octupole", "multilayer chip"),
        ("5 G/cm", "10 G/cm", "15 G/cm", "20 G/cm", "25 G/cm", "30 G/cm", "35 G/cm", "40 G/cm"),
        ("7 G/cm", "12 G/cm", "17 G/cm", "22 G/cm", "27 G/cm", "32 G/cm", "37 G/cm", "42 G/cm"),
        68501, 715000,
    ),
    _Domain(
        "quantum_capacitance_sensor", "quantum capacitance sensor", "QC", "dielectric", "sensitivity",
        ("hBN", "Al2O3", "HfO2", "SiO2", "Ta2O5", "diamond", "SiN", "mica"),
        ("monolayer hBN", "ALD alumina", "high-k hafnia", "thermal silica", "amorphous tantalum oxide", "CVD diamond", "LPCVD nitride", "exfoliated mica"),
        ("1 aF", "2 aF", "3 aF", "4 aF", "5 aF", "6 aF", "7 aF", "8 aF"),
        ("1.5 aF", "2.5 aF", "3.5 aF", "4.5 aF", "5.5 aF", "6.5 aF", "7.5 aF", "8.5 aF"),
        68601, 716000,
    ),
    _Domain(
        "fel_diagnostics", "free-electron-laser diagnostic", "FD", "screen", "bunch length",
        ("YAG", "OTR foil", "diamond", "LuAG", "Ce:YAG", "silicon", "GaAs", "phosphor"),
        ("thin YAG", "micro-OTR", "CVD diamond", "fast LuAG", "cerium garnet", "SOI screen", "LT-GaAs", "ultrafast phosphor"),
        ("10 fs", "20 fs", "30 fs", "40 fs", "50 fs", "60 fs", "70 fs", "80 fs"),
        ("15 fs", "25 fs", "35 fs", "45 fs", "55 fs", "65 fs", "75 fs", "85 fs"),
        68701, 717000,
    ),
    _Domain(
        "wgm_biosensor", "whispering-gallery biosensor", "WG", "resonator", "linewidth",
        ("silica sphere", "toroid", "disk", "bottle", "ring", "CaF2 disk", "MgF2 sphere", "polymer bead"),
        ("microtoroid", "wedge disk", "microbottle", "SiN ring", "crystalline CaF2", "crystalline MgF2", "hollow sphere", "hybrid polymer ring"),
        ("1 MHz", "2 MHz", "3 MHz", "4 MHz", "5 MHz", "6 MHz", "7 MHz", "8 MHz"),
        ("1.5 MHz", "2.5 MHz", "3.5 MHz", "4.5 MHz", "5.5 MHz", "6.5 MHz", "7.5 MHz", "8.5 MHz"),
        68801, 718000,
    ),
    _Domain(
        "cryo_cmos_controller", "cryo-CMOS controller", "CC", "process node", "clock rate",
        ("180 nm", "130 nm", "90 nm", "65 nm", "45 nm", "28 nm", "22 nm", "14 nm"),
        ("160 nm", "110 nm", "80 nm", "55 nm", "40 nm", "24 nm", "18 nm", "12 nm"),
        ("20 MHz", "40 MHz", "60 MHz", "80 MHz", "100 MHz", "120 MHz", "140 MHz", "160 MHz"),
        ("30 MHz", "50 MHz", "70 MHz", "90 MHz", "110 MHz", "130 MHz", "150 MHz", "170 MHz"),
        68901, 719000,
    ),
    _Domain(
        "thz_tds_sampler", "terahertz time-domain sampler", "TD", "emitter", "delay range",
        ("PCA", "ZnTe", "GaP", "organic crystal", "spintronic", "plasma", "QCL", "photomixer"),
        ("plasmonic PCA", "thin ZnTe", "orientation-patterned GaP", "DSTMS", "WCoFeB stack", "two-color plasma", "pulsed QCL", "dual-laser photomixer"),
        ("10 ps", "20 ps", "30 ps", "40 ps", "50 ps", "60 ps", "70 ps", "80 ps"),
        ("15 ps", "25 ps", "35 ps", "45 ps", "55 ps", "65 ps", "75 ps", "85 ps"),
        69001, 720000,
    ),
    _Domain(
        "neutron_reflectometer", "neutron reflectometer", "NR", "monochromator", "q range",
        ("graphite", "silicon", "germanium", "multilayer", "mosaic Cu", "mica", "sapphire", "supermirror"),
        ("pyrolytic graphite", "bent silicon", "graded Ge", "chirped multilayer", "Cu mosaic", "cleaved mica", "c-plane sapphire", "m=5 supermirror"),
        ("0.1 A-1", "0.2 A-1", "0.3 A-1", "0.4 A-1", "0.5 A-1", "0.6 A-1", "0.7 A-1", "0.8 A-1"),
        ("0.15 A-1", "0.25 A-1", "0.35 A-1", "0.45 A-1", "0.55 A-1", "0.65 A-1", "0.75 A-1", "0.85 A-1"),
        69101, 721000,
    ),
    _Domain(
        "ued_instrument", "ultrafast electron diffraction instrument", "FE", "gun mode", "pulse width",
        ("DC", "RF", "THz", "photoemission", "field emission", "plasma", "dielectric laser", "hybrid"),
        ("high-voltage DC", "S-band RF", "single-cycle THz", "multiphoton cathode", "cold field", "laser plasma", "DLA injector", "RF-THz hybrid"),
        ("20 fs", "40 fs", "60 fs", "80 fs", "100 fs", "120 fs", "140 fs", "160 fs"),
        ("30 fs", "50 fs", "70 fs", "90 fs", "110 fs", "130 fs", "150 fs", "170 fs"),
        69201, 722000,
    ),
    _Domain(
        "opm_array", "optically pumped magnetometer array", "OM", "vapor species", "noise floor",
        ("Rb87", "Rb85", "Cs133", "K39", "K41", "He3", "Xe129", "Na23"),
        ("enriched Rb87", "buffered Rb85", "microcell Cs", "SERF K39", "dual-isotope K", "polarized He3", "hyperpolarized Xe", "coated Na"),
        ("1 fT", "2 fT", "3 fT", "4 fT", "5 fT", "6 fT", "7 fT", "8 fT"),
        ("1.5 fT", "2.5 fT", "3.5 fT", "4.5 fT", "5.5 fT", "6.5 fT", "7.5 fT", "8.5 fT"),
        69301, 723000,
    ),
    _Domain(
        "nanomechanical_mass_spec", "nanomechanical mass spectrometer", "NM", "mode shape", "mass range",
        ("fundamental", "second flexural", "torsional", "breathing", "wineglass", "radial", "contour", "coupled"),
        ("localized fundamental", "split flexural", "dual torsional", "radial breathing", "high-order wineglass", "extensional radial", "contour overtone", "mode-coupled"),
        ("10 kDa", "20 kDa", "30 kDa", "40 kDa", "50 kDa", "60 kDa", "70 kDa", "80 kDa"),
        ("15 kDa", "25 kDa", "35 kDa", "45 kDa", "55 kDa", "65 kDa", "75 kDa", "85 kDa"),
        69401, 724000,
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
            f"S37-QueryGate {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S37-QueryGate record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S37-QueryGate {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S37-QueryGate record {code}?",
            f"For S37-QueryGate {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S37-QueryGate record {code}?",
        )
    return (
        f"S37-QueryGate audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S37-QueryGate audit {code}: what is {spec.field_a}?",
        f"In S37-QueryGate audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S37-QueryGate audit {code}: what is {spec.field_b}?",
        f"In S37-QueryGate audit {code}, which entry is tagged {spec.field_b}?",
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
    case_id = f"{split}-s37-{spec.name}-{index:03d}"
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
    return S37QueryGatedCase(
        case_id, split, spec.name, "en", sa, sb, qa1, qa2, qb1, qb2,
        texts, aliases, ids, ga, gb,
    )


def generate_s37_cases(split: Split) -> tuple[S37QueryGatedCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, i, split) for spec in _DOMAINS for i in range(count))


def validate_s37_partitions(train, dev):
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S37 partition size changed")
    domains = {d.name for d in _DOMAINS}
    if len(domains) != 12:
        raise RuntimeError("S37 domain count changed")
    if {r.domain for r in train} != domains or {r.domain for r in dev} != domains:
        raise RuntimeError("S37 domains changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S37 language changed")
    for rows in (train, dev):
        if len({r.case_id for r in rows}) != len(rows):
            raise RuntimeError("S37 duplicate case IDs")
        for r in rows:
            if r.state_a == r.state_b:
                raise RuntimeError("S37 state views must differ")
            if r.question_a1 == r.question_a2 or r.question_b1 == r.question_b2:
                raise RuntimeError("S37 question views must differ")
            if len(r.option_ids) != 4 or len(set(r.option_ids)) != 4:
                raise RuntimeError("S37 option IDs changed")
            if r.gold_a == r.gold_b:
                raise RuntimeError("S37 paired golds must differ")

    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    if ts & ds:
        raise RuntimeError("S37 TRAIN DEV state overlap")
    if tq & dq:
        raise RuntimeError("S37 TRAIN DEV question overlap")
    if to & do:
        raise RuntimeError("S37 TRAIN DEV option overlap")


__all__ = ["S37QueryGatedCase", "generate_s37_cases", "validate_s37_partitions"]
