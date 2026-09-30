from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S22FusionCase:
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
        row["option_texts"] = list(self.option_texts)
        row["option_aliases"] = list(self.option_aliases)
        row["option_ids"] = list(self.option_ids)
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
    _Domain("afm_stage","atomic force microscope stage","AF","scan mode","tip coating",
        ("contact","tapping","non-contact","PeakForce","lateral force","conductive","Kelvin probe","force-volume"),
        ("intermittent contact","amplitude modulation","frequency modulation","quantitative nanomechanical","friction force","current sensing","surface potential","force mapping"),
        ("gold","platinum-iridium","diamond-like carbon","silicon nitride","boron-doped diamond","chromium","aluminum","graphene"),
        ("AuCr","PtIr5","DLC-20","Si3N4","BDD","Cr20","Al30","monolayer graphene"),39001,281000),
    _Domain("ellipsometer","spectroscopic ellipsometer","EL","optical model","incidence angle",
        ("Cauchy","Tauc-Lorentz","Drude","EMA","Forouhi-Bloomer","Sellmeier","B-spline","Lorentz oscillator"),
        ("graded Cauchy","multi-oscillator Tauc","Drude-Lorentz","Bruggeman EMA"," Cody-Lorentz","anisotropic Sellmeier","Kramers-Kronig spline","Gaussian oscillator"),
        ("55 deg","58 deg","61 deg","64 deg","67 deg","70 deg","73 deg","76 deg"),
        ("56 deg","59 deg","62 deg","65 deg","68 deg","71 deg","74 deg","77 deg"),39101,282000),
    _Domain("neutron_counter","neutron detector counter","NC","converter layer","bias voltage",
        ("boron-10","lithium-6 fluoride","helium-3","gadolinium","boron carbide","lithium glass","uranium-235","scintillating fiber"),
        ("enriched B10","LiF-ZnS","pressurized He3","Gd foil","B4C multilayer","GS20 glass","fission chamber","Li-loaded fiber"),
        ("350 V","400 V","450 V","500 V","550 V","600 V","650 V","700 V"),
        ("375 V","425 V","475 V","525 V","575 V","625 V","675 V","725 V"),39201,283000),
    _Domain("impedance_analyzer","precision impedance analyzer","IA","excitation amplitude","frequency span",
        ("10 mV","20 mV","30 mV","40 mV","50 mV","60 mV","70 mV","80 mV"),
        ("15 mV","25 mV","35 mV","45 mV","55 mV","65 mV","75 mV","85 mV"),
        ("20 Hz-20 kHz","40 Hz-100 kHz","100 Hz-1 MHz","1 kHz-5 MHz","10 kHz-10 MHz","100 kHz-20 MHz","1 MHz-50 MHz","5 MHz-100 MHz"),
        ("30 Hz-30 kHz","60 Hz-200 kHz","200 Hz-2 MHz","2 kHz-6 MHz","20 kHz-12 MHz","200 kHz-25 MHz","2 MHz-60 MHz","6 MHz-110 MHz"),39301,284000),
    _Domain("spin_coater","thin-film spin coater","SC","ramp profile","final speed",
        ("linear","two-step","three-step","exponential","S-curve","pulse","slow-start","reverse-pulse"),
        ("piecewise linear","dual-ramp","triple-ramp","logarithmic","jerk-limited","burst-ramp","soft-start","bidirectional pulse"),
        ("1000 rpm","1500 rpm","2000 rpm","2500 rpm","3000 rpm","3500 rpm","4000 rpm","4500 rpm"),
        ("1250 rpm","1750 rpm","2250 rpm","2750 rpm","3250 rpm","3750 rpm","4250 rpm","4750 rpm"),39401,285000),
    _Domain("laser_interferometer","laser interferometer controller","LI","beam splitter","phase modulation",
        ("cube","plate","pellicle","polarizing cube","fiber coupler","dichroic","wedged plate","integrated MMI"),
        ("nonpolarizing cube","thin plate","membrane","PBS","2x2 fiber coupler","dual-band dichroic","wedged optic","silicon MMI"),
        ("2 kHz","4 kHz","6 kHz","8 kHz","10 kHz","12 kHz","14 kHz","16 kHz"),
        ("3 kHz","5 kHz","7 kHz","9 kHz","11 kHz","13 kHz","15 kHz","17 kHz"),39501,286000),
    _Domain("microcalorimeter","microcalorimeter platform","MCAL","sensor type","heating rate",
        ("thermopile","RTD","thermistor","TES","diode","pyroelectric","fiber Bragg","microcantilever"),
        ("thin-film thermopile","Pt100 RTD","NTC bead","transition-edge","silicon diode","PZT pyro","FBG array","bimaterial cantilever"),
        ("0.2 K/min","0.4 K/min","0.6 K/min","0.8 K/min","1.0 K/min","1.2 K/min","1.4 K/min","1.6 K/min"),
        ("0.3 K/min","0.5 K/min","0.7 K/min","0.9 K/min","1.1 K/min","1.3 K/min","1.5 K/min","1.7 K/min"),39601,287000),
    _Domain("stepper","photolithography stepper","ST","illumination mode","focus offset",
        ("conventional","annular","quadrupole","dipole","freeform","off-axis","coherent","partial coherence"),
        ("sigma conventional","thin annular","cross-quadrupole","x-dipole","source-mask optimized","oblique OAI","high-coherence","low-sigma"),
        ("-0.30 um","-0.20 um","-0.10 um","0.00 um","+0.10 um","+0.20 um","+0.30 um","+0.40 um"),
        ("-0.25 um","-0.15 um","-0.05 um","+0.05 um","+0.15 um","+0.25 um","+0.35 um","+0.45 um"),39701,288000),
    _Domain("gas_chromatograph","gas chromatograph","GC","column phase","carrier flow",
        ("5% phenyl","wax","cyanopropyl","PLOT-Q","dimethyl polysiloxane","trifluoropropyl","ionic liquid","molecular sieve"),
        ("low-bleed phenyl","PEG wax","biscyanopropyl","PLOT-U","PDMS","fluorosilicone","SLB-IL","5A sieve"),
        ("0.6 mL/min","0.8 mL/min","1.0 mL/min","1.2 mL/min","1.4 mL/min","1.6 mL/min","1.8 mL/min","2.0 mL/min"),
        ("0.7 mL/min","0.9 mL/min","1.1 mL/min","1.3 mL/min","1.5 mL/min","1.7 mL/min","1.9 mL/min","2.1 mL/min"),39801,289000),
    _Domain("squid_readout","SQUID readout controller","SQ","feedback mode","flux bias",
        ("flux-locked loop","open loop","digital feedback","analog feedback","two-stage","series array","RF SQUID","dispersive"),
        ("adaptive FLL","calibration open-loop","FPGA feedback","low-noise analog","two-stage cascade","array feedback","tank-circuit RF","microwave dispersive"),
        ("0.10 Phi0","0.15 Phi0","0.20 Phi0","0.25 Phi0","0.30 Phi0","0.35 Phi0","0.40 Phi0","0.45 Phi0"),
        ("0.12 Phi0","0.17 Phi0","0.22 Phi0","0.27 Phi0","0.32 Phi0","0.37 Phi0","0.42 Phi0","0.47 Phi0"),39901,290000),
    _Domain("flim_system","fluorescence lifetime imaging system","FL","excitation source","gate width",
        ("pulsed diode","supercontinuum","Ti:sapphire","LED","OPO","fiber laser","microchip laser","frequency-doubled diode"),
        ("gain-switched diode","filtered supercontinuum","mode-locked TiS","picosecond LED","synchronously pumped OPO","Er fiber","passively Q-switched","SHG diode"),
        ("100 ps","150 ps","200 ps","250 ps","300 ps","350 ps","400 ps","450 ps"),
        ("125 ps","175 ps","225 ps","275 ps","325 ps","375 ps","425 ps","475 ps"),40001,291000),
    _Domain("vibration_isolator","active vibration isolator","VI","control mode","corner frequency",
        ("skyhook","PID","feedforward","H-infinity","adaptive","notch control","state feedback","hybrid passive-active"),
        ("virtual skyhook","gain-scheduled PID","reference feedforward","robust Hinf","LMS adaptive","multi-notch","LQR feedback","semi-active hybrid"),
        ("0.5 Hz","0.8 Hz","1.1 Hz","1.4 Hz","1.7 Hz","2.0 Hz","2.3 Hz","2.6 Hz"),
        ("0.6 Hz","0.9 Hz","1.2 Hz","1.5 Hz","1.8 Hz","2.1 Hz","2.4 Hz","2.7 Hz"),40101,292000),
)


def _shuffle(*, case_id: str, noun: str, field_a: str, field_b: str,
             first: str, second: str, distractor_first: str,
             distractor_second: str, seed: int):
    rows = [
        ("a", field_a, first),
        ("b", field_b, second),
        ("x", field_a, distractor_first),
        ("y", field_b, distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts = tuple(f"for the {noun}, {field} is {value}" for _k, field, value in rows)
    aliases = tuple(f"{value} is the recorded {field} entry in this {noun}" for _k, field, value in rows)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i,(kind,_f,_v) in enumerate(rows) if kind == "a")
    gb = next(i for i,(kind,_f,_v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(spec: _Domain, *, split: Split, code: str, first: str, second: str):
    if split == "train":
        state_a = (
            f"S22-Priority {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"S22-Priority record {code} stores {second} beside {spec.field_b}. "
            f"The same {spec.noun} assigns {first} to {spec.field_a}."
        )
        qa1 = f"For S22-Priority {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S22-Priority record {code}, which entry is filed as {spec.field_a}?"
        qb1 = f"For S22-Priority {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S22-Priority record {code}, which entry is filed as {spec.field_b}?"
    else:
        state_a = (
            f"S22-Priority audit sheet {code} for the {spec.noun} assigns "
            f"{first} to {spec.field_a}, while {spec.field_b} is documented as {second}."
        )
        state_b = (
            f"On S22-Priority sheet {code}, {second} is the {spec.field_b} setting. "
            f"The {spec.noun} sheet separately records {first} for {spec.field_a}."
        )
        qa1 = f"Consult S22-Priority audit {code}: what is the {spec.field_a} entry?"
        qa2 = f"Which value carries the {spec.field_a} label on S22-Priority sheet {code}?"
        qb1 = f"Consult S22-Priority audit {code}: what is the {spec.field_b} entry?"
        qb2 = f"Which value carries the {spec.field_b} label on S22-Priority sheet {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S22FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s22-{spec.name}-{index:03d}"

    state_a, state_b, qa1, qa2, qb1, qb2 = _views(
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
    return S22FusionCase(
        case_id=case_id,
        split=split,
        domain=spec.name,
        language="en",
        state_a=state_a,
        state_b=state_b,
        question_a1=qa1,
        question_a2=qa2,
        question_b1=qb1,
        question_b2=qb2,
        option_texts=texts,
        option_aliases=aliases,
        option_ids=ids,
        gold_a=ga,
        gold_b=gb,
    )


def generate_s22_cases(split: Split) -> tuple[S22FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, index, split) for spec in _DOMAINS for index in range(count))


def validate_s22_partitions(
    train: tuple[S22FusionCase, ...],
    dev: tuple[S22FusionCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S22 partition size changed")
    if len({row.domain for row in train}) != 12 or len({row.domain for row in dev}) != 12:
        raise RuntimeError("S22 domain count changed")
    if any(row.split != "train" for row in train) or any(row.split != "dev" for row in dev):
        raise RuntimeError("S22 split labels changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("S22 language changed")
    if any(len(row.option_texts) != 4 or len(row.option_aliases) != 4 or len(row.option_ids) != 4 for row in (*train, *dev)):
        raise RuntimeError("S22 option cardinality changed")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    train_q = {x for row in train for x in (row.question_a1,row.question_a2,row.question_b1,row.question_b2)}
    dev_q = {x for row in dev for x in (row.question_a1,row.question_a2,row.question_b1,row.question_b2)}
    train_options = {x for row in train for x in (*row.option_texts,*row.option_aliases)}
    dev_options = {x for row in dev for x in (*row.option_texts,*row.option_aliases)}
    if train_states & dev_states:
        raise RuntimeError("S22 TRAIN/DEV exact state overlap")
    if train_q & dev_q:
        raise RuntimeError("S22 TRAIN/DEV exact question overlap")
    if train_options & dev_options:
        raise RuntimeError("S22 TRAIN/DEV exact option overlap")


__all__ = ["S22FusionCase", "generate_s22_cases", "validate_s22_partitions"]
