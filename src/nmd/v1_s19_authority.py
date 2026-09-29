from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S19FusionCase:
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
    _Domain("electron_microscope","electron microscope column","EM","electron source","accelerating voltage",
        ("LaB6","tungsten","Schottky FEG","cold FEG","CeB6","ZrO-coated W","photocathode","carbon nanotube"),
        ("Y2O3 cathode","IrCe emitter","NEA GaAs","diamond emitter","HfC tip","graphene edge","Cs3Sb photocathode","Mo field tip"),
        ("40 kV","60 kV","80 kV","100 kV","120 kV","140 kV","160 kV","180 kV"),
        ("50 kV","70 kV","90 kV","110 kV","130 kV","150 kV","170 kV","190 kV"),33001,221000),
    _Domain("molten_salt_loop","molten-salt test loop","MS","salt blend","circulation rate",
        ("FLiBe","FLiNaK","KCl-MgCl2","NaNO3-KNO3","LiCl-KCl","KF-ZrF4","NaCl-KCl-MgCl2","carbonate eutectic"),
        ("LiF-NaF-KF","chloride ternary","nitrate ternary","fluoroborate","LiF-BeF2-ZrF4","CsF-LiF","NaF-ZrF4","CaCl2-NaCl"),
        ("2 kg/s","4 kg/s","6 kg/s","8 kg/s","10 kg/s","12 kg/s","14 kg/s","16 kg/s"),
        ("3 kg/s","5 kg/s","7 kg/s","9 kg/s","11 kg/s","13 kg/s","15 kg/s","17 kg/s"),33101,222000),
    _Domain("gamma_spectrometer","gamma spectrometer","GS","detector crystal","shaping time",
        ("HPGe","NaI(Tl)","LaBr3","CeBr3","CZT","BGO","CsI(Tl)","LYSO"),
        ("SrI2","GAGG","CLYC","CdTe","plastic scintillator","BaF2","YAP","LuAG"),
        ("0.5 us","1 us","1.5 us","2 us","2.5 us","3 us","3.5 us","4 us"),
        ("0.75 us","1.25 us","1.75 us","2.25 us","2.75 us","3.25 us","3.75 us","4.25 us"),33201,223000),
    _Domain("seismic_isolator","seismic isolation rig","SI","bearing type","design displacement",
        ("lead rubber","friction pendulum","high-damping rubber","sliding PTFE","elastomeric","roller","spring-damper","hybrid pendulum"),
        ("triple pendulum","natural rubber","steel yielding","ball bearing","viscous-isolated","wire rope","negative stiffness","shape-memory"),
        ("50 mm","75 mm","100 mm","125 mm","150 mm","175 mm","200 mm","225 mm"),
        ("60 mm","85 mm","110 mm","135 mm","160 mm","185 mm","210 mm","235 mm"),33301,224000),
    _Domain("ultrasonic_welder","ultrasonic welding station","UW","sonotrode material","weld amplitude",
        ("titanium","tool steel","aluminum bronze","carbide","maraging steel","Inconel","H13","beryllium copper"),
        ("Ti-6Al-4V","M2 steel","Ampco bronze","WC-Co","D2 steel","Inconel 625","S7 steel","CuCrZr"),
        ("10 um","20 um","30 um","40 um","50 um","60 um","70 um","80 um"),
        ("15 um","25 um","35 um","45 um","55 um","65 um","75 um","85 um"),33401,225000),
    _Domain("cryogenic_pump","cryogenic pump skid","CP","pump principle","inlet temperature",
        ("centrifugal","reciprocating","screw","turbo","diaphragm","piston","magnetic bearing","regenerative"),
        ("gerotor","roots","scroll","linear piston","bellows","ejector","axial","peristaltic"),
        ("20 K","30 K","40 K","50 K","60 K","70 K","80 K","90 K"),
        ("25 K","35 K","45 K","55 K","65 K","75 K","85 K","95 K"),33501,226000),
    _Domain("mass_spectrometer","mass spectrometer","MSP","mass analyzer","scan rate",
        ("quadrupole","TOF","Orbitrap","ion trap","FT-ICR","magnetic sector","Q-TOF","triple quadrupole"),
        ("linear ion trap","reflectron TOF","electrostatic trap","Mattauch-Herzog","Q-Orbitrap","tandem TOF","cyclotron trap","Wien filter"),
        ("100 Da/s","200 Da/s","300 Da/s","400 Da/s","500 Da/s","600 Da/s","700 Da/s","800 Da/s"),
        ("150 Da/s","250 Da/s","350 Da/s","450 Da/s","550 Da/s","650 Da/s","750 Da/s","850 Da/s"),33601,227000),
    _Domain("hydrogen_compressor","hydrogen compressor","HC","compression method","outlet pressure",
        ("diaphragm","ionic piston","reciprocating","centrifugal","metal hydride","electrochemical","scroll","screw"),
        ("linear piston","liquid piston","roots","turboelectric","adsorption","cryogenic","bellows","oil-free rotary"),
        ("100 bar","200 bar","300 bar","400 bar","500 bar","600 bar","700 bar","800 bar"),
        ("150 bar","250 bar","350 bar","450 bar","550 bar","650 bar","750 bar","850 bar"),33701,228000),
    _Domain("fiber_draw_tower","optical fiber draw tower","FD","preform type","draw speed",
        ("MCVD","OVD","VAD","PCVD","stack-and-draw","rod-in-tube","sol-gel","powder-in-tube"),
        ("plasma CVD","outside plasma","direct nanoparticle","sleeve collapse","extruded preform","capillary stack","3D printed","molten-core"),
        ("5 m/s","10 m/s","15 m/s","20 m/s","25 m/s","30 m/s","35 m/s","40 m/s"),
        ("7.5 m/s","12.5 m/s","17.5 m/s","22.5 m/s","27.5 m/s","32.5 m/s","37.5 m/s","42.5 m/s"),33801,229000),
    _Domain("neural_probe_tester","neural probe tester","NP","electrode coating","stimulus current",
        ("platinum black","PEDOT:PSS","iridium oxide","gold","titanium nitride","graphene","CNT","diamond-like carbon"),
        ("PtIr","RuOx","polypyrrole","AuPt alloy","TaN","MXene","boron-doped diamond","conductive hydrogel"),
        ("5 uA","10 uA","15 uA","20 uA","25 uA","30 uA","35 uA","40 uA"),
        ("7.5 uA","12.5 uA","17.5 uA","22.5 uA","27.5 uA","32.5 uA","37.5 uA","42.5 uA"),33901,230000),
    _Domain("magnetic_separator","magnetic separator","MG","magnet topology","belt speed",
        ("drum","roll","overband","matrix","wet high-intensity","eddy current","rare-earth pulley","cross-belt"),
        ("induced roll","open-gradient","carousel","plate","suspended magnet","high-gradient","superconducting","slurry matrix"),
        ("0.2 m/s","0.4 m/s","0.6 m/s","0.8 m/s","1.0 m/s","1.2 m/s","1.4 m/s","1.6 m/s"),
        ("0.3 m/s","0.5 m/s","0.7 m/s","0.9 m/s","1.1 m/s","1.3 m/s","1.5 m/s","1.7 m/s"),34001,231000),
    _Domain("aeroacoustic_tunnel","aeroacoustic wind tunnel","AT","nozzle profile","test velocity",
        ("contraction bell","polynomial","Witozinsky","cubic","quintic","Bezier","spline","axisymmetric bell"),
        ("septic polynomial","optimized contour","elliptic","cosine","tanh","B-spline","superellipse","piecewise cubic"),
        ("20 m/s","30 m/s","40 m/s","50 m/s","60 m/s","70 m/s","80 m/s","90 m/s"),
        ("25 m/s","35 m/s","45 m/s","55 m/s","65 m/s","75 m/s","85 m/s","95 m/s"),34101,232000),
)


def _shuffle(
    *,
    case_id: str,
    noun: str,
    field_a: str,
    field_b: str,
    first: str,
    second: str,
    distractor_first: str,
    distractor_second: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int, int]:
    rows = [
        ("a", field_a, first),
        ("b", field_b, second),
        ("x", field_a, distractor_first),
        ("y", field_b, distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts = tuple(
        f"for the {noun}, {field} is {value}"
        for _kind, field, value in rows
    )
    aliases = tuple(
        f"{value} is the recorded {field} entry in this {noun}"
        for _kind, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _field, _value) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _field, _value) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(
    spec: _Domain,
    *,
    split: Split,
    code: str,
    first: str,
    second: str,
) -> tuple[str, str, str, str, str, str]:
    if split == "train":
        state_a = (
            f"S19-Triadic {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"S19-Triadic record {code} lists {second} beside {spec.field_b}. "
            f"In the same {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S19-Triadic {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S19-Triadic record {code}, which entry belongs under {spec.field_a}?"
        qb1 = f"For S19-Triadic {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S19-Triadic record {code}, which entry belongs under {spec.field_b}?"
    else:
        state_a = (
            f"S19-Triadic inspection dossier {code} for the {spec.noun} places {first} "
            f"under {spec.field_a}; the {spec.field_b} entry reads {second}."
        )
        state_b = (
            f"S19-Triadic dossier {code} marks {second} for {spec.field_b}. "
            f"The same {spec.noun} dossier tags {first} as {spec.field_a}."
        )
        qa1 = f"From S19-Triadic dossier {code}, identify {spec.field_a}."
        qa2 = f"Which S19-Triadic dossier entry is tagged {spec.field_a} for {code}?"
        qb1 = f"From S19-Triadic dossier {code}, identify {spec.field_b}."
        qb2 = f"Which S19-Triadic dossier entry is tagged {spec.field_b} for {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S19FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s19-{spec.name}-{index:03d}"

    state_a, state_b, qa1, qa2, qb1, qb2 = _views(
        spec,
        split=split,
        code=code,
        first=first,
        second=second,
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
    return S19FusionCase(
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


def generate_s19_cases(split: Split) -> tuple[S19FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s19_partitions(
    train: tuple[S19FusionCase, ...],
    dev: tuple[S19FusionCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S19 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S19 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S19 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S19 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("S19 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S19 state views must differ")
            if row.question_a1 == row.question_a2 or row.question_b1 == row.question_b2:
                raise RuntimeError("S19 question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S19 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S19 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S19 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S19 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S19 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S19 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S19 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S19 TRAIN/DEV option-alias overlap")


__all__ = [
    "S19FusionCase",
    "generate_s19_cases",
    "validate_s19_partitions",
]
