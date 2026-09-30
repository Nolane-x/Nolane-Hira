from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S20FusionCase:
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
    _Domain("atomic_clock","optical atomic clock","AC","interrogation scheme","cycle time",
        ("Rabi","Ramsey","hyper-Ramsey","auto-balanced Ramsey","composite pulse","spin echo","adiabatic rapid passage","frequency comb"),
        ("hyper-Ramsey B","synthetic frequency","phase-stepped Ramsey","zero-dead-time","interleaved Rabi","coherent population trapping","dynamic decoupling","dual-comb"),
        ("0.4 s","0.8 s","1.2 s","1.6 s","2.0 s","2.4 s","2.8 s","3.2 s"),
        ("0.6 s","1.0 s","1.4 s","1.8 s","2.2 s","2.6 s","3.0 s","3.4 s"),35001,241000),
    _Domain("desalination_membrane","desalination membrane skid","DM","membrane chemistry","recovery ratio",
        ("polyamide TFC","cellulose acetate","PES","PVDF","graphene oxide","aquaporin","polybenzimidazole","ceramic alumina"),
        ("polyamide nanocomposite","CTA","sulfonated PES","PTFE","MXene laminate","biomimetic channel","PEEK","zirconia ceramic"),
        ("30 percent","35 percent","40 percent","45 percent","50 percent","55 percent","60 percent","65 percent"),
        ("32 percent","37 percent","42 percent","47 percent","52 percent","57 percent","62 percent","67 percent"),35101,242000),
    _Domain("cryogenic_detector","cryogenic detector module","CD","sensor technology","bias voltage",
        ("TES","MKID","bolometer","SNSPD","NTD Ge","hot-electron bolometer","quantum capacitance","Josephson escape"),
        ("Ir-TES","KID array","CMB bolometer","WSi nanowire","Si thermistor","HEB mixer","graphene bolometer","SQUID threshold"),
        ("5 mV","10 mV","15 mV","20 mV","25 mV","30 mV","35 mV","40 mV"),
        ("7 mV","12 mV","17 mV","22 mV","27 mV","32 mV","37 mV","42 mV"),35201,243000),
    _Domain("turbomolecular_pump","turbomolecular pump","TP","bearing system","rotation speed",
        ("ceramic ball","magnetic","hybrid ceramic","oil-lubricated","gas bearing","active magnetic","passive magnetic","foil bearing"),
        ("Si3N4 ball","five-axis magnetic","dry hybrid","grease-packed","spiral-groove gas","superconducting magnetic","Halbach bearing","compliant foil"),
        ("24000 rpm","30000 rpm","36000 rpm","42000 rpm","48000 rpm","54000 rpm","60000 rpm","66000 rpm"),
        ("27000 rpm","33000 rpm","39000 rpm","45000 rpm","51000 rpm","57000 rpm","63000 rpm","69000 rpm"),35301,244000),
    _Domain("plasma_thruster","plasma thruster bench","PT","magnetic topology","discharge voltage",
        ("radial","cusp","mirror","solenoidal","quadrupole","Halbach","racetrack","diverging field"),
        ("nested cusp","magnetic nozzle","ECR mirror","dual solenoid","hexapole","segmented Halbach","closed drift","axial cusp"),
        ("150 V","200 V","250 V","300 V","350 V","400 V","450 V","500 V"),
        ("175 V","225 V","275 V","325 V","375 V","425 V","475 V","525 V"),35401,245000),
    _Domain("photonic_packager","photonic packaging cell","PP","coupling method","alignment tolerance",
        ("edge coupling","grating coupling","lensed fiber","fiber array","photonic wire bond","flip-chip optical","vertical coupler","free-space"),
        ("inverse taper","apodized grating","GRIN lens","V-groove array","polymer wire bond","micro-mirror coupling","3D coupler","micro-optic relay"),
        ("0.2 um","0.4 um","0.6 um","0.8 um","1.0 um","1.2 um","1.4 um","1.6 um"),
        ("0.3 um","0.5 um","0.7 um","0.9 um","1.1 um","1.3 um","1.5 um","1.7 um"),35501,246000),
    _Domain("geothermal_separator","geothermal steam separator","GT","separator geometry","inlet pressure",
        ("cyclone","horizontal drum","vertical drum","vane","swirl tube","flash vessel","mesh pad","multi-cyclone"),
        ("axial cyclone","tangential drum","compact vertical","chevron vane","helical tube","two-stage flash","wire-mesh","hydrocyclone bank"),
        ("4 bar","6 bar","8 bar","10 bar","12 bar","14 bar","16 bar","18 bar"),
        ("5 bar","7 bar","9 bar","11 bar","13 bar","15 bar","17 bar","19 bar"),35601,247000),
    _Domain("rna_synthesizer","RNA synthesis platform","RS","coupling chemistry","cycle duration",
        ("phosphoramidite","H-phosphonate","phosphotriester","enzymatic TdT","click ligation","solid-phase ligation","boranophosphate","thiophosphate"),
        ("2-cyanoethyl phosphoramidite","phosphonate ester","activated phosphate","template-free polymerase","SPAAC ligation","splinted ligation","borane-protected","phosphorothioate"),
        ("2 min","4 min","6 min","8 min","10 min","12 min","14 min","16 min"),
        ("3 min","5 min","7 min","9 min","11 min","13 min","15 min","17 min"),35701,248000),
    _Domain("laser_interferometer","laser interferometer","LI","beam splitter coating","arm length",
        ("dielectric HR","metallic silver","gold","broadband dielectric","ion-beam sputtered","sol-gel","multilayer oxide","crystalline"),
        ("Ta2O5-SiO2","protected silver","protected gold","chirped dielectric","IBS tantala","porous silica","hafnia-silica","AlGaAs crystalline"),
        ("0.5 m","1.0 m","1.5 m","2.0 m","2.5 m","3.0 m","3.5 m","4.0 m"),
        ("0.75 m","1.25 m","1.75 m","2.25 m","2.75 m","3.25 m","3.75 m","4.25 m"),35801,249000),
    _Domain("battery_calorimeter","battery calorimeter","BC","coolant medium","sampling interval",
        ("water","glycol","silicone oil","air","Novec","mineral oil","PAO","refrigerant"),
        ("water-glycol","propylene glycol","fluorinated oil","dry nitrogen","HFE fluid","transformer oil","synthetic ester","CO2 refrigerant"),
        ("1 s","2 s","3 s","4 s","5 s","6 s","7 s","8 s"),
        ("1.5 s","2.5 s","3.5 s","4.5 s","5.5 s","6.5 s","7.5 s","8.5 s"),35901,250000),
    _Domain("neutron_detector","neutron detector panel","ND","conversion layer","gas pressure",
        ("B-10","Li-6","Gd","He-3","BF3","LiF-ZnS","boron carbide","scintillating glass"),
        ("B4C enriched","LiF film","Gd2O3","He-3-CF4","boron-lined tube","Li glass fiber","boron nitride","GS20 glass"),
        ("1 atm","2 atm","3 atm","4 atm","5 atm","6 atm","7 atm","8 atm"),
        ("1.5 atm","2.5 atm","3.5 atm","4.5 atm","5.5 atm","6.5 atm","7.5 atm","8.5 atm"),36001,251000),
    _Domain("robotic_microscope","robotic microscope","RM","autofocus method","z-step",
        ("contrast sweep","phase detection","laser triangulation","image gradient","confocal peak","astigmatic","structured light","deep focus metric"),
        ("Tenengrad sweep","dual-pixel phase","chromatic confocal","Laplacian metric","pinhole peak","cylindrical lens","pattern projection","frequency-domain metric"),
        ("0.5 um","1 um","1.5 um","2 um","2.5 um","3 um","3.5 um","4 um"),
        ("0.75 um","1.25 um","1.75 um","2.25 um","2.75 um","3.25 um","3.75 um","4.25 um"),36101,252000),
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
            f"S20-Std {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"S20-Std record {code} lists {second} beside {spec.field_b}. "
            f"In the same {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S20-Std {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S20-Std record {code}, which entry belongs under {spec.field_a}?"
        qb1 = f"For S20-Std {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S20-Std record {code}, which entry belongs under {spec.field_b}?"
    else:
        state_a = (
            f"S20-Std inspection dossier {code} for the {spec.noun} places {first} "
            f"under {spec.field_a}; the {spec.field_b} entry reads {second}."
        )
        state_b = (
            f"S20-Std dossier {code} marks {second} for {spec.field_b}. "
            f"The same {spec.noun} dossier tags {first} as {spec.field_a}."
        )
        qa1 = f"From S20-Std dossier {code}, identify {spec.field_a}."
        qa2 = f"Which S20-Std dossier entry is tagged {spec.field_a} for {code}?"
        qb1 = f"From S20-Std dossier {code}, identify {spec.field_b}."
        qb2 = f"Which S20-Std dossier entry is tagged {spec.field_b} for {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S20FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s20-{spec.name}-{index:03d}"

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
    return S20FusionCase(
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


def generate_s20_cases(split: Split) -> tuple[S20FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s20_partitions(
    train: tuple[S20FusionCase, ...],
    dev: tuple[S20FusionCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S20 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S20 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S20 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S20 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("S20 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S20 state views must differ")
            if row.question_a1 == row.question_a2 or row.question_b1 == row.question_b2:
                raise RuntimeError("S20 question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S20 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S20 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S20 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S20 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S20 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S20 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S20 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S20 TRAIN/DEV option-alias overlap")


__all__ = [
    "S20FusionCase",
    "generate_s20_cases",
    "validate_s20_partitions",
]
