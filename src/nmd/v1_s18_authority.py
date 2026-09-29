from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S18FusionCase:
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
    _Domain("fusion_blanket","fusion blanket loop","FB","coolant medium","loop pressure",
        ("helium","water","PbLi","FLiBe","sodium","CO2","LiPb","molten salt"),
        ("supercritical helium","D2O","Li17Pb83","FLiNaK","lead-bismuth","nitrogen","chloride salt","potassium"),
        ("2 MPa","4 MPa","6 MPa","8 MPa","10 MPa","12 MPa","14 MPa","16 MPa"),
        ("3 MPa","5 MPa","7 MPa","9 MPa","11 MPa","13 MPa","15 MPa","17 MPa"),32001,221000),
    _Domain("microfluidic_sorter","microfluidic sorter","MS","actuation mode","channel flow",
        ("dielectrophoretic","acoustic","hydrodynamic","magnetic","optical","inertial","electroosmotic","pneumatic"),
        ("surface acoustic wave","travelling-wave DEP","viscoelastic","ferrofluidic","laser tweezer","Dean flow","DC electrokinetic","membrane valve"),
        ("5 uL/min","10 uL/min","15 uL/min","20 uL/min","25 uL/min","30 uL/min","35 uL/min","40 uL/min"),
        ("7 uL/min","12 uL/min","17 uL/min","22 uL/min","27 uL/min","32 uL/min","37 uL/min","42 uL/min"),32101,222000),
    _Domain("xray_detector","x-ray detector panel","XD","sensor material","frame rate",
        ("CsI","CdTe","silicon","GaAs","CZT","GOS","amorphous Se","perovskite"),
        ("CdZnTe","LGAD silicon","diamond","InGaAs","TlBr","LYSO","a-Si","lead-halide perovskite"),
        ("10 fps","20 fps","30 fps","40 fps","50 fps","60 fps","70 fps","80 fps"),
        ("15 fps","25 fps","35 fps","45 fps","55 fps","65 fps","75 fps","85 fps"),32201,223000),
    _Domain("district_heat","district-heating exchanger","DH","plate alloy","supply temperature",
        ("316L","titanium","duplex 2205","copper-nickel","carbon steel","Hastelloy C276","aluminum bronze","254SMO"),
        ("904L","superduplex","Inconel 625","CuNi 90/10","P265GH","Alloy 20","nickel 200","titanium grade 2"),
        ("60 C","70 C","80 C","90 C","100 C","110 C","120 C","130 C"),
        ("65 C","75 C","85 C","95 C","105 C","115 C","125 C","135 C"),32301,224000),
    _Domain("orbital_camera","orbital mapping camera","OC","focal design","line rate",
        ("pushbroom","whiskbroom","frame","TDI","panoramic","staring","multi-line","rolling"),
        ("dual-line pushbroom","step-stare","global frame","multi-stage TDI","fisheye mapping","snapshot mosaic","curved focal plane","event-based"),
        ("2 kHz","4 kHz","6 kHz","8 kHz","10 kHz","12 kHz","14 kHz","16 kHz"),
        ("3 kHz","5 kHz","7 kHz","9 kHz","11 kHz","13 kHz","15 kHz","17 kHz"),32401,225000),
    _Domain("electrospinning","electrospinning line","ES","polymer feed","collector speed",
        ("PCL","PLA","PAN","PVDF","PEO","nylon-6","PVA","PU"),
        ("PLGA","PBS","PEEK","PVDF-HFP","PEO-PPO","nylon-12","PVP","TPU"),
        ("100 rpm","200 rpm","300 rpm","400 rpm","500 rpm","600 rpm","700 rpm","800 rpm"),
        ("150 rpm","250 rpm","350 rpm","450 rpm","550 rpm","650 rpm","750 rpm","850 rpm"),32501,226000),
    _Domain("atomic_clock","atomic clock package","AC","reference species","servo bandwidth",
        ("cesium","rubidium","hydrogen","strontium","ytterbium","mercury","calcium","aluminum ion"),
        ("Rb-87 CPT","Cs fountain","Sr lattice","Yb ion","Hg ion","Mg ion","Ca+ ion","thorium nuclear"),
        ("1 Hz","2 Hz","3 Hz","4 Hz","5 Hz","6 Hz","7 Hz","8 Hz"),
        ("1.5 Hz","2.5 Hz","3.5 Hz","4.5 Hz","5.5 Hz","6.5 Hz","7.5 Hz","8.5 Hz"),32601,227000),
    _Domain("desalination_train","desalination membrane train","DT","membrane chemistry","feed flux",
        ("polyamide RO","cellulose acetate","TFC","graphene oxide","ceramic NF","PVDF UF","PES UF","ion-exchange"),
        ("aquaporin","polybenzimidazole","thin-film nanocomposite","MOF composite","alumina NF","PTFE MF","PAN UF","bipolar membrane"),
        ("10 LMH","15 LMH","20 LMH","25 LMH","30 LMH","35 LMH","40 LMH","45 LMH"),
        ("12 LMH","17 LMH","22 LMH","27 LMH","32 LMH","37 LMH","42 LMH","47 LMH"),32701,228000),
    _Domain("rail_brake","rail regenerative brake","RB","converter topology","regen limit",
        ("two-level IGBT","three-level NPC","SiC inverter","thyristor chopper","matrix converter","MMC","dual-active bridge","flying capacitor"),
        ("T-type NPC","GaN inverter","ANPC","IGCT chopper","current-source inverter","hybrid MMC","LLC bridge","cascaded H-bridge"),
        ("200 kW","300 kW","400 kW","500 kW","600 kW","700 kW","800 kW","900 kW"),
        ("250 kW","350 kW","450 kW","550 kW","650 kW","750 kW","850 kW","950 kW"),32801,229000),
    _Domain("neural_probe","neural recording probe","NP","electrode material","sample bandwidth",
        ("platinum","iridium oxide","gold","PEDOT:PSS","tungsten","graphene","titanium nitride","carbon fiber"),
        ("PtIr","ruthenium oxide","platinum black","PEDOT-CNT","stainless steel","boron-doped diamond","TaN","CNT yarn"),
        ("5 kHz","10 kHz","15 kHz","20 kHz","25 kHz","30 kHz","35 kHz","40 kHz"),
        ("7.5 kHz","12.5 kHz","17.5 kHz","22.5 kHz","27.5 kHz","32.5 kHz","37.5 kHz","42.5 kHz"),32901,230000),
    _Domain("hydrogen_storage","hydrogen storage vessel","HS","liner material","fill pressure",
        ("HDPE","aluminum","stainless steel","carbon composite","PEEK","polyamide","titanium","glass composite"),
        ("PA6","Al-Li","316LN","thermoplastic composite","PPS","PA12","Ti-6Al-4V","basalt composite"),
        ("20 MPa","30 MPa","40 MPa","50 MPa","60 MPa","70 MPa","80 MPa","90 MPa"),
        ("25 MPa","35 MPa","45 MPa","55 MPa","65 MPa","75 MPa","85 MPa","95 MPa"),33001,231000),
    _Domain("spectrograph","astronomical spectrograph","SG","disperser type","resolving power",
        ("echelle","VPH grating","prism","grism","ruled grating","Fabry-Perot","AWG","immersed grating"),
        ("R2 echelle","volume Bragg","cross-dispersing prism","silicon grism","holographic grating","VIPA","photonic lantern AWG","germanium immersed"),
        ("10000","20000","30000","40000","50000","60000","70000","80000"),
        ("15000","25000","35000","45000","55000","65000","75000","85000"),33101,232000),
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
            f"S18-Paired {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"S18-Paired record {code} lists {second} beside {spec.field_b}. "
            f"In the same {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S18-Paired {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S18-Paired record {code}, which entry belongs under {spec.field_a}?"
        qb1 = f"For S18-Paired {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S18-Paired record {code}, which entry belongs under {spec.field_b}?"
    else:
        state_a = (
            f"S18-Paired inspection dossier {code} for the {spec.noun} places {first} "
            f"under {spec.field_a}; the {spec.field_b} entry reads {second}."
        )
        state_b = (
            f"S18-Paired dossier {code} marks {second} for {spec.field_b}. "
            f"The same {spec.noun} dossier tags {first} as {spec.field_a}."
        )
        qa1 = f"From S18-Paired dossier {code}, identify {spec.field_a}."
        qa2 = f"Which S18-Paired dossier entry is tagged {spec.field_a} for {code}?"
        qb1 = f"From S18-Paired dossier {code}, identify {spec.field_b}."
        qb2 = f"Which S18-Paired dossier entry is tagged {spec.field_b} for {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S18FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s18-{spec.name}-{index:03d}"

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
    return S18FusionCase(
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


def generate_s18_cases(split: Split) -> tuple[S18FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s18_partitions(
    train: tuple[S18FusionCase, ...],
    dev: tuple[S18FusionCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S18 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S18 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S18 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S18 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("S18 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S18 state views must differ")
            if row.question_a1 == row.question_a2 or row.question_b1 == row.question_b2:
                raise RuntimeError("S18 question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S18 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S18 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S18 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S18 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S18 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S18 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S18 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S18 TRAIN/DEV option-alias overlap")


__all__ = [
    "S18FusionCase",
    "generate_s18_cases",
    "validate_s18_partitions",
]
