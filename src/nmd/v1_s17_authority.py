from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S17FusionCase:
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
    _Domain("laser_cooling","laser-cooling table","LC","trap geometry","detuning",
        ("MOT","optical molasses","dipole trap","lattice","Paul hybrid","ring trap","sheet trap","tweezer array"),
        ("dark SPOT","gray molasses","Bessel trap","crossed dipole","holographic array","axicon ring","standing wave","painted trap"),
        ("-2 MHz","-4 MHz","-6 MHz","-8 MHz","-10 MHz","-12 MHz","-14 MHz","-16 MHz"),
        ("-3 MHz","-5 MHz","-7 MHz","-9 MHz","-11 MHz","-13 MHz","-15 MHz","-17 MHz"),28001,201000),
    _Domain("biogas_scrubber","biogas scrubber","BS","packing media","gas flow",
        ("ceramic rings","plastic saddles","structured PP","activated carbon","zeolite","biochar","stainless mesh","foam ceramic"),
        ("Pall rings","metal saddles","woven mesh","graphitic carbon","molecular sieve","wood biochar","PTFE mesh","SiC foam"),
        ("40 Nm3/h","60 Nm3/h","80 Nm3/h","100 Nm3/h","120 Nm3/h","140 Nm3/h","160 Nm3/h","180 Nm3/h"),
        ("50 Nm3/h","70 Nm3/h","90 Nm3/h","110 Nm3/h","130 Nm3/h","150 Nm3/h","170 Nm3/h","190 Nm3/h"),28101,202000),
    _Domain("acoustic_array","underwater acoustic array","AA","hydrophone element","sample rate",
        ("piezo ceramic","PVDF","fiber optic","MEMS","quartz","piezocomposite","vector sensor","capacitive"),
        ("PZT-5H","PVDF-TrFE","FBG","CMUT","PMN-PT","single crystal","triaxial vector","optical interferometric"),
        ("24 kHz","48 kHz","72 kHz","96 kHz","120 kHz","144 kHz","168 kHz","192 kHz"),
        ("30 kHz","54 kHz","78 kHz","102 kHz","126 kHz","150 kHz","174 kHz","198 kHz"),28201,203000),
    _Domain("plasma_diagnostics","plasma diagnostics rack","PD","probe type","sweep voltage",
        ("Langmuir","Mach","B-dot","emissive","retarding field","triple probe","hairpin","microwave"),
        ("compensated Langmuir","double probe","Rogowski","heated emissive","energy analyzer","quad probe","resonant hairpin","reflectometry"),
        ("20 V","40 V","60 V","80 V","100 V","120 V","140 V","160 V"),
        ("30 V","50 V","70 V","90 V","110 V","130 V","150 V","170 V"),28301,204000),
    _Domain("precision_scale","precision mass comparator","PS","load cell","settling window",
        ("EMFC","strain gauge","capacitive","quartz","piezoelectric","magnetic force","optical","resonant"),
        ("monolithic EMFC","foil gauge","MEMS capacitive","QCM","PZT stack","voice-coil","interferometric","vibrating beam"),
        ("2 s","4 s","6 s","8 s","10 s","12 s","14 s","16 s"),
        ("3 s","5 s","7 s","9 s","11 s","13 s","15 s","17 s"),28401,205000),
    _Domain("solid_state_laser","solid-state laser head","SL","gain medium","rep rate",
        ("Nd:YAG","Yb:YAG","Ti:sapphire","Nd:YVO4","Er:YAG","Cr:ZnSe","Tm:YLF","Ho:YAG"),
        ("Yb:KGW","Nd:GdVO4","Cr:LiSAF","Yb:CALGO","Er:YLF","Cr:ZnS","Tm:YAG","Ho:YLF"),
        ("1 kHz","2 kHz","3 kHz","4 kHz","5 kHz","6 kHz","7 kHz","8 kHz"),
        ("1.5 kHz","2.5 kHz","3.5 kHz","4.5 kHz","5.5 kHz","6.5 kHz","7.5 kHz","8.5 kHz"),28501,206000),
    _Domain("water_electrolysis","water electrolysis skid","WE","separator type","cell current",
        ("PEM","AEM","diaphragm","ceramic","porous PTFE","zirfon","ionomer","woven polymer"),
        ("reinforced PEM","FAA-3","asbestos-free diaphragm","YSZ sheet","ePTFE","PPS felt","PBI membrane","PEEK mesh"),
        ("100 A","150 A","200 A","250 A","300 A","350 A","400 A","450 A"),
        ("125 A","175 A","225 A","275 A","325 A","375 A","425 A","475 A"),28601,207000),
    _Domain("satellite_link","satellite link terminal","STL","antenna type","symbol rate",
        ("parabolic","phased array","helical","horn","patch","reflectarray","lens","slot array"),
        ("flat panel","digital beamforming array","quadrifilar helix","corrugated horn","stacked patch","metasurface reflectarray","Luneburg lens","waveguide slot"),
        ("2 Msps","4 Msps","6 Msps","8 Msps","10 Msps","12 Msps","14 Msps","16 Msps"),
        ("3 Msps","5 Msps","7 Msps","9 Msps","11 Msps","13 Msps","15 Msps","17 Msps"),28701,208000),
    _Domain("powder_bed","powder-bed furnace","PB","powder alloy","soak temperature",
        ("316L","Inconel 718","Ti-6Al-4V","AlSi10Mg","CoCr","maraging 300","Hastelloy X","CuCrZr"),
        ("17-4PH","Inconel 625","Ti-5553","Scalmalloy","CoCrMo","M2 HSS","Haynes 282","GRCop-42"),
        ("600 C","700 C","800 C","900 C","1000 C","1100 C","1200 C","1300 C"),
        ("650 C","750 C","850 C","950 C","1050 C","1150 C","1250 C","1350 C"),28801,209000),
    _Domain("drone_payload","drone payload bay","DP","camera type","gimbal limit",
        ("RGB","thermal","multispectral","hyperspectral","event","SWIR","LiDAR camera","stereo"),
        ("global-shutter RGB","LWIR","five-band","pushbroom","neuromorphic","MWIR","flash LiDAR","polarimetric stereo"),
        ("20 deg","30 deg","40 deg","50 deg","60 deg","70 deg","80 deg","90 deg"),
        ("25 deg","35 deg","45 deg","55 deg","65 deg","75 deg","85 deg","95 deg"),28901,210000),
    _Domain("soil_reactor","soil remediation reactor","SR","oxidant","injection rate",
        ("persulfate","permanganate","peroxide","ozone","Fenton reagent","air","steam","nitrate"),
        ("activated persulfate","sodium permanganate","calcium peroxide","oxygen","chelated Fenton","biosparge air","superheated steam","sulfate"),
        ("2 L/min","4 L/min","6 L/min","8 L/min","10 L/min","12 L/min","14 L/min","16 L/min"),
        ("3 L/min","5 L/min","7 L/min","9 L/min","11 L/min","13 L/min","15 L/min","17 L/min"),29001,211000),
    _Domain("cryostat_stage","cryostat sample stage","CS","mount material","base temperature",
        ("copper","sapphire","aluminum","silicon","molybdenum","beryllium copper","Macor","diamond"),
        ("OFHC copper","SiC","6061 aluminum","fused silica","tungsten","phosphor bronze","alumina","CVD diamond"),
        ("0.1 K","0.2 K","0.3 K","0.4 K","0.5 K","0.6 K","0.7 K","0.8 K"),
        ("0.15 K","0.25 K","0.35 K","0.45 K","0.55 K","0.65 K","0.75 K","0.85 K"),29101,212000),
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
            f"S17-Balance {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"S17-Balance record {code} lists {second} beside {spec.field_b}. "
            f"In the same {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S17-Balance {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S17-Balance record {code}, which entry belongs under {spec.field_a}?"
        qb1 = f"For S17-Balance {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S17-Balance record {code}, which entry belongs under {spec.field_b}?"
    else:
        state_a = (
            f"S17-Balance inspection dossier {code} for the {spec.noun} places {first} "
            f"under {spec.field_a}; the {spec.field_b} entry reads {second}."
        )
        state_b = (
            f"S17-Balance dossier {code} marks {second} for {spec.field_b}. "
            f"The same {spec.noun} dossier tags {first} as {spec.field_a}."
        )
        qa1 = f"From S17-Balance dossier {code}, identify {spec.field_a}."
        qa2 = f"Which S17-Balance dossier entry is tagged {spec.field_a} for {code}?"
        qb1 = f"From S17-Balance dossier {code}, identify {spec.field_b}."
        qb2 = f"Which S17-Balance dossier entry is tagged {spec.field_b} for {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S17FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s17-{spec.name}-{index:03d}"

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
    return S17FusionCase(
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


def generate_s17_cases(split: Split) -> tuple[S17FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s17_partitions(
    train: tuple[S17FusionCase, ...],
    dev: tuple[S17FusionCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S17 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S17 must remain English-only")

    domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != domains:
        raise RuntimeError("Hira v1 S17 TRAIN domains changed")
    if {row.domain for row in dev} != domains:
        raise RuntimeError("Hira v1 S17 DEV domains changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("S17 duplicate case IDs")
        for row in rows:
            if row.state_a == row.state_b:
                raise RuntimeError("S17 state views must differ")
            if row.question_a1 == row.question_a2 or row.question_b1 == row.question_b2:
                raise RuntimeError("S17 question views must differ")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("S17 option IDs must be opaque unique K=4")
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("S17 requires two semantic views for K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("S17 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("S17 gold index out of range")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    if train_states & dev_states:
        raise RuntimeError("S17 TRAIN/DEV state-view overlap")

    train_questions = {
        x for row in train
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    dev_questions = {
        x for row in dev
        for x in (row.question_a1, row.question_a2, row.question_b1, row.question_b2)
    }
    if train_questions & dev_questions:
        raise RuntimeError("S17 TRAIN/DEV question-view overlap")

    train_options = {x for row in train for x in row.option_texts}
    dev_options = {x for row in dev for x in row.option_texts}
    if train_options & dev_options:
        raise RuntimeError("S17 TRAIN/DEV option-text overlap")

    train_aliases = {x for row in train for x in row.option_aliases}
    dev_aliases = {x for row in dev for x in row.option_aliases}
    if train_aliases & dev_aliases:
        raise RuntimeError("S17 TRAIN/DEV option-alias overlap")


__all__ = [
    "S17FusionCase",
    "generate_s17_cases",
    "validate_s17_partitions",
]
