from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]

@dataclass(frozen=True)
class S24FusionCase:
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
    _Domain("eis_cell","electrochemical impedance cell","EI","reference electrode","AC amplitude",
        ("Ag/AgCl","Hg/HgO","calomel","Li metal","Na metal","Pt pseudo","Au pseudo","RHE",),
        ("silver chloride","mercury oxide","saturated calomel","lithium foil","sodium foil","platinum wire","gold wire","reversible hydrogen",),
        ("5 mV","10 mV","15 mV","20 mV","25 mV","30 mV","35 mV","40 mV",),
        ("7 mV","12 mV","17 mV","22 mV","27 mV","32 mV","37 mV","42 mV",),43001,321000),
    _Domain("afm_force","AFM force-volume rig","AF","cantilever coating","approach speed",
        ("gold","platinum","silicon nitride","diamond","aluminum","chromium","titanium","graphene",),
        ("Au-coated","Pt-coated","SiN","diamond-like","Al-coated","Cr-coated","Ti-coated","graphitic",),
        ("0.2 um/s","0.5 um/s","1 um/s","2 um/s","5 um/s","10 um/s","20 um/s","50 um/s",),
        ("0.3 um/s","0.7 um/s","1.5 um/s","3 um/s","7 um/s","15 um/s","30 um/s","70 um/s",),43101,322000),
    _Domain("lif_detector","laser-induced fluorescence detector","LF","emission filter","PMT gain",
        ("450 nm","500 nm","550 nm","600 nm","650 nm","700 nm","750 nm","800 nm",),
        ("460 nm bandpass","510 nm bandpass","560 nm bandpass","610 nm bandpass","660 nm bandpass","710 nm bandpass","760 nm bandpass","810 nm bandpass",),
        ("400 V","450 V","500 V","550 V","600 V","650 V","700 V","750 V",),
        ("420 V","470 V","520 V","570 V","620 V","670 V","720 V","770 V",),43201,323000),
    _Domain("sorption_balance","gravimetric sorption analyzer","GS","carrier gas","equilibration criterion",
        ("nitrogen","argon","helium","carbon dioxide","methane","hydrogen","oxygen","air",),
        ("dry N2","ultrapure Ar","He carrier","CO2 stream","CH4 stream","H2 stream","O2 stream","zero air",),
        ("0.01 %/min","0.02 %/min","0.03 %/min","0.04 %/min","0.05 %/min","0.06 %/min","0.07 %/min","0.08 %/min",),
        ("0.015 %/min","0.025 %/min","0.035 %/min","0.045 %/min","0.055 %/min","0.065 %/min","0.075 %/min","0.085 %/min",),43301,324000),
    _Domain("microct","micro-CT scanner","CT","beam filter","voxel size",
        ("aluminum","copper","tin","brass","titanium","molybdenum","silver","none",),
        ("0.5 mm Al","0.2 mm Cu","0.1 mm Sn","0.3 mm brass","0.1 mm Ti","0.05 mm Mo","0.03 mm Ag","open beam",),
        ("2 um","4 um","6 um","8 um","10 um","12 um","14 um","16 um",),
        ("3 um","5 um","7 um","9 um","11 um","13 um","15 um","17 um",),43401,325000),
    _Domain("xps_stage","XPS analysis stage","XP","charge neutralizer","pass energy",
        ("electron flood","ion flood","dual beam","low-energy electron","Ar ion","none","UV neutralizer","plasma",),
        ("e-flood gun","low-energy ion gun","dual neutralizer","LEED flood","argon beam","disabled","UV lamp","plasma flood",),
        ("10 eV","20 eV","30 eV","40 eV","50 eV","60 eV","80 eV","100 eV",),
        ("15 eV","25 eV","35 eV","45 eV","55 eV","70 eV","90 eV","120 eV",),43501,326000),
    _Domain("capillary_electro","capillary electrophoresis unit","CE","capillary coating","separation voltage",
        ("bare silica","polyacrylamide","PEG","PVA","cationic","anionic","hydrophobic","zwitterionic",),
        ("fused silica","PA-coated","PEGylated","PVA-coated","amine-coated","sulfonate-coated","C18-modified","zwitterion-coated",),
        ("5 kV","10 kV","15 kV","20 kV","25 kV","30 kV","35 kV","40 kV",),
        ("7 kV","12 kV","17 kV","22 kV","27 kV","32 kV","37 kV","42 kV",),43601,327000),
    _Domain("oct_system","optical coherence tomography system","OC","scan protocol","A-scan rate",
        ("raster","radial","spiral","line","volumetric","angiography","Doppler","adaptive",),
        ("Cartesian raster","polar radial","Archimedean spiral","single line","3D volume","OCTA","phase Doppler","motion-adaptive",),
        ("20 kHz","40 kHz","60 kHz","80 kHz","100 kHz","120 kHz","160 kHz","200 kHz",),
        ("30 kHz","50 kHz","70 kHz","90 kHz","110 kHz","140 kHz","180 kHz","220 kHz",),43701,328000),
    _Domain("kelvin_probe","scanning Kelvin probe","KP","tip material","lift height",
        ("tungsten","platinum","gold","nickel","stainless steel","iridium","copper","carbon",),
        ("W tip","Pt tip","Au tip","Ni tip","steel tip","Ir tip","Cu tip","carbon fiber",),
        ("10 nm","20 nm","30 nm","40 nm","50 nm","60 nm","80 nm","100 nm",),
        ("15 nm","25 nm","35 nm","45 nm","55 nm","70 nm","90 nm","120 nm",),43801,329000),
    _Domain("flow_cytometer","spectral flow cytometer","FC","trigger channel","sample pressure",
        ("FSC","SSC","violet","blue","yellow-green","red","UV","near-IR",),
        ("forward scatter","side scatter","405 nm","488 nm","561 nm","640 nm","355 nm","730 nm",),
        ("5 psi","7 psi","9 psi","11 psi","13 psi","15 psi","17 psi","19 psi",),
        ("6 psi","8 psi","10 psi","12 psi","14 psi","16 psi","18 psi","20 psi",),43901,330000),
    _Domain("nanoindent","nanoindenter","NI","tip geometry","loading rate",
        ("Berkovich","cube corner","Vickers","spherical","conical","flat punch","Knoop","wedge",),
        ("three-sided Berkovich","cube-corner","four-sided Vickers","10 um sphere","60 deg cone","20 um flat punch","Knoop diamond","line wedge",),
        ("0.1 mN/s","0.2 mN/s","0.5 mN/s","1 mN/s","2 mN/s","5 mN/s","10 mN/s","20 mN/s",),
        ("0.15 mN/s","0.3 mN/s","0.7 mN/s","1.5 mN/s","3 mN/s","7 mN/s","15 mN/s","30 mN/s",),44001,331000),
    _Domain("flim","fluorescence lifetime microscope","FL","timing mode","bin width",
        ("TCSPC","frequency-domain","gated","streak","phase","rapidFLIM","time-gated SPAD","analog",),
        ("time-correlated counting","modulated phase","intensified gate","streak camera","phase shift","rapid lifetime","SPAD gating","analog decay",),
        ("4 ps","8 ps","16 ps","32 ps","64 ps","128 ps","256 ps","512 ps",),
        ("6 ps","12 ps","24 ps","48 ps","96 ps","192 ps","384 ps","768 ps",),44101,332000),
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
    aliases = tuple(f"{value} is the logged {field} setting for this {noun}" for _k, field, value in rows)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i,(kind,_f,_v) in enumerate(rows) if kind == "a")
    gb = next(i for i,(kind,_f,_v) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb

def _views(spec: _Domain, *, split: Split, code: str, first: str, second: str):
    if split == "train":
        state_a = (
            f"S24-Reliability {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"S24-Reliability record {code} lists {second} for {spec.field_b}. "
            f"The same {spec.noun} assigns {first} to {spec.field_a}."
        )
        qa1 = f"For S24-Reliability {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S24-Reliability record {code}, which entry is {spec.field_a}?"
        qb1 = f"For S24-Reliability {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S24-Reliability record {code}, which entry is {spec.field_b}?"
    else:
        state_a = (
            f"S24-Reliability audit {code} for the {spec.noun} documents "
            f"{first} as {spec.field_a} and {second} as {spec.field_b}."
        )
        state_b = (
            f"Audit {code} says the {spec.field_b} setting is {second}. "
            f"Separately, this {spec.noun} records {first} for {spec.field_a}."
        )
        qa1 = f"Read S24-Reliability audit {code}: what is the {spec.field_a} entry?"
        qa2 = f"Which value is tagged {spec.field_a} on S24-Reliability audit {code}?"
        qb1 = f"Read S24-Reliability audit {code}: what is the {spec.field_b} entry?"
        qb2 = f"Which value is tagged {spec.field_b} on S24-Reliability audit {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2

def _build(spec: _Domain, index: int, split: Split) -> S24FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s24-{spec.name}-{index:03d}"
    state_a, state_b, qa1, qa2, qb1, qb2 = _views(
        spec, split=split, code=code, first=first, second=second
    )
    texts, aliases, ids, ga, gb = _shuffle(
        case_id=case_id, noun=spec.noun, field_a=spec.field_a, field_b=spec.field_b,
        first=first, second=second, distractor_first=distractor_first,
        distractor_second=distractor_second, seed=spec.seed + offset,
    )
    return S24FusionCase(
        case_id=case_id, split=split, domain=spec.name, language="en",
        state_a=state_a, state_b=state_b, question_a1=qa1, question_a2=qa2,
        question_b1=qb1, question_b2=qb2, option_texts=texts,
        option_aliases=aliases, option_ids=ids, gold_a=ga, gold_b=gb,
    )

def generate_s24_cases(split: Split) -> tuple[S24FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(_build(spec, index, split) for spec in _DOMAINS for index in range(count))

def validate_s24_partitions(train: tuple[S24FusionCase, ...], dev: tuple[S24FusionCase, ...]) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S24 partition size changed")
    if len({r.domain for r in train}) != 12 or len({r.domain for r in dev}) != 12:
        raise RuntimeError("S24 domain count changed")
    if any(r.split != "train" for r in train) or any(r.split != "dev" for r in dev):
        raise RuntimeError("S24 split labels changed")
    if any(r.language != "en" for r in (*train, *dev)):
        raise RuntimeError("S24 language changed")
    if any(len(r.option_texts)!=4 or len(r.option_aliases)!=4 or len(r.option_ids)!=4 for r in (*train,*dev)):
        raise RuntimeError("S24 option cardinality changed")
    train_states={x for r in train for x in (r.state_a,r.state_b)}
    dev_states={x for r in dev for x in (r.state_a,r.state_b)}
    train_q={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dev_q={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    train_opts={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    dev_opts={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if train_states & dev_states: raise RuntimeError("S24 TRAIN/DEV exact state overlap")
    if train_q & dev_q: raise RuntimeError("S24 TRAIN/DEV exact question overlap")
    if train_opts & dev_opts: raise RuntimeError("S24 TRAIN/DEV exact option overlap")

__all__=["S24FusionCase","generate_s24_cases","validate_s24_partitions"]
