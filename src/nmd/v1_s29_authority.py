from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train","dev"]


@dataclass(frozen=True)
class S29FullBlockCase:
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
    option_texts: tuple[str,...]
    option_aliases: tuple[str,...]
    option_ids: tuple[str,...]
    gold_a: int
    gold_b: int

    def to_dict(self) -> dict:
        row=asdict(self)
        row["option_texts"]=list(self.option_texts)
        row["option_aliases"]=list(self.option_aliases)
        row["option_ids"]=list(self.option_ids)
        return row


@dataclass(frozen=True)
class _Domain:
    name: str
    noun: str
    prefix: str
    field_a: str
    field_b: str
    train_a: tuple[str,...]
    dev_a: tuple[str,...]
    train_b: tuple[str,...]
    dev_b: tuple[str,...]
    seed: int
    dev_offset: int


_DOMAINS=(
    _Domain("quantum_magnetometer","quantum magnetometer","QM","cell species","bias field",
        ("Rb-87","Cs-133","K-39","He-3","Xe-129","Ne-21","Rb-85","K-41"),
        ("Cs-131","Xe-131","He-4","Ne-22","Kr-83","Ar-40","Rb-82","K-40"),
        ("2 uT","4 uT","6 uT","8 uT","10 uT","12 uT","14 uT","16 uT"),
        ("3 uT","5 uT","7 uT","9 uT","11 uT","13 uT","15 uT","17 uT"),51301,463000),
    _Domain("photonic_switch","photonic switch fabric","PF","switch mode","slot time",
        ("bar","cross","broadcast","drop","add","loopback","split","merge"),
        ("bar-lock","cross-lock","multicast","bypass","insert","return","fanout","combine"),
        ("2 ns","4 ns","6 ns","8 ns","10 ns","12 ns","14 ns","16 ns"),
        ("3 ns","5 ns","7 ns","9 ns","11 ns","13 ns","15 ns","17 ns"),51401,464000),
    _Domain("micro_gc","micro gas chromatograph","MG","column coating","carrier pressure",
        ("PDMS","PEG","molecular sieve","PLOT-Q","OV-1","Carbowax","Tenax","Porapak"),
        ("PDMS-DVB","PEG-20M","Molsieve 5A","PLOT-U","SE-30","FFAP","Tenax TA","Hayesep"),
        ("20 kPa","30 kPa","40 kPa","50 kPa","60 kPa","70 kPa","80 kPa","90 kPa"),
        ("25 kPa","35 kPa","45 kPa","55 kPa","65 kPa","75 kPa","85 kPa","95 kPa"),51501,465000),
    _Domain("beam_steerer","acousto-optic beam steerer","AB","crystal","drive frequency",
        ("TeO2","quartz","LiNbO3","Ge","GaP","KDP","BBO","fused silica"),
        ("PbMoO4","LiTaO3","ZnO","Si","GaAs","ADP","LBO","sapphire"),
        ("40 MHz","60 MHz","80 MHz","100 MHz","120 MHz","140 MHz","160 MHz","180 MHz"),
        ("50 MHz","70 MHz","90 MHz","110 MHz","130 MHz","150 MHz","170 MHz","190 MHz"),51601,466000),
    _Domain("nanoindenter","nanoindenter controller","NI","tip geometry","loading rate",
        ("Berkovich","cube corner","spherical","Vickers","flat punch","conical","Knoop","wedge"),
        ("modified Berkovich","cono-spherical","diamond sphere","micro-Vickers","circular punch","sharp cone","elongated Knoop","blade"),
        ("1 mN/s","2 mN/s","3 mN/s","4 mN/s","5 mN/s","6 mN/s","7 mN/s","8 mN/s"),
        ("1.5 mN/s","2.5 mN/s","3.5 mN/s","4.5 mN/s","5.5 mN/s","6.5 mN/s","7.5 mN/s","8.5 mN/s"),51701,467000),
    _Domain("frequency_comb","frequency-comb controller","FC","lock reference","repetition rate",
        ("Rb clock","GPSDO","maser","optical cavity","Cs clock","OCXO","hydrogen line","ULE cavity"),
        ("dual Rb","GNSSDO","active maser","cryogenic cavity","fountain Cs","SC-cut OCXO","ammonia line","silicon cavity"),
        ("80 MHz","100 MHz","120 MHz","140 MHz","160 MHz","180 MHz","200 MHz","220 MHz"),
        ("90 MHz","110 MHz","130 MHz","150 MHz","170 MHz","190 MHz","210 MHz","230 MHz"),51801,468000),
    _Domain("micro_ct","micro-CT scanner","CT","filter","tube current",
        ("aluminum","copper","tin","silver","molybdenum","tungsten","beryllium","carbon"),
        ("Al-Cu","Cu-Sn","brass","Ag-Pd","Mo-Re","W-Cu","Be-Cu","graphite"),
        ("50 uA","100 uA","150 uA","200 uA","250 uA","300 uA","350 uA","400 uA"),
        ("75 uA","125 uA","175 uA","225 uA","275 uA","325 uA","375 uA","425 uA"),51901,469000),
    _Domain("ion_source","ion source controller","IS","source type","extraction voltage",
        ("duoplasmatron","ECR","Penning","RF","Kaufman","hollow cathode","field emission","laser ablation"),
        ("multicusp","helicon","magnetron","ICP","gridded","thermionic","Spindt","MALDI"),
        ("2 kV","4 kV","6 kV","8 kV","10 kV","12 kV","14 kV","16 kV"),
        ("3 kV","5 kV","7 kV","9 kV","11 kV","13 kV","15 kV","17 kV"),52001,470000),
    _Domain("micro_calorimeter","microcalorimeter array","MC","absorber","bath temperature",
        ("gold","bismuth","tin","copper","tantalum","platinum","aluminum","silicon"),
        ("Au-Bi","Bi-Sn","indium","OFHC copper","niobium","Pt-Ir","Al-Mn","SiN"),
        ("40 mK","50 mK","60 mK","70 mK","80 mK","90 mK","100 mK","110 mK"),
        ("45 mK","55 mK","65 mK","75 mK","85 mK","95 mK","105 mK","115 mK"),52101,471000),
    _Domain("waveguide_tuner","microwave waveguide tuner","WT","stub mode","phase offset",
        ("single","double","triple","sliding","rotary","iris","plunger","hybrid"),
        ("single-lock","double-lock","triple-lock","vernier","rotary-lock","dual-iris","fine plunger","hybrid-lock"),
        ("10 deg","20 deg","30 deg","40 deg","50 deg","60 deg","70 deg","80 deg"),
        ("15 deg","25 deg","35 deg","45 deg","55 deg","65 deg","75 deg","85 deg"),52201,472000),
    _Domain("spectral_camera","spectral camera","SC","sensor","frame rate",
        ("InGaAs","sCMOS","EMCCD","HgCdTe","Si CCD","QWIP","microbolometer","SPAD"),
        ("extended InGaAs","BSI sCMOS","deep EMCCD","MCT","back-thinned CCD","T2SL","VOx","SiPM"),
        ("20 fps","40 fps","60 fps","80 fps","100 fps","120 fps","140 fps","160 fps"),
        ("30 fps","50 fps","70 fps","90 fps","110 fps","130 fps","150 fps","170 fps"),52301,473000),
    _Domain("thinfilm_monitor","thin-film monitor","TM","crystal","sampling period",
        ("quartz","sapphire","silicon","alumina","LiNbO3","SiC","diamond","glass"),
        ("AT-cut quartz","YAG","SOI","zirconia","LiTaO3","GaN","CVD diamond","fused silica"),
        ("0.1 s","0.2 s","0.3 s","0.4 s","0.5 s","0.6 s","0.7 s","0.8 s"),
        ("0.15 s","0.25 s","0.35 s","0.45 s","0.55 s","0.65 s","0.75 s","0.85 s"),52401,474000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the recorded {field} entry for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S29-FullBlock {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S29-FullBlock record {code} puts {second} under {spec.field_b}. The same {spec.noun} assigns {first} to {spec.field_a}.",
            f"For S29-FullBlock {spec.noun} {code}, identify {spec.field_a}.",
            f"In S29-FullBlock record {code}, which entry belongs under {spec.field_a}?",
            f"For S29-FullBlock {spec.noun} {code}, identify {spec.field_b}.",
            f"In S29-FullBlock record {code}, which entry belongs under {spec.field_b}?",
        )
    return (
        f"S29-FullBlock audit {code} for the {spec.noun} records {first} as {spec.field_a} and {second} as {spec.field_b}.",
        f"Audit {code} marks {second} for {spec.field_b}; separately, the S29-FullBlock {spec.noun} lists {first} for {spec.field_a}.",
        f"Read S29-FullBlock audit {code}: what is {spec.field_a}?",
        f"Which audit value is tagged {spec.field_a} for {code}?",
        f"Read S29-FullBlock audit {code}: what is {spec.field_b}?",
        f"Which audit value is tagged {spec.field_b} for {code}?",
    )


def _build(spec,index,split):
    firsts=spec.train_a if split=="train" else spec.dev_a
    seconds=spec.train_b if split=="train" else spec.dev_b
    first=firsts[index%len(firsts)]
    second=seconds[(index*3+1)%len(seconds)]
    df=firsts[(index+3)%len(firsts)]
    ds=seconds[(index*5+2)%len(seconds)]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    case_id=f"{split}-s29-{spec.name}-{index:03d}"
    state_a,state_b,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(case_id=case_id,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,first=first,second=second,distractor_first=df,distractor_second=ds,seed=spec.seed+offset)
    return S29FullBlockCase(case_id,split,spec.name,"en",state_a,state_b,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb)


def generate_s29_cases(split: Split) -> tuple[S29FullBlockCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s29_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S29 partition size changed")
    if len({r.domain for r in train})!=12 or len({r.domain for r in dev})!=12: raise RuntimeError("S29 domain count changed")
    if any(r.language!="en" for r in (*train,*dev)): raise RuntimeError("S29 language changed")
    if any(len(r.option_ids)!=4 or len(set(r.option_ids))!=4 for r in (*train,*dev)): raise RuntimeError("S29 option IDs changed")
    ts={x for r in train for x in (r.state_a,r.state_b)}
    ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds: raise RuntimeError("S29 TRAIN DEV state overlap")
    if tq&dq: raise RuntimeError("S29 TRAIN DEV question overlap")
    if to&do: raise RuntimeError("S29 TRAIN DEV option overlap")


__all__=["S29FullBlockCase","generate_s29_cases","validate_s29_partitions"]
