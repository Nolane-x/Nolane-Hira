from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S31GlobalCase:
    case_id:str
    split:Split
    domain:str
    language:str
    state_a:str
    state_b:str
    question_a1:str
    question_a2:str
    question_b1:str
    question_b2:str
    option_texts:tuple[str,...]
    option_aliases:tuple[str,...]
    option_ids:tuple[str,...]
    gold_a:int
    gold_b:int

    def to_dict(self)->dict:
        row=asdict(self)
        row["option_texts"]=list(self.option_texts)
        row["option_aliases"]=list(self.option_aliases)
        row["option_ids"]=list(self.option_ids)
        return row


@dataclass(frozen=True)
class _Domain:
    name:str
    noun:str
    prefix:str
    field_a:str
    field_b:str
    train_a:tuple[str,...]
    dev_a:tuple[str,...]
    train_b:tuple[str,...]
    dev_b:tuple[str,...]
    seed:int
    dev_offset:int


_DOMAINS=(
    _Domain("raman_microscope","Raman microscope","RM","excitation","spectral resolution",
        ("488 nm","514 nm","532 nm","633 nm","660 nm","785 nm","830 nm","1064 nm"),
        ("457 nm","505 nm","561 nm","594 nm","685 nm","730 nm","850 nm","980 nm"),
        ("0.5 cm-1","1 cm-1","1.5 cm-1","2 cm-1","2.5 cm-1","3 cm-1","3.5 cm-1","4 cm-1"),
        ("0.75 cm-1","1.25 cm-1","1.75 cm-1","2.25 cm-1","2.75 cm-1","3.25 cm-1","3.75 cm-1","4.25 cm-1"),56101,521000),
    _Domain("atomic_clock_servo","atomic-clock servo","AS","reference transition","loop bandwidth",
        ("Rb D1","Rb D2","Cs D1","Cs D2","Sr clock","Yb clock","Hg clock","Ca clock"),
        ("Rb CPT","Cs CPT","Sr intercombination","Yb intercombination","Al+ clock","In+ clock","Mg clock","Ca+ clock"),
        ("1 Hz","2 Hz","3 Hz","4 Hz","5 Hz","6 Hz","7 Hz","8 Hz"),
        ("1.5 Hz","2.5 Hz","3.5 Hz","4.5 Hz","5.5 Hz","6.5 Hz","7.5 Hz","8.5 Hz"),56201,522000),
    _Domain("microreactor","microreactor","MR","catalyst","residence time",
        ("Pt","Pd","Ni","Cu","Ru","Rh","Co","Fe"),
        ("PtRu","PdAu","NiMo","CuZn","RuC","RhAl","CoFe","FeCr"),
        ("2 s","4 s","6 s","8 s","10 s","12 s","14 s","16 s"),
        ("3 s","5 s","7 s","9 s","11 s","13 s","15 s","17 s"),56301,523000),
    _Domain("terahertz_source","terahertz source","TH","emitter","center frequency",
        ("photoconductive","optical rectification","QCL","Gunn","Schottky multiplier","BWO","plasma","spintronic"),
        ("LT-GaAs PCA","ZnTe OR","GaAs QCL","IMPATT","diode chain","gyro-BWO","laser plasma","CoFeB spintronic"),
        ("0.2 THz","0.4 THz","0.6 THz","0.8 THz","1.0 THz","1.2 THz","1.4 THz","1.6 THz"),
        ("0.3 THz","0.5 THz","0.7 THz","0.9 THz","1.1 THz","1.3 THz","1.5 THz","1.7 THz"),56401,524000),
    _Domain("force_sensor","precision force sensor","FS","transducer","range",
        ("strain gauge","piezoelectric","capacitive","optical fiber","magnetic","quartz","MEMS","resonant"),
        ("foil strain","PZT","differential capacitive","FBG","Hall flexure","quartz fork","silicon MEMS","vibrating beam"),
        ("1 N","2 N","3 N","4 N","5 N","6 N","7 N","8 N"),
        ("1.5 N","2.5 N","3.5 N","4.5 N","5.5 N","6.5 N","7.5 N","8.5 N"),56501,525000),
    _Domain("optical_modulator","optical modulator","OM","modulator type","half-wave voltage",
        ("LiNbO3","silicon MZI","EAM","Pockels cell","AOM","MEMS","polymer EO","plasmonic"),
        ("thin-film LiNbO3","SiN MZI","InP EAM","KDP cell","TeO2 AOM","DMD","organic EO","ITO plasmonic"),
        ("1 V","2 V","3 V","4 V","5 V","6 V","7 V","8 V"),
        ("1.5 V","2.5 V","3.5 V","4.5 V","5.5 V","6.5 V","7.5 V","8.5 V"),56601,526000),
    _Domain("neutron_detector","neutron detector","ND","converter","bias voltage",
        ("He-3","B-10","Li-6","Gd","BF3","scintillator","diamond","fission chamber"),
        ("B4C","LiF","Gd2O3","boron film","ZnS-LiF","plastic scintillator","CVD diamond","U-235 chamber"),
        ("100 V","200 V","300 V","400 V","500 V","600 V","700 V","800 V"),
        ("150 V","250 V","350 V","450 V","550 V","650 V","750 V","850 V"),56701,527000),
    _Domain("mass_flow_controller","mass-flow controller","MF","sensor","full scale",
        ("thermal","Coriolis","dP","ultrasonic","laminar","MEMS thermal","vortex","capillary"),
        ("MEMS calorimetric","micro-Coriolis","silicon dP","transit-time","LFE","hot-film","micro-vortex","bubble-free capillary"),
        ("10 sccm","20 sccm","30 sccm","40 sccm","50 sccm","60 sccm","70 sccm","80 sccm"),
        ("15 sccm","25 sccm","35 sccm","45 sccm","55 sccm","65 sccm","75 sccm","85 sccm"),56801,528000),
    _Domain("electron_spectrometer","electron spectrometer","ES","analyzer","pass energy",
        ("hemispherical","cylindrical mirror","TOF","retarding field","magnetic sector","electrostatic sector","Wien","velocity map"),
        ("double hemisphere","CMA","reflectron TOF","RFA grid","sector magnet","ESA","crossed-field","VMI lens"),
        ("5 eV","10 eV","15 eV","20 eV","25 eV","30 eV","35 eV","40 eV"),
        ("7 eV","12 eV","17 eV","22 eV","27 eV","32 eV","37 eV","42 eV"),56901,529000),
    _Domain("laser_rangefinder","laser rangefinder","LR","wavelength","pulse energy",
        ("905 nm","1064 nm","1310 nm","1550 nm","532 nm","650 nm","808 nm","940 nm"),
        ("850 nm","980 nm","1270 nm","1625 nm","515 nm","635 nm","780 nm","1030 nm"),
        ("1 uJ","2 uJ","3 uJ","4 uJ","5 uJ","6 uJ","7 uJ","8 uJ"),
        ("1.5 uJ","2.5 uJ","3.5 uJ","4.5 uJ","5.5 uJ","6.5 uJ","7.5 uJ","8.5 uJ"),57001,530000),
    _Domain("magnet_power_supply","magnet power supply","MS","topology","current ripple",
        ("linear","buck","full bridge","interleaved","resonant","SCR","PWM","four-quadrant"),
        ("pass-bank","multiphase buck","H-bridge","six-phase","LLC","thyristor","sigma-delta PWM","regenerative"),
        ("1 ppm","2 ppm","3 ppm","4 ppm","5 ppm","6 ppm","7 ppm","8 ppm"),
        ("1.5 ppm","2.5 ppm","3.5 ppm","4.5 ppm","5.5 ppm","6.5 ppm","7.5 ppm","8.5 ppm"),57101,531000),
    _Domain("fluorescence_reader","fluorescence reader","FR","detector","integration",
        ("PMT","SiPM","CCD","CMOS","APD","photodiode","EMCCD","SPAD array"),
        ("GaAsP PMT","digital SiPM","sCMOS","BSI CMOS","InGaAs APD","PIN diode","deep EMCCD","dSiPM array"),
        ("10 ms","20 ms","30 ms","40 ms","50 ms","60 ms","70 ms","80 ms"),
        ("15 ms","25 ms","35 ms","45 ms","55 ms","65 ms","75 ms","85 ms"),57201,532000),
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
            f"S31-Global {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S31-Global record {code} stores {second} under {spec.field_b}. The same {spec.noun} assigns {first} to {spec.field_a}.",
            f"For S31-Global {spec.noun} {code}, identify {spec.field_a}.",
            f"In S31-Global record {code}, which entry belongs under {spec.field_a}?",
            f"For S31-Global {spec.noun} {code}, identify {spec.field_b}.",
            f"In S31-Global record {code}, which entry belongs under {spec.field_b}?",
        )
    return (
        f"S31-Global audit {code} for the {spec.noun} records {first} as {spec.field_a} and {second} as {spec.field_b}.",
        f"Audit {code} marks {second} for {spec.field_b}; separately, the S31-Global {spec.noun} lists {first} for {spec.field_a}.",
        f"Read S31-Global audit {code}: what is {spec.field_a}?",
        f"Which audit value is tagged {spec.field_a} for {code}?",
        f"Read S31-Global audit {code}: what is {spec.field_b}?",
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
    case_id=f"{split}-s31-{spec.name}-{index:03d}"
    state_a,state_b,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(case_id=case_id,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,first=first,second=second,distractor_first=df,distractor_second=ds,seed=spec.seed+offset)
    return S31GlobalCase(case_id,split,spec.name,"en",state_a,state_b,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb)


def generate_s31_cases(split:Split)->tuple[S31GlobalCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s31_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S31 partition size changed")
    domains={d.name for d in _DOMAINS}
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains: raise RuntimeError("S31 domains changed")
    if any(r.language!="en" for r in (*train,*dev)): raise RuntimeError("S31 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows): raise RuntimeError("S31 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S31 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2: raise RuntimeError("S31 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4: raise RuntimeError("S31 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S31 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}
    ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds: raise RuntimeError("S31 TRAIN DEV state overlap")
    if tq&dq: raise RuntimeError("S31 TRAIN DEV question overlap")
    if to&do: raise RuntimeError("S31 TRAIN DEV option overlap")


__all__=["S31GlobalCase","generate_s31_cases","validate_s31_partitions"]
