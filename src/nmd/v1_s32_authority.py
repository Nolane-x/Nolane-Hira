from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S32MatrixCase:
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
    _Domain("cavity_ringdown","cavity ring-down spectrometer","CR","mirror coating","ring-down time",
        ("Ta2O5","SiO2","HfO2","Al2O3","TiO2","ZrO2","MgF2","CaF2"),
        ("Ta2O5-SiO2","HfO2-SiO2","AlN","Y2O3","Nb2O5","LaF3","BaF2","LiF"),
        ("12 us","16 us","20 us","24 us","28 us","32 us","36 us","40 us"),
        ("14 us","18 us","22 us","26 us","30 us","34 us","38 us","42 us"),59101,541000),
    _Domain("spin_resonance","spin-resonance console","SR","nucleus","pulse width",
        ("1H","13C","19F","31P","15N","29Si","7Li","23Na"),
        ("2H","11B","17O","33S","39K","25Mg","27Al","43Ca"),
        ("2 us","4 us","6 us","8 us","10 us","12 us","14 us","16 us"),
        ("3 us","5 us","7 us","9 us","11 us","13 us","15 us","17 us"),59201,542000),
    _Domain("nanopore_sensor","nanopore sensor","NP","membrane","bias voltage",
        ("SiN","graphene","MoS2","hBN","Al2O3","SiO2","polymer","mica"),
        ("SiC","WS2","WSe2","diamond","TiO2","quartz","PET","sapphire"),
        ("80 mV","100 mV","120 mV","140 mV","160 mV","180 mV","200 mV","220 mV"),
        ("90 mV","110 mV","130 mV","150 mV","170 mV","190 mV","210 mV","230 mV"),59301,543000),
    _Domain("optical_tweezer","optical tweezer","OT","trap wavelength","trap stiffness",
        ("780 nm","808 nm","850 nm","940 nm","980 nm","1030 nm","1064 nm","1550 nm"),
        ("760 nm","820 nm","880 nm","915 nm","1010 nm","1047 nm","1120 nm","1310 nm"),
        ("0.1 pN/nm","0.2 pN/nm","0.3 pN/nm","0.4 pN/nm","0.5 pN/nm","0.6 pN/nm","0.7 pN/nm","0.8 pN/nm"),
        ("0.15 pN/nm","0.25 pN/nm","0.35 pN/nm","0.45 pN/nm","0.55 pN/nm","0.65 pN/nm","0.75 pN/nm","0.85 pN/nm"),59401,544000),
    _Domain("lockin_amplifier","lock-in amplifier","LA","reference mode","time constant",
        ("internal","external TTL","external sine","dual harmonic","PLL","chopper","heterodyne","quadrature"),
        ("internal DDS","external CMOS","external RF","third harmonic","digital PLL","optical sync","dual heterodyne","IQ"),
        ("1 ms","3 ms","5 ms","7 ms","9 ms","11 ms","13 ms","15 ms"),
        ("2 ms","4 ms","6 ms","8 ms","10 ms","12 ms","14 ms","16 ms"),59501,545000),
    _Domain("microheater","microheater controller","MH","heater material","power setpoint",
        ("Pt","NiCr","TiN","W","Mo","graphene","ITO","polysilicon"),
        ("PtRh","CrAl","TaN","WRe","MoSi2","CNT","AZO","SiC"),
        ("20 mW","40 mW","60 mW","80 mW","100 mW","120 mW","140 mW","160 mW"),
        ("30 mW","50 mW","70 mW","90 mW","110 mW","130 mW","150 mW","170 mW"),59601,546000),
    _Domain("electron_gun","electron gun","EG","cathode","beam energy",
        ("LaB6","CeB6","tungsten","Schottky","field emitter","dispenser","photocathode","CNT"),
        ("TaB6","HfC","Re","ZrO","cold field","oxide cathode","GaAs photocathode","graphene emitter"),
        ("1 keV","2 keV","3 keV","4 keV","5 keV","6 keV","7 keV","8 keV"),
        ("1.5 keV","2.5 keV","3.5 keV","4.5 keV","5.5 keV","6.5 keV","7.5 keV","8.5 keV"),59701,547000),
    _Domain("polarization_controller","polarization controller","PC","actuator","retardance",
        ("fiber squeezer","LiNbO3","MEMS","liquid crystal","piezo paddle","Faraday","electro-optic","magneto-optic"),
        ("three-paddle","thin-film LN","silicon MEMS","LCOS","PZT coil","Terbium glass","KTP","YIG"),
        ("20 deg","40 deg","60 deg","80 deg","100 deg","120 deg","140 deg","160 deg"),
        ("30 deg","50 deg","70 deg","90 deg","110 deg","130 deg","150 deg","170 deg"),59801,548000),
    _Domain("cryo_digitizer","cryogenic digitizer","CD","ADC type","sample rate",
        ("SAR","pipeline","sigma-delta","flash","dual-slope","time-interleaved","Josephson","successive approximation"),
        ("charge-redistribution SAR","folding pipeline","CT sigma-delta","subranging flash","integrating","four-way interleaved","RSFQ","capacitive SAR"),
        ("10 MS/s","20 MS/s","30 MS/s","40 MS/s","50 MS/s","60 MS/s","70 MS/s","80 MS/s"),
        ("15 MS/s","25 MS/s","35 MS/s","45 MS/s","55 MS/s","65 MS/s","75 MS/s","85 MS/s"),59901,549000),
    _Domain("tunable_filter","tunable optical filter","TF","filter mechanism","bandwidth",
        ("Fabry-Perot","AOTF","LC filter","MEMS grating","ring resonator","Bragg grating","etalon","acousto-optic"),
        ("micro-etalon","TeO2 AOTF","LC Lyot","DMD grating","SiN ring","chirped FBG","solid etalon","quartz AOTF"),
        ("0.1 nm","0.2 nm","0.3 nm","0.4 nm","0.5 nm","0.6 nm","0.7 nm","0.8 nm"),
        ("0.15 nm","0.25 nm","0.35 nm","0.45 nm","0.55 nm","0.65 nm","0.75 nm","0.85 nm"),60001,550000),
    _Domain("acoustic_resonator","acoustic resonator","AR","mode","quality factor",
        ("longitudinal","shear","radial","wineglass","Lamb","Love","SAW","BAW"),
        ("thickness-extensional","torsional","breathing","elliptic","A0 Lamb","SH Love","Rayleigh SAW","FBAR"),
        ("1000","2000","3000","4000","5000","6000","7000","8000"),
        ("1500","2500","3500","4500","5500","6500","7500","8500"),60101,551000),
    _Domain("lidar_receiver","lidar receiver","LRX","detector","gate width",
        ("Si APD","InGaAs APD","SPAD","PMT","SiPM","PIN","MCT","SNSPD"),
        ("linear APD","Geiger InGaAs","SPAD array","MCP-PMT","digital SiPM","balanced PIN","HOT MCT","nanowire array"),
        ("2 ns","4 ns","6 ns","8 ns","10 ns","12 ns","14 ns","16 ns"),
        ("3 ns","5 ns","7 ns","9 ns","11 ns","13 ns","15 ns","17 ns"),60201,552000),
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
            f"S32-Matrix {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S32-Matrix record {code} stores {second} under {spec.field_b}. The same {spec.noun} assigns {first} to {spec.field_a}.",
            f"For S32-Matrix {spec.noun} {code}, identify {spec.field_a}.",
            f"In S32-Matrix record {code}, which entry belongs under {spec.field_a}?",
            f"For S32-Matrix {spec.noun} {code}, identify {spec.field_b}.",
            f"In S32-Matrix record {code}, which entry belongs under {spec.field_b}?",
        )
    return (
        f"S32-Matrix audit {code} for the {spec.noun} records {first} as {spec.field_a} and {second} as {spec.field_b}.",
        f"Audit {code} marks {second} for {spec.field_b}; separately, the S32-Matrix {spec.noun} lists {first} for {spec.field_a}.",
        f"Read S32-Matrix audit {code}: what is {spec.field_a}?",
        f"Which audit value is tagged {spec.field_a} for {code}?",
        f"Read S32-Matrix audit {code}: what is {spec.field_b}?",
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
    case_id=f"{split}-s32-{spec.name}-{index:03d}"
    state_a,state_b,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(case_id=case_id,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,first=first,second=second,distractor_first=df,distractor_second=ds,seed=spec.seed+offset)
    return S32MatrixCase(case_id,split,spec.name,"en",state_a,state_b,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb)


def generate_s32_cases(split:Split)->tuple[S32MatrixCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s32_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S32 partition size changed")
    domains={d.name for d in _DOMAINS}
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains: raise RuntimeError("S32 domains changed")
    if any(r.language!="en" for r in (*train,*dev)): raise RuntimeError("S32 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows): raise RuntimeError("S32 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S32 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2: raise RuntimeError("S32 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4: raise RuntimeError("S32 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S32 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}
    ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds: raise RuntimeError("S32 TRAIN DEV state overlap")
    if tq&dq: raise RuntimeError("S32 TRAIN DEV question overlap")
    if to&do: raise RuntimeError("S32 TRAIN DEV option overlap")


__all__=["S32MatrixCase","generate_s32_cases","validate_s32_partitions"]
