from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S33QueryExplicitCase:
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
    _Domain("atom_interferometer","atom interferometer","AI","beam splitter","pulse separation",
        ("Raman","Bragg","microwave","Bloch","Kapitza-Dirac","STIRAP","RF","optical lattice"),
        ("double-Raman","large-momentum Bragg","chirped microwave","accelerated Bloch","pulsed KD","adiabatic Raman","multi-RF","moving lattice"),
        ("8 ms","12 ms","16 ms","20 ms","24 ms","28 ms","32 ms","36 ms"),
        ("10 ms","14 ms","18 ms","22 ms","26 ms","30 ms","34 ms","38 ms"),61101,561000),
    _Domain("terahertz_source","terahertz source","TH","emitter","center frequency",
        ("photomixer","PCA","QCL","Schottky multiplier","BWO","gyrotron","OR crystal","Josephson"),
        ("UTC photomixer","LT-GaAs PCA","dual-comb QCL","HBV multiplier","solid-state BWO","micro-gyrotron","DSTMS","stacked Josephson"),
        ("0.2 THz","0.4 THz","0.6 THz","0.8 THz","1.0 THz","1.2 THz","1.4 THz","1.6 THz"),
        ("0.3 THz","0.5 THz","0.7 THz","0.9 THz","1.1 THz","1.3 THz","1.5 THz","1.7 THz"),61201,562000),
    _Domain("micro_rheometer","micro-rheometer","MR","geometry","shear rate",
        ("cone plate","parallel plate","Couette","double gap","vane","capillary","microchannel","torsion"),
        ("truncated cone","serrated plate","Taylor-Couette","triple gap","four-blade vane","slit capillary","cross-slot","oscillatory torsion"),
        ("10 1/s","20 1/s","30 1/s","40 1/s","50 1/s","60 1/s","70 1/s","80 1/s"),
        ("15 1/s","25 1/s","35 1/s","45 1/s","55 1/s","65 1/s","75 1/s","85 1/s"),61301,563000),
    _Domain("muon_detector","muon detector","MD","sensor medium","threshold",
        ("plastic scintillator","RPC","GEM","drift tube","Cherenkov","silicon","diamond","liquid scintillator"),
        ("WLS scintillator","MRPC","triple GEM","straw tube","aerogel Cherenkov","LGAD","CVD diamond","LAB scintillator"),
        ("2 MeV","4 MeV","6 MeV","8 MeV","10 MeV","12 MeV","14 MeV","16 MeV"),
        ("3 MeV","5 MeV","7 MeV","9 MeV","11 MeV","13 MeV","15 MeV","17 MeV"),61401,564000),
    _Domain("electrooptic_sampler","electro-optic sampler","EO","crystal","delay step",
        ("ZnTe","GaP","LiTaO3","LiNbO3","BBO","KTP","DAST","GaSe"),
        ("CdTe","GaAs","MgO:LiNbO3","KTA","LBO","RTP","OH1","InP"),
        ("5 fs","10 fs","15 fs","20 fs","25 fs","30 fs","35 fs","40 fs"),
        ("7 fs","12 fs","17 fs","22 fs","27 fs","32 fs","37 fs","42 fs"),61501,565000),
    _Domain("micro_pcr","micro-PCR controller","PCR","channel material","anneal temperature",
        ("PDMS","COC","PMMA","glass","silicon","PEEK","polycarbonate","cyclic olefin polymer"),
        ("Teflon AF","COP","PS","quartz","SiN","PPS","PET","FEP"),
        ("48 C","50 C","52 C","54 C","56 C","58 C","60 C","62 C"),
        ("49 C","51 C","53 C","55 C","57 C","59 C","61 C","63 C"),61601,566000),
    _Domain("magnetron_supply","magnetron power supply","MS","pulse mode","peak voltage",
        ("square","half-sine","bipolar","burst","chirped","flat-top","Gaussian","dual pulse"),
        ("trapezoid","raised cosine","alternating bipolar","packet burst","FM chirp","regulated flat","super-Gaussian","doublet"),
        ("2 kV","4 kV","6 kV","8 kV","10 kV","12 kV","14 kV","16 kV"),
        ("3 kV","5 kV","7 kV","9 kV","11 kV","13 kV","15 kV","17 kV"),61701,567000),
    _Domain("squid_readout","SQUID readout","SQ","feedback mode","flux range",
        ("open loop","FLL","digital FLL","AC bias","DC bias","two-stage","series array","rf-SQUID"),
        ("flux ramp","dual FLL","FPGA FLL","MHz bias","low-noise DC","cascade","parallel array","microwave SQUID"),
        ("1 Phi0","2 Phi0","3 Phi0","4 Phi0","5 Phi0","6 Phi0","7 Phi0","8 Phi0"),
        ("1.5 Phi0","2.5 Phi0","3.5 Phi0","4.5 Phi0","5.5 Phi0","6.5 Phi0","7.5 Phi0","8.5 Phi0"),61801,568000),
    _Domain("optical_frequency_divider","optical frequency divider","OD","reference cavity","division ratio",
        ("ULE","silicon","sapphire","Zerodur","fused silica","CaF2","MgF2","crystalline coating"),
        ("cryogenic silicon","single-crystal sapphire","ULE spacer","SiO2","fluoride cavity","whispering CaF2","MgF2 WGM","AlGaAs coating"),
        ("10","20","30","40","50","60","70","80"),
        ("15","25","35","45","55","65","75","85"),61901,569000),
    _Domain("micro_mass_spec","micro mass spectrometer","MM","analyzer","scan span",
        ("quadrupole","TOF","ion trap","sector","Orbitrap","FTICR","Wien filter","cyclotron"),
        ("linear ion trap","reflectron TOF","Paul trap","double focusing","mini-Orbitrap","compact ICR","E×B filter","Penning"),
        ("20 Da","40 Da","60 Da","80 Da","100 Da","120 Da","140 Da","160 Da"),
        ("30 Da","50 Da","70 Da","90 Da","110 Da","130 Da","150 Da","170 Da"),62001,570000),
    _Domain("quantum_dot_driver","quantum-dot driver","QD","gate mode","pulse amplitude",
        ("plunger","barrier","exchange","sensor","reservoir","detuning","virtual gate","rf gate"),
        ("fast plunger","symmetric barrier","J gate","charge sensor","lead gate","epsilon gate","orthogonal virtual","reflectometry gate"),
        ("2 mV","4 mV","6 mV","8 mV","10 mV","12 mV","14 mV","16 mV"),
        ("3 mV","5 mV","7 mV","9 mV","11 mV","13 mV","15 mV","17 mV"),62101,571000),
    _Domain("laser_ultrasound","laser-ultrasound receiver","LU","interferometer","bandwidth",
        ("Michelson","Fabry-Perot","Sagnac","Mach-Zehnder","photorefractive","heterodyne","confocal","speckle"),
        ("polarization Michelson","confocal FP","fiber Sagnac","balanced MZ","two-wave mixing","IQ heterodyne","dual confocal","adaptive speckle"),
        ("2 MHz","4 MHz","6 MHz","8 MHz","10 MHz","12 MHz","14 MHz","16 MHz"),
        ("3 MHz","5 MHz","7 MHz","9 MHz","11 MHz","13 MHz","15 MHz","17 MHz"),62201,572000),
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
            f"S33-Query {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S33-Query record {code} stores {second} under {spec.field_b}. The same {spec.noun} assigns {first} to {spec.field_a}.",
            f"For S33-Query {spec.noun} {code}, identify {spec.field_a}.",
            f"In S33-Query record {code}, which entry belongs under {spec.field_a}?",
            f"For S33-Query {spec.noun} {code}, identify {spec.field_b}.",
            f"In S33-Query record {code}, which entry belongs under {spec.field_b}?",
        )
    return (
        f"S33-Query audit {code} for the {spec.noun} records {first} as {spec.field_a} and {second} as {spec.field_b}.",
        f"Audit {code} marks {second} for {spec.field_b}; separately, the S33-Query {spec.noun} lists {first} for {spec.field_a}.",
        f"Read S33-Query audit {code}: what is {spec.field_a}?",
        f"Which audit value is tagged {spec.field_a} for {code}?",
        f"Read S33-Query audit {code}: what is {spec.field_b}?",
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
    case_id=f"{split}-s33-{spec.name}-{index:03d}"
    state_a,state_b,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(case_id=case_id,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,first=first,second=second,distractor_first=df,distractor_second=ds,seed=spec.seed+offset)
    return S33QueryExplicitCase(case_id,split,spec.name,"en",state_a,state_b,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb)


def generate_s33_cases(split:Split)->tuple[S33QueryExplicitCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s33_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S33 partition size changed")
    domains={d.name for d in _DOMAINS}
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains: raise RuntimeError("S33 domains changed")
    if any(r.language!="en" for r in (*train,*dev)): raise RuntimeError("S33 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows): raise RuntimeError("S33 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S33 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2: raise RuntimeError("S33 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4: raise RuntimeError("S33 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S33 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}
    ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds: raise RuntimeError("S33 TRAIN DEV state overlap")
    if tq&dq: raise RuntimeError("S33 TRAIN DEV question overlap")
    if to&do: raise RuntimeError("S33 TRAIN DEV option overlap")


__all__=["S33QueryExplicitCase","generate_s33_cases","validate_s33_partitions"]
