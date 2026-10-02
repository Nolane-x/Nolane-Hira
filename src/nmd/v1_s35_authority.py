from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S35NativeCase:
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
        for key in ("option_texts","option_aliases","option_ids"):
            row[key]=list(row[key])
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
    _Domain("terahertz_imager","terahertz imager","TH","emitter","scan pitch",
        ("photomixer","QCL","PCA","multiplier","UTC-PD","BWO","FEL","Schottky"),
        ("plasmonic PCA","dual-comb QCL","LT-GaAs PCA","harmonic chain","uni-traveling PD","gyro-BWO","compact FEL","graphene mixer"),
        ("20 um","30 um","40 um","50 um","60 um","70 um","80 um","90 um"),
        ("25 um","35 um","45 um","55 um","65 um","75 um","85 um","95 um"),65101,601000),
    _Domain("muon_telescope","muon telescope","MU","detector plane","coincidence window",
        ("scintillator","RPC","GEM","drift tube","Micromegas","silicon","Cherenkov","emulsion"),
        ("LYSO array","MRPC","triple-GEM","straw tube","resistive Micromegas","LGAD","aerogel","nuclear emulsion"),
        ("5 ns","10 ns","15 ns","20 ns","25 ns","30 ns","35 ns","40 ns"),
        ("7 ns","12 ns","17 ns","22 ns","27 ns","32 ns","37 ns","42 ns"),65201,602000),
    _Domain("raman_mapper","Raman mapper","RM","excitation","mapping step",
        ("405 nm","488 nm","514 nm","532 nm","633 nm","785 nm","830 nm","1064 nm"),
        ("442 nm","473 nm","515 nm","561 nm","660 nm","730 nm","980 nm","1310 nm"),
        ("0.2 um","0.4 um","0.6 um","0.8 um","1.0 um","1.2 um","1.4 um","1.6 um"),
        ("0.3 um","0.5 um","0.7 um","0.9 um","1.1 um","1.3 um","1.5 um","1.7 um"),65301,603000),
    _Domain("electron_diffractometer","electron diffractometer","ED","accelerating voltage","camera length",
        ("40 kV","60 kV","80 kV","100 kV","120 kV","160 kV","200 kV","300 kV"),
        ("50 kV","70 kV","90 kV","110 kV","140 kV","180 kV","250 kV","350 kV"),
        ("50 mm","75 mm","100 mm","125 mm","150 mm","175 mm","200 mm","225 mm"),
        ("60 mm","85 mm","110 mm","135 mm","160 mm","185 mm","210 mm","235 mm"),65401,604000),
    _Domain("optical_frequency_comb","optical frequency comb","OC","comb source","repetition rate",
        ("Er fiber","Yb fiber","Ti:sapphire","microresonator","EO comb","OPO comb","Cr:ZnS","Tm fiber"),
        ("dual-comb Er","thin-disk Yb","Kerr TiSa","SiN microcomb","integrated EO","synchronously-pumped OPO","Cr:ZnSe","Ho fiber"),
        ("50 MHz","100 MHz","150 MHz","200 MHz","250 MHz","300 MHz","350 MHz","400 MHz"),
        ("75 MHz","125 MHz","175 MHz","225 MHz","275 MHz","325 MHz","375 MHz","425 MHz"),65501,605000),
    _Domain("atomic_force_microscope","atomic force microscope","AF","probe mode","line rate",
        ("contact","tapping","noncontact","PeakForce","KPFM","MFM","conductive","force-volume"),
        ("lateral-force","bimodal","FM-AFM","FastScan","AM-KPFM","lift-mode MFM","TUNA","QI mode"),
        ("1 Hz","2 Hz","3 Hz","4 Hz","5 Hz","6 Hz","7 Hz","8 Hz"),
        ("1.5 Hz","2.5 Hz","3.5 Hz","4.5 Hz","5.5 Hz","6.5 Hz","7.5 Hz","8.5 Hz"),65601,606000),
    _Domain("gamma_spectrometer","gamma spectrometer","GS","detector crystal","shaping time",
        ("HPGe","NaI","LaBr3","BGO","CsI","CdZnTe","CeBr3","LYSO"),
        ("BEGe","SrI2","CLYC","GAGG","LaCl3","TlBr","LuAG","YAP"),
        ("1 us","2 us","3 us","4 us","5 us","6 us","7 us","8 us"),
        ("1.5 us","2.5 us","3.5 us","4.5 us","5.5 us","6.5 us","7.5 us","8.5 us"),65701,607000),
    _Domain("micro_piv_system","micro-PIV system","PV","tracer","frame interval",
        ("polystyrene","silica","liposome","fluorescent bead","oil droplet","hollow glass","quantum dot","nanodiamond"),
        ("PMMA bead","alumina","vesicle","upconversion bead","emulsion droplet","silvered sphere","carbon dot","SiC nanoparticle"),
        ("10 us","20 us","30 us","40 us","50 us","60 us","70 us","80 us"),
        ("15 us","25 us","35 us","45 us","55 us","65 us","75 us","85 us"),65801,608000),
    _Domain("ellipsometer","spectroscopic ellipsometer","EL","polarizer","incidence angle",
        ("Glan-Taylor","wire grid","calcite","quartz","polymer film","MgF2","Si prism","ZnSe grid"),
        ("Glan-laser","nanowire grid","BBO","achromatic quartz","LC polymer","CaF2","Ge prism","KRS5 grid"),
        ("55 deg","57 deg","59 deg","61 deg","63 deg","65 deg","67 deg","69 deg"),
        ("56 deg","58 deg","60 deg","62 deg","64 deg","66 deg","68 deg","70 deg"),65901,609000),
    _Domain("magnetic_resonance_probe","magnetic resonance probe","MR","coil geometry","tuning frequency",
        ("solenoid","saddle","birdcage","surface loop","Helmholtz","microcoil","stripline","spiral"),
        ("Alderman-Grant","TEM birdcage","phased loop","cryoprobe loop","Maxwell pair","planar microcoil","coplanar line","meander"),
        ("10 MHz","20 MHz","30 MHz","40 MHz","50 MHz","60 MHz","70 MHz","80 MHz"),
        ("15 MHz","25 MHz","35 MHz","45 MHz","55 MHz","65 MHz","75 MHz","85 MHz"),66001,610000),
    _Domain("mass_flow_controller","mass-flow controller","MF","sensor type","full scale",
        ("thermal","Coriolis","differential pressure","ultrasonic","laminar","MEMS","vortex","capillary"),
        ("micro-Coriolis","hot-film","Venturi","time-of-flight","LFE","CMOS MEMS","micro-vortex","bubble"),
        ("10 sccm","20 sccm","30 sccm","40 sccm","50 sccm","60 sccm","70 sccm","80 sccm"),
        ("15 sccm","25 sccm","35 sccm","45 sccm","55 sccm","65 sccm","75 sccm","85 sccm"),66101,611000),
    _Domain("optical_coherence_tomograph","optical coherence tomograph","OT","sweep source","A-scan rate",
        ("VCSEL","FDML","MEMS laser","ECDL","supercontinuum","SLD","DBR","DFB"),
        ("MEMS-VCSEL","akinetic laser","polygon FDML","sampled-grating DBR","Kerr comb","quantum-dot SLD","tunable DBR","external-cavity DFB"),
        ("20 kHz","40 kHz","60 kHz","80 kHz","100 kHz","120 kHz","140 kHz","160 kHz"),
        ("30 kHz","50 kHz","70 kHz","90 kHz","110 kHz","130 kHz","150 kHz","170 kHz"),66201,612000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the recorded {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S35-Native {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S35-Native record {code} stores {second} for {spec.field_b}. The same {spec.noun} records {first} for {spec.field_a}.",
            f"For S35-Native {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S35-Native record {code}?",
            f"For S35-Native {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S35-Native record {code}?",
        )
    return (
        f"S35-Native audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; separately, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S35-Native audit {code}: what is {spec.field_a}?",
        f"In S35-Native audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S35-Native audit {code}: what is {spec.field_b}?",
        f"In S35-Native audit {code}, which entry is tagged {spec.field_b}?",
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
    case_id=f"{split}-s35-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=case_id,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=df,distractor_second=ds,
        seed=spec.seed+offset,
    )
    return S35NativeCase(
        case_id,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,
        texts,aliases,ids,ga,gb,
    )


def generate_s35_cases(split:Split)->tuple[S35NativeCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s35_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192:
        raise RuntimeError("S35 partition size changed")
    domains={d.name for d in _DOMAINS}
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S35 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S35 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S35 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b:
                raise RuntimeError("S35 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S35 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S35 option IDs changed")
            if r.gold_a==r.gold_b:
                raise RuntimeError("S35 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}
    ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds:
        raise RuntimeError("S35 TRAIN DEV state overlap")
    if tq&dq:
        raise RuntimeError("S35 TRAIN DEV question overlap")
    if to&do:
        raise RuntimeError("S35 TRAIN DEV option overlap")


__all__=["S35NativeCase","generate_s35_cases","validate_s35_partitions"]
