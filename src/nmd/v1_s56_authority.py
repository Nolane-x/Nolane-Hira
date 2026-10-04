from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

from .v1_s55_authority import generate_s55_cases

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S56DecisionConsistencyCase:
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

    def to_dict(self):
        row=asdict(self)
        for key in ("option_texts","option_aliases","option_ids"):
            row[key]=list(row[key])
        return row


@dataclass(frozen=True)
class _Domain:
    name:str; noun:str; prefix:str; field_a:str; field_b:str
    train_a:tuple[str,...]; dev_a:tuple[str,...]
    train_b:tuple[str,...]; dev_b:tuple[str,...]
    seed:int; dev_offset:int


def _seq(prefix,unit,start,step):
    return tuple(f"{prefix}{start+i*step:g} {unit}" for i in range(8))


_DOMAINS=(
    _Domain("decision_axion_gradient_scope","decision-consistency axion gradient scope","AG","resonator","gradient floor",
        ("Nb helix","Ta helix","Cu toroid","Si toroid","Al loop","quartz ring","graphene loop","sapphire toroid"),
        ("flux Nb helix","shielded Ta helix","annealed Cu toroid","cryogenic Si toroid","epitaxial Al loop","polished quartz ring","ballistic graphene loop","whispering sapphire toroid"),
        _seq("", "V/m2",1.1,0.6),_seq("", "V/m2",1.4,0.6),221101,1711000),
    _Domain("decision_spin_pressure_camera","decision-consistency spin pressure camera","SP","film","pressure floor",
        ("PtCo film","CoFeB film","NiFe film","YIG film","FeGa film","CrI3 film","WTe2 film","FePS3 film"),
        ("DMI PtCo film","biased CoFeB film","low-loss NiFe film","epitaxial YIG film","strained FeGa film","layered CrI3 film","helical WTe2 film","AFM FePS3 film"),
        _seq("", "nPa",1.3,0.5),_seq("", "nPa",1.55,0.5),221201,1712000),
    _Domain("decision_phonon_phase_bridge","decision-consistency phonon phase bridge","PH","guide","phase floor",
        ("SiN ridge","AlN ridge","GaN ridge","diamond ridge","SiC ridge","quartz ridge","LiNbO3 ridge","Si ridge"),
        ("soft SiN ridge","piezo AlN ridge","suspended GaN ridge","nanobeam diamond ridge","4H-SiC ridge","cryogenic quartz ridge","periodic LiNbO3 ridge","phononic Si ridge"),
        _seq("", "rad",0.03,0.04),_seq("", "rad",0.05,0.04),221301,1713000),
    _Domain("decision_rydberg_curvature_clock","decision-consistency Rydberg curvature clock","RC","ensemble","curvature floor",
        ("Rb 52D","Cs 58S","Rb 67P","Cs 74D","K 45D","Na 50S","Sr 61D","Yb 65P"),
        ("dressed Rb 52D","shielded Cs 58S","circular Rb 67P","clocked Cs 74D","ultracold K 45D","collimated Na 50S","lattice Sr 61D","trapped Yb 65P"),
        _seq("", "nrad/mm",2.0,1.0),_seq("", "nrad/mm",2.5,1.0),221401,1714000),
    _Domain("decision_moire_recoil_map","decision-consistency moire recoil map","MR","stack","recoil floor",
        ("WSe2-WS2","WS2-MoS2","MoTe2-MoSe2","graphene-hBN","graphene-MoS2","MoSe2-WSe2","MoS2-hBN","WTe2-WS2"),
        ("twist WSe2-WS2","gated WS2-MoS2","strained MoTe2-MoSe2","aligned graphene-hBN","hybrid graphene-MoS2","encapsulated MoSe2-WSe2","defect MoS2-hBN","moire WTe2-WS2"),
        _seq("", "pm",0.12,0.06),_seq("", "pm",0.15,0.06),221501,1715000),
    _Domain("decision_neutron_phase_lens","decision-consistency neutron phase lens","NP","analyzer","phase blur",
        ("He3 tube","Fe lattice","Co lattice","Si grating","Ni grating","supermirror","diamond grid","quartz grid"),
        ("polarized He3 tube","exchange Fe lattice","epitaxial Co lattice","perfect-Si grating","magnetized Ni grating","m=7 supermirror","CVD diamond grid","etched quartz grid"),
        _seq("", "mrad",0.08,0.05),_seq("", "mrad",0.105,0.05),221601,1716000),
    _Domain("decision_topological_pressure_radar","decision-consistency topological pressure radar","TP","channel","pressure leakage",
        ("kagome channel","honeycomb channel","Chern channel","valley channel","Floquet channel","ring channel","synthetic channel","magnon channel"),
        ("breathing kagome channel","domain-wall honeycomb channel","magnetic Chern channel","valley-Hall channel","driven Floquet channel","coupled-ring channel","frequency synthetic channel","chiral magnon channel"),
        ("-25 dB","-29 dB","-33 dB","-37 dB","-41 dB","-45 dB","-49 dB","-53 dB"),
        ("-27 dB","-31 dB","-35 dB","-39 dB","-43 dB","-47 dB","-51 dB","-55 dB"),221701,1717000),
    _Domain("decision_molecular_gradient_compass","decision-consistency molecular gradient compass","MG","beam","gradient drift",
        ("HfF beam","ThO beam","YbF beam","BaF beam","CaF beam","SrF beam","OCS beam","HCN beam"),
        ("oriented HfF beam","selected ThO beam","slow YbF beam","cold BaF beam","cooled CaF beam","collimated SrF beam","decelerated OCS beam","oriented HCN beam"),
        _seq("", "uT/mm",0.7,0.4),_seq("", "uT/mm",0.9,0.4),221801,1718000),
    _Domain("decision_vacuum_torque_scope","decision-consistency vacuum torque scope","VT","surface","torque floor",
        ("Au plate","Ag plate","Si plate","graphene plate","Cu plate","Al plate","ITO plate","diamond plate"),
        ("annealed Au plate","template Ag plate","passivated Si plate","gated graphene plate","oxide-free Cu plate","epitaxial Al plate","ENZ ITO plate","CVD diamond plate"),
        _seq("", "zNm",0.6,0.5),_seq("", "zNm",0.85,0.5),221901,1719000),
    _Domain("decision_atomic_curvature_camera","decision-consistency atomic curvature camera","AC","species","curvature floor",
        ("Rb fountain","Cs fountain","Sr lattice","Yb lattice","K cloud","Li cloud","Na cloud","Ca cloud"),
        ("delta-kicked Rb fountain","dual-zone Cs fountain","interleaved Sr lattice","clocked Yb lattice","spin K cloud","degenerate Li cloud","collimated Na cloud","cooled Ca cloud"),
        _seq("", "urad/m",1.4,0.8),_seq("", "urad/m",1.8,0.8),222001,1720000),
    _Domain("decision_superfluid_phase_scope","decision-consistency superfluid phase scope","SU","fluid","phase noise",
        ("He4 layer","He3 layer","Rb condensate","Na condensate","K condensate","Li condensate","exciton fluid","polariton fluid"),
        ("vortex-free He4 layer","B-phase He3 layer","ring Rb condensate","spinor Na condensate","paired K condensate","unitary Li condensate","dipolar exciton fluid","driven polariton fluid"),
        _seq("", "rad",0.025,0.03),_seq("", "rad",0.04,0.03),222101,1721000),
    _Domain("decision_ferroelectric_recoil_bridge","decision-consistency ferroelectric recoil bridge","FR","crystal","recoil jitter",
        ("BaTiO3 chip","KTaO3 chip","LiNbO3 chip","PbTiO3 chip","BiFeO3 chip","SrTiO3 chip","KNbO3 chip","PZT chip"),
        ("strained BaTiO3 chip","quantum KTaO3 chip","poled LiNbO3 chip","epitaxial PbTiO3 chip","multiferroic BiFeO3 chip","isotopic SrTiO3 chip","domain KNbO3 chip","textured PZT chip"),
        _seq("", "nm",0.4,0.4),_seq("", "nm",0.6,0.4),222201,1722000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the S56 decision-consistency {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S56-Decision {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S56-Decision record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S56-Decision {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S56-Decision record {code}?",
            f"For S56-Decision {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S56-Decision record {code}?",
        )
    return (
        f"S56-Decision audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S56-Decision audit {code}: what is {spec.field_a}?",
        f"In S56-Decision audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S56-Decision audit {code}: what is {spec.field_b}?",
        f"In S56-Decision audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    aa=spec.train_a if split=="train" else spec.dev_a
    bb=spec.train_b if split=="train" else spec.dev_b
    first=aa[index%8]; second=bb[(index*3+1)%8]
    wrong_a=aa[(index+3)%8]; wrong_b=bb[(index*5+2)%8]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    cid=f"{split}-s56-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S56DecisionConsistencyCase(
        cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb
    )


def generate_s56_cases(split:Split)->tuple[S56DecisionConsistencyCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def _states(rows): return {x for r in rows for x in (r.state_a,r.state_b)}
def _questions(rows): return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
def _options(rows): return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def validate_s56_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S56 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12: raise RuntimeError("S56 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S56 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S56 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S56 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S56 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S56 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S56 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S56 paired golds must differ")
    if _states(train)&_states(dev): raise RuntimeError("S56 TRAIN DEV state overlap")
    if _questions(train)&_questions(dev): raise RuntimeError("S56 TRAIN DEV question overlap")
    if _options(train)&_options(dev): raise RuntimeError("S56 TRAIN DEV option overlap")

    prior=(*generate_s55_cases("train"),*generate_s55_cases("dev"))
    current=(*train,*dev)
    if _states(current)&_states(prior): raise RuntimeError("S56 exact S55 state overlap")
    if _questions(current)&_questions(prior): raise RuntimeError("S56 exact S55 question overlap")
    if _options(current)&_options(prior): raise RuntimeError("S56 exact S55 option overlap")


__all__=["S56DecisionConsistencyCase","generate_s56_cases","validate_s56_partitions"]
