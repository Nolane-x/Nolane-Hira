from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

from .v1_s54_authority import generate_s54_cases

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S55LearnedJointCase:
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


_DOMAINS=(
    _Domain("learned_joint_muon_pressure_map","learned-joint muon pressure map","MP","absorber","pressure floor",
        ("silica tile","diamond tile","SiC tile","quartz tile","sapphire tile","graphene sheet","AlN tile","Si tile"),
        ("porous silica tile","CVD diamond tile","4H-SiC tile","etched quartz tile","cryogenic sapphire tile","suspended graphene sheet","piezo AlN tile","isotopic Si tile"),
        ("1.1 nPa","1.7 nPa","2.3 nPa","2.9 nPa","3.5 nPa","4.1 nPa","4.7 nPa","5.3 nPa"),
        ("1.4 nPa","2.0 nPa","2.6 nPa","3.2 nPa","3.8 nPa","4.4 nPa","5.0 nPa","5.6 nPa"),201101,1611000),
    _Domain("learned_joint_spin_recoil_clock","learned-joint spin recoil clock","SR","film","recoil floor",
        ("PtCo strip","CoFeB strip","NiFe strip","YIG strip","FeGa strip","CrI3 strip","WTe2 strip","FePS3 strip"),
        ("DMI PtCo strip","biased CoFeB strip","low-loss NiFe strip","epitaxial YIG strip","strained FeGa strip","layered CrI3 strip","helical WTe2 strip","AFM FePS3 strip"),
        ("0.31 um","0.37 um","0.43 um","0.49 um","0.55 um","0.61 um","0.67 um","0.73 um"),
        ("0.34 um","0.40 um","0.46 um","0.52 um","0.58 um","0.64 um","0.70 um","0.76 um"),201201,1612000),
    _Domain("learned_joint_phonon_curvature_bridge","learned-joint phonon curvature bridge","PC","guide","curvature floor",
        ("SiN beam","AlN beam","GaN beam","diamond beam","SiC beam","quartz beam","LiNbO3 beam","Si beam"),
        ("soft SiN beam","piezo AlN beam","suspended GaN beam","nanobeam diamond","4H-SiC beam","cryogenic quartz beam","periodic LiNbO3 beam","phononic Si beam"),
        ("0.021 mm-1","0.027 mm-1","0.033 mm-1","0.039 mm-1","0.045 mm-1","0.051 mm-1","0.057 mm-1","0.063 mm-1"),
        ("0.024 mm-1","0.030 mm-1","0.036 mm-1","0.042 mm-1","0.048 mm-1","0.054 mm-1","0.060 mm-1","0.066 mm-1"),201301,1613000),
    _Domain("learned_joint_rydberg_gradient_camera","learned-joint Rydberg gradient camera","RG","ensemble","gradient floor",
        ("Rb 50D","Cs 56S","Rb 65P","Cs 72D","K 43D","Na 48S","Sr 59D","Yb 63P"),
        ("dressed Rb 50D","shielded Cs 56S","circular Rb 65P","clocked Cs 72D","ultracold K 43D","collimated Na 48S","lattice Sr 59D","trapped Yb 63P"),
        ("1.2 V/m2","1.8 V/m2","2.4 V/m2","3.0 V/m2","3.6 V/m2","4.2 V/m2","4.8 V/m2","5.4 V/m2"),
        ("1.5 V/m2","2.1 V/m2","2.7 V/m2","3.3 V/m2","3.9 V/m2","4.5 V/m2","5.1 V/m2","5.7 V/m2"),201401,1614000),
    _Domain("learned_joint_moire_torque_scope","learned-joint moire torque scope","MT","stack","torque floor",
        ("WSe2-MoS2","WS2-MoSe2","MoTe2-WS2","graphene-hBN","graphene-WSe2","MoSe2-WS2","MoS2-hBN","WTe2-hBN"),
        ("twist WSe2-MoS2","gated WS2-MoSe2","strained MoTe2-WS2","aligned graphene-hBN","hybrid graphene-WSe2","encapsulated MoSe2-WS2","defect MoS2-hBN","moire WTe2-hBN"),
        ("1.3 zNm","1.9 zNm","2.5 zNm","3.1 zNm","3.7 zNm","4.3 zNm","4.9 zNm","5.5 zNm"),
        ("1.6 zNm","2.2 zNm","2.8 zNm","3.4 zNm","4.0 zNm","4.6 zNm","5.2 zNm","5.8 zNm"),201501,1615000),
    _Domain("learned_joint_neutron_pressure_lens","learned-joint neutron pressure lens","NP","analyzer","pressure blur",
        ("He3 tube","Fe mesh","Co mesh","Si grating","Ni grating","supermirror","diamond grid","quartz grid"),
        ("polarized He3 tube","exchange Fe mesh","epitaxial Co mesh","perfect-Si grating","magnetized Ni grating","m=6 supermirror","CVD diamond grid","etched quartz grid"),
        ("0.14 uPa","0.20 uPa","0.26 uPa","0.32 uPa","0.38 uPa","0.44 uPa","0.50 uPa","0.56 uPa"),
        ("0.17 uPa","0.23 uPa","0.29 uPa","0.35 uPa","0.41 uPa","0.47 uPa","0.53 uPa","0.59 uPa"),201601,1616000),
    _Domain("learned_joint_topological_phase_compass","learned-joint topological phase compass","TP","channel","phase leakage",
        ("kagome mode","honeycomb mode","Chern mode","valley mode","Floquet mode","ring mode","synthetic mode","magnon mode"),
        ("breathing kagome mode","domain-wall honeycomb mode","magnetic Chern mode","valley-Hall mode","driven Floquet mode","coupled-ring mode","frequency synthetic mode","chiral magnon mode"),
        ("-26 dB","-30 dB","-34 dB","-38 dB","-42 dB","-46 dB","-50 dB","-54 dB"),
        ("-28 dB","-32 dB","-36 dB","-40 dB","-44 dB","-48 dB","-52 dB","-56 dB"),201701,1617000),
    _Domain("learned_joint_molecular_curvature_radar","learned-joint molecular curvature radar","MC","beam","curvature drift",
        ("HfF beam","ThO beam","YbF beam","BaF beam","CaF beam","SrF beam","OCS beam","HCN beam"),
        ("oriented HfF beam","selected ThO beam","slow YbF beam","cold BaF beam","cooled CaF beam","collimated SrF beam","decelerated OCS beam","oriented HCN beam"),
        ("0.12 mm-1","0.18 mm-1","0.24 mm-1","0.30 mm-1","0.36 mm-1","0.42 mm-1","0.48 mm-1","0.54 mm-1"),
        ("0.15 mm-1","0.21 mm-1","0.27 mm-1","0.33 mm-1","0.39 mm-1","0.45 mm-1","0.51 mm-1","0.57 mm-1"),201801,1618000),
    _Domain("learned_joint_vacuum_recoil_clock","learned-joint vacuum recoil clock","VR","surface","recoil floor",
        ("Au disk","Ag disk","Si disk","graphene disk","Cu disk","Al disk","ITO disk","diamond disk"),
        ("annealed Au disk","template Ag disk","passivated Si disk","gated graphene disk","oxide-free Cu disk","epitaxial Al disk","ENZ ITO disk","CVD diamond disk"),
        ("0.11 pm","0.17 pm","0.23 pm","0.29 pm","0.35 pm","0.41 pm","0.47 pm","0.53 pm"),
        ("0.14 pm","0.20 pm","0.26 pm","0.32 pm","0.38 pm","0.44 pm","0.50 pm","0.56 pm"),201901,1619000),
    _Domain("learned_joint_atomic_pressure_scope","learned-joint atomic pressure scope","AP","species","pressure floor",
        ("Rb fountain","Cs fountain","Sr lattice","Yb lattice","K cloud","Li cloud","Na cloud","Ca cloud"),
        ("delta-kicked Rb fountain","dual-zone Cs fountain","interleaved Sr lattice","clocked Yb lattice","spin K cloud","degenerate Li cloud","collimated Na cloud","cooled Ca cloud"),
        ("2.2 nPa","2.8 nPa","3.4 nPa","4.0 nPa","4.6 nPa","5.2 nPa","5.8 nPa","6.4 nPa"),
        ("2.5 nPa","3.1 nPa","3.7 nPa","4.3 nPa","4.9 nPa","5.5 nPa","6.1 nPa","6.7 nPa"),202001,1620000),
    _Domain("learned_joint_superfluid_torque_map","learned-joint superfluid torque map","ST","fluid","torque noise",
        ("He4 layer","He3 layer","Rb condensate","Na condensate","K condensate","Li condensate","exciton fluid","polariton fluid"),
        ("vortex-free He4 layer","B-phase He3 layer","ring Rb condensate","spinor Na condensate","paired K condensate","unitary Li condensate","dipolar exciton fluid","driven polariton fluid"),
        ("0.03 zNm","0.05 zNm","0.07 zNm","0.09 zNm","0.11 zNm","0.13 zNm","0.15 zNm","0.17 zNm"),
        ("0.04 zNm","0.06 zNm","0.08 zNm","0.10 zNm","0.12 zNm","0.14 zNm","0.16 zNm","0.18 zNm"),202101,1621000),
    _Domain("learned_joint_ferroelectric_gradient_bridge","learned-joint ferroelectric gradient bridge","FG","crystal","gradient jitter",
        ("BaTiO3 plate","KTaO3 plate","LiNbO3 plate","PbTiO3 plate","BiFeO3 plate","SrTiO3 plate","KNbO3 plate","PZT plate"),
        ("strained BaTiO3 plate","quantum KTaO3 plate","poled LiNbO3 plate","epitaxial PbTiO3 plate","multiferroic BiFeO3 plate","isotopic SrTiO3 plate","domain KNbO3 plate","textured PZT plate"),
        ("0.7 V/mm","1.0 V/mm","1.3 V/mm","1.6 V/mm","1.9 V/mm","2.2 V/mm","2.5 V/mm","2.8 V/mm"),
        ("0.85 V/mm","1.15 V/mm","1.45 V/mm","1.75 V/mm","2.05 V/mm","2.35 V/mm","2.65 V/mm","2.95 V/mm"),202201,1622000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the S55 learned-joint {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S55-Learned {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S55-Learned record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S55-Learned {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S55-Learned record {code}?",
            f"For S55-Learned {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S55-Learned record {code}?",
        )
    return (
        f"S55-Learned audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S55-Learned audit {code}: what is {spec.field_a}?",
        f"In S55-Learned audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S55-Learned audit {code}: what is {spec.field_b}?",
        f"In S55-Learned audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    aa=spec.train_a if split=="train" else spec.dev_a
    bb=spec.train_b if split=="train" else spec.dev_b
    first=aa[index%8]; second=bb[(index*3+1)%8]
    wrong_a=aa[(index+3)%8]; wrong_b=bb[(index*5+2)%8]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    cid=f"{split}-s55-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S55LearnedJointCase(cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb)


def generate_s55_cases(split:Split)->tuple[S55LearnedJointCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def _states(rows): return {x for r in rows for x in (r.state_a,r.state_b)}
def _questions(rows): return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
def _options(rows): return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def validate_s55_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S55 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12: raise RuntimeError("S55 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S55 domains changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows): raise RuntimeError("S55 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S55 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S55 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S55 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S55 paired golds must differ")
    if _states(train)&_states(dev): raise RuntimeError("S55 TRAIN DEV state overlap")
    if _questions(train)&_questions(dev): raise RuntimeError("S55 TRAIN DEV question overlap")
    if _options(train)&_options(dev): raise RuntimeError("S55 TRAIN DEV option overlap")

    prior=(*generate_s54_cases("train"),*generate_s54_cases("dev"))
    current=(*train,*dev)
    if _states(current)&_states(prior): raise RuntimeError("S55 exact S54 state overlap")
    if _questions(current)&_questions(prior): raise RuntimeError("S55 exact S54 question overlap")
    if _options(current)&_options(prior): raise RuntimeError("S55 exact S54 option overlap")


__all__=["S55LearnedJointCase","generate_s55_cases","validate_s55_partitions"]
