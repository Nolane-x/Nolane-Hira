from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S52QueryRelationCase:
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
    _Domain("query_relation_spin_noise_scope","query-relation spin-noise scope","QS","sensor","noise floor",
        ("NV ring","SiV disk","SiC slab","GaAs well","graphene dot","YIG strip","AlN drum","quartz fork"),
        ("vector NV ring","isotopic SiV disk","4H-SiC slab","gated GaAs well","encapsulated graphene dot","low-loss YIG strip","piezo AlN drum","cryogenic quartz fork"),
        ("2 nT","3 nT","4 nT","5 nT","6 nT","7 nT","8 nT","9 nT"),
        ("2.5 nT","3.5 nT","4.5 nT","5.5 nT","6.5 nT","7.5 nT","8.5 nT","9.5 nT"),151101,1301000),
    _Domain("query_relation_phonon_delay_map","query-relation phonon-delay map","QP","guide","delay jitter",
        ("Si ridge","SiN ridge","AlN ridge","GaN ridge","diamond ridge","SiC ridge","LiNbO3 ridge","quartz ridge"),
        ("phononic Si ridge","soft SiN ridge","piezo AlN ridge","suspended GaN ridge","nanobeam diamond ridge","4H-SiC ridge","periodic LiNbO3 ridge","etched quartz ridge"),
        ("8 ps","12 ps","16 ps","20 ps","24 ps","28 ps","32 ps","36 ps"),
        ("10 ps","14 ps","18 ps","22 ps","26 ps","30 ps","34 ps","38 ps"),151201,1302000),
    _Domain("query_relation_axion_phase_clock","query-relation axion-phase clock","QA","cavity","phase drift",
        ("Nb cavity","Cu cavity","Al cavity","sapphire cavity","Si cavity","quartz cavity","graphene cavity","diamond cavity"),
        ("flux-locked Nb cavity","annealed Cu cavity","epitaxial Al cavity","whispering sapphire cavity","cryogenic Si cavity","superpolished quartz cavity","gated graphene cavity","CVD diamond cavity"),
        ("0.7 urad","1.0 urad","1.3 urad","1.6 urad","1.9 urad","2.2 urad","2.5 urad","2.8 urad"),
        ("0.85 urad","1.15 urad","1.45 urad","1.75 urad","2.05 urad","2.35 urad","2.65 urad","2.95 urad"),151301,1303000),
    _Domain("query_relation_moire_spin_camera","query-relation moire-spin camera","QM","stack","spin blur",
        ("WSe2-MoSe2","WS2-WSe2","MoS2-WS2","graphene-hBN","graphene-graphene","MoTe2-WSe2","MoSe2-hBN","WS2-hBN"),
        ("twist-locked WSe2-MoSe2","gated WS2-WSe2","encapsulated MoS2-WS2","aligned graphene-hBN","magic-angle graphene-graphene","strain-tuned MoTe2-WSe2","defect MoSe2-hBN","moire WS2-hBN"),
        ("0.06 deg","0.09 deg","0.12 deg","0.15 deg","0.18 deg","0.21 deg","0.24 deg","0.27 deg"),
        ("0.075 deg","0.105 deg","0.135 deg","0.165 deg","0.195 deg","0.225 deg","0.255 deg","0.285 deg"),151401,1304000),
    _Domain("query_relation_quantum_heat_lens","query-relation quantum-heat lens","QH","detector","thermal floor",
        ("TES pixel","nanoSQUID","NV sensor","SiV sensor","graphene drum","MoS2 drum","nanowire pixel","bolometer"),
        ("transition-edge pixel","gradiometric nanoSQUID","vector NV sensor","isotopic SiV sensor","suspended graphene drum","encapsulated MoS2 drum","proximity nanowire pixel","athermal bolometer"),
        ("4 nK","6 nK","8 nK","10 nK","12 nK","14 nK","16 nK","18 nK"),
        ("5 nK","7 nK","9 nK","11 nK","13 nK","15 nK","17 nK","19 nK"),151501,1305000),
    _Domain("query_relation_nuclear_recoil_clock","query-relation nuclear-recoil clock","QR","target","recoil threshold",
        ("Ge crystal","Si crystal","Xe cell","Ar cell","CaWO4 crystal","NaI crystal","TeO2 crystal","Al2O3 crystal"),
        ("enriched Ge crystal","isotopic Si crystal","dual-phase Xe cell","ultrapure Ar cell","cryogenic CaWO4 crystal","low-background NaI crystal","bolometric TeO2 crystal","sapphire Al2O3 crystal"),
        ("6 eV","9 eV","12 eV","15 eV","18 eV","21 eV","24 eV","27 eV"),
        ("7.5 eV","10.5 eV","13.5 eV","16.5 eV","19.5 eV","22.5 eV","25.5 eV","28.5 eV"),151601,1306000),
    _Domain("query_relation_superfluid_torque_scope","query-relation superfluid-torque scope","QT","fluid","torque floor",
        ("He4 film","He3 film","Rb condensate","Na condensate","K condensate","Li condensate","exciton fluid","polariton fluid"),
        ("vortex-free He4 film","B-phase He3 film","ring Rb condensate","spinor Na condensate","paired K condensate","unitary Li condensate","dipolar exciton fluid","driven polariton fluid"),
        ("3 zNm","5 zNm","7 zNm","9 zNm","11 zNm","13 zNm","15 zNm","17 zNm"),
        ("4 zNm","6 zNm","8 zNm","10 zNm","12 zNm","14 zNm","16 zNm","18 zNm"),151701,1307000),
    _Domain("query_relation_topological_pressure_map","query-relation topological-pressure map","QX","channel","pressure leakage",
        ("kagome edge","honeycomb edge","Chern edge","valley edge","Floquet edge","ring edge","synthetic edge","magnon edge"),
        ("breathing kagome edge","domain-wall honeycomb edge","magnetic Chern edge","valley-Hall edge","driven Floquet edge","coupled-ring edge","frequency synthetic edge","chiral magnon edge"),
        ("-29 dB","-33 dB","-37 dB","-41 dB","-45 dB","-49 dB","-53 dB","-57 dB"),
        ("-31 dB","-35 dB","-39 dB","-43 dB","-47 dB","-51 dB","-55 dB","-59 dB"),151801,1308000),
    _Domain("query_relation_optical_spin_ruler","query-relation optical-spin ruler","QO","interface","phase loss",
        ("NV cavity","SiV cavity","QD cavity","Er cavity","Yb cavity","Rb cell","Cs cell","Tm crystal"),
        ("impedance NV cavity","nanophotonic SiV cavity","charged-QD cavity","rare-earth Er cavity","spectral Yb cavity","EIT Rb cell","spin-wave Cs cell","cryogenic Tm crystal"),
        ("0.04 rad","0.06 rad","0.08 rad","0.10 rad","0.12 rad","0.14 rad","0.16 rad","0.18 rad"),
        ("0.05 rad","0.07 rad","0.09 rad","0.11 rad","0.13 rad","0.15 rad","0.17 rad","0.19 rad"),151901,1309000),
    _Domain("query_relation_molecular_phase_scope","query-relation molecular-phase scope","QF","species","phase jitter",
        ("HfF+ packet","ThO beam","YbF beam","BaF beam","CaF beam","SrF beam","OCS beam","HCN beam"),
        ("oriented HfF+ packet","state-selected ThO beam","slow YbF beam","cold BaF beam","laser-cooled CaF beam","collimated SrF beam","Stark-decelerated OCS beam","oriented HCN beam"),
        ("9 urad","13 urad","17 urad","21 urad","25 urad","29 urad","33 urad","37 urad"),
        ("11 urad","15 urad","19 urad","23 urad","27 urad","31 urad","35 urad","39 urad"),152001,1310000),
    _Domain("query_relation_casimir_force_camera","query-relation Casimir-force camera","QG","surface","force floor",
        ("Au sphere","Ag sphere","Si plate","graphene sheet","Cu sphere","Al sphere","ITO plate","diamond plate"),
        ("annealed Au sphere","template Ag sphere","passivated Si plate","gated graphene sheet","oxide-free Cu sphere","epitaxial Al sphere","ENZ ITO plate","CVD diamond plate"),
        ("0.15 pN","0.21 pN","0.27 pN","0.33 pN","0.39 pN","0.45 pN","0.51 pN","0.57 pN"),
        ("0.18 pN","0.24 pN","0.30 pN","0.36 pN","0.42 pN","0.48 pN","0.54 pN","0.60 pN"),152101,1311000),
    _Domain("query_relation_atomic_gradient_compass","query-relation atomic-gradient compass","QJ","atom source","gradient noise",
        ("Rb fountain","Cs fountain","Sr lattice","Yb lattice","K cloud","Li cloud","Na cloud","Ca cloud"),
        ("delta-kicked Rb fountain","dual-zone Cs fountain","interleaved Sr lattice","clocked Yb lattice","spin-polarized K cloud","degenerate Li cloud","collimated Na cloud","laser-cooled Ca cloud"),
        ("1.8 E","2.4 E","3.0 E","3.6 E","4.2 E","4.8 E","5.4 E","6.0 E"),
        ("2.1 E","2.7 E","3.3 E","3.9 E","4.5 E","5.1 E","5.7 E","6.3 E"),152201,1312000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the S52 query-relation {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S52-Canon {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S52-Canon record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S52-Canon {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S52-Canon record {code}?",
            f"For S52-Canon {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S52-Canon record {code}?",
        )
    return (
        f"S52-Canon audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S52-Canon audit {code}: what is {spec.field_a}?",
        f"In S52-Canon audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S52-Canon audit {code}: what is {spec.field_b}?",
        f"In S52-Canon audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    aa=spec.train_a if split=="train" else spec.dev_a
    bb=spec.train_b if split=="train" else spec.dev_b
    first=aa[index%8]; second=bb[(index*3+1)%8]
    wrong_a=aa[(index+3)%8]; wrong_b=bb[(index*5+2)%8]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    cid=f"{split}-s52-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S52QueryRelationCase(cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb)


def generate_s52_cases(split:Split)->tuple[S52QueryRelationCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s52_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S52 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12: raise RuntimeError("S52 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains: raise RuntimeError("S52 domains changed")
    if any(r.language!="en" for r in (*train,*dev)): raise RuntimeError("S52 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows): raise RuntimeError("S52 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S52 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2: raise RuntimeError("S52 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4: raise RuntimeError("S52 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S52 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}; ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}; dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}; do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds: raise RuntimeError("S52 TRAIN DEV state overlap")
    if tq&dq: raise RuntimeError("S52 TRAIN DEV question overlap")
    if to&do: raise RuntimeError("S52 TRAIN DEV option overlap")


__all__=["S52QueryRelationCase","generate_s52_cases","validate_s52_partitions"]
