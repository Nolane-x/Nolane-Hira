from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S53LateInteractionCase:
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
    _Domain("late_interaction_muon_phase_scope","late-interaction muon-phase scope","LM","target","phase floor",
        ("silica aerogel","carbon foil","beryllium plate","diamond wafer","quartz tile","sapphire plate","silicon slab","graphene sheet"),
        ("graded silica aerogel","ultrathin carbon foil","annealed beryllium plate","CVD diamond wafer","superpolished quartz tile","cryogenic sapphire plate","isotopic silicon slab","encapsulated graphene sheet"),
        ("1.1 mrad","1.6 mrad","2.1 mrad","2.6 mrad","3.1 mrad","3.6 mrad","4.1 mrad","4.6 mrad"),
        ("1.35 mrad","1.85 mrad","2.35 mrad","2.85 mrad","3.35 mrad","3.85 mrad","4.35 mrad","4.85 mrad"),161101,1401000),
    _Domain("late_interaction_spin_heat_map","late-interaction spin-heat map","LS","film","thermal floor",
        ("PtCo film","WCoFe film","YIG film","NiFe film","FeGa film","CoTb film","MnGa film","CrI3 film"),
        ("DMI PtCo film","compensated WCoFe film","low-loss YIG film","exchange NiFe film","strained FeGa film","ferrimagnetic CoTb film","L10 MnGa film","encapsulated CrI3 film"),
        ("5 nK","8 nK","11 nK","14 nK","17 nK","20 nK","23 nK","26 nK"),
        ("6.5 nK","9.5 nK","12.5 nK","15.5 nK","18.5 nK","21.5 nK","24.5 nK","27.5 nK"),161201,1402000),
    _Domain("late_interaction_phonon_torque_clock","late-interaction phonon-torque clock","LP","resonator","torque floor",
        ("SiN beam","AlN beam","diamond beam","SiC beam","GaAs disk","LiNbO3 disk","quartz fork","graphene drum"),
        ("soft SiN beam","piezo AlN beam","phononic diamond beam","4H-SiC beam","whispering GaAs disk","periodic LiNbO3 disk","cryogenic quartz fork","tensioned graphene drum"),
        ("2 zNm","4 zNm","6 zNm","8 zNm","10 zNm","12 zNm","14 zNm","16 zNm"),
        ("3 zNm","5 zNm","7 zNm","9 zNm","11 zNm","13 zNm","15 zNm","17 zNm"),161301,1403000),
    _Domain("late_interaction_rydberg_pressure_camera","late-interaction Rydberg-pressure camera","LR","ensemble","pressure floor",
        ("Rb 50D cloud","Cs 55S cloud","Rb 63P cloud","Cs 70D cloud","K 42D cloud","Na 47S cloud","Sr 58D cloud","Yb 62P cloud"),
        ("dressed Rb 50D cloud","shielded Cs 55S cloud","circular Rb 63P cloud","clocked Cs 70D cloud","ultracold K 42D cloud","collimated Na 47S cloud","lattice Sr 58D cloud","trapped Yb 62P cloud"),
        ("4 nPa","7 nPa","10 nPa","13 nPa","16 nPa","19 nPa","22 nPa","25 nPa"),
        ("5.5 nPa","8.5 nPa","11.5 nPa","14.5 nPa","17.5 nPa","20.5 nPa","23.5 nPa","26.5 nPa"),161401,1404000),
    _Domain("late_interaction_moire_recoil_radar","late-interaction moire-recoil radar","LC","stack","recoil blur",
        ("MoTe2-MoSe2","WTe2-WSe2","MoS2-hBN","WS2-hBN","graphene-MoS2","graphene-WSe2","MoSe2-WS2","WSe2-hBN"),
        ("twist-locked MoTe2-MoSe2","gated WTe2-WSe2","encapsulated MoS2-hBN","moire WS2-hBN","aligned graphene-MoS2","strained graphene-WSe2","dual-gated MoSe2-WS2","defect WSe2-hBN"),
        ("0.04 nm-1","0.07 nm-1","0.10 nm-1","0.13 nm-1","0.16 nm-1","0.19 nm-1","0.22 nm-1","0.25 nm-1"),
        ("0.055 nm-1","0.085 nm-1","0.115 nm-1","0.145 nm-1","0.175 nm-1","0.205 nm-1","0.235 nm-1","0.265 nm-1"),161501,1405000),
    _Domain("late_interaction_neutron_heat_compass","late-interaction neutron-heat compass","LN","analyzer","thermal drift",
        ("He3 cell","Fe foil","Co foil","Si grating","Ni grating","supermirror","diamond grating","quartz grating"),
        ("polarized He3 cell","exchange Fe foil","epitaxial Co foil","perfect-Si grating","magnetized Ni grating","m=6 supermirror","CVD diamond grating","etched quartz grating"),
        ("7 nK","10 nK","13 nK","16 nK","19 nK","22 nK","25 nK","28 nK"),
        ("8.5 nK","11.5 nK","14.5 nK","17.5 nK","20.5 nK","23.5 nK","26.5 nK","29.5 nK"),161601,1406000),
    _Domain("late_interaction_superfluid_gradient_scope","late-interaction superfluid-gradient scope","LG","fluid","gradient floor",
        ("He4 ring","He3-B ring","Rb torus","Na torus","K torus","Li torus","exciton ring","polariton ring"),
        ("vortex-free He4 ring","textured He3-B ring","persistent Rb torus","spinor Na torus","paired K torus","unitary Li torus","dipolar exciton ring","driven polariton ring"),
        ("0.3 E","0.5 E","0.7 E","0.9 E","1.1 E","1.3 E","1.5 E","1.7 E"),
        ("0.4 E","0.6 E","0.8 E","1.0 E","1.2 E","1.4 E","1.6 E","1.8 E"),161701,1407000),
    _Domain("late_interaction_topological_spin_lens","late-interaction topological-spin lens","LL","edge","spin leakage",
        ("kagome channel","honeycomb channel","Chern channel","valley channel","Floquet channel","ring channel","synthetic channel","magnon channel"),
        ("breathing kagome channel","domain-wall honeycomb channel","magnetic Chern channel","valley-Hall channel","driven Floquet channel","coupled-ring channel","frequency synthetic channel","chiral magnon channel"),
        ("-30 dB","-34 dB","-38 dB","-42 dB","-46 dB","-50 dB","-54 dB","-58 dB"),
        ("-32 dB","-36 dB","-40 dB","-44 dB","-48 dB","-52 dB","-56 dB","-60 dB"),161801,1408000),
    _Domain("late_interaction_optical_pressure_bridge","late-interaction optical-pressure bridge","LO","interface","pressure loss",
        ("NV cavity","SiV cavity","QD cavity","Er cavity","Yb cavity","Rb cell","Cs cell","Tm crystal"),
        ("critical NV cavity","nanophotonic SiV cavity","charged-QD cavity","rare-earth Er cavity","spectral Yb cavity","EIT Rb cell","spin-wave Cs cell","cryogenic Tm crystal"),
        ("3 nPa","5 nPa","7 nPa","9 nPa","11 nPa","13 nPa","15 nPa","17 nPa"),
        ("4 nPa","6 nPa","8 nPa","10 nPa","12 nPa","14 nPa","16 nPa","18 nPa"),161901,1409000),
    _Domain("late_interaction_molecular_spin_clock","late-interaction molecular-spin clock","LD","species","spin jitter",
        ("HfF+ packet","ThO beam","YbF beam","BaF beam","CaF beam","SrF beam","OCS beam","HCN beam"),
        ("oriented HfF+ packet","state-selected ThO beam","slow YbF beam","cold BaF beam","laser-cooled CaF beam","collimated SrF beam","Stark-decelerated OCS beam","oriented HCN beam"),
        ("0.05 rad","0.08 rad","0.11 rad","0.14 rad","0.17 rad","0.20 rad","0.23 rad","0.26 rad"),
        ("0.065 rad","0.095 rad","0.125 rad","0.155 rad","0.185 rad","0.215 rad","0.245 rad","0.275 rad"),162001,1410000),
    _Domain("late_interaction_vacuum_recoil_camera","late-interaction vacuum-recoil camera","LV","surface","recoil floor",
        ("Au sphere","Ag sphere","Si plate","graphene sheet","Cu sphere","Al sphere","ITO plate","diamond plate"),
        ("annealed Au sphere","template Ag sphere","passivated Si plate","gated graphene sheet","oxide-free Cu sphere","epitaxial Al sphere","ENZ ITO plate","CVD diamond plate"),
        ("0.12 pm","0.18 pm","0.24 pm","0.30 pm","0.36 pm","0.42 pm","0.48 pm","0.54 pm"),
        ("0.15 pm","0.21 pm","0.27 pm","0.33 pm","0.39 pm","0.45 pm","0.51 pm","0.57 pm"),162101,1411000),
    _Domain("late_interaction_atomic_phase_compass","late-interaction atomic-phase compass","LA","atom source","phase noise",
        ("Rb fountain","Cs fountain","Sr lattice","Yb lattice","K cloud","Li cloud","Na cloud","Ca cloud"),
        ("delta-kicked Rb fountain","dual-zone Cs fountain","interleaved Sr lattice","clocked Yb lattice","spin-polarized K cloud","degenerate Li cloud","collimated Na cloud","laser-cooled Ca cloud"),
        ("0.6 urad","0.9 urad","1.2 urad","1.5 urad","1.8 urad","2.1 urad","2.4 urad","2.7 urad"),
        ("0.75 urad","1.05 urad","1.35 urad","1.65 urad","1.95 urad","2.25 urad","2.55 urad","2.85 urad"),162201,1412000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the S53 late-interaction {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S53-Late {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S53-Late record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S53-Late {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S53-Late record {code}?",
            f"For S53-Late {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S53-Late record {code}?",
        )
    return (
        f"S53-Late audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S53-Late audit {code}: what is {spec.field_a}?",
        f"In S53-Late audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S53-Late audit {code}: what is {spec.field_b}?",
        f"In S53-Late audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    aa=spec.train_a if split=="train" else spec.dev_a
    bb=spec.train_b if split=="train" else spec.dev_b
    first=aa[index%8]; second=bb[(index*3+1)%8]
    wrong_a=aa[(index+3)%8]; wrong_b=bb[(index*5+2)%8]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    cid=f"{split}-s53-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S53LateInteractionCase(cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb)


def generate_s53_cases(split:Split)->tuple[S53LateInteractionCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s53_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S53 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12: raise RuntimeError("S53 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains: raise RuntimeError("S53 domains changed")
    if any(r.language!="en" for r in (*train,*dev)): raise RuntimeError("S53 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows): raise RuntimeError("S53 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S53 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2: raise RuntimeError("S53 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4: raise RuntimeError("S53 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S53 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}; ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}; dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}; do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds: raise RuntimeError("S53 TRAIN DEV state overlap")
    if tq&dq: raise RuntimeError("S53 TRAIN DEV question overlap")
    if to&do: raise RuntimeError("S53 TRAIN DEV option overlap")


__all__=["S53LateInteractionCase","generate_s53_cases","validate_s53_partitions"]
