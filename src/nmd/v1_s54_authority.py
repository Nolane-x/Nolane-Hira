from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S54JointInteractionCase:
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
    _Domain("triadic_axion_recoil_scope","triadic axion recoil scope","AR","resonator","recoil floor",
        ("Nb loop","Ta loop","Cu ring","Si ring","Al ring","quartz loop","graphene ring","sapphire ring"),
        ("flux-locked Nb loop","shielded Ta loop","annealed Cu ring","cryogenic Si ring","epitaxial Al ring","superpolished quartz loop","ballistic graphene ring","whispering sapphire ring"),
        ("1.6 neV","2.2 neV","2.8 neV","3.4 neV","4.0 neV","4.6 neV","5.2 neV","5.8 neV"),
        ("1.9 neV","2.5 neV","3.1 neV","3.7 neV","4.3 neV","4.9 neV","5.5 neV","6.1 neV"),181101,1511000),
    _Domain("triadic_spin_curvature_camera","triadic spin curvature camera","SC","film","curvature blur",
        ("PtCo film","CoFeB film","NiFe film","FeGa film","YIG film","CrI3 film","WTe2 film","FePS3 film"),
        ("DMI PtCo film","biased CoFeB film","low-loss NiFe film","strained FeGa film","epitaxial YIG film","layered CrI3 film","helical WTe2 film","AFM FePS3 film"),
        ("0.07 mm-1","0.10 mm-1","0.13 mm-1","0.16 mm-1","0.19 mm-1","0.22 mm-1","0.25 mm-1","0.28 mm-1"),
        ("0.085 mm-1","0.115 mm-1","0.145 mm-1","0.175 mm-1","0.205 mm-1","0.235 mm-1","0.265 mm-1","0.295 mm-1"),181201,1512000),
    _Domain("triadic_phonon_pressure_lens","triadic phonon pressure lens","PP","guide","pressure floor",
        ("SiN ridge","AlN ridge","GaN ridge","SiC ridge","diamond ridge","quartz ridge","LiNbO3 ridge","Si ridge"),
        ("soft SiN ridge","piezo AlN ridge","suspended GaN ridge","4H-SiC ridge","nanobeam diamond ridge","cryogenic quartz ridge","periodic LiNbO3 ridge","phononic Si ridge"),
        ("2 nPa","3 nPa","4 nPa","5 nPa","6 nPa","7 nPa","8 nPa","9 nPa"),
        ("2.5 nPa","3.5 nPa","4.5 nPa","5.5 nPa","6.5 nPa","7.5 nPa","8.5 nPa","9.5 nPa"),181301,1513000),
    _Domain("triadic_rydberg_torque_map","triadic Rydberg torque map","RT","ensemble","torque floor",
        ("Rb 48D","Cs 54S","Rb 63P","Cs 70D","K 41D","Na 46S","Sr 57D","Yb 61P"),
        ("dressed Rb 48D","shielded Cs 54S","circular Rb 63P","clocked Cs 70D","ultracold K 41D","collimated Na 46S","lattice Sr 57D","trapped Yb 61P"),
        ("2 zNm","3 zNm","4 zNm","5 zNm","6 zNm","7 zNm","8 zNm","9 zNm"),
        ("2.5 zNm","3.5 zNm","4.5 zNm","5.5 zNm","6.5 zNm","7.5 zNm","8.5 zNm","9.5 zNm"),181401,1514000),
    _Domain("triadic_moire_gradient_clock","triadic moire gradient clock","MG","stack","gradient floor",
        ("WSe2-MoSe2","WS2-WSe2","MoS2-WS2","graphene-hBN","graphene-graphene","MoTe2-WSe2","MoSe2-hBN","WS2-hBN"),
        ("twist WSe2-MoSe2","gated WS2-WSe2","encapsulated MoS2-WS2","aligned graphene-hBN","magic-angle graphene-graphene","strained MoTe2-WSe2","defect MoSe2-hBN","moire WS2-hBN"),
        ("0.04 V/mm","0.06 V/mm","0.08 V/mm","0.10 V/mm","0.12 V/mm","0.14 V/mm","0.16 V/mm","0.18 V/mm"),
        ("0.05 V/mm","0.07 V/mm","0.09 V/mm","0.11 V/mm","0.13 V/mm","0.15 V/mm","0.17 V/mm","0.19 V/mm"),181501,1515000),
    _Domain("triadic_neutron_torque_compass","triadic neutron torque compass","NT","analyzer","torque blur",
        ("He3 cell","Fe foil","Co foil","Si grating","Ni grating","supermirror","diamond grating","quartz grating"),
        ("polarized He3 cell","exchange Fe foil","epitaxial Co foil","perfect-Si grating","magnetized Ni grating","m=5 supermirror","CVD diamond grating","etched quartz grating"),
        ("0.12 zNm","0.18 zNm","0.24 zNm","0.30 zNm","0.36 zNm","0.42 zNm","0.48 zNm","0.54 zNm"),
        ("0.15 zNm","0.21 zNm","0.27 zNm","0.33 zNm","0.39 zNm","0.45 zNm","0.51 zNm","0.57 zNm"),181601,1516000),
    _Domain("triadic_topological_recoil_radar","triadic topological recoil radar","TR","channel","recoil leakage",
        ("kagome edge","honeycomb edge","Chern edge","valley edge","Floquet edge","ring edge","synthetic edge","magnon edge"),
        ("breathing kagome edge","domain-wall honeycomb edge","magnetic Chern edge","valley-Hall edge","driven Floquet edge","coupled-ring edge","frequency synthetic edge","chiral magnon edge"),
        ("-27 dB","-31 dB","-35 dB","-39 dB","-43 dB","-47 dB","-51 dB","-55 dB"),
        ("-29 dB","-33 dB","-37 dB","-41 dB","-45 dB","-49 dB","-53 dB","-57 dB"),181701,1517000),
    _Domain("triadic_molecular_pressure_scope","triadic molecular pressure scope","MP","beam","pressure drift",
        ("HfF+ beam","ThO beam","YbF beam","BaF beam","CaF beam","SrF beam","OCS beam","HCN beam"),
        ("oriented HfF+ beam","selected ThO beam","slow YbF beam","cold BaF beam","cooled CaF beam","collimated SrF beam","decelerated OCS beam","oriented HCN beam"),
        ("4 uPa","6 uPa","8 uPa","10 uPa","12 uPa","14 uPa","16 uPa","18 uPa"),
        ("5 uPa","7 uPa","9 uPa","11 uPa","13 uPa","15 uPa","17 uPa","19 uPa"),181801,1518000),
    _Domain("triadic_vacuum_gradient_bridge","triadic vacuum gradient bridge","VG","surface","gradient floor",
        ("Au sphere","Ag sphere","Si plate","graphene sheet","Cu sphere","Al sphere","ITO plate","diamond plate"),
        ("annealed Au sphere","template Ag sphere","passivated Si plate","gated graphene sheet","oxide-free Cu sphere","epitaxial Al sphere","ENZ ITO plate","CVD diamond plate"),
        ("0.13 pN/mm","0.19 pN/mm","0.25 pN/mm","0.31 pN/mm","0.37 pN/mm","0.43 pN/mm","0.49 pN/mm","0.55 pN/mm"),
        ("0.16 pN/mm","0.22 pN/mm","0.28 pN/mm","0.34 pN/mm","0.40 pN/mm","0.46 pN/mm","0.52 pN/mm","0.58 pN/mm"),181901,1519000),
    _Domain("triadic_atomic_recoil_camera","triadic atomic recoil camera","AC","species","recoil floor",
        ("Rb cloud","Cs cloud","Sr lattice","Yb lattice","K cloud","Li cloud","Na cloud","Ca cloud"),
        ("delta-kicked Rb cloud","dual-zone Cs cloud","interleaved Sr lattice","clocked Yb lattice","spin K cloud","degenerate Li cloud","collimated Na cloud","cooled Ca cloud"),
        ("3 mm/s","4 mm/s","5 mm/s","6 mm/s","7 mm/s","8 mm/s","9 mm/s","10 mm/s"),
        ("3.5 mm/s","4.5 mm/s","5.5 mm/s","6.5 mm/s","7.5 mm/s","8.5 mm/s","9.5 mm/s","10.5 mm/s"),182001,1520000),
    _Domain("triadic_superfluid_curvature_clock","triadic superfluid curvature clock","SU","fluid","curvature noise",
        ("He4 film","He3 film","Rb condensate","Na condensate","K condensate","Li condensate","exciton fluid","polariton fluid"),
        ("vortex-free He4 film","B-phase He3 film","ring Rb condensate","spinor Na condensate","paired K condensate","unitary Li condensate","dipolar exciton fluid","driven polariton fluid"),
        ("0.05 mm-1","0.08 mm-1","0.11 mm-1","0.14 mm-1","0.17 mm-1","0.20 mm-1","0.23 mm-1","0.26 mm-1"),
        ("0.065 mm-1","0.095 mm-1","0.125 mm-1","0.155 mm-1","0.185 mm-1","0.215 mm-1","0.245 mm-1","0.275 mm-1"),182101,1521000),
    _Domain("triadic_ferroelectric_pressure_map","triadic ferroelectric pressure map","FP","crystal","pressure jitter",
        ("BaTiO3 slab","KTaO3 slab","LiNbO3 slab","PbTiO3 slab","BiFeO3 slab","SrTiO3 slab","KNbO3 slab","PZT slab"),
        ("strained BaTiO3 slab","quantum KTaO3 slab","poled LiNbO3 slab","epitaxial PbTiO3 slab","multiferroic BiFeO3 slab","isotopic SrTiO3 slab","domain KNbO3 slab","textured PZT slab"),
        ("0.6 nPa","0.9 nPa","1.2 nPa","1.5 nPa","1.8 nPa","2.1 nPa","2.4 nPa","2.7 nPa"),
        ("0.75 nPa","1.05 nPa","1.35 nPa","1.65 nPa","1.95 nPa","2.25 nPa","2.55 nPa","2.85 nPa"),182201,1522000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the S54 triadic {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S54-Triadic {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S54-Triadic record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S54-Triadic {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S54-Triadic record {code}?",
            f"For S54-Triadic {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S54-Triadic record {code}?",
        )
    return (
        f"S54-Triadic audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S54-Triadic audit {code}: what is {spec.field_a}?",
        f"In S54-Triadic audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S54-Triadic audit {code}: what is {spec.field_b}?",
        f"In S54-Triadic audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    aa=spec.train_a if split=="train" else spec.dev_a
    bb=spec.train_b if split=="train" else spec.dev_b
    first=aa[index%8]
    second=bb[(index*3+1)%8]
    wrong_a=aa[(index+3)%8]
    wrong_b=bb[(index*5+2)%8]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    cid=f"{split}-s54-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S54JointInteractionCase(
        cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,
        texts,aliases,ids,ga,gb
    )


def generate_s54_cases(split:Split)->tuple[S54JointInteractionCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s54_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S54 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12: raise RuntimeError("S54 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S54 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S54 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S54 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b:
                raise RuntimeError("S54 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S54 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S54 option IDs changed")
            if r.gold_a==r.gold_b:
                raise RuntimeError("S54 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}
    ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds: raise RuntimeError("S54 TRAIN DEV state overlap")
    if tq&dq: raise RuntimeError("S54 TRAIN DEV question overlap")
    if to&do: raise RuntimeError("S54 TRAIN DEV option overlap")


__all__=["S54JointInteractionCase","generate_s54_cases","validate_s54_partitions"]
