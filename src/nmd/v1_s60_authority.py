from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

from .v1_s60_authority import generate_s60_cases

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S60HybridCase:
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


def _seq(unit,start,step):
    return tuple(f"{start+i*step:g} {unit}" for i in range(8))


_DOMAINS=(
    _Domain("hybrid60_anyon_phase_scope","anyon phase scope","AP","anyon substrate","braid threshold",
        ("kagome-107 tile","Chern-109 tile","moire-113 tile","QAH-127 tile","spin-131 tile","flux-137 tile","valley-139 tile","ring-149 tile"),
        ("braided kagome-151 tile","gated Chern-157 tile","twisted moire-163 tile","cold QAH-167 tile","dressed spin-173 tile","threaded flux-179 tile","split valley-181 tile","coupled ring-191 tile"),
        _seq("mrad",0.43,0.31),_seq("mrad",0.585,0.31),251101,2111000),
    _Domain("hybrid60_graviton_shear_lens","graviton shear lens","GS","shear resonator","strain threshold",
        ("Si-107 beam","SiN-109 beam","AlN-113 beam","GaN-127 beam","diamond-131 beam","SiC-137 beam","quartz-139 beam","sapphire-149 beam"),
        ("etched Si-151 beam","soft SiN-157 beam","piezo AlN-163 beam","strained GaN-167 beam","CVD diamond-173 beam","4H-SiC-179 beam","cold quartz-181 beam","whispering sapphire-191 beam"),
        _seq("fstrain",1.4,0.62),_seq("fstrain",1.71,0.62),251201,2112000),
    _Domain("hybrid60_axion_impedance_map","axion impedance map","AI","conversion cavity","impedance threshold",
        ("Nb-107 cavity","Al-109 cavity","Cu-113 cavity","NbTi-127 cavity","MgB2-131 cavity","YBCO-137 cavity","Ta-139 cavity","MoRe-149 cavity"),
        ("shielded Nb-151 cavity","clean Al-157 cavity","annealed Cu-163 cavity","high-Q NbTi-167 cavity","thin MgB2-173 cavity","film YBCO-179 cavity","etched Ta-181 cavity","alloy MoRe-191 cavity"),
        _seq("uOhm",0.29,0.16),_seq("uOhm",0.37,0.16),251301,2113000),
    _Domain("hybrid60_magnon_vorticity_clock","magnon vorticity clock","MV","magnon lattice","vorticity threshold",
        ("YIG-107 lattice","NiFe-109 lattice","CoFeB-113 lattice","FeGa-127 lattice","PtCo-131 lattice","CrI3-137 lattice","FePS3-139 lattice","WTe2-149 lattice"),
        ("low-loss YIG-151 lattice","annealed NiFe-157 lattice","biased CoFeB-163 lattice","strained FeGa-167 lattice","DMI PtCo-173 lattice","layered CrI3-179 lattice","AFM FePS3-181 lattice","helical WTe2-191 lattice"),
        _seq("mHz",0.23,0.11),_seq("mHz",0.285,0.11),251401,2114000),
    _Domain("hybrid60_neutrino_phase_bridge","neutrino phase bridge","NP","phase detector","phase threshold",
        ("Ge-107 detector","Xe-109 detector","Ar-113 detector","Ne-127 detector","He-131 detector","Si-137 detector","Te-139 detector","Mo-149 detector"),
        ("enriched Ge-151 detector","dual Xe-157 detector","cold Ar-163 detector","low-noise Ne-167 detector","polarized He-173 detector","isotopic Si-179 detector","bolometric Te-181 detector","crystal Mo-191 detector"),
        _seq("prad",0.61,0.27),_seq("prad",0.745,0.27),251501,2115000),
    _Domain("hybrid60_photon_drag_compass","photon drag compass","PD","optical guide","drag threshold",
        ("SiN-107 guide","LiNbO3-109 guide","AlN-113 guide","GaP-127 guide","SiC-131 guide","diamond-137 guide","silica-139 guide","Ta2O5-149 guide"),
        ("slow SiN-151 guide","poled LiNbO3-157 guide","piezo AlN-163 guide","etched GaP-167 guide","4H-SiC-173 guide","NV diamond-179 guide","hollow silica-181 guide","low-loss Ta2O5-191 guide"),
        _seq("fm/s",1.1,0.48),_seq("fm/s",1.34,0.48),251601,2116000),
    _Domain("hybrid60_fermi_pressure_scope","Fermi pressure scope","FP","quantum gas","pressure threshold",
        ("Li6-107 gas","K40-109 gas","Sr87-113 gas","Yb171-127 gas","Yb173-131 gas","Dy161-137 gas","Er167-139 gas","Cr53-149 gas"),
        ("unitary Li6-151 gas","paired K40-157 gas","lattice Sr87-163 gas","clock Yb171-167 gas","spin Yb173-173 gas","dipolar Dy161-179 gas","dipolar Er167-181 gas","polarized Cr53-191 gas"),
        _seq("pPa",0.74,0.39),_seq("pPa",0.935,0.39),251701,2117000),
    _Domain("hybrid60_vacuum_torque_radar","vacuum torque radar","VT","surface pair","torque threshold",
        ("AuSi-107 pair","AgSi-109 pair","CuSi-113 pair","AlSi-127 pair","grapheneSi-131 pair","ITOSi-137 pair","diamondSi-139 pair","hBNSi-149 pair"),
        ("clean AuSi-151 pair","template AgSi-157 pair","annealed CuSi-163 pair","epitaxial AlSi-167 pair","gated grapheneSi-173 pair","ENZ ITOSi-179 pair","CVD diamondSi-181 pair","aligned hBNSi-191 pair"),
        _seq("zNm",0.12,0.07),_seq("zNm",0.155,0.07),251801,2118000),
    _Domain("hybrid60_atomic_recoil_camera","atomic recoil camera","AR","atom array","recoil threshold",
        ("Rb-107 array","Cs-109 array","Sr-113 array","Yb-127 array","K-131 array","Na-137 array","Ca-139 array","Li-149 array"),
        ("tweezer Rb-151 array","clock Cs-157 array","lattice Sr-163 array","clock Yb-167 array","spin K-173 array","cooled Na-179 array","ion Ca-181 array","degenerate Li-191 array"),
        _seq("fm",0.33,0.14),_seq("fm",0.4,0.14),251901,2119000),
    _Domain("hybrid60_molecular_flux_scope","molecular flux scope","MF","molecular beam","flux threshold",
        ("ThO-107 beam","HfF-109 beam","YbF-113 beam","BaF-127 beam","CaF-131 beam","SrF-137 beam","OCS-139 beam","NH3-149 beam"),
        ("selected ThO-151 beam","oriented HfF-157 beam","slow YbF-163 beam","cold BaF-167 beam","cooled CaF-173 beam","collimated SrF-179 beam","decelerated OCS-181 beam","focused NH3-191 beam"),
        _seq("a.u.",1.3,0.57),_seq("a.u.",1.585,0.57),252001,2120000),
    _Domain("hybrid60_superfluid_drag_map","superfluid drag map","SD","quantum fluid","drag threshold",
        ("He4-107 film","He3-109 film","Rb-113 ring","Na-127 ring","K-131 ring","Li-137 ring","exciton-139 fluid","polariton-149 fluid"),
        ("vortex He4-151 film","B-phase He3-157 film","persistent Rb-163 ring","spinor Na-167 ring","paired K-173 ring","unitary Li-179 ring","dipolar exciton-181 fluid","driven polariton-191 fluid"),
        _seq("nN",0.15,0.075),_seq("nN",0.1875,0.075),252101,2121000),
    _Domain("hybrid60_ferroelectric_phase_lens","ferroelectric phase lens","FL","ferro stack","phase threshold",
        ("BaTiO3-107 stack","KTaO3-109 stack","LiNbO3-113 stack","PbTiO3-127 stack","BiFeO3-131 stack","SrTiO3-137 stack","KNbO3-139 stack","PZT-149 stack"),
        ("strained BaTiO3-151 stack","quantum KTaO3-157 stack","poled LiNbO3-163 stack","epitaxial PbTiO3-167 stack","multiferroic BiFeO3-173 stack","isotopic SrTiO3-179 stack","domain KNbO3-181 stack","textured PZT-191 stack"),
        _seq("urad",0.41,0.22),_seq("urad",0.52,0.22),252201,2122000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the S60 bounded-hybrid {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the S60 hybrid-composer {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S60-Hybrid {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S60-Hybrid record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S60-Hybrid {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S60-Hybrid record {code}?",
            f"For S60-Hybrid {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S60-Hybrid record {code}?",
        )
    return (
        f"S60-Hybrid audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Pair-head audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S60-Hybrid audit {code}: what is {spec.field_a}?",
        f"In S60-Hybrid audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S60-Hybrid audit {code}: what is {spec.field_b}?",
        f"In S60-Hybrid audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    aa=spec.train_a if split=="train" else spec.dev_a
    bb=spec.train_b if split=="train" else spec.dev_b
    first=aa[index%8]; second=bb[(index*5+1)%8]
    wrong_a=aa[(index+3)%8]; wrong_b=bb[(index*7+2)%8]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    cid=f"{split}-s60-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S60HybridCase(
        cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb
    )


def generate_s60_cases(split:Split)->tuple[S60HybridCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def _states(rows): return {x for r in rows for x in (r.state_a,r.state_b)}
def _questions(rows): return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
def _options(rows): return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def validate_s60_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S60 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12: raise RuntimeError("S60 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S60 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S60 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S60 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S60 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S60 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S60 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S60 paired golds must differ")
    if _states(train)&_states(dev): raise RuntimeError("S60 TRAIN DEV state overlap")
    if _questions(train)&_questions(dev): raise RuntimeError("S60 TRAIN DEV question overlap")
    if _options(train)&_options(dev): raise RuntimeError("S60 TRAIN DEV option overlap")

    prior=(*generate_s60_cases("train"),*generate_s60_cases("dev"))
    current=(*train,*dev)
    if _states(current)&_states(prior): raise RuntimeError("S60 exact S58 state overlap")
    if _questions(current)&_questions(prior): raise RuntimeError("S60 exact S58 question overlap")
    if _options(current)&_options(prior): raise RuntimeError("S60 exact S58 option overlap")


__all__=["S60HybridCase","generate_s60_cases","validate_s60_partitions"]
