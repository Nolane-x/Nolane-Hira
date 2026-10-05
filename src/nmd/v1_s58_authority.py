from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

from .v1_s57_authority import generate_s57_cases

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S58ConsensusTeacherCase:
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
    _Domain("teacher_muon_tensor_scope","muon tensor scope","MT","tensor channel","phase threshold",
        ("diamond-11 cavity","SiC-13 cavity","quartz-17 cavity","sapphire-19 cavity","AlN-23 cavity","GaN-29 cavity","SiN-31 cavity","LiNbO3-37 cavity"),
        ("shielded diamond-41 cavity","cold SiC-43 cavity","etched quartz-47 cavity","whispering sapphire-53 cavity","piezo AlN-59 cavity","strained GaN-61 cavity","soft SiN-67 cavity","poled LiNbO3-71 cavity"),
        _seq("mrad",0.7,0.45),_seq("mrad",0.93,0.45),241101,1911000),
    _Domain("teacher_spin_flux_lattice","spin flux lattice","SF","lattice film","flux threshold",
        ("PtCo-11 film","CoFeB-13 film","NiFe-17 film","YIG-19 film","FeGa-23 film","CrI3-29 film","WTe2-31 film","FePS3-37 film"),
        ("DMI PtCo-41 film","biased CoFeB-43 film","annealed NiFe-47 film","low-loss YIG-53 film","strained FeGa-59 film","layered CrI3-61 film","helical WTe2-67 film","AFM FePS3-71 film"),
        _seq("fT",2.3,0.7),_seq("fT",2.65,0.7),241201,1912000),
    _Domain("teacher_phonon_shear_bridge","phonon shear bridge","PS","bridge guide","shear threshold",
        ("SiN-11 ridge","AlN-13 ridge","GaN-17 ridge","diamond-19 ridge","SiC-23 ridge","quartz-29 ridge","LiNbO3-31 ridge","Si-37 ridge"),
        ("soft SiN-41 ridge","piezo AlN-43 ridge","suspended GaN-47 ridge","nanobeam diamond-53 ridge","4H-SiC-59 ridge","cold quartz-61 ridge","periodic LiNbO3-67 ridge","phononic Si-71 ridge"),
        _seq("nstrain",0.11,0.07),_seq("nstrain",0.145,0.07),241301,1913000),
    _Domain("teacher_rydberg_vorticity_clock","Rydberg vorticity clock","RV","clock ensemble","vorticity threshold",
        ("Rb 81D","Cs 83S","Rb 87P","Cs 89D","K 91D","Na 97S","Sr 101D","Yb 103P"),
        ("dressed Rb 107D","shielded Cs 109S","circular Rb 113P","clocked Cs 127D","ultracold K 131D","collimated Na 137S","lattice Sr 139D","trapped Yb 149P"),
        _seq("nrad/mm2",1.7,0.9),_seq("nrad/mm2",2.15,0.9),241401,1914000),
    _Domain("teacher_moire_strain_map","moire strain map","MS","heterostack","strain threshold",
        ("WSe2-WS2-A","WS2-MoS2-B","MoTe2-MoSe2-C","graphene-hBN-D","graphene-MoS2-E","MoSe2-WSe2-F","MoS2-hBN-G","WTe2-WS2-H"),
        ("twist WSe2-WS2-I","gated WS2-MoS2-J","strained MoTe2-MoSe2-K","aligned graphene-hBN-L","hybrid graphene-MoS2-M","encapsulated MoSe2-WSe2-N","defect MoS2-hBN-O","moire WTe2-WS2-P"),
        _seq("ppm",0.31,0.13),_seq("ppm",0.375,0.13),241501,1915000),
    _Domain("teacher_neutron_torsion_lens","neutron torsion lens","NT","torsion analyzer","torsion threshold",
        ("He3-11 tube","Fe-13 lattice","Co-17 lattice","Si-19 grating","Ni-23 grating","mirror-29","diamond-31 grid","quartz-37 grid"),
        ("polarized He3-41 tube","exchange Fe-43 lattice","epitaxial Co-47 lattice","perfect-Si-53 grating","magnetized Ni-59 grating","m9 mirror-61","CVD diamond-67 grid","etched quartz-71 grid"),
        _seq("urad",0.19,0.08),_seq("urad",0.23,0.08),241601,1916000),
    _Domain("teacher_topological_impedance_radar","topological impedance radar","TI","edge channel","impedance threshold",
        ("kagome-11 channel","honeycomb-13 channel","Chern-17 channel","valley-19 channel","Floquet-23 channel","ring-29 channel","synthetic-31 channel","magnon-37 channel"),
        ("breathing kagome-41 channel","wall honeycomb-43 channel","magnetic Chern-47 channel","valley-Hall-53 channel","driven Floquet-59 channel","coupled-ring-61 channel","frequency synthetic-67 channel","chiral magnon-71 channel"),
        _seq("mOhm",1.6,0.55),_seq("mOhm",1.875,0.55),241701,1917000),
    _Domain("teacher_molecular_torsion_compass","molecular torsion compass","MC","molecular beam","torsion threshold",
        ("HfF-11 beam","ThO-13 beam","YbF-17 beam","BaF-19 beam","CaF-23 beam","SrF-29 beam","OCS-31 beam","HCN-37 beam"),
        ("oriented HfF-41 beam","selected ThO-43 beam","slow YbF-47 beam","cold BaF-53 beam","cooled CaF-59 beam","collimated SrF-61 beam","decelerated OCS-67 beam","oriented HCN-71 beam"),
        _seq("prad",0.8,0.38),_seq("prad",0.99,0.38),241801,1918000),
    _Domain("teacher_vacuum_susceptibility_scope","vacuum susceptibility scope","VS","surface cell","susceptibility threshold",
        ("Au-11 plate","Ag-13 plate","Si-17 plate","graphene-19 plate","Cu-23 plate","Al-29 plate","ITO-31 plate","diamond-37 plate"),
        ("annealed Au-41 plate","template Ag-43 plate","passivated Si-47 plate","gated graphene-53 plate","clean Cu-59 plate","epitaxial Al-61 plate","ENZ ITO-67 plate","CVD diamond-71 plate"),
        _seq("ppt",0.9,0.6),_seq("ppt",1.2,0.6),241901,1919000),
    _Domain("teacher_atomic_shear_camera","atomic shear camera","AS","atomic species","shear threshold",
        ("Rb-11 fountain","Cs-13 fountain","Sr-17 lattice","Yb-19 lattice","K-23 cloud","Li-29 cloud","Na-31 cloud","Ca-37 cloud"),
        ("kicked Rb-41 fountain","dual Cs-43 fountain","interleaved Sr-47 lattice","clocked Yb-53 lattice","spin K-59 cloud","degenerate Li-61 cloud","collimated Na-67 cloud","cooled Ca-71 cloud"),
        _seq("nstrain",1.8,0.72),_seq("nstrain",2.16,0.72),242001,1920000),
    _Domain("teacher_superfluid_vorticity_scope","superfluid vorticity scope","SV","quantum fluid","vorticity threshold",
        ("He4-11 layer","He3-13 layer","Rb-17 condensate","Na-19 condensate","K-23 condensate","Li-29 condensate","exciton-31 fluid","polariton-37 fluid"),
        ("vortex-free He4-41 layer","B-phase He3-43 layer","ring Rb-47 condensate","spinor Na-53 condensate","paired K-59 condensate","unitary Li-61 condensate","dipolar exciton-67 fluid","driven polariton-71 fluid"),
        _seq("mHz",0.17,0.09),_seq("mHz",0.215,0.09),242101,1921000),
    _Domain("teacher_ferroelectric_shear_bridge","ferroelectric shear bridge","FS","ferro crystal","shear threshold",
        ("BaTiO3-11 chip","KTaO3-13 chip","LiNbO3-17 chip","PbTiO3-19 chip","BiFeO3-23 chip","SrTiO3-29 chip","KNbO3-31 chip","PZT-37 chip"),
        ("strained BaTiO3-41 chip","quantum KTaO3-43 chip","poled LiNbO3-47 chip","epitaxial PbTiO3-53 chip","multiferroic BiFeO3-59 chip","isotopic SrTiO3-61 chip","domain KNbO3-67 chip","textured PZT-71 chip"),
        _seq("nstrain",0.52,0.33),_seq("nstrain",0.685,0.33),242201,1922000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the S58 consensus-teacher {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the S58 teacher-consensus {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S58-Teacher {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S58-Teacher record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S58-Teacher {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S58-Teacher record {code}?",
            f"For S58-Teacher {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S58-Teacher record {code}?",
        )
    return (
        f"S58-Teacher audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Teacher audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S58-Teacher audit {code}: what is {spec.field_a}?",
        f"In S58-Teacher audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S58-Teacher audit {code}: what is {spec.field_b}?",
        f"In S58-Teacher audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    aa=spec.train_a if split=="train" else spec.dev_a
    bb=spec.train_b if split=="train" else spec.dev_b
    first=aa[index%8]; second=bb[(index*3+1)%8]
    wrong_a=aa[(index+3)%8]; wrong_b=bb[(index*5+2)%8]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    cid=f"{split}-s58-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S58ConsensusTeacherCase(
        cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb
    )


def generate_s58_cases(split:Split)->tuple[S58ConsensusTeacherCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def _states(rows): return {x for r in rows for x in (r.state_a,r.state_b)}
def _questions(rows): return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
def _options(rows): return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def validate_s58_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S58 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12: raise RuntimeError("S58 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S58 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S58 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S58 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S58 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S58 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S58 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S58 paired golds must differ")
    if _states(train)&_states(dev): raise RuntimeError("S58 TRAIN DEV state overlap")
    if _questions(train)&_questions(dev): raise RuntimeError("S58 TRAIN DEV question overlap")
    if _options(train)&_options(dev): raise RuntimeError("S58 TRAIN DEV option overlap")

    prior=(*generate_s57_cases("train"),*generate_s57_cases("dev"))
    current=(*train,*dev)
    if _states(current)&_states(prior): raise RuntimeError("S58 exact S57 state overlap")
    if _questions(current)&_questions(prior): raise RuntimeError("S58 exact S57 question overlap")
    if _options(current)&_options(prior): raise RuntimeError("S58 exact S57 option overlap")


__all__=["S58ConsensusTeacherCase","generate_s58_cases","validate_s58_partitions"]
