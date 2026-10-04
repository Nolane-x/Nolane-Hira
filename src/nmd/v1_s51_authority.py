from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S51PersistedNativeCase:
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
    _Domain("persisted_native_axion_phase_array","persisted-native axion phase array","XA","resonator","phase floor",
        ("niobium toroid","copper toroid","sapphire ring","silicon ring","alumina ring","tantalum loop","quartz loop","graphene ring"),
        ("flux-locked niobium toroid","annealed copper toroid","whispering sapphire ring","cryogenic silicon ring","low-loss alumina ring","shielded tantalum loop","superpolished quartz loop","ballistic graphene ring"),
        ("1.7 urad","2.3 urad","2.9 urad","3.5 urad","4.1 urad","4.7 urad","5.3 urad","5.9 urad"),
        ("1.95 urad","2.55 urad","3.15 urad","3.75 urad","4.35 urad","4.95 urad","5.55 urad","6.15 urad"),131101,1201000),
    _Domain("persisted_native_gravity_gradiometer","persisted-native gravity gradiometer","GG","atom source","gradient floor",
        ("Rb fountain","Cs fountain","Sr lattice","Yb lattice","K cloud","Li cloud","Na cloud","Ca cloud"),
        ("delta-kicked Rb fountain","dual-zone Cs fountain","interleaved Sr lattice","clocked Yb lattice","spin-polarized K cloud","degenerate Li cloud","collimated Na cloud","laser-cooled Ca cloud"),
        ("2.4 E","3.2 E","4.0 E","4.8 E","5.6 E","6.4 E","7.2 E","8.0 E"),
        ("2.8 E","3.6 E","4.4 E","5.2 E","6.0 E","6.8 E","7.6 E","8.4 E"),131201,1202000),
    _Domain("persisted_native_chiral_phonon_scope","persisted-native chiral phonon scope","CP","waveguide","chirality leak",
        ("Si ridge","SiN ridge","AlN ridge","GaN ridge","diamond ridge","SiC ridge","LiNbO3 ridge","quartz ridge"),
        ("phononic Si ridge","soft-clamped SiN ridge","piezoelectric AlN ridge","suspended GaN ridge","nanobeam diamond ridge","4H-SiC ridge","periodic LiNbO3 ridge","cryogenic quartz ridge"),
        ("-31 dB","-35 dB","-39 dB","-43 dB","-47 dB","-51 dB","-55 dB","-59 dB"),
        ("-33 dB","-37 dB","-41 dB","-45 dB","-49 dB","-53 dB","-57 dB","-61 dB"),131301,1203000),
    _Domain("persisted_native_spin_orbit_clock","persisted-native spin-orbit clock","SO","channel","precession drift",
        ("InSb wire","InAs wire","GaAs wire","Ge wire","Si wire","WTe2 edge","HgTe edge","graphene edge"),
        ("ballistic InSb wire","epitaxial InAs wire","gated GaAs wire","strained Ge wire","isotopic Si wire","helical WTe2 edge","inverted HgTe edge","valley graphene edge"),
        ("0.05 rad","0.08 rad","0.11 rad","0.14 rad","0.17 rad","0.20 rad","0.23 rad","0.26 rad"),
        ("0.065 rad","0.095 rad","0.125 rad","0.155 rad","0.185 rad","0.215 rad","0.245 rad","0.275 rad"),131401,1204000),
    _Domain("persisted_native_rydberg_microwave_lens","persisted-native Rydberg microwave lens","RM","ensemble","field floor",
        ("Rb 46D","Cs 52S","Rb 61P","Cs 68D","K 39D","Na 44S","Sr 55D","Yb 59P"),
        ("dressed Rb 46D","shielded Cs 52S","circular Rb 61P","clocked Cs 68D","ultracold K 39D","collimated Na 44S","lattice Sr 55D","trapped Yb 59P"),
        ("7 nV/cm","10 nV/cm","13 nV/cm","16 nV/cm","19 nV/cm","22 nV/cm","25 nV/cm","28 nV/cm"),
        ("8.5 nV/cm","11.5 nV/cm","14.5 nV/cm","17.5 nV/cm","20.5 nV/cm","23.5 nV/cm","26.5 nV/cm","29.5 nV/cm"),131501,1205000),
    _Domain("persisted_native_vacuum_force_camera","persisted-native vacuum force camera","VF","surface pair","force floor",
        ("Au-Si pair","Ag-Si pair","Au-graphene pair","Cu-quartz pair","Al-sapphire pair","ITO-Si pair","diamond-Au pair","SiC-Au pair"),
        ("template Au-Si pair","annealed Ag-Si pair","gated Au-graphene pair","oxide-free Cu-quartz pair","epitaxial Al-sapphire pair","ENZ ITO-Si pair","CVD diamond-Au pair","4H-SiC-Au pair"),
        ("0.18 pN","0.24 pN","0.30 pN","0.36 pN","0.42 pN","0.48 pN","0.54 pN","0.60 pN"),
        ("0.21 pN","0.27 pN","0.33 pN","0.39 pN","0.45 pN","0.51 pN","0.57 pN","0.63 pN"),131601,1206000),
    _Domain("persisted_native_neutron_spin_mapper","persisted-native neutron spin mapper","NM","analyzer","spin blur",
        ("He3 cell","Fe foil","Co foil","Si grating","Ni grating","supermirror","diamond grating","quartz grating"),
        ("polarized He3 cell","exchange Fe foil","epitaxial Co foil","perfect-Si grating","magnetized Ni grating","m=5 supermirror","CVD diamond grating","etched quartz grating"),
        ("0.16 mrad","0.22 mrad","0.28 mrad","0.34 mrad","0.40 mrad","0.46 mrad","0.52 mrad","0.58 mrad"),
        ("0.19 mrad","0.25 mrad","0.31 mrad","0.37 mrad","0.43 mrad","0.49 mrad","0.55 mrad","0.61 mrad"),131701,1207000),
    _Domain("persisted_native_moire_charge_holograph","persisted-native moire charge holograph","MC","stack","charge blur",
        ("WSe2-MoSe2","WS2-WSe2","MoS2-WS2","graphene-hBN","graphene-graphene","MoTe2-WSe2","MoSe2-hBN","WS2-hBN"),
        ("twist-locked WSe2-MoSe2","gated WS2-WSe2","encapsulated MoS2-WS2","aligned graphene-hBN","magic-angle graphene-graphene","strain-tuned MoTe2-WSe2","defect MoSe2-hBN","moire WS2-hBN"),
        ("0.09 e","0.13 e","0.17 e","0.21 e","0.25 e","0.29 e","0.33 e","0.37 e"),
        ("0.11 e","0.15 e","0.19 e","0.23 e","0.27 e","0.31 e","0.35 e","0.39 e"),131801,1208000),
    _Domain("persisted_native_optomechanical_accelerometer","persisted-native optomechanical accelerometer","OA","oscillator","acceleration floor",
        ("SiN trampoline","silica sphere","diamond beam","SiC beam","GaAs disk","AlN disk","graphene drum","quartz fork"),
        ("soft-clamped SiN trampoline","levitated silica sphere","phononic diamond beam","4H-SiC beam","whispering GaAs disk","piezo AlN disk","tensioned graphene drum","cryogenic quartz fork"),
        ("3 ng","5 ng","7 ng","9 ng","11 ng","13 ng","15 ng","17 ng"),
        ("4 ng","6 ng","8 ng","10 ng","12 ng","14 ng","16 ng","18 ng"),131901,1209000),
    _Domain("persisted_native_nuclear_quadrupole_clock","persisted-native nuclear quadrupole clock","NQ","isotope","quadrupole drift",
        ("Lu176 ion","Yb171 ion","Sr87 atom","Al27 ion","Hg199 ion","Ca43 ion","Th229 crystal","U235 crystal"),
        ("hyperfine Lu176 ion","logic-readout Yb171 ion","lattice Sr87 atom","clock Al27 ion","cryogenic Hg199 ion","trapped Ca43 ion","isomer Th229 crystal","shielded U235 crystal"),
        ("0.8 mHz","1.1 mHz","1.4 mHz","1.7 mHz","2.0 mHz","2.3 mHz","2.6 mHz","2.9 mHz"),
        ("0.95 mHz","1.25 mHz","1.55 mHz","1.85 mHz","2.15 mHz","2.45 mHz","2.75 mHz","3.05 mHz"),132001,1210000),
    _Domain("persisted_native_quantum_acoustic_tomograph","persisted-native quantum acoustic tomograph","QA","transducer","displacement floor",
        ("AlN resonator","LiNbO3 resonator","SiN membrane","diamond disk","GaAs disk","SiC disk","quartz plate","graphene drum"),
        ("piezo AlN resonator","periodic LiNbO3 resonator","soft-clamped SiN membrane","nanobeam diamond disk","phononic GaAs disk","4H-SiC disk","low-loss quartz plate","suspended graphene drum"),
        ("1.4 fm","1.9 fm","2.4 fm","2.9 fm","3.4 fm","3.9 fm","4.4 fm","4.9 fm"),
        ("1.65 fm","2.15 fm","2.65 fm","3.15 fm","3.65 fm","4.15 fm","4.65 fm","5.15 fm"),132101,1211000),
    _Domain("persisted_native_topological_heat_compass","persisted-native topological heat compass","HT","channel","thermal leakage",
        ("kagome edge","honeycomb edge","Chern edge","valley edge","Floquet edge","ring edge","synthetic edge","magnon edge"),
        ("breathing kagome edge","domain-wall honeycomb edge","magnetic Chern edge","valley-Hall edge","driven Floquet edge","coupled-ring edge","frequency synthetic edge","chiral magnon edge"),
        ("-28 dB","-32 dB","-36 dB","-40 dB","-44 dB","-48 dB","-52 dB","-56 dB"),
        ("-30 dB","-34 dB","-38 dB","-42 dB","-46 dB","-50 dB","-54 dB","-58 dB"),132201,1212000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the S51 persisted-authority {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S51-Persisted {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S51-Persisted record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S51-Persisted {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S51-Persisted record {code}?",
            f"For S51-Persisted {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S51-Persisted record {code}?",
        )
    return (
        f"S51-Persisted audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S51-Persisted audit {code}: what is {spec.field_a}?",
        f"In S51-Persisted audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S51-Persisted audit {code}: what is {spec.field_b}?",
        f"In S51-Persisted audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    aa=spec.train_a if split=="train" else spec.dev_a
    bb=spec.train_b if split=="train" else spec.dev_b
    first=aa[index%8]; second=bb[(index*3+1)%8]
    wrong_a=aa[(index+3)%8]; wrong_b=bb[(index*5+2)%8]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    cid=f"{split}-s51-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S51PersistedNativeCase(cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb)


def generate_s51_train_cases()->tuple[S51PersistedNativeCase,...]:
    return tuple(_build(spec,i,"train") for spec in _DOMAINS for i in range(64))


def generate_s51_dev_cases()->tuple[S51PersistedNativeCase,...]:
    return tuple(_build(spec,i,"dev") for spec in _DOMAINS for i in range(16))


def validate_s51_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S51 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12: raise RuntimeError("S51 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains: raise RuntimeError("S51 domains changed")
    if any(r.language!="en" for r in (*train,*dev)): raise RuntimeError("S51 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows): raise RuntimeError("S51 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S51 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2: raise RuntimeError("S51 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4: raise RuntimeError("S51 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S51 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}; ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}; dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}; do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds: raise RuntimeError("S51 TRAIN DEV state overlap")
    if tq&dq: raise RuntimeError("S51 TRAIN DEV question overlap")
    if to&do: raise RuntimeError("S51 TRAIN DEV option overlap")


__all__=[
    "S51PersistedNativeCase",
    "generate_s51_train_cases",
    "generate_s51_dev_cases",
    "validate_s51_partitions",
]
