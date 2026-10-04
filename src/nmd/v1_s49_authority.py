from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S49IdentityCase:
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
        for k in ("option_texts","option_aliases","option_ids"): row[k]=list(row[k])
        return row


@dataclass(frozen=True)
class _Domain:
    name:str; noun:str; prefix:str; field_a:str; field_b:str
    train_a:tuple[str,...]; dev_a:tuple[str,...]
    train_b:tuple[str,...]; dev_b:tuple[str,...]
    seed:int; dev_offset:int


_DOMAINS=(
    _Domain("quantum_flux_cartographer","quantum flux cartographer","QF","pickup","flux floor",
        ("MoRe loop","NbTi loop","Al loop","NbN loop","YBCO loop","graphene loop","TiN loop","Ta loop"),
        ("gradiometric MoRe loop","persistent NbTi loop","nanobridge Al loop","kinetic NbN loop","grain-boundary YBCO loop","ballistic graphene loop","high-Q TiN loop","shielded Ta loop"),
        ("12 nPhi0","16 nPhi0","20 nPhi0","24 nPhi0","28 nPhi0","32 nPhi0","36 nPhi0","40 nPhi0"),
        ("14 nPhi0","18 nPhi0","22 nPhi0","26 nPhi0","30 nPhi0","34 nPhi0","38 nPhi0","42 nPhi0"),120101,1101000),
    _Domain("exciton_phase_tracker","exciton phase tracker","EP","heterostructure","phase blur",
        ("MoSe2 pair","WSe2 pair","WS2 pair","MoS2 pair","GaAs well","InGaAs well","graphene bilayer","hBN stack"),
        ("moire MoSe2 pair","twist-locked WSe2 pair","gated WS2 pair","encapsulated MoS2 pair","indirect GaAs well","strain-tuned InGaAs well","biased graphene bilayer","defect hBN stack"),
        ("0.07 rad","0.10 rad","0.13 rad","0.16 rad","0.19 rad","0.22 rad","0.25 rad","0.28 rad"),
        ("0.085 rad","0.115 rad","0.145 rad","0.175 rad","0.205 rad","0.235 rad","0.265 rad","0.295 rad"),120201,1102000),
    _Domain("atomic_rotation_holograph","atomic rotation holograph","AR","species","rotation floor",
        ("Rb87 cloud","Cs133 cloud","Sr88 cloud","Yb174 cloud","K39 cloud","Na23 cloud","Li7 cloud","Ca40 cloud"),
        ("delta-kicked Rb87 cloud","fountain Cs133 cloud","lattice Sr88 cloud","clocked Yb174 cloud","spinor K39 cloud","collimated Na23 cloud","degenerate Li7 cloud","laser-cooled Ca40 cloud"),
        ("2 nrad/s","3 nrad/s","4 nrad/s","5 nrad/s","6 nrad/s","7 nrad/s","8 nrad/s","9 nrad/s"),
        ("2.5 nrad/s","3.5 nrad/s","4.5 nrad/s","5.5 nrad/s","6.5 nrad/s","7.5 nrad/s","8.5 nrad/s","9.5 nrad/s"),120301,1103000),
    _Domain("magnon_recoil_interferometer","magnon recoil interferometer","MI","guide","recoil blur",
        ("YIG guide","CoFeB guide","NiFe guide","FeGa guide","hematite guide","NiO guide","CoO guide","Cr2O3 guide"),
        ("low-loss YIG guide","exchange-biased CoFeB guide","narrow NiFe guide","strained FeGa guide","canted hematite guide","antiferromagnetic NiO guide","epitaxial CoO guide","magnetoelectric Cr2O3 guide"),
        ("0.11 um-1","0.15 um-1","0.19 um-1","0.23 um-1","0.27 um-1","0.31 um-1","0.35 um-1","0.39 um-1"),
        ("0.13 um-1","0.17 um-1","0.21 um-1","0.25 um-1","0.29 um-1","0.33 um-1","0.37 um-1","0.41 um-1"),120401,1104000),
    _Domain("polariton_vector_camera","polariton vector camera","PV","cavity","vector linewidth",
        ("perovskite cavity","GaAs cavity","GaN cavity","ZnO cavity","organic cavity","TMD cavity","SiC cavity","AlN cavity"),
        ("high-Q perovskite cavity","strong-coupling GaAs cavity","nitride GaN cavity","low-loss ZnO cavity","ordered organic cavity","moire TMD cavity","phononic SiC cavity","piezo AlN cavity"),
        ("0.31 meV","0.43 meV","0.55 meV","0.67 meV","0.79 meV","0.91 meV","1.03 meV","1.15 meV"),
        ("0.37 meV","0.49 meV","0.61 meV","0.73 meV","0.85 meV","0.97 meV","1.09 meV","1.21 meV"),120501,1105000),
    _Domain("nanostrain_phase_scope","nanostrain phase scope","NS","probe","strain floor",
        ("NV center","SiV center","GeV center","SnV center","graphene gauge","SiC divacancy","MoS2 exciton","GaAs dot"),
        ("vector NV center","isotopic SiV center","strained GeV center","clocked SnV center","suspended graphene gauge","4H-SiC divacancy","encapsulated MoS2 exciton","charge-stable GaAs dot"),
        ("5 neps","7 neps","9 neps","11 neps","13 neps","15 neps","17 neps","19 neps"),
        ("6 neps","8 neps","10 neps","12 neps","14 neps","16 neps","18 neps","20 neps"),120601,1106000),
    _Domain("terahertz_spin_mapper","terahertz spin mapper","TH","detector","field noise",
        ("graphene FET","InGaAs FET","GaN HEMT","Si MOSFET","MoS2 FET","WSe2 FET","CNT FET","diamond FET"),
        ("dual-gate graphene FET","low-noise InGaAs FET","polar GaN HEMT","cryogenic Si MOSFET","encapsulated MoS2 FET","ambipolar WSe2 FET","aligned CNT FET","NV diamond FET"),
        ("0.6 uV/cm","0.9 uV/cm","1.2 uV/cm","1.5 uV/cm","1.8 uV/cm","2.1 uV/cm","2.4 uV/cm","2.7 uV/cm"),
        ("0.75 uV/cm","1.05 uV/cm","1.35 uV/cm","1.65 uV/cm","1.95 uV/cm","2.25 uV/cm","2.55 uV/cm","2.85 uV/cm"),120701,1107000),
    _Domain("molecular_torque_clock","molecular torque clock","MT","species","torque drift",
        ("OCS packet","HCN packet","NH3 packet","CH3F packet","H2CO packet","SO2 packet","CaF packet","SrF packet"),
        ("state-selected OCS packet","oriented HCN packet","inversion-cooled NH3 packet","symmetric-top CH3F packet","para-H2CO packet","supersonic SO2 packet","laser-cooled CaF packet","collimated SrF packet"),
        ("3 yNm","4 yNm","5 yNm","6 yNm","7 yNm","8 yNm","9 yNm","10 yNm"),
        ("3.5 yNm","4.5 yNm","5.5 yNm","6.5 yNm","7.5 yNm","8.5 yNm","9.5 yNm","10.5 yNm"),120801,1108000),
    _Domain("topological_charge_router","topological charge router","TC","channel","charge leakage",
        ("WTe2 edge","HgTe edge","graphene edge","MoTe2 edge","Bi2Se3 edge","InAs edge","GaSb edge","MnBi2Te4 edge"),
        ("helical WTe2 edge","inverted HgTe edge","valley graphene edge","protected MoTe2 edge","topological Bi2Se3 edge","epitaxial InAs edge","broken-gap GaSb edge","axion MnBi2Te4 edge"),
        ("-33 dB","-37 dB","-41 dB","-45 dB","-49 dB","-53 dB","-57 dB","-61 dB"),
        ("-35 dB","-39 dB","-43 dB","-47 dB","-51 dB","-55 dB","-59 dB","-63 dB"),120901,1109000),
    _Domain("cryogenic_field_tomograph","cryogenic field tomograph","CF","sensor","field floor",
        ("SQUID pixel","NV pixel","Hall pixel","GMR pixel","TMR pixel","graphene pixel","SiV pixel","fluxgate pixel"),
        ("nanoSQUID pixel","vector NV pixel","ballistic Hall pixel","low-noise GMR pixel","cryogenic TMR pixel","gated graphene pixel","isotopic SiV pixel","micro-fluxgate pixel"),
        ("4 pT","6 pT","8 pT","10 pT","12 pT","14 pT","16 pT","18 pT"),
        ("5 pT","7 pT","9 pT","11 pT","13 pT","15 pT","17 pT","19 pT"),121001,1110000),
    _Domain("quantum_pressure_clock","quantum pressure clock","QC","membrane","pressure drift",
        ("SiN drum","graphene drum","diamond membrane","SiC beam","AlN disk","GaAs disk","quartz plate","MoS2 drum"),
        ("soft-clamped SiN drum","tensioned graphene drum","CVD diamond membrane","4H-SiC beam","piezo AlN disk","phononic GaAs disk","low-loss quartz plate","encapsulated MoS2 drum"),
        ("7 nPa","10 nPa","13 nPa","16 nPa","19 nPa","22 nPa","25 nPa","28 nPa"),
        ("8.5 nPa","11.5 nPa","14.5 nPa","17.5 nPa","20.5 nPa","23.5 nPa","26.5 nPa","29.5 nPa"),121101,1111000),
    _Domain("vacuum_phase_radar","vacuum phase radar","VP","optic","phase noise",
        ("Si mirror","sapphire mirror","SiN mirror","GaAs mirror","AlGaAs mirror","diamond mirror","quartz mirror","CaF2 mirror"),
        ("crystalline Si mirror","cryogenic sapphire mirror","membrane SiN mirror","epitaxial GaAs mirror","Bragg AlGaAs mirror","CVD diamond mirror","superpolished quartz mirror","fluoride CaF2 mirror"),
        ("9 urad","12 urad","15 urad","18 urad","21 urad","24 urad","27 urad","30 urad"),
        ("10.5 urad","13.5 urad","16.5 urad","19.5 urad","22.5 urad","25.5 urad","28.5 urad","31.5 urad"),121201,1112000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the S49 identity-authority {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S49-Identity {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S49-Identity record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S49-Identity {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S49-Identity record {code}?",
            f"For S49-Identity {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S49-Identity record {code}?",
        )
    return (
        f"S49-Identity audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S49-Identity audit {code}: what is {spec.field_a}?",
        f"In S49-Identity audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S49-Identity audit {code}: what is {spec.field_b}?",
        f"In S49-Identity audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    aa=spec.train_a if split=="train" else spec.dev_a
    bb=spec.train_b if split=="train" else spec.dev_b
    first=aa[index%8]; second=bb[(index*3+1)%8]
    wrong_a=aa[(index+3)%8]; wrong_b=bb[(index*5+2)%8]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    cid=f"{split}-s49-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,seed=spec.seed+offset)
    return S49IdentityCase(cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb)


def generate_s49_cases(split:Split)->tuple[S49IdentityCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s49_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S49 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12: raise RuntimeError("S49 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains: raise RuntimeError("S49 domains changed")
    if any(r.language!="en" for r in (*train,*dev)): raise RuntimeError("S49 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows): raise RuntimeError("S49 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S49 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2: raise RuntimeError("S49 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4: raise RuntimeError("S49 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S49 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}; ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}; dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}; do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds: raise RuntimeError("S49 TRAIN DEV state overlap")
    if tq&dq: raise RuntimeError("S49 TRAIN DEV question overlap")
    if to&do: raise RuntimeError("S49 TRAIN DEV option overlap")


__all__=["S49IdentityCase","generate_s49_cases","validate_s49_partitions"]
