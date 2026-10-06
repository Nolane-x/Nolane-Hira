from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

from .v1_s73_authority import generate_s73_cases


Split=Literal["train","dev"]


@dataclass(frozen=True)
class S74DirectSetCase:
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


def _labels(stem,start):
    return tuple(f"{stem}-{start+i*13+1}" for i in range(8))


def _seq(unit,start,step):
    return tuple(f"{start+i*step:g} {unit}" for i in range(8))


_DOMAINS=(
    _Domain(
        "dsac74_andreev_bound_catalog","Andreev-bound-state catalog","AB",
        "junction class","bound-state threshold",
        _labels("junction",701),_labels("phase-junction",901),
        _seq("meV",0.44,0.12),_seq("meV",0.5,0.12),311101,6101000,
    ),
    _Domain(
        "dsac74_coulomb_drag_ledger","Coulomb-drag ledger","CD",
        "bilayer sample","drag threshold",
        _labels("bilayer",709),_labels("gated-bilayer",909),
        _seq("ohm",1.8,0.41),_seq("ohm",2.005,0.41),311201,6102000,
    ),
    _Domain(
        "dsac74_dark_exciton_register","dark-exciton register","DE",
        "quantum well","dark-state threshold",
        _labels("well",719),_labels("strained-well",919),
        _seq("ps",6.2,1.31),_seq("ps",6.855,1.31),311301,6103000,
    ),
    _Domain(
        "dsac74_fractional_charge_table","fractional-charge table","FC",
        "edge channel","charge threshold",
        _labels("edge",727),_labels("biased-edge",927),
        _seq("e",0.21,0.05),_seq("e",0.235,0.05),311401,6104000,
    ),
    _Domain(
        "dsac74_gruneisen_parameter_map","Gruneisen-parameter map","GP",
        "crystal batch","anharmonic threshold",
        _labels("crystal",733),_labels("compressed-crystal",933),
        _seq("a.u.",1.2,0.27),_seq("a.u.",1.335,0.27),311501,6105000,
    ),
    _Domain(
        "dsac74_hofstadter_gap_atlas","Hofstadter-gap atlas","HG",
        "flux lattice","gap threshold",
        _labels("flux",739),_labels("tilted-flux",939),
        _seq("meV",2.7,0.58),_seq("meV",2.99,0.58),311601,6106000,
    ),
    _Domain(
        "dsac74_isotope_shift_book","isotope-shift book","IS",
        "spectral line","shift threshold",
        _labels("line",743),_labels("isotope-line",943),
        _seq("MHz",11.0,2.4),_seq("MHz",12.2,2.4),311701,6107000,
    ),
    _Domain(
        "dsac74_kondo_scale_chart","Kondo-scale chart","KS",
        "impurity sample","temperature threshold",
        _labels("impurity",751),_labels("screened-impurity",951),
        _seq("K",3.3,0.74),_seq("K",3.67,0.74),311801,6108000,
    ),
    _Domain(
        "dsac74_lamb_shift_grid","Lamb-shift grid","LS",
        "atomic transition","shift threshold",
        _labels("transition",757),_labels("dressed-transition",957),
        _seq("MHz",0.84,0.19),_seq("MHz",0.935,0.19),311901,6109000,
    ),
    _Domain(
        "dsac74_orbital_hall_notebook","orbital-Hall notebook","OH",
        "transport sample","conductivity threshold",
        _labels("transport",761),_labels("spin-orbit-transport",961),
        _seq("S/cm",14.0,3.1),_seq("S/cm",15.55,3.1),312001,6110000,
    ),
    _Domain(
        "dsac74_pseudogap_registry","pseudogap registry","PG",
        "correlated layer","gap threshold",
        _labels("layer",769),_labels("doped-layer",969),
        _seq("meV",9.1,1.83),_seq("meV",10.015,1.83),312101,6111000,
    ),
    _Domain(
        "dsac74_valley_zeeman_index","valley-Zeeman index","VZ",
        "monolayer sample","splitting threshold",
        _labels("mono",773),_labels("gated-mono",973),
        _seq("meV",1.05,0.24),_seq("meV",1.17,0.24),312201,6112000,
    ),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[
        ("a",field_a,first),
        ("b",field_b,second),
        ("x",field_a,distractor_first),
        ("y",field_b,distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts=tuple(
        f"for the S74 direct-set {noun}, {field} is {value}"
        for _kind,field,value in rows
    )
    aliases=tuple(
        f"{value} is the S74 direct-set {field} value for this {noun}"
        for _kind,field,value in rows
    )
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S74-DSAC {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S74-DSAC record {code} stores {second} for {spec.field_b}; independently the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S74-DSAC {spec.noun} {code}, identify {spec.field_a}.",
            f"Which entry belongs to {spec.field_a} in S74-DSAC record {code}?",
            f"For S74-DSAC {spec.noun} {code}, identify {spec.field_b}.",
            f"Which entry belongs to {spec.field_b} in S74-DSAC record {code}?",
        )
    return (
        f"S74-DSAC audit {code} of the {spec.noun}: {spec.field_a} equals {first}, whereas {spec.field_b} equals {second}.",
        f"S74-DSAC audit {code} assigns {second} to {spec.field_b}; separately the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S74-DSAC audit {code}: what is {spec.field_a}?",
        f"In S74-DSAC audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S74-DSAC audit {code}: what is {spec.field_b}?",
        f"In S74-DSAC audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    aa=spec.train_a if split=="train" else spec.dev_a
    bb=spec.train_b if split=="train" else spec.dev_b
    first=aa[index%8]
    second=bb[(index*5+1)%8]
    wrong_a=aa[(index+3)%8]
    wrong_b=bb[(index*7+2)%8]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    cid=f"{split}-s74-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S74DirectSetCase(
        cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,
        texts,aliases,ids,ga,gb,
    )


def generate_s74_cases(split:Split)->tuple[S74DirectSetCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def _states(rows):
    return {x for r in rows for x in (r.state_a,r.state_b)}


def _questions(rows):
    return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}


def _options(rows):
    return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def validate_s74_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192:
        raise RuntimeError("S74 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12:
        raise RuntimeError("S74 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S74 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S74 language changed")

    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S74 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b:
                raise RuntimeError("S74 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S74 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S74 option IDs changed")
            if r.gold_a==r.gold_b:
                raise RuntimeError("S74 paired golds must differ")

    if _states(train)&_states(dev):
        raise RuntimeError("S74 TRAIN DEV state overlap")
    if _questions(train)&_questions(dev):
        raise RuntimeError("S74 TRAIN DEV question overlap")
    if _options(train)&_options(dev):
        raise RuntimeError("S74 TRAIN DEV option overlap")

    prior=(*generate_s73_cases("train"),*generate_s73_cases("dev"))
    current=(*train,*dev)
    if _states(current)&_states(prior):
        raise RuntimeError("S74 exact S73 state overlap")
    if _questions(current)&_questions(prior):
        raise RuntimeError("S74 exact S73 question overlap")
    if _options(current)&_options(prior):
        raise RuntimeError("S74 exact S73 option overlap")


__all__=["S74DirectSetCase","generate_s74_cases","validate_s74_partitions"]
