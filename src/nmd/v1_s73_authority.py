from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

from .v1_s72_authority import generate_s72_cases


Split=Literal["train","dev"]


@dataclass(frozen=True)
class S73SafetyVetoCase:
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
    return tuple(f"{stem}-{start+i*11+1}" for i in range(8))


def _seq(unit,start,step):
    return tuple(f"{start+i*step:g} {unit}" for i in range(8))


_DOMAINS=(
    _Domain(
        "veto73_berry_curvature_index","Berry-curvature index","BC",
        "crystal sector","curvature threshold",
        _labels("sector",601),_labels("tilted-sector",801),
        _seq("1/nm2",0.18,0.07),_seq("1/nm2",0.215,0.07),301101,5101000,
    ),
    _Domain(
        "veto73_exciton_lifetime_book","exciton lifetime book","EL",
        "heterostructure batch","lifetime threshold",
        _labels("hetero",613),_labels("strained-hetero",813),
        _seq("ps",4.6,1.13),_seq("ps",5.165,1.13),301201,5102000,
    ),
    _Domain(
        "veto73_magnon_dispersion_ledger","magnon dispersion ledger","MD",
        "magnetic stack","dispersion threshold",
        _labels("magnon",617),_labels("biased-magnon",817),
        _seq("meV",1.9,0.47),_seq("meV",2.135,0.47),301301,5103000,
    ),
    _Domain(
        "veto73_phonon_scattering_log","phonon scattering log","PS",
        "phonon branch","scattering threshold",
        _labels("branch",631),_labels("anharmonic-branch",831),
        _seq("THz",2.8,0.66),_seq("THz",3.13,0.66),301401,5104000,
    ),
    _Domain(
        "veto73_plasmon_resonance_notebook","plasmon resonance notebook","PR",
        "nanostructure array","resonance threshold",
        _labels("array",641),_labels("gated-array",841),
        _seq("eV",0.72,0.16),_seq("eV",0.8,0.16),301501,5105000,
    ),
    _Domain(
        "veto73_polariton_coupling_atlas","polariton coupling atlas","PC",
        "microcavity sample","coupling threshold",
        _labels("cavity",653),_labels("hybrid-cavity",853),
        _seq("meV",7.4,1.21),_seq("meV",8.005,1.21),301601,5106000,
    ),
    _Domain(
        "veto73_quantum_capacitance_grid","quantum capacitance grid","QC",
        "device channel","capacitance threshold",
        _labels("channel",659),_labels("dual-gate-channel",859),
        _seq("fF",0.43,0.095),_seq("fF",0.4775,0.095),301701,5107000,
    ),
    _Domain(
        "veto73_rydberg_blockade_table","Rydberg blockade table","RB",
        "atom ensemble","blockade threshold",
        _labels("ensemble",661),_labels("dressed-ensemble",861),
        _seq("MHz",5.5,1.08),_seq("MHz",6.04,1.08),301801,5108000,
    ),
    _Domain(
        "veto73_skyrmion_radius_chart","skyrmion radius chart","SR",
        "magnetic film","radius threshold",
        _labels("film",673),_labels("chiral-film",873),
        _seq("nm",8.2,1.46),_seq("nm",8.93,1.46),301901,5109000,
    ),
    _Domain(
        "veto73_spin_seebeck_map","spin Seebeck map","SS",
        "thermal stack","voltage threshold",
        _labels("thermal",677),_labels("gradient-thermal",877),
        _seq("uV",1.15,0.29),_seq("uV",1.295,0.29),302001,5110000,
    ),
    _Domain(
        "veto73_topological_gap_registry","topological gap registry","TG",
        "insulator sample","gap threshold",
        _labels("sample",683),_labels("inverted-sample",883),
        _seq("meV",3.7,0.79),_seq("meV",4.095,0.79),302101,5111000,
    ),
    _Domain(
        "veto73_weyl_node_register","Weyl node register","WN",
        "semimetal crystal","node-separation threshold",
        _labels("crystal",691),_labels("strained-crystal",891),
        _seq("1/A",0.026,0.006),_seq("1/A",0.029,0.006),302201,5112000,
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
        f"for the S73 safety-veto {noun}, {field} is {value}"
        for _kind,field,value in rows
    )
    aliases=tuple(
        f"{value} is the S73 safety-veto {field} value for this {noun}"
        for _kind,field,value in rows
    )
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S73-Veto {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S73-Veto record {code} stores {second} for {spec.field_b}; independently the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S73-Veto {spec.noun} {code}, identify {spec.field_a}.",
            f"Which entry belongs to {spec.field_a} in S73-Veto record {code}?",
            f"For S73-Veto {spec.noun} {code}, identify {spec.field_b}.",
            f"Which entry belongs to {spec.field_b} in S73-Veto record {code}?",
        )
    return (
        f"S73-Veto audit {code} of the {spec.noun}: {spec.field_a} equals {first}, whereas {spec.field_b} equals {second}.",
        f"S73-Veto audit {code} assigns {second} to {spec.field_b}; separately the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S73-Veto audit {code}: what is {spec.field_a}?",
        f"In S73-Veto audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S73-Veto audit {code}: what is {spec.field_b}?",
        f"In S73-Veto audit {code}, which entry is tagged {spec.field_b}?",
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
    cid=f"{split}-s73-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S73SafetyVetoCase(
        cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,
        texts,aliases,ids,ga,gb,
    )


def generate_s73_cases(split:Split)->tuple[S73SafetyVetoCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def _states(rows):
    return {x for r in rows for x in (r.state_a,r.state_b)}


def _questions(rows):
    return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}


def _options(rows):
    return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def validate_s73_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192:
        raise RuntimeError("S73 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12:
        raise RuntimeError("S73 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S73 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S73 language changed")

    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S73 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b:
                raise RuntimeError("S73 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S73 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S73 option IDs changed")
            if r.gold_a==r.gold_b:
                raise RuntimeError("S73 paired golds must differ")

    if _states(train)&_states(dev):
        raise RuntimeError("S73 TRAIN DEV state overlap")
    if _questions(train)&_questions(dev):
        raise RuntimeError("S73 TRAIN DEV question overlap")
    if _options(train)&_options(dev):
        raise RuntimeError("S73 TRAIN DEV option overlap")

    prior=(*generate_s72_cases("train"),*generate_s72_cases("dev"))
    current=(*train,*dev)
    if _states(current)&_states(prior):
        raise RuntimeError("S73 exact S72 state overlap")
    if _questions(current)&_questions(prior):
        raise RuntimeError("S73 exact S72 question overlap")
    if _options(current)&_options(prior):
        raise RuntimeError("S73 exact S72 option overlap")


__all__=["S73SafetyVetoCase","generate_s73_cases","validate_s73_partitions"]
