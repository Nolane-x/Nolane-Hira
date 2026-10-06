from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

from .v1_s68_authority import generate_s68_cases


Split=Literal["train","dev"]


@dataclass(frozen=True)
class S69QueryGatedInteractionCase:
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
    return tuple(f"{stem}-{start+i*6+1}" for i in range(8))


def _seq(unit,start,step):
    return tuple(f"{start+i*step:g} {unit}" for i in range(8))


_DOMAINS=(
    _Domain(
        "interact69_polariton_detuning_atlas","polariton detuning atlas","PD",
        "cavity lattice","detuning threshold",
        _labels("Bragg",211),_labels("chirped-Bragg",311),
        _seq("MHz",2.7,1.13),_seq("MHz",3.265,1.13),269101,3101000,
    ),
    _Domain(
        "interact69_phonon_topology_ledger","phonon topology ledger","PT",
        "phononic substrate","edge-gap threshold",
        _labels("phonon",223),_labels("etched-phonon",323),
        _seq("kHz",18.4,3.7),_seq("kHz",20.25,3.7),269201,3102000,
    ),
    _Domain(
        "interact69_spinon_flux_register","spinon flux register","SF",
        "spin-liquid host","flux threshold",
        _labels("spinon",227),_labels("frustrated-spinon",327),
        _seq("mPhi",0.73,0.19),_seq("mPhi",0.825,0.19),269301,3103000,
    ),
    _Domain(
        "interact69_exciton_stark_chart","exciton Stark chart","ES",
        "exciton stack","Stark threshold",
        _labels("exciton",229),_labels("gated-exciton",329),
        _seq("meV",4.1,0.83),_seq("meV",4.515,0.83),269401,3104000,
    ),
    _Domain(
        "interact69_casimir_mode_table","Casimir mode table","CM",
        "mirror pair","force threshold",
        _labels("mirror",233),_labels("coated-mirror",333),
        _seq("pN",1.9,0.47),_seq("pN",2.135,0.47),269501,3105000,
    ),
    _Domain(
        "interact69_magnon_phase_index","magnon phase index","MP",
        "magnonic crystal","phase threshold",
        _labels("magnon",239),_labels("patterned-magnon",339),
        _seq("mrad",0.54,0.23),_seq("mrad",0.655,0.23),269601,3106000,
    ),
    _Domain(
        "interact69_neutrino_baseline_map","neutrino baseline map","NB",
        "detector medium","baseline threshold",
        _labels("detector",241),_labels("doped-detector",341),
        _seq("km",17.0,5.5),_seq("km",19.75,5.5),269701,3107000,
    ),
    _Domain(
        "interact69_superfluid_vortex_sheet","superfluid vortex sheet","SV",
        "superfluid cell","vortex threshold",
        _labels("cell",251),_labels("cooled-cell",351),
        _seq("Hz",6.2,1.41),_seq("Hz",6.905,1.41),269801,3108000,
    ),
    _Domain(
        "interact69_quantum_capacitance_log","quantum capacitance log","QC",
        "2D channel","capacitance threshold",
        _labels("channel",257),_labels("encapsulated-channel",357),
        _seq("fF",12.5,2.8),_seq("fF",13.9,2.8),269901,3109000,
    ),
    _Domain(
        "interact69_gravity_gradient_card","gravity gradient card","GG",
        "sensor frame","gradient threshold",
        _labels("frame",263),_labels("isolated-frame",363),
        _seq("E",0.81,0.17),_seq("E",0.895,0.17),270001,3110000,
    ),
    _Domain(
        "interact69_raman_coherence_grid","Raman coherence grid","RC",
        "Raman medium","coherence threshold",
        _labels("medium",269),_labels("pumped-medium",369),
        _seq("us",3.4,0.76),_seq("us",3.78,0.76),270101,3111000,
    ),
    _Domain(
        "interact69_rydberg_blockade_book","Rydberg blockade book","RB",
        "atom array","blockade threshold",
        _labels("array",271),_labels("tweezer-array",371),
        _seq("um",2.2,0.52),_seq("um",2.46,0.52),270201,3112000,
    ),
)


def _shuffle(
    *,
    case_id,
    noun,
    field_a,
    field_b,
    first,
    second,
    distractor_first,
    distractor_second,
    seed,
):
    rows=[
        ("a",field_a,first),
        ("b",field_b,second),
        ("x",field_a,distractor_first),
        ("y",field_b,distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts=tuple(
        f"for the S69 interaction representation {noun}, {field} is {value}"
        for _kind,field,value in rows
    )
    aliases=tuple(
        f"{value} is the S69 query-gated {field} value for this {noun}"
        for _kind,field,value in rows
    )
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S69-Interact {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S69-Interact record {code} stores {second} for {spec.field_b}; independently the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S69-Interact {spec.noun} {code}, identify {spec.field_a}.",
            f"Which entry belongs to {spec.field_a} in S69-Interact record {code}?",
            f"For S69-Interact {spec.noun} {code}, identify {spec.field_b}.",
            f"Which entry belongs to {spec.field_b} in S69-Interact record {code}?",
        )
    return (
        f"S69-Interact audit {code} of the {spec.noun}: {spec.field_a} equals {first}, whereas {spec.field_b} equals {second}.",
        f"S69-Interact audit {code} assigns {second} to {spec.field_b}; separately the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S69-Interact audit {code}: what is {spec.field_a}?",
        f"In S69-Interact audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S69-Interact audit {code}: what is {spec.field_b}?",
        f"In S69-Interact audit {code}, which entry is tagged {spec.field_b}?",
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
    cid=f"{split}-s69-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(
        spec,split=split,code=code,first=first,second=second
    )
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,
        noun=spec.noun,
        field_a=spec.field_a,
        field_b=spec.field_b,
        first=first,
        second=second,
        distractor_first=wrong_a,
        distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S69QueryGatedInteractionCase(
        cid,split,spec.name,"en",
        sa,sb,qa1,qa2,qb1,qb2,
        texts,aliases,ids,ga,gb,
    )


def generate_s69_cases(split:Split)->tuple[S69QueryGatedInteractionCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def _states(rows):
    return {x for r in rows for x in (r.state_a,r.state_b)}


def _questions(rows):
    return {
        x for r in rows
        for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)
    }


def _options(rows):
    return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def validate_s69_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192:
        raise RuntimeError("S69 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12:
        raise RuntimeError("S69 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S69 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S69 language changed")

    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S69 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b:
                raise RuntimeError("S69 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S69 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S69 option IDs changed")
            if r.gold_a==r.gold_b:
                raise RuntimeError("S69 paired golds must differ")

    if _states(train)&_states(dev):
        raise RuntimeError("S69 TRAIN DEV state overlap")
    if _questions(train)&_questions(dev):
        raise RuntimeError("S69 TRAIN DEV question overlap")
    if _options(train)&_options(dev):
        raise RuntimeError("S69 TRAIN DEV option overlap")

    prior=(*generate_s68_cases("train"),*generate_s68_cases("dev"))
    current=(*train,*dev)
    if _states(current)&_states(prior):
        raise RuntimeError("S69 exact S68 state overlap")
    if _questions(current)&_questions(prior):
        raise RuntimeError("S69 exact S68 question overlap")
    if _options(current)&_options(prior):
        raise RuntimeError("S69 exact S68 option overlap")


__all__=[
    "S69QueryGatedInteractionCase",
    "generate_s69_cases",
    "validate_s69_partitions",
]
