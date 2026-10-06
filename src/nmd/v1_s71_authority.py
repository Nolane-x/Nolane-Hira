from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

from .v1_s70_authority import generate_s70_cases


Split=Literal["train","dev"]


@dataclass(frozen=True)
class S71MultiStatComposerCase:
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
    return tuple(f"{stem}-{start+i*7+1}" for i in range(8))


def _seq(unit,start,step):
    return tuple(f"{start+i*step:g} {unit}" for i in range(8))


_DOMAINS=(
    _Domain(
        "composer71_axion_cavity_registry","axion cavity registry","AC",
        "resonator batch","scan threshold",
        _labels("resonator",401),_labels("cryogenic-resonator",501),
        _seq("GHz",1.71,0.31),_seq("GHz",1.865,0.31),291101,4101000,
    ),
    _Domain(
        "composer71_bloch_band_catalog","Bloch band catalog","BB",
        "optical lattice","band-gap threshold",
        _labels("lattice",409),_labels("phase-lattice",509),
        _seq("kHz",8.4,1.37),_seq("kHz",9.085,1.37),291201,4102000,
    ),
    _Domain(
        "composer71_charge_density_ledger","charge-density ledger","CD",
        "layer stack","density-wave threshold",
        _labels("stack",419),_labels("twisted-stack",519),
        _seq("meV",2.3,0.61),_seq("meV",2.605,0.61),291301,4103000,
    ),
    _Domain(
        "composer71_dirac_cone_register","Dirac cone register","DC",
        "Dirac material","cone-splitting threshold",
        _labels("Dirac",431),_labels("strained-Dirac",531),
        _seq("meV",5.2,0.94),_seq("meV",5.67,0.94),291401,4104000,
    ),
    _Domain(
        "composer71_floquet_mode_atlas","Floquet mode atlas","FM",
        "driven lattice","quasienergy threshold",
        _labels("Floquet",439),_labels("driven-Floquet",539),
        _seq("MHz",7.1,1.52),_seq("MHz",7.86,1.52),291501,4105000,
    ),
    _Domain(
        "composer71_hall_viscosity_chart","Hall viscosity chart","HV",
        "quantum fluid","viscosity threshold",
        _labels("fluid",443),_labels("Hall-fluid",543),
        _seq("nPa s",0.81,0.22),_seq("nPa s",0.92,0.22),291601,4106000,
    ),
    _Domain(
        "composer71_josephson_phase_book","Josephson phase book","JP",
        "junction array","phase-slip threshold",
        _labels("junction",449),_labels("biased-junction",549),
        _seq("mrad",1.4,0.37),_seq("mrad",1.585,0.37),291701,4107000,
    ),
    _Domain(
        "composer71_kerr_rotation_table","Kerr rotation table","KR",
        "magneto-optic film","rotation threshold",
        _labels("film",457),_labels("doped-film",557),
        _seq("urad",3.2,0.73),_seq("urad",3.565,0.73),291801,4108000,
    ),
    _Domain(
        "composer71_luttinger_parameter_map","Luttinger parameter map","LP",
        "1D channel","interaction threshold",
        _labels("channel",461),_labels("gated-channel",561),
        _seq("K",0.42,0.09),_seq("K",0.465,0.09),291901,4109000,
    ),
    _Domain(
        "composer71_moire_gap_notebook","moiré gap notebook","MG",
        "moiré superlattice","gap threshold",
        _labels("moire",463),_labels("aligned-moire",563),
        _seq("meV",6.8,1.17),_seq("meV",7.385,1.17),292001,4110000,
    ),
    _Domain(
        "composer71_nematic_order_grid","nematic order grid","NO",
        "electronic nematic","order threshold",
        _labels("nematic",467),_labels("strained-nematic",567),
        _seq("a.u.",0.34,0.08),_seq("a.u.",0.38,0.08),292101,4111000,
    ),
    _Domain(
        "composer71_orbital_torque_index","orbital torque index","OT",
        "spin-orbit stack","torque threshold",
        _labels("torque",479),_labels("pulsed-torque",579),
        _seq("pN nm",2.6,0.58),_seq("pN nm",2.89,0.58),292201,4112000,
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
        f"for the S71 multi-stat composer {noun}, {field} is {value}"
        for _kind,field,value in rows
    )
    aliases=tuple(
        f"{value} is the S71 multi-stat {field} value for this {noun}"
        for _kind,field,value in rows
    )
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S71-Composer {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S71-Composer record {code} stores {second} for {spec.field_b}; independently the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S71-Composer {spec.noun} {code}, identify {spec.field_a}.",
            f"Which entry belongs to {spec.field_a} in S71-Composer record {code}?",
            f"For S71-Composer {spec.noun} {code}, identify {spec.field_b}.",
            f"Which entry belongs to {spec.field_b} in S71-Composer record {code}?",
        )
    return (
        f"S71-Composer audit {code} of the {spec.noun}: {spec.field_a} equals {first}, whereas {spec.field_b} equals {second}.",
        f"S71-Composer audit {code} assigns {second} to {spec.field_b}; separately the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S71-Composer audit {code}: what is {spec.field_a}?",
        f"In S71-Composer audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S71-Composer audit {code}: what is {spec.field_b}?",
        f"In S71-Composer audit {code}, which entry is tagged {spec.field_b}?",
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
    cid=f"{split}-s71-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=cid,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=wrong_a,distractor_second=wrong_b,
        seed=spec.seed+offset,
    )
    return S71MultiStatComposerCase(
        cid,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,
        texts,aliases,ids,ga,gb,
    )


def generate_s71_cases(split:Split)->tuple[S71MultiStatComposerCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def _states(rows):
    return {x for r in rows for x in (r.state_a,r.state_b)}


def _questions(rows):
    return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}


def _options(rows):
    return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def validate_s71_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192:
        raise RuntimeError("S71 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12:
        raise RuntimeError("S71 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S71 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S71 language changed")

    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S71 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b:
                raise RuntimeError("S71 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S71 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S71 option IDs changed")
            if r.gold_a==r.gold_b:
                raise RuntimeError("S71 paired golds must differ")

    if _states(train)&_states(dev):
        raise RuntimeError("S71 TRAIN DEV state overlap")
    if _questions(train)&_questions(dev):
        raise RuntimeError("S71 TRAIN DEV question overlap")
    if _options(train)&_options(dev):
        raise RuntimeError("S71 TRAIN DEV option overlap")

    prior=(*generate_s70_cases("train"),*generate_s70_cases("dev"))
    current=(*train,*dev)
    if _states(current)&_states(prior):
        raise RuntimeError("S71 exact S70 state overlap")
    if _questions(current)&_questions(prior):
        raise RuntimeError("S71 exact S70 question overlap")
    if _options(current)&_options(prior):
        raise RuntimeError("S71 exact S70 option overlap")


__all__=["S71MultiStatComposerCase","generate_s71_cases","validate_s71_partitions"]
