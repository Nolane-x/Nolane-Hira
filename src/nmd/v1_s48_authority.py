from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S48QuotientCase:
    case_id: str
    split: Split
    domain: str
    language: str
    state_a: str
    state_b: str
    question_a1: str
    question_a2: str
    question_b1: str
    question_b2: str
    option_texts: tuple[str, ...]
    option_aliases: tuple[str, ...]
    option_ids: tuple[str, ...]
    gold_a: int
    gold_b: int

    def to_dict(self) -> dict:
        row=asdict(self)
        for key in ("option_texts","option_aliases","option_ids"):
            row[key]=list(row[key])
        return row


@dataclass(frozen=True)
class _Domain:
    name: str
    noun: str
    prefix: str
    field_a: str
    field_b: str
    train_a: tuple[str,...]
    dev_a: tuple[str,...]
    train_b: tuple[str,...]
    dev_b: tuple[str,...]
    seed: int
    dev_offset: int


_DOMAINS=(
    _Domain(
        "quantum_capacitance_microscope","quantum capacitance microscope","QC","electrode","capacitance noise",
        ("graphene gate","Au gate","Al gate","TiN gate","ITO gate","MoS2 gate","WSe2 gate","NbN gate"),
        ("encapsulated graphene gate","template Au gate","oxide-free Al gate","superconducting TiN gate","ENZ ITO gate","dual-gated MoS2 gate","twisted WSe2 gate","cryogenic NbN gate"),
        ("3.4 aF","4.6 aF","5.8 aF","7.0 aF","8.2 aF","9.4 aF","10.6 aF","11.8 aF"),
        ("4.0 aF","5.2 aF","6.4 aF","7.6 aF","8.8 aF","10.0 aF","11.2 aF","12.4 aF"),103101,993000,
    ),
    _Domain(
        "polariton_flow_mapper","polariton flow mapper","PF","cavity","flow blur",
        ("GaAs cavity","perovskite cavity","GaN cavity","ZnO cavity","organic cavity","SiC cavity","TMD cavity","microdisk cavity"),
        ("high-Q GaAs cavity","patterned perovskite cavity","strong-coupling GaN cavity","cryogenic ZnO cavity","ordered organic cavity","4H-SiC cavity","moire TMD cavity","whispering microdisk cavity"),
        ("0.12 um/s","0.18 um/s","0.24 um/s","0.30 um/s","0.36 um/s","0.42 um/s","0.48 um/s","0.54 um/s"),
        ("0.15 um/s","0.21 um/s","0.27 um/s","0.33 um/s","0.39 um/s","0.45 um/s","0.51 um/s","0.57 um/s"),103201,994000,
    ),
    _Domain(
        "spin_noise_tomograph","spin noise tomograph","SN","probe","noise floor",
        ("Faraday probe","Kerr probe","NV probe","SQUID probe","Hall probe","MOKE probe","Raman probe","ESR probe"),
        ("balanced Faraday probe","shot-noise Kerr probe","vector NV probe","nanoSQUID probe","ballistic Hall probe","heterodyne MOKE probe","stimulated Raman probe","pulsed ESR probe"),
        ("2.2 nT","3.0 nT","3.8 nT","4.6 nT","5.4 nT","6.2 nT","7.0 nT","7.8 nT"),
        ("2.6 nT","3.4 nT","4.2 nT","5.0 nT","5.8 nT","6.6 nT","7.4 nT","8.2 nT"),103301,995000,
    ),
    _Domain(
        "quantum_vorticity_clock","quantum vorticity clock","QV","fluid","vorticity drift",
        ("He3 film","He4 film","BEC cloud","exciton fluid","polariton fluid","electron fluid","photon fluid","magnon fluid"),
        ("superfluid He3 film","ultracold He4 film","rotating BEC cloud","dipolar exciton fluid","driven polariton fluid","hydrodynamic electron fluid","nonlinear photon fluid","coherent magnon fluid"),
        ("0.09 Hz","0.13 Hz","0.17 Hz","0.21 Hz","0.25 Hz","0.29 Hz","0.33 Hz","0.37 Hz"),
        ("0.11 Hz","0.15 Hz","0.19 Hz","0.23 Hz","0.27 Hz","0.31 Hz","0.35 Hz","0.39 Hz"),103401,996000,
    ),
    _Domain(
        "nanophotonic_force_radar","nanophotonic force radar","NF","resonator","force floor",
        ("Si ring","SiN ring","GaAs disk","diamond disk","AlN disk","SiC ring","LiNbO3 disk","quartz disk"),
        ("slot Si ring","soft-clamped SiN ring","phononic GaAs disk","NV diamond disk","piezo AlN disk","4H-SiC ring","periodic LiNbO3 disk","low-loss quartz disk"),
        ("4 aN","6 aN","8 aN","10 aN","12 aN","14 aN","16 aN","18 aN"),
        ("5 aN","7 aN","9 aN","11 aN","13 aN","15 aN","17 aN","19 aN"),103501,997000,
    ),
    _Domain(
        "molecular_dipole_imager","molecular dipole imager","MD","species","dipole spread",
        ("HCN beam","OCS beam","NH3 beam","CH3F beam","H2CO beam","SO2 beam","CO beam","N2O beam"),
        ("oriented HCN beam","state-selected OCS beam","inversion-cooled NH3 beam","Stark-focused CH3F beam","para-H2CO beam","supersonic SO2 beam","decelerated CO beam","cold N2O beam"),
        ("0.12 D","0.16 D","0.20 D","0.24 D","0.28 D","0.32 D","0.36 D","0.40 D"),
        ("0.14 D","0.18 D","0.22 D","0.26 D","0.30 D","0.34 D","0.38 D","0.42 D"),103601,998000,
    ),
    _Domain(
        "quantum_current_holograph","quantum current holograph","QH","channel","current noise",
        ("graphene ribbon","InAs wire","InSb wire","GaAs channel","WTe2 edge","HgTe edge","MoS2 channel","Si MOS channel"),
        ("encapsulated graphene ribbon","ballistic InAs wire","Majorana-tuned InSb wire","high-mobility GaAs channel","helical WTe2 edge","inverted HgTe edge","gated MoS2 channel","isotopic Si MOS channel"),
        ("7 pA","10 pA","13 pA","16 pA","19 pA","22 pA","25 pA","28 pA"),
        ("8.5 pA","11.5 pA","14.5 pA","17.5 pA","20.5 pA","23.5 pA","26.5 pA","29.5 pA"),103701,999000,
    ),
    _Domain(
        "superfluid_phase_camera","superfluid phase camera","SF","sample","phase blur",
        ("He3 cell","He4 cell","Li6 gas","K40 gas","Rb87 gas","Na23 gas","exciton condensate","polariton condensate"),
        ("confined He3 cell","film He4 cell","unitary Li6 gas","paired K40 gas","vortex Rb87 gas","spinor Na23 gas","bilayer exciton condensate","driven polariton condensate"),
        ("0.16 mrad","0.22 mrad","0.28 mrad","0.34 mrad","0.40 mrad","0.46 mrad","0.52 mrad","0.58 mrad"),
        ("0.19 mrad","0.25 mrad","0.31 mrad","0.37 mrad","0.43 mrad","0.49 mrad","0.55 mrad","0.61 mrad"),103801,1000000,
    ),
    _Domain(
        "electron_hydrodynamic_compass","electron hydrodynamic compass","EH","device","viscosity blur",
        ("graphene Hall bar","GaAs strip","PdCoO2 strip","WTe2 strip","MoP strip","PtSn4 strip","WP2 strip","Cd3As2 strip"),
        ("ultraclean graphene Hall bar","ballistic GaAs strip","delafossite PdCoO2 strip","encapsulated WTe2 strip","hydrodynamic MoP strip","high-purity PtSn4 strip","Weyl WP2 strip","Dirac Cd3As2 strip"),
        ("0.8 um2/s","1.1 um2/s","1.4 um2/s","1.7 um2/s","2.0 um2/s","2.3 um2/s","2.6 um2/s","2.9 um2/s"),
        ("0.95 um2/s","1.25 um2/s","1.55 um2/s","1.85 um2/s","2.15 um2/s","2.45 um2/s","2.75 um2/s","3.05 um2/s"),103901,1001000,
    ),
    _Domain(
        "quantum_strain_interferometer","quantum strain interferometer","QS","sensor","strain floor",
        ("Si beam","SiN beam","diamond beam","SiC beam","quartz beam","GaAs beam","AlN beam","graphene beam"),
        ("phononic Si beam","soft-clamped SiN beam","NV diamond beam","4H-SiC beam","low-loss quartz beam","piezo GaAs beam","suspended AlN beam","tensioned graphene beam"),
        ("4 peps","6 peps","8 peps","10 peps","12 peps","14 peps","16 peps","18 peps"),
        ("5 peps","7 peps","9 peps","11 peps","13 peps","15 peps","17 peps","19 peps"),104001,1002000,
    ),
    _Domain(
        "terahertz_phase_tomograph","terahertz phase tomograph","TH","detector","phase noise",
        ("graphene FET","HEMT array","Schottky array","bolometer array","QCL mixer","Josephson mixer","InSb detector","GaAs detector"),
        ("plasmonic graphene FET","cryogenic HEMT array","balanced Schottky array","TES bolometer array","dual-comb QCL mixer","Josephson heterodyne mixer","hot-electron InSb detector","photoconductive GaAs detector"),
        ("0.22 deg","0.30 deg","0.38 deg","0.46 deg","0.54 deg","0.62 deg","0.70 deg","0.78 deg"),
        ("0.26 deg","0.34 deg","0.42 deg","0.50 deg","0.58 deg","0.66 deg","0.74 deg","0.82 deg"),104101,1003000,
    ),
    _Domain(
        "quantum_magnetothermal_mapper","quantum magnetothermal mapper","QM","sensor","thermal drift",
        ("NV tile","SiV tile","SQUID loop","Hall cross","graphene bolometer","MoS2 sensor","Johnson probe","TES pixel"),
        ("vector NV tile","isotopic SiV tile","nanoSQUID loop","ballistic Hall cross","suspended graphene bolometer","encapsulated MoS2 sensor","cross-correlated Johnson probe","transition-edge TES pixel"),
        ("0.7 mK","1.0 mK","1.3 mK","1.6 mK","1.9 mK","2.2 mK","2.5 mK","2.8 mK"),
        ("0.85 mK","1.15 mK","1.45 mK","1.75 mK","2.05 mK","2.35 mK","2.65 mK","2.95 mK"),104201,1004000,
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
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(
        f"{value} is the S48 quotient-authority {field} value for this {noun}"
        for _kind,field,value in rows
    )
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S48-Quotient {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S48-Quotient record {code} stores {second} for {spec.field_b}; the same {spec.noun} stores {first} for {spec.field_a}.",
            f"For S48-Quotient {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value belongs to {spec.field_a} in S48-Quotient record {code}?",
            f"For S48-Quotient {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value belongs to {spec.field_b} in S48-Quotient record {code}?",
        )
    return (
        f"S48-Quotient audit {code} for the {spec.noun}: {spec.field_a} is {first}, while {spec.field_b} is {second}.",
        f"Audit {code} assigns {second} to {spec.field_b}; independently, the {spec.noun} assigns {first} to {spec.field_a}.",
        f"Read S48-Quotient audit {code}: what is {spec.field_a}?",
        f"In S48-Quotient audit {code}, which entry is tagged {spec.field_a}?",
        f"Read S48-Quotient audit {code}: what is {spec.field_b}?",
        f"In S48-Quotient audit {code}, which entry is tagged {spec.field_b}?",
    )


def _build(spec,index,split):
    firsts=spec.train_a if split=="train" else spec.dev_a
    seconds=spec.train_b if split=="train" else spec.dev_b
    first=firsts[index%len(firsts)]
    second=seconds[(index*3+1)%len(seconds)]
    distractor_first=firsts[(index+3)%len(firsts)]
    distractor_second=seconds[(index*5+2)%len(seconds)]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    case_id=f"{split}-s48-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=case_id,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=distractor_first,
        distractor_second=distractor_second,seed=spec.seed+offset,
    )
    return S48QuotientCase(
        case_id,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,
        texts,aliases,ids,ga,gb,
    )


def generate_s48_cases(split: Split) -> tuple[S48QuotientCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s48_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192:
        raise RuntimeError("S48 partition size changed")
    domains={d.name for d in _DOMAINS}
    if len(domains)!=12:
        raise RuntimeError("S48 domain count changed")
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S48 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S48 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S48 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b:
                raise RuntimeError("S48 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S48 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S48 option IDs changed")
            if r.gold_a==r.gold_b:
                raise RuntimeError("S48 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}
    ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds:
        raise RuntimeError("S48 TRAIN DEV state overlap")
    if tq&dq:
        raise RuntimeError("S48 TRAIN DEV question overlap")
    if to&do:
        raise RuntimeError("S48 TRAIN DEV option overlap")


__all__=["S48QuotientCase","generate_s48_cases","validate_s48_partitions"]
