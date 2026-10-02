from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split=Literal["train","dev"]


@dataclass(frozen=True)
class S34TransportCase:
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

    def to_dict(self)->dict:
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
    _Domain("neutron_tomograph","neutron tomograph","NT","collimator","exposure",
        ("pinhole","slit","radial","multi-hole","coded","fan","parallel","hex"),
        ("micro-slit","lamellar","spiral","MURA","annular","cross-fan","stacked","honeycomb"),
        ("2 s","4 s","6 s","8 s","10 s","12 s","14 s","16 s"),
        ("3 s","5 s","7 s","9 s","11 s","13 s","15 s","17 s"),63101,581000),
    _Domain("phonon_spectrometer","phonon spectrometer","PS","analyzer","resolution",
        ("quartz","silicon","germanium","sapphire","mica","diamond","LiF","CaF2"),
        ("SiGe","strained silicon","CVD diamond","AlN","GaN","MgO","YAG","spinel"),
        ("1 meV","2 meV","3 meV","4 meV","5 meV","6 meV","7 meV","8 meV"),
        ("1.5 meV","2.5 meV","3.5 meV","4.5 meV","5.5 meV","6.5 meV","7.5 meV","8.5 meV"),63201,582000),
    _Domain("micro_gravimeter","micro-gravimeter","MG","proof mass","bandwidth",
        ("silicon","quartz","tungsten","gold","platinum","SiC","diamond","glass"),
        ("SiN","fused quartz","iridium","osmium","molybdenum","sapphire","AlN","Zerodur"),
        ("1 Hz","2 Hz","3 Hz","4 Hz","5 Hz","6 Hz","7 Hz","8 Hz"),
        ("1.5 Hz","2.5 Hz","3.5 Hz","4.5 Hz","5.5 Hz","6.5 Hz","7.5 Hz","8.5 Hz"),63301,583000),
    _Domain("plasma_interferometer","plasma interferometer","PI","wavelength","path length",
        ("532 nm","633 nm","780 nm","1064 nm","1310 nm","1550 nm","2 um","3 um"),
        ("515 nm","660 nm","795 nm","1030 nm","1280 nm","1625 nm","2.2 um","3.3 um"),
        ("2 cm","4 cm","6 cm","8 cm","10 cm","12 cm","14 cm","16 cm"),
        ("3 cm","5 cm","7 cm","9 cm","11 cm","13 cm","15 cm","17 cm"),63401,584000),
    _Domain("ion_mobility_cell","ion-mobility cell","IM","drift gas","field strength",
        ("helium","nitrogen","argon","neon","CO2","air","hydrogen","xenon"),
        ("He-Ne","N2-He","Ar-He","Kr","dry air","CO2-N2","D2","Xe-Ne"),
        ("10 V/cm","20 V/cm","30 V/cm","40 V/cm","50 V/cm","60 V/cm","70 V/cm","80 V/cm"),
        ("15 V/cm","25 V/cm","35 V/cm","45 V/cm","55 V/cm","65 V/cm","75 V/cm","85 V/cm"),63501,585000),
    _Domain("magnetooptic_trap","magneto-optic trap","MT","atom species","detuning",
        ("Rb87","Rb85","Cs133","Na23","K39","Li7","Sr88","Yb174"),
        ("K40","Li6","Sr87","Yb171","Ca40","Ba138","Er168","Dy164"),
        ("-1 MHz","-2 MHz","-3 MHz","-4 MHz","-5 MHz","-6 MHz","-7 MHz","-8 MHz"),
        ("-1.5 MHz","-2.5 MHz","-3.5 MHz","-4.5 MHz","-5.5 MHz","-6.5 MHz","-7.5 MHz","-8.5 MHz"),63601,586000),
    _Domain("nanopore_reader","nanopore reader","NR","membrane","bias voltage",
        ("SiN","graphene","MoS2","hBN","Al2O3","glass","polymer","diamond"),
        ("WS2","mica","SiO2","hafnia","zirconia","PET","Parylene","SiC"),
        ("20 mV","40 mV","60 mV","80 mV","100 mV","120 mV","140 mV","160 mV"),
        ("30 mV","50 mV","70 mV","90 mV","110 mV","130 mV","150 mV","170 mV"),63701,587000),
    _Domain("spin_echo_console","spin-echo console","SE","pulse sequence","echo time",
        ("Hahn","CPMG","stimulated","gradient echo","RARE","EPI","UTE","ZTE"),
        ("Carr-Purcell","multi-echo","STEAM","balanced GRE","FSE","spiral EPI","SWIFT","PETRA"),
        ("2 ms","4 ms","6 ms","8 ms","10 ms","12 ms","14 ms","16 ms"),
        ("3 ms","5 ms","7 ms","9 ms","11 ms","13 ms","15 ms","17 ms"),63801,588000),
    _Domain("microcalorimeter","microcalorimeter","MC","absorber","thermal time",
        ("gold","bismuth","tin","copper","tungsten","platinum","aluminum","silicon"),
        ("AuBi","iridium","molybdenum","tantalum","hafnium","niobium","TiN","SiC"),
        ("1 us","2 us","3 us","4 us","5 us","6 us","7 us","8 us"),
        ("1.5 us","2.5 us","3.5 us","4.5 us","5.5 us","6.5 us","7.5 us","8.5 us"),63901,589000),
    _Domain("laser_doppler_vibrometer","laser Doppler vibrometer","LV","laser line","velocity range",
        ("HeNe","Nd:YAG","fiber","diode","Ti:sapphire","Er fiber","Yb fiber","DFB"),
        ("frequency-doubled YAG","VCSEL","ECDL","OPO","Cr:ZnSe","Tm fiber","Ho fiber","DBR"),
        ("1 mm/s","2 mm/s","3 mm/s","4 mm/s","5 mm/s","6 mm/s","7 mm/s","8 mm/s"),
        ("1.5 mm/s","2.5 mm/s","3.5 mm/s","4.5 mm/s","5.5 mm/s","6.5 mm/s","7.5 mm/s","8.5 mm/s"),64001,590000),
    _Domain("cryogenic_amplifier","cryogenic amplifier","CA","device","gain",
        ("HEMT","SiGe","parametric","SQUID","JPA","TWPA","MMIC","maser"),
        ("InP HEMT","GaAs HEMT","JPC","SNAIL","KIT","Josephson TWPA","cryo-CMOS","diamond maser"),
        ("10 dB","12 dB","14 dB","16 dB","18 dB","20 dB","22 dB","24 dB"),
        ("11 dB","13 dB","15 dB","17 dB","19 dB","21 dB","23 dB","25 dB"),64101,591000),
    _Domain("photoacoustic_cell","photoacoustic cell","PA","buffer gas","modulation",
        ("nitrogen","helium","argon","neon","air","CO2","xenon","krypton"),
        ("N2-He","He-Ne","Ar-N2","dry air","CO2-He","Xe-Ne","Kr-Ar","D2"),
        ("100 Hz","200 Hz","300 Hz","400 Hz","500 Hz","600 Hz","700 Hz","800 Hz"),
        ("150 Hz","250 Hz","350 Hz","450 Hz","550 Hz","650 Hz","750 Hz","850 Hz"),64201,592000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the recorded {field} value for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S34-Transport {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S34-Transport record {code} stores {second} for {spec.field_b}. The {spec.noun} also records {first} for {spec.field_a}.",
            f"For S34-Transport {spec.noun} {code}, identify {spec.field_a}.",
            f"Which value is filed under {spec.field_a} in S34-Transport record {code}?",
            f"For S34-Transport {spec.noun} {code}, identify {spec.field_b}.",
            f"Which value is filed under {spec.field_b} in S34-Transport record {code}?",
        )
    return (
        f"S34-Transport audit {code} for the {spec.noun} lists {first} as {spec.field_a} and {second} as {spec.field_b}.",
        f"Audit {code}: {spec.field_b} carries {second}; separately {spec.field_a} carries {first} for this {spec.noun}.",
        f"Read S34-Transport audit {code}: what is {spec.field_a}?",
        f"In audit {code}, which entry corresponds to {spec.field_a}?",
        f"Read S34-Transport audit {code}: what is {spec.field_b}?",
        f"In audit {code}, which entry corresponds to {spec.field_b}?",
    )


def _build(spec,index,split):
    firsts=spec.train_a if split=="train" else spec.dev_a
    seconds=spec.train_b if split=="train" else spec.dev_b
    first=firsts[index%len(firsts)]
    second=seconds[(index*3+1)%len(seconds)]
    df=firsts[(index+3)%len(firsts)]
    ds=seconds[(index*5+2)%len(seconds)]
    offset=index+(spec.dev_offset if split=="dev" else 0)
    code=f"{spec.prefix}{offset:05d}"
    case_id=f"{split}-s34-{spec.name}-{index:03d}"
    sa,sb,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(
        case_id=case_id,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,
        first=first,second=second,distractor_first=df,distractor_second=ds,
        seed=spec.seed+offset,
    )
    return S34TransportCase(
        case_id,split,spec.name,"en",sa,sb,qa1,qa2,qb1,qb2,
        texts,aliases,ids,ga,gb,
    )


def generate_s34_cases(split:Split)->tuple[S34TransportCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s34_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192:
        raise RuntimeError("S34 partition size changed")
    domains={d.name for d in _DOMAINS}
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains:
        raise RuntimeError("S34 domains changed")
    if any(r.language!="en" for r in (*train,*dev)):
        raise RuntimeError("S34 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows):
            raise RuntimeError("S34 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b:
                raise RuntimeError("S34 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2:
                raise RuntimeError("S34 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4:
                raise RuntimeError("S34 option IDs changed")
            if r.gold_a==r.gold_b:
                raise RuntimeError("S34 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}
    ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds:
        raise RuntimeError("S34 TRAIN DEV state overlap")
    if tq&dq:
        raise RuntimeError("S34 TRAIN DEV question overlap")
    if to&do:
        raise RuntimeError("S34 TRAIN DEV option overlap")


__all__=["S34TransportCase","generate_s34_cases","validate_s34_partitions"]
