from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal

Split = Literal["train","dev"]


@dataclass(frozen=True)
class S30MatchedCase:
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
    option_texts: tuple[str,...]
    option_aliases: tuple[str,...]
    option_ids: tuple[str,...]
    gold_a: int
    gold_b: int

    def to_dict(self) -> dict:
        row=asdict(self)
        row["option_texts"]=list(self.option_texts)
        row["option_aliases"]=list(self.option_aliases)
        row["option_ids"]=list(self.option_ids)
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
    _Domain("fiber_interferometer","fiber interferometer","FI","fiber type","phase bias",
        ("SMF-28","PM fiber","HNLF","DCF","PCF","erbium fiber","fluoride fiber","multicore fiber"),
        ("PANDA PM","bow-tie PM","tellurite HNLF","chirped DCF","hollow-core PCF","ytterbium fiber","ZBLAN","seven-core fiber"),
        ("10 deg","20 deg","30 deg","40 deg","50 deg","60 deg","70 deg","80 deg"),
        ("15 deg","25 deg","35 deg","45 deg","55 deg","65 deg","75 deg","85 deg"),53101,501000),
    _Domain("rf_synthesizer","RF synthesizer","RS","reference source","output level",
        ("OCXO","TCXO","rubidium","GPSDO","maser","crystal","PLL ref","external 10 MHz"),
        ("SC-cut OCXO","oven TCXO","Rb mini-clock","GNSSDO","active maser","BVA crystal","DDS ref","external 100 MHz"),
        ("-20 dBm","-15 dBm","-10 dBm","-5 dBm","0 dBm","5 dBm","10 dBm","15 dBm"),
        ("-18 dBm","-13 dBm","-8 dBm","-3 dBm","2 dBm","7 dBm","12 dBm","17 dBm"),53201,502000),
    _Domain("cold_atom_source","cold-atom source","CA","atom species","oven temperature",
        ("Rb-87","Cs-133","K-39","Li-7","Sr-88","Yb-174","Na-23","Ca-40"),
        ("Rb-85","Cs-135","K-41","Li-6","Sr-87","Yb-171","Na-22","Ca-44"),
        ("80 C","100 C","120 C","140 C","160 C","180 C","200 C","220 C"),
        ("90 C","110 C","130 C","150 C","170 C","190 C","210 C","230 C"),53301,503000),
    _Domain("microfluidic_pump","microfluidic pump","MP","actuator","flow setpoint",
        ("piezo","syringe","peristaltic","electroosmotic","pneumatic","diaphragm","gear","capillary"),
        ("stack piezo","dual syringe","roller","AC electroosmotic","pressure servo","MEMS diaphragm","microgear","electrowetting"),
        ("5 uL/min","10 uL/min","15 uL/min","20 uL/min","25 uL/min","30 uL/min","35 uL/min","40 uL/min"),
        ("7 uL/min","12 uL/min","17 uL/min","22 uL/min","27 uL/min","32 uL/min","37 uL/min","42 uL/min"),53401,504000),
    _Domain("xray_monochromator","X-ray monochromator","XM","crystal","Bragg angle",
        ("Si111","Si220","Ge111","diamond111","Si311","Ge220","quartz","sapphire"),
        ("Si400","Si422","Ge311","diamond220","Si511","Ge400","calcite","YAG"),
        ("10 deg","15 deg","20 deg","25 deg","30 deg","35 deg","40 deg","45 deg"),
        ("12 deg","17 deg","22 deg","27 deg","32 deg","37 deg","42 deg","47 deg"),53501,505000),
    _Domain("laser_diode_driver","laser-diode driver","LD","control mode","current limit",
        ("constant current","constant power","pulse","analog modulation","digital modulation","soft start","burst","bias tee"),
        ("CC servo","optical APC","nanosecond pulse","linear modulation","PWM modulation","slew-limited","packet burst","dual-bias"),
        ("100 mA","200 mA","300 mA","400 mA","500 mA","600 mA","700 mA","800 mA"),
        ("150 mA","250 mA","350 mA","450 mA","550 mA","650 mA","750 mA","850 mA"),53601,506000),
    _Domain("vacuum_stage","vacuum translation stage","VS","bearing","travel",
        ("cross roller","air bearing","flexure","magnetic","ceramic slide","ball screw","linear motor","piezo walk"),
        ("vacuum roller","porous air","compound flexure","maglev","alumina slide","roller screw","ironless motor","inchworm"),
        ("5 mm","10 mm","15 mm","20 mm","25 mm","30 mm","35 mm","40 mm"),
        ("7 mm","12 mm","17 mm","22 mm","27 mm","32 mm","37 mm","42 mm"),53701,507000),
    _Domain("thermal_imager","thermal imager","TI","detector","integration time",
        ("microbolometer","InSb","MCT","InGaAs","QWIP","T2SL","thermopile","pyroelectric"),
        ("VOx microbolometer","cooled InSb","HOT MCT","extended InGaAs","superlattice QWIP","type-II SL","MEMS thermopile","LiTaO3"),
        ("1 ms","2 ms","3 ms","4 ms","5 ms","6 ms","7 ms","8 ms"),
        ("1.5 ms","2.5 ms","3.5 ms","4.5 ms","5.5 ms","6.5 ms","7.5 ms","8.5 ms"),53801,508000),
    _Domain("acoustic_calibrator","acoustic calibrator","AC","reference element","tone level",
        ("condenser mic","MEMS mic","piezo disc","hydrophone","laser vibrometer","pressure sensor","electret","fiber sensor"),
        ("lab condenser","digital MEMS","PZT ring","reference hydrophone","heterodyne vibrometer","quartz pressure","prepolarized electret","FBG acoustic"),
        ("70 dB","75 dB","80 dB","85 dB","90 dB","95 dB","100 dB","105 dB"),
        ("72 dB","77 dB","82 dB","87 dB","92 dB","97 dB","102 dB","107 dB"),53901,509000),
    _Domain("electrochem_station","electrochemistry station","ES","working electrode","scan rate",
        ("glassy carbon","platinum","gold","graphite","boron diamond","ITO","carbon cloth","nickel"),
        ("pyrolytic carbon","Pt disk","Au mesh","HOPG","BDD film","FTO","carbon paper","Ni foam"),
        ("10 mV/s","20 mV/s","30 mV/s","40 mV/s","50 mV/s","60 mV/s","70 mV/s","80 mV/s"),
        ("15 mV/s","25 mV/s","35 mV/s","45 mV/s","55 mV/s","65 mV/s","75 mV/s","85 mV/s"),54001,510000),
    _Domain("photodiode_receiver","photodiode receiver","PR","detector type","transimpedance",
        ("Si PIN","InGaAs PIN","APD","Ge","balanced pair","UV diode","SPAD","MCT"),
        ("large-area Si","extended InGaAs","Si APD","strained Ge","balanced InGaAs","GaN UV","SNSPD interface","cooled MCT"),
        ("1 kOhm","2 kOhm","3 kOhm","4 kOhm","5 kOhm","6 kOhm","7 kOhm","8 kOhm"),
        ("1.5 kOhm","2.5 kOhm","3.5 kOhm","4.5 kOhm","5.5 kOhm","6.5 kOhm","7.5 kOhm","8.5 kOhm"),54101,511000),
    _Domain("inertial_sensor","inertial sensor","IN","gyro type","sample rate",
        ("MEMS","FOG","RLG","HRG","tuning fork","vibrating ring","atomic","optical cavity"),
        ("closed-loop MEMS","IFOG","four-mirror RLG","hemispherical","quartz fork","silicon ring","cold-atom","microresonator"),
        ("100 Hz","200 Hz","300 Hz","400 Hz","500 Hz","600 Hz","700 Hz","800 Hz"),
        ("150 Hz","250 Hz","350 Hz","450 Hz","550 Hz","650 Hz","750 Hz","850 Hz"),54201,512000),
)


def _shuffle(*,case_id,noun,field_a,field_b,first,second,distractor_first,distractor_second,seed):
    rows=[("a",field_a,first),("b",field_b,second),("x",field_a,distractor_first),("y",field_b,distractor_second)]
    random.Random(seed).shuffle(rows)
    texts=tuple(f"for the {noun}, {field} is {value}" for _kind,field,value in rows)
    aliases=tuple(f"{value} is the recorded {field} entry for this {noun}" for _kind,field,value in rows)
    ids=tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
    gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
    return texts,aliases,ids,ga,gb


def _views(spec,*,split,code,first,second):
    if split=="train":
        return (
            f"S30-Matched {spec.noun} record {code}: {spec.field_a} {first}; {spec.field_b} {second}.",
            f"S30-Matched record {code} stores {second} for {spec.field_b}. The same {spec.noun} assigns {first} to {spec.field_a}.",
            f"For S30-Matched {spec.noun} {code}, identify {spec.field_a}.",
            f"In S30-Matched record {code}, which entry belongs under {spec.field_a}?",
            f"For S30-Matched {spec.noun} {code}, identify {spec.field_b}.",
            f"In S30-Matched record {code}, which entry belongs under {spec.field_b}?",
        )
    return (
        f"S30-Matched audit {code} for the {spec.noun} lists {first} as {spec.field_a} and {second} as {spec.field_b}.",
        f"Audit {code} tags {second} under {spec.field_b}; elsewhere the same S30-Matched {spec.noun} tags {first} under {spec.field_a}.",
        f"From S30-Matched audit {code}, what is {spec.field_a}?",
        f"Which audit entry carries the {spec.field_a} tag for {code}?",
        f"From S30-Matched audit {code}, what is {spec.field_b}?",
        f"Which audit entry carries the {spec.field_b} tag for {code}?",
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
    case_id=f"{split}-s30-{spec.name}-{index:03d}"
    state_a,state_b,qa1,qa2,qb1,qb2=_views(spec,split=split,code=code,first=first,second=second)
    texts,aliases,ids,ga,gb=_shuffle(case_id=case_id,noun=spec.noun,field_a=spec.field_a,field_b=spec.field_b,first=first,second=second,distractor_first=df,distractor_second=ds,seed=spec.seed+offset)
    return S30MatchedCase(case_id,split,spec.name,"en",state_a,state_b,qa1,qa2,qb1,qb2,texts,aliases,ids,ga,gb)


def generate_s30_cases(split: Split) -> tuple[S30MatchedCase,...]:
    count=64 if split=="train" else 16
    return tuple(_build(spec,i,split) for spec in _DOMAINS for i in range(count))


def validate_s30_partitions(train,dev):
    if len(train)!=768 or len(dev)!=192: raise RuntimeError("S30 partition size changed")
    domains={d.name for d in _DOMAINS}
    if {r.domain for r in train}!=domains or {r.domain for r in dev}!=domains: raise RuntimeError("S30 domains changed")
    if any(r.language!="en" for r in (*train,*dev)): raise RuntimeError("S30 language changed")
    for rows in (train,dev):
        if len({r.case_id for r in rows})!=len(rows): raise RuntimeError("S30 duplicate case IDs")
        for r in rows:
            if r.state_a==r.state_b: raise RuntimeError("S30 state views must differ")
            if r.question_a1==r.question_a2 or r.question_b1==r.question_b2: raise RuntimeError("S30 question views must differ")
            if len(r.option_ids)!=4 or len(set(r.option_ids))!=4: raise RuntimeError("S30 option IDs changed")
            if r.gold_a==r.gold_b: raise RuntimeError("S30 paired golds must differ")
    ts={x for r in train for x in (r.state_a,r.state_b)}
    ds={x for r in dev for x in (r.state_a,r.state_b)}
    tq={x for r in train for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    dq={x for r in dev for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
    to={x for r in train for x in (*r.option_texts,*r.option_aliases)}
    do={x for r in dev for x in (*r.option_texts,*r.option_aliases)}
    if ts&ds: raise RuntimeError("S30 TRAIN DEV state overlap")
    if tq&dq: raise RuntimeError("S30 TRAIN DEV question overlap")
    if to&do: raise RuntimeError("S30 TRAIN DEV option overlap")


__all__=["S30MatchedCase","generate_s30_cases","validate_s30_partitions"]
