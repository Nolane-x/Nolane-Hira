from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S21FusionCase:
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
        row = asdict(self)
        row["option_texts"] = list(self.option_texts)
        row["option_aliases"] = list(self.option_aliases)
        row["option_ids"] = list(self.option_ids)
        return row


@dataclass(frozen=True)
class _Domain:
    name: str
    noun: str
    prefix: str
    field_a: str
    field_b: str
    train_a: tuple[str, ...]
    dev_a: tuple[str, ...]
    train_b: tuple[str, ...]
    dev_b: tuple[str, ...]
    seed: int
    dev_offset: int


_DOMAINS = (
    _Domain("mot_controller","magneto-optic trap controller","MC","cooling transition","detuning",
        ("D2 cycling","D1 gray molasses","repump assisted","sigma-plus","lin-perp-lin","Raman sideband","narrow-line","dual-frequency"),
        ("F=2 to F'=3","D1 Lambda","dark-state molasses","circular MOT","polarization gradient","resolved-sideband","intercombination line","bichromatic"),
        ("-8 MHz","-12 MHz","-16 MHz","-20 MHz","-24 MHz","-28 MHz","-32 MHz","-36 MHz"),
        ("-10 MHz","-14 MHz","-18 MHz","-22 MHz","-26 MHz","-30 MHz","-34 MHz","-38 MHz"),37001,261000),
    _Domain("mass_spec_source","mass spectrometer source","MS","ionization mode","source voltage",
        ("ESI","APCI","MALDI","EI","CI","FAB","DESI","ICP"),
        ("nanoESI","photoionization","MALDI-2","field ionization","negative CI","LSIMS","DART","glow discharge"),
        ("1.8 kV","2.2 kV","2.6 kV","3.0 kV","3.4 kV","3.8 kV","4.2 kV","4.6 kV"),
        ("2.0 kV","2.4 kV","2.8 kV","3.2 kV","3.6 kV","4.0 kV","4.4 kV","4.8 kV"),37101,262000),
    _Domain("frequency_synth","optical frequency synthesizer","FS","comb architecture","servo bandwidth",
        ("Er:fiber","Yb:fiber","Ti:sapphire","microcomb","EO comb","difference-frequency","dual-comb","soliton crystal"),
        ("NALM Er","Yb thin-disk","Kerr-lens TiS","SiN DKS","phase-modulated EO","DFG offset-free","asynchronous dual","dark-pulse microcomb"),
        ("20 kHz","40 kHz","60 kHz","80 kHz","100 kHz","120 kHz","140 kHz","160 kHz"),
        ("30 kHz","50 kHz","70 kHz","90 kHz","110 kHz","130 kHz","150 kHz","170 kHz"),37201,263000),
    _Domain("mems_tester","MEMS accelerometer tester","MT","excitation profile","sample rate",
        ("sine sweep","random vibration","shock pulse","chirp","multi-sine","step acceleration","centrifuge","modal tap"),
        ("log sweep","pink-noise vibration","half-sine shock","exponential chirp","orthogonal multi-tone","ramp acceleration","rate table","impulse hammer"),
        ("2 kHz","4 kHz","6 kHz","8 kHz","10 kHz","12 kHz","14 kHz","16 kHz"),
        ("3 kHz","5 kHz","7 kHz","9 kHz","11 kHz","13 kHz","15 kHz","17 kHz"),37301,264000),
    _Domain("cryo_pump","cryogenic pump controller","CP","regeneration mode","heater power",
        ("timed warmup","pressure-triggered","manual bake","staged regen","gas-assisted","closed-loop","pulse heat","dual-zone"),
        ("adaptive warmup","load-triggered","vacuum bake","multi-stage regen","helium purge","PID regeneration","burst heating","zoned ramp"),
        ("40 W","60 W","80 W","100 W","120 W","140 W","160 W","180 W"),
        ("50 W","70 W","90 W","110 W","130 W","150 W","170 W","190 W"),37401,265000),
    _Domain("xrf_analyzer","X-ray fluorescence analyzer","XA","anode material","tube current",
        ("rhodium","silver","tungsten","molybdenum","chromium","copper","scandium","gold"),
        ("Rh thin-window","Ag microfocus","W transmission","Mo sealed","Cr long-fine-focus","Cu rotating","Sc low-energy","Au microtube"),
        ("10 mA","15 mA","20 mA","25 mA","30 mA","35 mA","40 mA","45 mA"),
        ("12 mA","17 mA","22 mA","27 mA","32 mA","37 mA","42 mA","47 mA"),37501,266000),
    _Domain("probe_station","semiconductor probe station","PS","chuck mode","contact force",
        ("vacuum","thermal","RF","high-voltage","cryogenic","manual","semi-auto","fully-auto"),
        ("guarded vacuum","hot-cold thermal","mmWave RF","pulsed HV","closed-cycle cryo","micropositioned","vision-assisted","wafer-map auto"),
        ("5 mN","10 mN","15 mN","20 mN","25 mN","30 mN","35 mN","40 mN"),
        ("7 mN","12 mN","17 mN","22 mN","27 mN","32 mN","37 mN","42 mN"),37601,267000),
    _Domain("magnetometer_cal","magnetometer calibration rig","MG","field waveform","rotation rate",
        ("DC bias","triangle","sine","square","chirped","rotating vector","pseudo-random","multi-axis step"),
        ("offset-cancelled DC","trapezoid","phase-locked sine","bipolar square","log chirp","conical vector","PRBS","3D staircase"),
        ("0.5 deg/s","1 deg/s","1.5 deg/s","2 deg/s","2.5 deg/s","3 deg/s","3.5 deg/s","4 deg/s"),
        ("0.75 deg/s","1.25 deg/s","1.75 deg/s","2.25 deg/s","2.75 deg/s","3.25 deg/s","3.75 deg/s","4.25 deg/s"),37701,268000),
    _Domain("microreactor","continuous-flow microreactor","MR","mixing element","residence time",
        ("T-mixer","herringbone","split-recombine","packed bed","coiled tube","jet mixer","static helix","droplet slug"),
        ("impinging T","staggered herringbone","SAR laminate","catalyst monolith","Dean-flow coil","confined jet","Kenics insert","segmented flow"),
        ("2 s","4 s","6 s","8 s","10 s","12 s","14 s","16 s"),
        ("3 s","5 s","7 s","9 s","11 s","13 s","15 s","17 s"),37801,269000),
    _Domain("acoustic_emission","acoustic emission monitor","AE","sensor coupling","threshold",
        ("grease","wax","epoxy","dry clamp","magnetic shoe","water film","gel","waveguide"),
        ("silicone grease","hot-melt wax","low-shrink epoxy","spring clamp","magnetic base","thin water layer","ultrasound gel","steel waveguide"),
        ("35 dB","40 dB","45 dB","50 dB","55 dB","60 dB","65 dB","70 dB"),
        ("37 dB","42 dB","47 dB","52 dB","57 dB","62 dB","67 dB","72 dB"),37901,270000),
    _Domain("thz_tds","terahertz time-domain spectrometer","TH","detector type","delay step",
        ("ZnTe EO","GaP EO","PCA","Schottky","bolometric","air-biased","DSTMS","organic crystal"),
        ("thin-ZnTe EO","GaSe sampling","LT-GaAs PCA","zero-bias diode","Golay cell","ABCD","DAST","OH1 crystal"),
        ("2 fs","4 fs","6 fs","8 fs","10 fs","12 fs","14 fs","16 fs"),
        ("3 fs","5 fs","7 fs","9 fs","11 fs","13 fs","15 fs","17 fs"),38001,271000),
    _Domain("sem_detector","electron microscope detector","ED","detector mode","dwell time",
        ("SE","BSE","EDS","EBSD","STEM bright-field","HAADF","CL","in-lens"),
        ("Everhart-Thornley","annular BSE","windowless EDS","direct EBSD","BF STEM","ADF STEM","parabolic CL","through-lens"),
        ("1 us","2 us","3 us","4 us","5 us","6 us","7 us","8 us"),
        ("1.5 us","2.5 us","3.5 us","4.5 us","5.5 us","6.5 us","7.5 us","8.5 us"),38101,272000),
)


def _shuffle(
    *,
    case_id: str,
    noun: str,
    field_a: str,
    field_b: str,
    first: str,
    second: str,
    distractor_first: str,
    distractor_second: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int, int]:
    rows = [
        ("a", field_a, first),
        ("b", field_b, second),
        ("x", field_a, distractor_first),
        ("y", field_b, distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts = tuple(
        f"for the {noun}, {field} is {value}"
        for _kind, field, value in rows
    )
    aliases = tuple(
        f"{value} is the recorded {field} entry in this {noun}"
        for _kind, field, value in rows
    )
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _field, _value) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _field, _value) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _views(
    spec: _Domain,
    *,
    split: Split,
    code: str,
    first: str,
    second: str,
) -> tuple[str, str, str, str, str, str]:
    if split == "train":
        state_a = (
            f"S21-Factor {spec.noun} record {code}: {spec.field_a} {first}; "
            f"{spec.field_b} {second}."
        )
        state_b = (
            f"S21-Factor record {code} lists {second} beside {spec.field_b}. "
            f"In the same {spec.noun}, {spec.field_a} is {first}."
        )
        qa1 = f"For S21-Factor {spec.noun} {code}, identify {spec.field_a}."
        qa2 = f"In S21-Factor record {code}, which entry belongs under {spec.field_a}?"
        qb1 = f"For S21-Factor {spec.noun} {code}, identify {spec.field_b}."
        qb2 = f"In S21-Factor record {code}, which entry belongs under {spec.field_b}?"
    else:
        state_a = (
            f"S21-Factor validation dossier {code} for the {spec.noun} places "
            f"{first} under {spec.field_a}; the {spec.field_b} entry reads {second}."
        )
        state_b = (
            f"S21-Factor dossier {code} marks {second} for {spec.field_b}. "
            f"The same {spec.noun} dossier tags {first} as {spec.field_a}."
        )
        qa1 = f"From S21-Factor dossier {code}, identify {spec.field_a}."
        qa2 = f"Which S21-Factor dossier entry is tagged {spec.field_a} for {code}?"
        qb1 = f"From S21-Factor dossier {code}, identify {spec.field_b}."
        qb2 = f"Which S21-Factor dossier entry is tagged {spec.field_b} for {code}?"
    return state_a, state_b, qa1, qa2, qb1, qb2


def _build(spec: _Domain, index: int, split: Split) -> S21FusionCase:
    firsts = spec.train_a if split == "train" else spec.dev_a
    seconds = spec.train_b if split == "train" else spec.dev_b
    first = firsts[index % len(firsts)]
    second = seconds[(index * 3 + 1) % len(seconds)]
    distractor_first = firsts[(index + 3) % len(firsts)]
    distractor_second = seconds[(index * 5 + 2) % len(seconds)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    code = f"{spec.prefix}{offset:05d}"
    case_id = f"{split}-s21-{spec.name}-{index:03d}"

    state_a, state_b, qa1, qa2, qb1, qb2 = _views(
        spec,
        split=split,
        code=code,
        first=first,
        second=second,
    )
    texts, aliases, ids, ga, gb = _shuffle(
        case_id=case_id,
        noun=spec.noun,
        field_a=spec.field_a,
        field_b=spec.field_b,
        first=first,
        second=second,
        distractor_first=distractor_first,
        distractor_second=distractor_second,
        seed=spec.seed + offset,
    )
    return S21FusionCase(
        case_id=case_id,
        split=split,
        domain=spec.name,
        language="en",
        state_a=state_a,
        state_b=state_b,
        question_a1=qa1,
        question_a2=qa2,
        question_b1=qb1,
        question_b2=qb2,
        option_texts=texts,
        option_aliases=aliases,
        option_ids=ids,
        gold_a=ga,
        gold_b=gb,
    )


def generate_s21_cases(split: Split) -> tuple[S21FusionCase, ...]:
    count = 64 if split == "train" else 16
    return tuple(
        _build(spec, index, split)
        for spec in _DOMAINS
        for index in range(count)
    )


def validate_s21_partitions(
    train: tuple[S21FusionCase, ...],
    dev: tuple[S21FusionCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("S21 partition size changed")
    if len({row.domain for row in train}) != 12:
        raise RuntimeError("S21 TRAIN domain count changed")
    if len({row.domain for row in dev}) != 12:
        raise RuntimeError("S21 DEV domain count changed")
    if any(row.split != "train" for row in train):
        raise RuntimeError("S21 TRAIN split label changed")
    if any(row.split != "dev" for row in dev):
        raise RuntimeError("S21 DEV split label changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("S21 language changed")
    if any(len(row.option_texts) != 4 for row in (*train, *dev)):
        raise RuntimeError("S21 K changed")
    if any(len(row.option_aliases) != 4 for row in (*train, *dev)):
        raise RuntimeError("S21 option alias count changed")
    if any(len(row.option_ids) != 4 for row in (*train, *dev)):
        raise RuntimeError("S21 option ID count changed")

    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    train_q = {
        x
        for row in train
        for x in (
            row.question_a1,
            row.question_a2,
            row.question_b1,
            row.question_b2,
        )
    }
    dev_q = {
        x
        for row in dev
        for x in (
            row.question_a1,
            row.question_a2,
            row.question_b1,
            row.question_b2,
        )
    }
    if train_states & dev_states:
        raise RuntimeError("S21 TRAIN/DEV exact state overlap")
    if train_q & dev_q:
        raise RuntimeError("S21 TRAIN/DEV exact question overlap")


__all__ = [
    "S21FusionCase",
    "generate_s21_cases",
    "validate_s21_partitions",
]
