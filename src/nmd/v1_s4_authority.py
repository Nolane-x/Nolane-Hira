from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Callable, Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S4PairCase:
    case_id: str
    split: Split
    domain: str
    language: str
    state: str
    question_a: str
    question_b: str
    option_texts: tuple[str, ...]
    option_ids: tuple[str, ...]
    gold_a: int
    gold_b: int

    def to_dict(self) -> dict:
        row = asdict(self)
        row["option_texts"] = list(self.option_texts)
        row["option_ids"] = list(self.option_ids)
        return row


def _shuffle(
    *,
    case_id: str,
    first: str,
    second: str,
    distractor_first: str,
    distractor_second: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    rows = [
        ("a", first),
        ("b", second),
        ("x", distractor_first),
        ("y", distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts = tuple(text for _, text in rows)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _) in enumerate(rows) if kind == "b")
    return texts, ids, ga, gb


def _build_case(
    *,
    index: int,
    split: Split,
    domain: str,
    prefix: str,
    first_values: tuple[str, ...],
    second_values: tuple[str, ...],
    state_train: str,
    state_dev: str,
    question_a_train: str,
    question_b_train: str,
    question_a_dev: str,
    question_b_dev: str,
    answer_a: Callable[[str], str],
    answer_b: Callable[[str], str],
    seed_base: int,
    dev_offset: int,
) -> S4PairCase:
    first = first_values[index % len(first_values)]
    second = second_values[(index * 3 + 1) % len(second_values)]
    other_first = first_values[(index + 3) % len(first_values)]
    other_second = second_values[(index * 5 + 2) % len(second_values)]
    offset = index + (dev_offset if split == "dev" else 0)
    case_id = f"{split}-s4-{domain}-{index:03d}"

    state = (state_train if split == "train" else state_dev).format(
        code=f"{prefix}{offset:04d}",
        first=first,
        second=second,
    )
    qa = question_a_train if split == "train" else question_a_dev
    qb = question_b_train if split == "train" else question_b_dev

    texts, ids, ga, gb = _shuffle(
        case_id=case_id,
        first=answer_a(first),
        second=answer_b(second),
        distractor_first=answer_a(other_first),
        distractor_second=answer_b(other_second),
        seed=seed_base + offset,
    )
    return S4PairCase(
        case_id,
        split,
        domain,
        "en",
        state,
        qa,
        qb,
        texts,
        ids,
        ga,
        gb,
    )


def _solar(index: int, split: Split) -> S4PairCase:
    train_a = ("mono-PERC", "bifacial", "thin-film", "HJT", "TOPCon", "IBC", "poly-Si", "CdTe")
    dev_a = ("CIGS", "perovskite", "a-Si", "tandem", "GaAs", "organic", "CZTS", "PERC-X")
    train_b = ("12 degrees", "18 degrees", "23 degrees", "27 degrees", "31 degrees", "35 degrees", "39 degrees", "44 degrees")
    dev_b = ("14 degrees", "20 degrees", "25 degrees", "29 degrees", "33 degrees", "37 degrees", "41 degrees", "46 degrees")
    return _build_case(
        index=index, split=split, domain="solar_array", prefix="SA",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Solar record {code} specifies module type {first} and tilt angle {second}.",
        state_dev="Array dossier {code} lists {first} modules; the mounting inclination is {second}.",
        question_a_train="Which module type is specified in the solar record?",
        question_b_train="What tilt angle is specified in the solar record?",
        question_a_dev="What module technology does this array dossier list?",
        question_b_dev="What mounting inclination is recorded for this array?",
        answer_a=lambda x: f"the requested answer is {x}",
        answer_b=lambda x: f"the requested answer is {x}",
        seed_base=9401, dev_offset=13000,
    )


def _pharmacy(index: int, split: Split) -> S4PairCase:
    train_a = ("azurin", "belorin", "cerafin", "doramed", "elixin", "feradol", "gantrix", "helovin")
    dev_a = ("ivarol", "jurexin", "kalmed", "luminex", "merovin", "norafin", "optilen", "praxol")
    train_b = ("5 mg", "8 mg", "12 mg", "16 mg", "20 mg", "24 mg", "30 mg", "40 mg")
    dev_b = ("6 mg", "10 mg", "14 mg", "18 mg", "22 mg", "26 mg", "32 mg", "44 mg")
    return _build_case(
        index=index, split=split, domain="pharmacy_batch", prefix="PB",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Pharmacy batch {code} contains compound {first} at dose {second}.",
        state_dev="Dispensing ledger {code} names active agent {first}; unit strength is {second}.",
        question_a_train="Which compound is contained in the pharmacy batch?",
        question_b_train="What dose is recorded for the pharmacy batch?",
        question_a_dev="What active agent is named in this dispensing ledger?",
        question_b_dev="What unit strength does this ledger record?",
        answer_a=lambda x: f"the requested answer is {x}",
        answer_b=lambda x: f"the requested answer is {x}",
        seed_base=9501, dev_offset=14000,
    )


def _satellite(index: int, split: Split) -> S4PairCase:
    train_a = ("polar-3", "equator-7", "aurora-2", "zenith-9", "vector-4", "apex-6", "nova-1", "orbit-8")
    dev_a = ("lunar-5", "cosmos-3", "stellar-7", "horizon-2", "pulsar-6", "meridian-4", "galaxy-9", "comet-1")
    train_b = ("X-band", "S-band", "Ka-band", "Ku-band", "L-band", "C-band", "UHF", "VHF")
    dev_b = ("Q-band", "K-band", "P-band", "W-band", "HF", "SHF", "EHF", "X2-band")
    return _build_case(
        index=index, split=split, domain="satellite_task", prefix="ST",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Satellite task {code} targets track {first} and uses downlink {second}.",
        state_dev="Mission sheet {code} assigns trajectory {first}; telemetry channel is {second}.",
        question_a_train="Which track is targeted by the satellite task?",
        question_b_train="Which downlink is used by the satellite task?",
        question_a_dev="What trajectory does the mission sheet assign?",
        question_b_dev="What telemetry channel does the mission sheet specify?",
        answer_a=lambda x: f"the requested answer is {x}",
        answer_b=lambda x: f"the requested answer is {x}",
        seed_base=9601, dev_offset=15000,
    )


def _quarry(index: int, split: Split) -> S4PairCase:
    train_a = ("diorite", "shale", "limestone", "gneiss", "slate", "quartzite", "marble", "dolomite")
    dev_a = ("andesite", "rhyolite", "serpentinite", "phyllite", "travertine", "hornfels", "pegmatite", "breccia")
    train_b = ("bench 3", "bench 5", "bench 8", "bench 11", "bench 14", "bench 17", "bench 20", "bench 23")
    dev_b = ("bench 4", "bench 6", "bench 9", "bench 12", "bench 15", "bench 18", "bench 21", "bench 24")
    return _build_case(
        index=index, split=split, domain="quarry_sample", prefix="QS",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Quarry sample {code} is classified as {first} and was taken from {second}.",
        state_dev="Extraction note {code} identifies rock type {first}; collection point is {second}.",
        question_a_train="What rock classification is recorded for the quarry sample?",
        question_b_train="From which bench was the quarry sample taken?",
        question_a_dev="What rock type does the extraction note identify?",
        question_b_dev="What collection point is listed in the extraction note?",
        answer_a=lambda x: f"the requested answer is {x}",
        answer_b=lambda x: f"the requested answer is {x}",
        seed_base=9701, dev_offset=16000,
    )


def _music(index: int, split: Split) -> S4PairCase:
    train_a = ("cello", "oboe", "harp", "clarinet", "viola", "bassoon", "flute", "trumpet")
    dev_a = ("mandolin", "sitar", "dulcimer", "cor anglais", "trombone", "piccolo", "euphonium", "banjo")
    train_b = ("D minor", "A major", "F major", "C minor", "G major", "E minor", "B flat", "E flat")
    dev_b = ("B minor", "D major", "A minor", "G minor", "C major", "F minor", "A flat", "D flat")
    return _build_case(
        index=index, split=split, domain="music_catalog", prefix="MC",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Music catalog {code} lists lead instrument {first} and tonal center {second}.",
        state_dev="Performance index {code} names {first} as principal instrument; written key is {second}.",
        question_a_train="Which lead instrument is listed in the music catalog?",
        question_b_train="What tonal center is listed in the music catalog?",
        question_a_dev="What principal instrument does the performance index name?",
        question_b_dev="What written key is recorded in the performance index?",
        answer_a=lambda x: f"the requested answer is {x}",
        answer_b=lambda x: f"the requested answer is {x}",
        seed_base=9801, dev_offset=17000,
    )


def _drill(index: int, split: Split) -> S4PairCase:
    train_a = ("sector Alpha", "sector Bravo", "sector Charlie", "sector Delta", "sector Echo", "sector Foxtrot", "sector Golf", "sector Hotel")
    dev_a = ("sector India", "sector Juliet", "sector Kilo", "sector Lima", "sector Mike", "sector November", "sector Oscar", "sector Papa")
    train_b = ("07:15", "08:30", "09:45", "11:00", "12:15", "13:30", "14:45", "16:00")
    dev_b = ("07:25", "08:40", "09:55", "11:10", "12:25", "13:40", "14:55", "16:10")
    return _build_case(
        index=index, split=split, domain="emergency_drill", prefix="ED",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Emergency drill {code} assigns assembly area {first} and start time {second}.",
        state_dev="Response exercise {code} directs participants to {first}; activation begins at {second}.",
        question_a_train="Which assembly area is assigned for the emergency drill?",
        question_b_train="What start time is assigned for the emergency drill?",
        question_a_dev="Where does the response exercise direct participants?",
        question_b_dev="When does activation begin for the response exercise?",
        answer_a=lambda x: f"the requested answer is {x}",
        answer_b=lambda x: f"the requested answer is {x}",
        seed_base=9901, dev_offset=18000,
    )


def generate_s4_pairs(split: Split) -> tuple[S4PairCase, ...]:
    count = 64 if split == "train" else 16
    rows = []
    for builder in (_solar, _pharmacy, _satellite, _quarry, _music, _drill):
        rows.extend(builder(i, split) for i in range(count))
    return tuple(rows)


def validate_s4_partitions(
    train: tuple[S4PairCase, ...],
    dev: tuple[S4PairCase, ...],
) -> None:
    if len(train) != 384 or len(dev) != 96:
        raise RuntimeError("Hira v1 S4 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S4-A must remain English-only")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S4 duplicate case IDs")
        for row in rows:
            if len(row.option_texts) != 4 or len(row.option_ids) != 4:
                raise RuntimeError("Hira v1 S4 requires K=4")
            if len(set(row.option_ids)) != 4:
                raise RuntimeError("Hira v1 S4 option IDs must be unique")
            if row.gold_a == row.gold_b:
                raise RuntimeError("Hira v1 S4 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("Hira v1 S4 gold index out of range")

    train_states = {row.state for row in train}
    dev_states = {row.state for row in dev}
    if train_states & dev_states:
        raise RuntimeError("Hira v1 S4 TRAIN/DEV state overlap")

    train_q = {q for row in train for q in (row.question_a, row.question_b)}
    dev_q = {q for row in dev for q in (row.question_a, row.question_b)}
    if train_q & dev_q:
        raise RuntimeError("Hira v1 S4 TRAIN/DEV question overlap")


__all__ = ["S4PairCase", "generate_s4_pairs", "validate_s4_partitions"]
