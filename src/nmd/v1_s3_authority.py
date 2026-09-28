from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S3PairCase:
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


_TRAIN_MINERALS = ("basalt", "quartzite", "gneiss", "dolomite", "shale", "granite", "slate", "marble")
_DEV_MINERALS = ("andesite", "rhyolite", "schist", "limestone", "sandstone", "serpentinite", "phyllite", "obsidian")
_TRAIN_DEPTHS = ("12 meters", "19 meters", "27 meters", "34 meters", "41 meters", "53 meters", "68 meters", "76 meters")
_DEV_DEPTHS = ("14 meters", "23 meters", "31 meters", "38 meters", "47 meters", "59 meters", "64 meters", "82 meters")

_TRAIN_FREQ = ("88.4 MHz", "91.7 MHz", "94.2 MHz", "97.6 MHz", "101.3 MHz", "104.8 MHz", "107.1 MHz", "109.5 MHz")
_DEV_FREQ = ("89.6 MHz", "92.9 MHz", "95.4 MHz", "98.8 MHz", "102.5 MHz", "105.6 MHz", "108.3 MHz", "111.7 MHz")
_TRAIN_SECTOR = ("sector Alpha", "sector Bravo", "sector Cedar", "sector Delta", "sector Ember", "sector Fjord", "sector Grove", "sector Harbor")
_DEV_SECTOR = ("sector Indigo", "sector Jasper", "sector Kilo", "sector Lagoon", "sector Meadow", "sector Nimbus", "sector Opal", "sector Prairie")

_TRAIN_AISLE = ("aisle A3", "aisle B6", "aisle C9", "aisle D12", "aisle E15", "aisle F18", "aisle G21", "aisle H24")
_DEV_AISLE = ("aisle J2", "aisle K5", "aisle L8", "aisle M11", "aisle N14", "aisle P17", "aisle Q20", "aisle R23")
_TRAIN_PAYLOAD = ("14 kilograms", "22 kilograms", "31 kilograms", "39 kilograms", "47 kilograms", "56 kilograms", "63 kilograms", "71 kilograms")
_DEV_PAYLOAD = ("17 kilograms", "26 kilograms", "34 kilograms", "43 kilograms", "52 kilograms", "59 kilograms", "67 kilograms", "74 kilograms")

_TRAIN_MEMBRANE = ("ceramic M1", "polymer P2", "carbon C3", "fiber F4", "mesh G5", "ceramic M6", "polymer P7", "carbon C8")
_DEV_MEMBRANE = ("fiber X1", "mesh Y2", "ceramic Z3", "polymer W4", "carbon V5", "fiber U6", "mesh T7", "ceramic S8")
_TRAIN_FLOW = ("18 L/min", "25 L/min", "33 L/min", "42 L/min", "51 L/min", "60 L/min", "69 L/min", "78 L/min")
_DEV_FLOW = ("21 L/min", "29 L/min", "37 L/min", "46 L/min", "55 L/min", "64 L/min", "73 L/min", "82 L/min")


def _shuffle(
    *,
    case_id: str,
    first: str,
    second: str,
    distractor_first: str,
    distractor_second: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    values = [
        ("a", first),
        ("b", second),
        ("x", distractor_first),
        ("y", distractor_second),
    ]
    random.Random(seed).shuffle(values)
    texts = tuple(text for _, text in values)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    gold_a = next(i for i, (kind, _) in enumerate(values) if kind == "a")
    gold_b = next(i for i, (kind, _) in enumerate(values) if kind == "b")
    return texts, ids, gold_a, gold_b


def _geology(index: int, split: Split) -> S3PairCase:
    minerals = _TRAIN_MINERALS if split == "train" else _DEV_MINERALS
    depths = _TRAIN_DEPTHS if split == "train" else _DEV_DEPTHS
    mineral = minerals[index % 8]
    depth = depths[(index * 3 + 1) % 8]
    other_mineral = minerals[(index + 3) % 8]
    other_depth = depths[(index * 5 + 2) % 8]
    offset = index + (13000 if split == "dev" else 0)
    cid = f"{split}-s3-geology-{index:03d}"
    if split == "train":
        state = f"Core sample GS{offset:04d} is classified as {mineral} and was recovered at depth {depth}."
        qa = "Which rock type is recorded for this core sample?"
        qb = "At what depth was this core sample recovered?"
    else:
        state = f"Survey specimen DG{offset:04d} identifies lithology {mineral}; its collection depth is {depth}."
        qa = "What lithology does the survey specimen identify?"
        qb = "What collection depth is listed for the survey specimen?"
    texts, ids, ga, gb = _shuffle(
        case_id=cid,
        first=f"the requested answer is {mineral}",
        second=f"the requested answer is {depth}",
        distractor_first=f"the requested answer is {other_mineral}",
        distractor_second=f"the requested answer is {other_depth}",
        seed=8301 + offset,
    )
    return S3PairCase(cid, split, "geology_sample", "en", state, qa, qb, texts, ids, ga, gb)


def _broadcast(index: int, split: Split) -> S3PairCase:
    freqs = _TRAIN_FREQ if split == "train" else _DEV_FREQ
    sectors = _TRAIN_SECTOR if split == "train" else _DEV_SECTOR
    freq = freqs[index % 8]
    sector = sectors[(index * 3 + 2) % 8]
    other_freq = freqs[(index + 4) % 8]
    other_sector = sectors[(index * 5 + 1) % 8]
    offset = index + (14000 if split == "dev" else 0)
    cid = f"{split}-s3-broadcast-{index:03d}"
    if split == "train":
        state = f"Broadcast node BN{offset:04d} operates at {freq} and points toward {sector}."
        qa = "What frequency does this broadcast node use?"
        qb = "Toward which antenna sector is this broadcast node pointed?"
    else:
        state = f"Transmission record DB{offset:04d} lists carrier frequency {freq}; antenna coverage is aimed at {sector}."
        qa = "Which carrier frequency appears in the transmission record?"
        qb = "Which antenna sector receives the listed coverage?"
    texts, ids, ga, gb = _shuffle(
        case_id=cid,
        first=f"the requested answer is {freq}",
        second=f"the requested answer is {sector}",
        distractor_first=f"the requested answer is {other_freq}",
        distractor_second=f"the requested answer is {other_sector}",
        seed=8401 + offset,
    )
    return S3PairCase(cid, split, "broadcast_station", "en", state, qa, qb, texts, ids, ga, gb)


def _robot(index: int, split: Split) -> S3PairCase:
    aisles = _TRAIN_AISLE if split == "train" else _DEV_AISLE
    payloads = _TRAIN_PAYLOAD if split == "train" else _DEV_PAYLOAD
    aisle = aisles[index % 8]
    payload = payloads[(index * 3 + 3) % 8]
    other_aisle = aisles[(index + 5) % 8]
    other_payload = payloads[(index * 5 + 2) % 8]
    offset = index + (15000 if split == "dev" else 0)
    cid = f"{split}-s3-robot-{index:03d}"
    if split == "train":
        state = f"Inventory robot IR{offset:04d} is assigned to {aisle} with payload limit {payload}."
        qa = "Which aisle is assigned to the inventory robot?"
        qb = "What payload limit is recorded for the inventory robot?"
    else:
        state = f"Warehouse unit DR{offset:04d} patrols {aisle}; its maximum carried load is {payload}."
        qa = "Which aisle does the warehouse unit patrol?"
        qb = "What maximum load can the warehouse unit carry?"
    texts, ids, ga, gb = _shuffle(
        case_id=cid,
        first=f"the requested answer is {aisle}",
        second=f"the requested answer is {payload}",
        distractor_first=f"the requested answer is {other_aisle}",
        distractor_second=f"the requested answer is {other_payload}",
        seed=8501 + offset,
    )
    return S3PairCase(cid, split, "robot_inventory", "en", state, qa, qb, texts, ids, ga, gb)


def _water(index: int, split: Split) -> S3PairCase:
    membranes = _TRAIN_MEMBRANE if split == "train" else _DEV_MEMBRANE
    flows = _TRAIN_FLOW if split == "train" else _DEV_FLOW
    membrane = membranes[index % 8]
    flow = flows[(index * 3 + 4) % 8]
    other_membrane = membranes[(index + 2) % 8]
    other_flow = flows[(index * 5 + 3) % 8]
    offset = index + (16000 if split == "dev" else 0)
    cid = f"{split}-s3-water-{index:03d}"
    if split == "train":
        state = f"Treatment line WT{offset:04d} uses membrane {membrane} and nominal flow {flow}."
        qa = "Which membrane is installed on this treatment line?"
        qb = "What nominal flow is specified for this treatment line?"
    else:
        state = f"Filtration ledger DW{offset:04d} records filter element {membrane}; operating flow is {flow}."
        qa = "What filter element does the filtration ledger record?"
        qb = "What operating flow is recorded in the filtration ledger?"
    texts, ids, ga, gb = _shuffle(
        case_id=cid,
        first=f"the requested answer is {membrane}",
        second=f"the requested answer is {flow}",
        distractor_first=f"the requested answer is {other_membrane}",
        distractor_second=f"the requested answer is {other_flow}",
        seed=8601 + offset,
    )
    return S3PairCase(cid, split, "water_treatment", "en", state, qa, qb, texts, ids, ga, gb)


def generate_s3_pairs(split: Split) -> tuple[S3PairCase, ...]:
    count = 64 if split == "train" else 16
    rows = []
    for builder in (_geology, _broadcast, _robot, _water):
        rows.extend(builder(i, split) for i in range(count))
    return tuple(rows)


def validate_s3_partitions(
    train: tuple[S3PairCase, ...],
    dev: tuple[S3PairCase, ...],
) -> None:
    if len(train) != 256 or len(dev) != 64:
        raise RuntimeError("Hira v1 S3 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S3-A must remain English-only")
    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S3 duplicate case IDs")
        for row in rows:
            if len(row.option_texts) != 4 or len(row.option_ids) != 4:
                raise RuntimeError("Hira v1 S3 requires K=4")
            if len(set(row.option_ids)) != 4:
                raise RuntimeError("Hira v1 S3 option IDs must be unique")
            if row.gold_a == row.gold_b:
                raise RuntimeError("Hira v1 S3 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("Hira v1 S3 gold index out of range")

    if {row.state for row in train} & {row.state for row in dev}:
        raise RuntimeError("Hira v1 S3 TRAIN/DEV state overlap")
    train_q = {q for row in train for q in (row.question_a, row.question_b)}
    dev_q = {q for row in dev for q in (row.question_a, row.question_b)}
    if train_q & dev_q:
        raise RuntimeError("Hira v1 S3 TRAIN/DEV question overlap")


__all__ = ["S3PairCase", "generate_s3_pairs", "validate_s3_partitions"]
