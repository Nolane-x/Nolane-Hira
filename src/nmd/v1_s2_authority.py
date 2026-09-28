from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S2PairCase:
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


_TRAIN_SENSOR = ("ion", "optical", "thermal", "acoustic", "magnetic", "pressure", "humidity", "vibration")
_DEV_SENSOR = ("infrared", "ultrasonic", "capacitive", "inductive", "chemical", "strain", "flow", "radiation")
_TRAIN_CAL = ("2026-01-14", "2026-02-19", "2026-03-23", "2026-04-11", "2026-05-27", "2026-06-08", "2026-07-16", "2026-08-29")
_DEV_CAL = ("2026-09-03", "2026-09-17", "2026-10-06", "2026-10-21", "2026-11-09", "2026-11-24", "2026-12-05", "2026-12-18")

_TRAIN_INGREDIENT = ("cardamom", "sesame", "hazelnut", "coconut", "ginger", "cinnamon", "almond", "pistachio")
_DEV_INGREDIENT = ("saffron", "molasses", "anise", "pecan", "cranberry", "vanilla", "walnut", "chestnut")
_TRAIN_DURATION = ("18 minutes", "24 minutes", "31 minutes", "37 minutes", "43 minutes", "49 minutes", "56 minutes", "62 minutes")
_DEV_DURATION = ("21 minutes", "27 minutes", "34 minutes", "40 minutes", "46 minutes", "53 minutes", "59 minutes", "66 minutes")

_TRAIN_RUNWAY = ("runway 04", "runway 09", "runway 13", "runway 18", "runway 22", "runway 27", "runway 31", "runway 36")
_DEV_RUNWAY = ("runway 05", "runway 10", "runway 14", "runway 19", "runway 23", "runway 28", "runway 32", "runway 35")
_TRAIN_SQUAWK = ("2147", "3261", "4385", "5502", "6714", "7830", "1456", "8923")
_DEV_SQUAWK = ("2053", "3179", "4296", "5418", "6532", "7641", "8765", "1904")

_TRAIN_INSTITUTION = ("Arbor Institute", "Beacon Archive", "Crescent Gallery", "Drift Museum", "Ember Foundation", "Fjord Center", "Grove Trust", "Helix Collection")
_DEV_INSTITUTION = ("Iris Institute", "Jasper Archive", "Kepler Gallery", "Lagoon Museum", "Mosaic Foundation", "Nimbus Center", "Opal Trust", "Prairie Collection")
_TRAIN_CLASS = ("class A1", "class B2", "class C3", "class D4", "class E5", "class F6", "class G7", "class H8")
_DEV_CLASS = ("class J1", "class K2", "class L3", "class M4", "class N5", "class P6", "class Q7", "class R8")


def _shuffle(
    *,
    case_id: str,
    first: str,
    second: str,
    distractor_first: str,
    distractor_second: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    rows = [("a", first), ("b", second), ("x", distractor_first), ("y", distractor_second)]
    random.Random(seed).shuffle(rows)
    texts = tuple(text for _, text in rows)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _) in enumerate(rows) if kind == "b")
    return texts, ids, ga, gb


def _sensor(index: int, split: Split) -> S2PairCase:
    kinds = _TRAIN_SENSOR if split == "train" else _DEV_SENSOR
    dates = _TRAIN_CAL if split == "train" else _DEV_CAL
    kind = kinds[index % 8]
    date = dates[(index * 3 + 1) % 8]
    other_kind = kinds[(index + 3) % 8]
    other_date = dates[(index * 5 + 2) % 8]
    offset = index + (9000 if split == "dev" else 0)
    cid = f"{split}-s2-sensor-{index:03d}"
    if split == "train":
        state = f"Registry SR{offset:04d} identifies sensor family {kind} and calibration date {date}."
        qa = "Which sensor family is identified in this registry entry?"
        qb = "What calibration date is recorded for this sensor?"
    else:
        state = f"Device ledger DS{offset:04d} categorizes the unit as {kind}; its most recent calibration is dated {date}."
        qa = "How does the device ledger categorize this unit?"
        qb = "When was the unit most recently calibrated?"
    texts, ids, ga, gb = _shuffle(
        case_id=cid,
        first=f"the requested answer is {kind}",
        second=f"the requested answer is {date}",
        distractor_first=f"the requested answer is {other_kind}",
        distractor_second=f"the requested answer is {other_date}",
        seed=7201 + offset,
    )
    return S2PairCase(cid, split, "sensor_registry", "en", state, qa, qb, texts, ids, ga, gb)


def _recipe(index: int, split: Split) -> S2PairCase:
    ingredients = _TRAIN_INGREDIENT if split == "train" else _DEV_INGREDIENT
    durations = _TRAIN_DURATION if split == "train" else _DEV_DURATION
    ingredient = ingredients[index % 8]
    duration = durations[(index * 3 + 2) % 8]
    other_ingredient = ingredients[(index + 4) % 8]
    other_duration = durations[(index * 5 + 1) % 8]
    offset = index + (10000 if split == "dev" else 0)
    cid = f"{split}-s2-recipe-{index:03d}"
    if split == "train":
        state = f"Recipe RC{offset:04d} features {ingredient} and specifies a baking duration of {duration}."
        qa = "Which featured ingredient is listed in the recipe?"
        qb = "What baking duration does the recipe specify?"
    else:
        state = f"Kitchen card DK{offset:04d} names {ingredient} as its signature ingredient; the bake lasts {duration}."
        qa = "What signature ingredient does this kitchen card name?"
        qb = "How long does the bake last according to the kitchen card?"
    texts, ids, ga, gb = _shuffle(
        case_id=cid,
        first=f"the requested answer is {ingredient}",
        second=f"the requested answer is {duration}",
        distractor_first=f"the requested answer is {other_ingredient}",
        distractor_second=f"the requested answer is {other_duration}",
        seed=7301 + offset,
    )
    return S2PairCase(cid, split, "recipe_card", "en", state, qa, qb, texts, ids, ga, gb)


def _flight(index: int, split: Split) -> S2PairCase:
    runways = _TRAIN_RUNWAY if split == "train" else _DEV_RUNWAY
    squawks = _TRAIN_SQUAWK if split == "train" else _DEV_SQUAWK
    runway = runways[index % 8]
    squawk = squawks[(index * 3 + 3) % 8]
    other_runway = runways[(index + 5) % 8]
    other_squawk = squawks[(index * 5 + 2) % 8]
    offset = index + (11000 if split == "dev" else 0)
    cid = f"{split}-s2-flight-{index:03d}"
    if split == "train":
        state = f"Clearance CL{offset:04d} assigns {runway} and transponder code {squawk}."
        qa = "Which runway is assigned by this clearance?"
        qb = "What transponder code is assigned?"
    else:
        state = f"Departure authorization DF{offset:04d} directs the aircraft to {runway}; the squawk code is {squawk}."
        qa = "Which runway does the departure authorization direct the aircraft to?"
        qb = "Which squawk code is stated in the authorization?"
    texts, ids, ga, gb = _shuffle(
        case_id=cid,
        first=f"the requested answer is {runway}",
        second=f"the requested answer is {squawk}",
        distractor_first=f"the requested answer is {other_runway}",
        distractor_second=f"the requested answer is {other_squawk}",
        seed=7401 + offset,
    )
    return S2PairCase(cid, split, "flight_clearance", "en", state, qa, qb, texts, ids, ga, gb)


def _loan(index: int, split: Split) -> S2PairCase:
    institutions = _TRAIN_INSTITUTION if split == "train" else _DEV_INSTITUTION
    classes = _TRAIN_CLASS if split == "train" else _DEV_CLASS
    institution = institutions[index % 8]
    insurance = classes[(index * 3 + 4) % 8]
    other_institution = institutions[(index + 2) % 8]
    other_class = classes[(index * 5 + 3) % 8]
    offset = index + (12000 if split == "dev" else 0)
    cid = f"{split}-s2-loan-{index:03d}"
    if split == "train":
        state = f"Loan file LF{offset:04d} lists origin institution {institution} and insurance category {insurance}."
        qa = "Which institution is listed as the origin of this loan?"
        qb = "What insurance category is recorded for the loan?"
    else:
        state = f"Exhibition transfer DE{offset:04d} arrived from {institution}; its coverage classification is {insurance}."
        qa = "Where did this exhibition transfer arrive from?"
        qb = "What coverage classification is assigned to the transfer?"
    texts, ids, ga, gb = _shuffle(
        case_id=cid,
        first=f"the requested answer is {institution}",
        second=f"the requested answer is {insurance}",
        distractor_first=f"the requested answer is {other_institution}",
        distractor_second=f"the requested answer is {other_class}",
        seed=7501 + offset,
    )
    return S2PairCase(cid, split, "museum_loan", "en", state, qa, qb, texts, ids, ga, gb)


def generate_s2_pairs(split: Split) -> tuple[S2PairCase, ...]:
    count = 64 if split == "train" else 16
    rows = []
    for builder in (_sensor, _recipe, _flight, _loan):
        rows.extend(builder(i, split) for i in range(count))
    return tuple(rows)


def validate_s2_partitions(
    train: tuple[S2PairCase, ...],
    dev: tuple[S2PairCase, ...],
) -> None:
    if len(train) != 256 or len(dev) != 64:
        raise RuntimeError("Hira v1 S2 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S2-A must remain English-only")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S2 duplicate case IDs")
        for row in rows:
            if len(row.option_texts) != 4 or len(row.option_ids) != 4:
                raise RuntimeError("Hira v1 S2 requires K=4")
            if len(set(row.option_ids)) != 4:
                raise RuntimeError("Hira v1 S2 option IDs must be unique")
            if row.gold_a == row.gold_b:
                raise RuntimeError("Hira v1 S2 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("Hira v1 S2 gold index out of range")

    if {row.state for row in train} & {row.state for row in dev}:
        raise RuntimeError("Hira v1 S2 TRAIN/DEV state overlap")
    train_q = {q for row in train for q in (row.question_a, row.question_b)}
    dev_q = {q for row in dev for q in (row.question_a, row.question_b)}
    if train_q & dev_q:
        raise RuntimeError("Hira v1 S2 TRAIN/DEV question overlap")


__all__ = ["S2PairCase", "generate_s2_pairs", "validate_s2_partitions"]
