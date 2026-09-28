from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S1PairCase:
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


_TRAIN_ORIGINS = ("Orion", "Lumen", "Cobalt", "Juniper", "Sable", "Vega", "Cinder", "Mosaic")
_TRAIN_SEALS = ("ZX-41", "KP-73", "LM-28", "QR-66", "NV-15", "HT-92", "BC-57", "DF-84")
_DEV_ORIGINS = ("Aster", "Beryl", "Comet", "Dahlia", "Elm", "Flint", "Garnet", "Haven")
_DEV_SEALS = ("UA-34", "WB-61", "XC-79", "YD-26", "ZE-53", "AF-88", "BG-17", "CH-45")

_TRAIN_REAGENTS = ("citrate", "acetate", "borate", "glycine", "phosphate", "lactate", "malate", "succinate")
_TRAIN_TEMPS = ("18 C", "22 C", "27 C", "31 C", "35 C", "39 C", "43 C", "47 C")
_DEV_REAGENTS = ("tartrate", "formate", "oxalate", "pyruvate", "carbonate", "sulfate", "nitrate", "chloride")
_DEV_TEMPS = ("19 C", "24 C", "29 C", "33 C", "37 C", "41 C", "45 C", "49 C")

_TRAIN_VLANS = ("VLAN 112", "VLAN 224", "VLAN 336", "VLAN 448", "VLAN 560", "VLAN 672", "VLAN 784", "VLAN 896")
_TRAIN_PORTS = ("Gi1/0/4", "Gi1/0/9", "Gi1/0/14", "Gi1/0/19", "Gi1/0/24", "Gi1/0/29", "Gi1/0/34", "Gi1/0/39")
_DEV_VLANS = ("VLAN 135", "VLAN 247", "VLAN 359", "VLAN 461", "VLAN 573", "VLAN 685", "VLAN 797", "VLAN 909")
_DEV_PORTS = ("Gi2/0/3", "Gi2/0/8", "Gi2/0/13", "Gi2/0/18", "Gi2/0/23", "Gi2/0/28", "Gi2/0/33", "Gi2/0/38")

_TRAIN_ZONES = ("Amber", "Breeze", "Cedar", "Drift", "Ember", "Frost", "Grove", "Harbor")
_TRAIN_WINDOWS = ("08:10", "09:25", "10:40", "11:55", "13:10", "14:25", "15:40", "16:55")
_DEV_ZONES = ("Ivory", "Jade", "Kestrel", "Lagoon", "Meadow", "Nimbus", "Oasis", "Prairie")
_DEV_WINDOWS = ("08:20", "09:35", "10:50", "12:05", "13:20", "14:35", "15:50", "17:05")


def _shuffle_options(
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
    rng = random.Random(seed)
    rng.shuffle(values)
    option_texts = tuple(text for _, text in values)
    option_ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    gold_a = next(i for i, (kind, _) in enumerate(values) if kind == "a")
    gold_b = next(i for i, (kind, _) in enumerate(values) if kind == "b")
    return option_texts, option_ids, gold_a, gold_b


def _shipment(index: int, split: Split) -> S1PairCase:
    origins = _TRAIN_ORIGINS if split == "train" else _DEV_ORIGINS
    seals = _TRAIN_SEALS if split == "train" else _DEV_SEALS
    origin = origins[index % len(origins)]
    seal = seals[(index * 3 + 1) % len(seals)]
    other_origin = origins[(index + 3) % len(origins)]
    other_seal = seals[(index * 5 + 2) % len(seals)]
    offset = index + (5000 if split == "dev" else 0)
    case_id = f"{split}-s1-shipment-{index:03d}"
    if split == "train":
        state = f"Shipment SM{offset:04d} originated at terminal {origin} and carries seal {seal}."
        qa = "Which terminal is listed as the shipment origin?"
        qb = "What seal code is attached to the shipment?"
    else:
        state = f"Manifest DM{offset:04d} records departure terminal {origin}; the container seal reads {seal}."
        qa = "From which terminal did this manifest say the container departed?"
        qb = "Which seal identifier is recorded on this manifest?"
    texts, ids, ga, gb = _shuffle_options(
        case_id=case_id,
        first=f"the requested answer is terminal {origin}",
        second=f"the requested answer is {seal}",
        distractor_first=f"the requested answer is terminal {other_origin}",
        distractor_second=f"the requested answer is {other_seal}",
        seed=6101 + offset,
    )
    return S1PairCase(case_id, split, "shipment_manifest", "en", state, qa, qb, texts, ids, ga, gb)


def _lab(index: int, split: Split) -> S1PairCase:
    reagents = _TRAIN_REAGENTS if split == "train" else _DEV_REAGENTS
    temps = _TRAIN_TEMPS if split == "train" else _DEV_TEMPS
    reagent = reagents[index % len(reagents)]
    temp = temps[(index * 3 + 2) % len(temps)]
    other_reagent = reagents[(index + 4) % len(reagents)]
    other_temp = temps[(index * 5 + 1) % len(temps)]
    offset = index + (6000 if split == "dev" else 0)
    case_id = f"{split}-s1-lab-{index:03d}"
    if split == "train":
        state = f"Specimen SP{offset:04d} uses reagent {reagent} and incubates at {temp}."
        qa = "Which reagent is used for this specimen?"
        qb = "At what temperature does this specimen incubate?"
    else:
        state = f"Lab card DL{offset:04d} specifies {reagent} as the reagent; incubation is maintained at {temp}."
        qa = "What reagent does the lab card specify?"
        qb = "What incubation temperature is written on the lab card?"
    texts, ids, ga, gb = _shuffle_options(
        case_id=case_id,
        first=f"the requested answer is {reagent}",
        second=f"the requested answer is {temp}",
        distractor_first=f"the requested answer is {other_reagent}",
        distractor_second=f"the requested answer is {other_temp}",
        seed=6201 + offset,
    )
    return S1PairCase(case_id, split, "lab_specimen", "en", state, qa, qb, texts, ids, ga, gb)


def _network(index: int, split: Split) -> S1PairCase:
    vlans = _TRAIN_VLANS if split == "train" else _DEV_VLANS
    ports = _TRAIN_PORTS if split == "train" else _DEV_PORTS
    vlan = vlans[index % len(vlans)]
    port = ports[(index * 3 + 3) % len(ports)]
    other_vlan = vlans[(index + 5) % len(vlans)]
    other_port = ports[(index * 5 + 2) % len(ports)]
    offset = index + (7000 if split == "dev" else 0)
    case_id = f"{split}-s1-network-{index:03d}"
    if split == "train":
        state = f"Node ND{offset:04d} belongs to {vlan} and connects through switch port {port}."
        qa = "Which VLAN is assigned to the node?"
        qb = "Which switch port connects the node?"
    else:
        state = f"Topology entry DN{offset:04d} maps the node into {vlan}; its physical uplink is port {port}."
        qa = "What VLAN does the topology entry assign?"
        qb = "What physical uplink port is recorded?"
    texts, ids, ga, gb = _shuffle_options(
        case_id=case_id,
        first=f"the requested answer is {vlan}",
        second=f"the requested answer is {port}",
        distractor_first=f"the requested answer is {other_vlan}",
        distractor_second=f"the requested answer is {other_port}",
        seed=6301 + offset,
    )
    return S1PairCase(case_id, split, "network_node", "en", state, qa, qb, texts, ids, ga, gb)


def _festival(index: int, split: Split) -> S1PairCase:
    zones = _TRAIN_ZONES if split == "train" else _DEV_ZONES
    windows = _TRAIN_WINDOWS if split == "train" else _DEV_WINDOWS
    zone = zones[index % len(zones)]
    window = windows[(index * 3 + 4) % len(windows)]
    other_zone = zones[(index + 2) % len(zones)]
    other_window = windows[(index * 5 + 3) % len(windows)]
    offset = index + (8000 if split == "dev" else 0)
    case_id = f"{split}-s1-festival-{index:03d}"
    if split == "train":
        state = f"Credential CR{offset:04d} assigns entry zone {zone} and check-in time {window}."
        qa = "Which entry zone is assigned to the credential?"
        qb = "What check-in time is assigned to the credential?"
    else:
        state = f"Access record DF{offset:04d} directs the holder to zone {zone}; the allowed check-in begins at {window}."
        qa = "To which zone does this access record direct the holder?"
        qb = "When does the allowed check-in begin?"
    texts, ids, ga, gb = _shuffle_options(
        case_id=case_id,
        first=f"the requested answer is zone {zone}",
        second=f"the requested answer is {window}",
        distractor_first=f"the requested answer is zone {other_zone}",
        distractor_second=f"the requested answer is {other_window}",
        seed=6401 + offset,
    )
    return S1PairCase(case_id, split, "festival_access", "en", state, qa, qb, texts, ids, ga, gb)


def generate_s1_pairs(split: Split) -> tuple[S1PairCase, ...]:
    per_domain = 64 if split == "train" else 16
    builders = (_shipment, _lab, _network, _festival)
    rows = []
    for builder in builders:
        rows.extend(builder(i, split) for i in range(per_domain))
    return tuple(rows)


def validate_s1_partitions(
    train: tuple[S1PairCase, ...],
    dev: tuple[S1PairCase, ...],
) -> None:
    if len(train) != 256 or len(dev) != 64:
        raise RuntimeError("Hira v1 S1 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S1-A must remain English-only")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S1 duplicate case IDs")
        for row in rows:
            if len(row.option_texts) != 4 or len(row.option_ids) != 4:
                raise RuntimeError("Hira v1 S1 requires K=4")
            if len(set(row.option_ids)) != 4:
                raise RuntimeError("Hira v1 S1 option IDs must be unique")
            if row.gold_a == row.gold_b:
                raise RuntimeError("Hira v1 S1 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("Hira v1 S1 gold index out of range")

    train_states = {row.state for row in train}
    dev_states = {row.state for row in dev}
    if train_states & dev_states:
        raise RuntimeError("Hira v1 S1 TRAIN/DEV state overlap")

    train_questions = {q for row in train for q in (row.question_a, row.question_b)}
    dev_questions = {q for row in dev for q in (row.question_a, row.question_b)}
    if train_questions & dev_questions:
        raise RuntimeError("Hira v1 S1 TRAIN/DEV question overlap")


__all__ = ["S1PairCase", "generate_s1_pairs", "validate_s1_partitions"]
