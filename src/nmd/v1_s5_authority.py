from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Callable, Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S5PairCase:
    case_id: str
    split: Split
    domain: str
    language: str
    state: str
    question_a: str
    question_b: str
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


def _shuffle(
    *,
    case_id: str,
    first: str,
    second: str,
    distractor_first: str,
    distractor_second: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int, int]:
    rows = [
        ("a", first),
        ("b", second),
        ("x", distractor_first),
        ("y", distractor_second),
    ]
    random.Random(seed).shuffle(rows)
    texts = tuple(f"the requested answer is {value}" for _, value in rows)
    aliases = tuple(f"{value} is the requested value" for _, value in rows)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


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
    qa_train: str,
    qb_train: str,
    qa_dev: str,
    qb_dev: str,
    seed_base: int,
    dev_offset: int,
) -> S5PairCase:
    first = first_values[index % len(first_values)]
    second = second_values[(index * 3 + 1) % len(second_values)]
    other_first = first_values[(index + 3) % len(first_values)]
    other_second = second_values[(index * 5 + 2) % len(second_values)]
    offset = index + (dev_offset if split == "dev" else 0)
    case_id = f"{split}-s5-{domain}-{index:03d}"

    template = state_train if split == "train" else state_dev
    state = template.format(
        code=f"{prefix}{offset:04d}",
        first=first,
        second=second,
    )
    qa = qa_train if split == "train" else qa_dev
    qb = qb_train if split == "train" else qb_dev

    texts, aliases, ids, ga, gb = _shuffle(
        case_id=case_id,
        first=first,
        second=second,
        distractor_first=other_first,
        distractor_second=other_second,
        seed=seed_base + offset,
    )
    return S5PairCase(
        case_id,
        split,
        domain,
        "en",
        state,
        qa,
        qb,
        texts,
        aliases,
        ids,
        ga,
        gb,
    )


def _battery(index: int, split: Split) -> S5PairCase:
    train_a = ("LFP", "NMC-811", "NCA", "LMO", "LCO", "sodium-ion", "zinc-air", "solid-state")
    dev_a = ("LMFP", "NMC-622", "LTO", "sulfur-ion", "iron-air", "magnesium-ion", "aluminum-ion", "silicon-anode")
    train_b = ("42 kWh", "58 kWh", "64 kWh", "76 kWh", "88 kWh", "96 kWh", "110 kWh", "124 kWh")
    dev_b = ("47 kWh", "61 kWh", "69 kWh", "81 kWh", "91 kWh", "103 kWh", "117 kWh", "131 kWh")
    return _build_case(
        index=index, split=split, domain="battery_pack", prefix="BP",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Battery record {code} lists cell chemistry {first} and usable capacity {second}.",
        state_dev="Pack dossier {code} identifies electrochemistry {first}; available energy is {second}.",
        qa_train="Which cell chemistry is listed in this battery record?",
        qb_train="What usable capacity is listed in this battery record?",
        qa_dev="What electrochemistry does this pack dossier identify?",
        qb_dev="How much available energy does this pack dossier report?",
        seed_base=10501, dev_offset=19000,
    )


def _weather(index: int, split: Split) -> S5PairCase:
    train_a = ("ridge North", "ridge East", "basin Cedar", "basin Flint", "plateau Amber", "plateau Quartz", "valley Opal", "valley Birch")
    dev_a = ("ridge South", "ridge West", "basin Hazel", "basin Onyx", "plateau Cobalt", "plateau Ivory", "valley Maple", "valley Pearl")
    train_b = ("812 hPa", "834 hPa", "856 hPa", "878 hPa", "900 hPa", "922 hPa", "944 hPa", "966 hPa")
    dev_b = ("821 hPa", "843 hPa", "865 hPa", "887 hPa", "909 hPa", "931 hPa", "953 hPa", "975 hPa")
    return _build_case(
        index=index, split=split, domain="weather_station", prefix="WS",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Weather station {code} is located at {first} and reports pressure {second}.",
        state_dev="Meteorological ledger {code} places the sensor at {first}; barometric reading is {second}.",
        qa_train="Where is this weather station located?",
        qb_train="What pressure does this weather station report?",
        qa_dev="Where does the meteorological ledger place the sensor?",
        qb_dev="What barometric reading is recorded in the ledger?",
        seed_base=10601, dev_offset=20000,
    )


def _textile(index: int, split: Split) -> S5PairCase:
    train_a = ("merino", "linen", "bamboo", "modal", "hemp", "cashmere", "alpaca", "ramie")
    dev_a = ("tencel", "mohair", "angora", "jute", "viscose", "cupro", "qiviut", "seacell")
    train_b = ("weave 18", "weave 24", "weave 30", "weave 36", "weave 42", "weave 48", "weave 54", "weave 60")
    dev_b = ("weave 21", "weave 27", "weave 33", "weave 39", "weave 45", "weave 51", "weave 57", "weave 63")
    return _build_case(
        index=index, split=split, domain="textile_batch", prefix="TB",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Textile batch {code} uses fiber {first} and construction code {second}.",
        state_dev="Fabric ledger {code} names material {first}; its structure designation is {second}.",
        qa_train="Which fiber is used in this textile batch?",
        qb_train="What construction code is recorded for this textile batch?",
        qa_dev="What material does this fabric ledger name?",
        qb_dev="What structure designation does this fabric ledger record?",
        seed_base=10701, dev_offset=21000,
    )


def _drone(index: int, split: Split) -> S5PairCase:
    train_a = ("corridor Aster", "corridor Beacon", "corridor Cinder", "corridor Drift", "corridor Ember", "corridor Fjord", "corridor Grove", "corridor Helix")
    dev_a = ("corridor Iris", "corridor Jasper", "corridor Kepler", "corridor Lagoon", "corridor Mosaic", "corridor Nimbus", "corridor Opal", "corridor Prairie")
    train_b = ("payload 3 kg", "payload 5 kg", "payload 7 kg", "payload 9 kg", "payload 11 kg", "payload 13 kg", "payload 15 kg", "payload 17 kg")
    dev_b = ("payload 4 kg", "payload 6 kg", "payload 8 kg", "payload 10 kg", "payload 12 kg", "payload 14 kg", "payload 16 kg", "payload 18 kg")
    return _build_case(
        index=index, split=split, domain="cargo_drone", prefix="CD",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Cargo drone mission {code} uses flight {first} and carries {second}.",
        state_dev="Aerial delivery sheet {code} assigns route {first}; certified load is {second}.",
        qa_train="Which flight corridor is used by this cargo drone mission?",
        qb_train="What payload does this cargo drone mission carry?",
        qa_dev="What route is assigned by this aerial delivery sheet?",
        qb_dev="What certified load is listed on this aerial delivery sheet?",
        seed_base=10801, dev_offset=22000,
    )


def _aquaculture(index: int, split: Split) -> S5PairCase:
    train_a = ("spirulina", "krill", "soy meal", "algae blend", "fishmeal", "pea protein", "insect meal", "wheat germ")
    dev_a = ("duckweed", "copepod", "lupin meal", "seaweed blend", "yeast protein", "rice bran", "mussel meal", "sunflower meal")
    train_b = ("2.1 percent", "2.5 percent", "2.9 percent", "3.3 percent", "3.7 percent", "4.1 percent", "4.5 percent", "4.9 percent")
    dev_b = ("2.3 percent", "2.7 percent", "3.1 percent", "3.5 percent", "3.9 percent", "4.3 percent", "4.7 percent", "5.1 percent")
    return _build_case(
        index=index, split=split, domain="aquaculture_feed", prefix="AF",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Aquaculture feed lot {code} uses protein source {first} and ration rate {second}.",
        state_dev="Hatchery nutrition card {code} specifies ingredient {first}; daily allocation is {second}.",
        qa_train="Which protein source is used in this feed lot?",
        qb_train="What ration rate is recorded for this feed lot?",
        qa_dev="What ingredient does this hatchery nutrition card specify?",
        qb_dev="What daily allocation is recorded on this nutrition card?",
        seed_base=10901, dev_offset=23000,
    )


def _telescope(index: int, split: Split) -> S5PairCase:
    train_a = ("M31", "M42", "M51", "M81", "NGC 253", "NGC 1300", "NGC 4565", "NGC 7331")
    dev_a = ("M33", "M63", "M82", "M101", "NGC 891", "NGC 1365", "NGC 2903", "NGC 6946")
    train_b = ("slot 01:20", "slot 02:10", "slot 03:00", "slot 03:50", "slot 04:40", "slot 05:30", "slot 06:20", "slot 07:10")
    dev_b = ("slot 01:35", "slot 02:25", "slot 03:15", "slot 04:05", "slot 04:55", "slot 05:45", "slot 06:35", "slot 07:25")
    return _build_case(
        index=index, split=split, domain="telescope_schedule", prefix="TS",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Telescope schedule {code} targets object {first} and reserves {second}.",
        state_dev="Observing queue {code} names source {first}; allocated window is {second}.",
        qa_train="Which object is targeted in this telescope schedule?",
        qb_train="Which time slot is reserved in this telescope schedule?",
        qa_dev="What source does this observing queue name?",
        qb_dev="What allocated window is listed in this observing queue?",
        seed_base=11001, dev_offset=24000,
    )


def _ceramic(index: int, split: Split) -> S5PairCase:
    train_a = ("porcelain", "stoneware", "earthenware", "raku", "bone china", "terracotta", "celadon", "faience")
    dev_a = ("majolica", "gres", "biscuit porcelain", "paper clay", "agateware", "jasperware", "blackware", "parian")
    train_b = ("cone 4", "cone 5", "cone 6", "cone 7", "cone 8", "cone 9", "cone 10", "cone 11")
    dev_b = ("cone 3", "cone 4.5", "cone 5.5", "cone 6.5", "cone 7.5", "cone 8.5", "cone 9.5", "cone 10.5")
    return _build_case(
        index=index, split=split, domain="ceramic_kiln", prefix="CK",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Ceramic kiln card {code} identifies body {first} and firing target {second}.",
        state_dev="Firing dossier {code} lists ceramic type {first}; maturation setting is {second}.",
        qa_train="Which ceramic body is identified on this kiln card?",
        qb_train="What firing target is identified on this kiln card?",
        qa_dev="What ceramic type is listed in this firing dossier?",
        qb_dev="What maturation setting is listed in this firing dossier?",
        seed_base=11101, dev_offset=25000,
    )


def _datacenter(index: int, split: Split) -> S5PairCase:
    train_a = ("pod Alder", "pod Birch", "pod Cedar", "pod Elm", "pod Fir", "pod Grove", "pod Hazel", "pod Juniper")
    dev_a = ("pod Maple", "pod Oak", "pod Pine", "pod Rowan", "pod Spruce", "pod Walnut", "pod Willow", "pod Yew")
    train_b = ("tier Gold", "tier Silver", "tier Bronze", "tier Platinum", "tier Quartz", "tier Ruby", "tier Sapphire", "tier Emerald")
    dev_b = ("tier Onyx", "tier Pearl", "tier Amber", "tier Jade", "tier Topaz", "tier Opal", "tier Garnet", "tier Cobalt")
    return _build_case(
        index=index, split=split, domain="data_center", prefix="DC",
        first_values=train_a if split == "train" else dev_a,
        second_values=train_b if split == "train" else dev_b,
        state_train="Data-center allocation {code} assigns compute {first} and resilience {second}.",
        state_dev="Infrastructure ledger {code} places the workload in {first}; redundancy class is {second}.",
        qa_train="Which compute pod is assigned in this data-center allocation?",
        qb_train="Which resilience tier is assigned in this data-center allocation?",
        qa_dev="Where does the infrastructure ledger place the workload?",
        qb_dev="What redundancy class does the infrastructure ledger specify?",
        seed_base=11201, dev_offset=26000,
    )


def generate_s5_pairs(split: Split) -> tuple[S5PairCase, ...]:
    count = 64 if split == "train" else 16
    rows = []
    for builder in (
        _battery,
        _weather,
        _textile,
        _drone,
        _aquaculture,
        _telescope,
        _ceramic,
        _datacenter,
    ):
        rows.extend(builder(i, split) for i in range(count))
    return tuple(rows)


def validate_s5_partitions(
    train: tuple[S5PairCase, ...],
    dev: tuple[S5PairCase, ...],
) -> None:
    if len(train) != 512 or len(dev) != 128:
        raise RuntimeError("Hira v1 S5 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S5-A must remain English-only")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S5 duplicate case IDs")
        for row in rows:
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("Hira v1 S5 requires four options with aliases")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("Hira v1 S5 option IDs must be unique K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("Hira v1 S5 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("Hira v1 S5 gold index out of range")

    train_states = {row.state for row in train}
    dev_states = {row.state for row in dev}
    if train_states & dev_states:
        raise RuntimeError("Hira v1 S5 TRAIN/DEV state overlap")

    train_q = {q for row in train for q in (row.question_a, row.question_b)}
    dev_q = {q for row in dev for q in (row.question_a, row.question_b)}
    if train_q & dev_q:
        raise RuntimeError("Hira v1 S5 TRAIN/DEV question overlap")


__all__ = ["S5PairCase", "generate_s5_pairs", "validate_s5_partitions"]
