from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S6PairCase:
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


@dataclass(frozen=True)
class _Domain:
    name: str
    prefix: str
    train_first: tuple[str, ...]
    dev_first: tuple[str, ...]
    train_second: tuple[str, ...]
    dev_second: tuple[str, ...]
    state_train: str
    state_dev: str
    qa_train: str
    qb_train: str
    qa_dev: str
    qb_dev: str
    seed: int
    dev_offset: int


_DOMAINS = (
    _Domain(
        "centrifuge_run", "CF",
        ("fixed-angle", "swing-bucket", "vertical", "zonal", "microplate", "hematocrit", "continuous-flow", "ultracentrifuge"),
        ("elutriation", "high-capacity", "low-speed", "refrigerated", "cytology", "clinical", "industrial", "analytical"),
        ("4200 rpm", "5600 rpm", "7100 rpm", "8600 rpm", "10200 rpm", "11800 rpm", "13400 rpm", "15000 rpm"),
        ("4700 rpm", "6200 rpm", "7700 rpm", "9200 rpm", "10700 rpm", "12300 rpm", "13900 rpm", "15500 rpm"),
        "Centrifuge run {code} uses rotor {first} at speed {second}.",
        "Spin ledger {code} records rotor class {first}; operating speed is {second}.",
        "Which rotor is used in this centrifuge run?",
        "What speed is used in this centrifuge run?",
        "What rotor class does this spin ledger record?",
        "What operating speed is recorded in this spin ledger?",
        11601, 30000,
    ),
    _Domain(
        "vineyard_lot", "VL",
        ("Syrah", "Riesling", "Tempranillo", "Viognier", "Malbec", "Grenache", "Nebbiolo", "Sangiovese"),
        ("Mourvedre", "Semillon", "Carmenere", "Barbera", "Pinotage", "Vermentino", "Touriga", "Gamay"),
        ("19 Brix", "20 Brix", "21 Brix", "22 Brix", "23 Brix", "24 Brix", "25 Brix", "26 Brix"),
        ("18.5 Brix", "19.5 Brix", "20.5 Brix", "21.5 Brix", "22.5 Brix", "23.5 Brix", "24.5 Brix", "25.5 Brix"),
        "Vineyard lot {code} lists grape variety {first} and sugar reading {second}.",
        "Harvest dossier {code} identifies cultivar {first}; must reading is {second}.",
        "Which grape variety is listed for this vineyard lot?",
        "What sugar reading is listed for this vineyard lot?",
        "Which cultivar does this harvest dossier identify?",
        "What must reading is reported in this harvest dossier?",
        11701, 31000,
    ),
    _Domain(
        "marine_engine", "ME",
        ("diesel", "methanol", "LNG", "biodiesel", "ammonia", "ethanol", "hydrogen", "dual-fuel"),
        ("synthetic diesel", "biomethane", "e-methanol", "green ammonia", "HVO", "DME", "fuel-cell hydrogen", "hybrid electric"),
        ("720 rpm", "810 rpm", "900 rpm", "990 rpm", "1080 rpm", "1170 rpm", "1260 rpm", "1350 rpm"),
        ("765 rpm", "855 rpm", "945 rpm", "1035 rpm", "1125 rpm", "1215 rpm", "1305 rpm", "1395 rpm"),
        "Marine engine log {code} specifies fuel {first} and cruise speed {second}.",
        "Propulsion sheet {code} names energy source {first}; shaft rate is {second}.",
        "Which fuel is specified in this marine engine log?",
        "What cruise speed is specified in this marine engine log?",
        "What energy source does this propulsion sheet name?",
        "What shaft rate is listed in this propulsion sheet?",
        11801, 32000,
    ),
    _Domain(
        "weather_balloon", "WB",
        ("ozonesonde", "radiosonde", "hygrometer", "thermistor", "barometer", "GPS package", "aerosol counter", "radiometer"),
        ("dewpoint sensor", "UV photometer", "cloud probe", "wind package", "pressure transducer", "gas sampler", "particle counter", "infrared radiometer"),
        ("18 km", "21 km", "24 km", "27 km", "30 km", "33 km", "36 km", "39 km"),
        ("19.5 km", "22.5 km", "25.5 km", "28.5 km", "31.5 km", "34.5 km", "37.5 km", "40.5 km"),
        "Weather balloon mission {code} carries payload {first} and bursts at {second}.",
        "Atmospheric flight card {code} lists instrument {first}; burst altitude is {second}.",
        "Which payload is carried by this weather balloon mission?",
        "At what altitude does this weather balloon burst?",
        "What instrument is listed on this atmospheric flight card?",
        "What burst altitude is listed on this atmospheric flight card?",
        11901, 33000,
    ),
    _Domain(
        "digitization_job", "DG",
        ("grayscale", "bitonal", "true-color", "multispectral", "infrared", "ultraviolet", "HDR", "reflectance"),
        ("polarized", "fluorescence", "transmitted-light", "raking-light", "thermal", "x-ray", "hyperspectral", "photogrammetric"),
        ("300 dpi", "400 dpi", "500 dpi", "600 dpi", "700 dpi", "800 dpi", "900 dpi", "1000 dpi"),
        ("350 dpi", "450 dpi", "550 dpi", "650 dpi", "750 dpi", "850 dpi", "950 dpi", "1050 dpi"),
        "Digitization job {code} uses scan mode {first} at resolution {second}.",
        "Imaging register {code} records capture method {first}; sampling resolution is {second}.",
        "Which scan mode is used for this digitization job?",
        "What resolution is used for this digitization job?",
        "What capture method does this imaging register record?",
        "What sampling resolution does this imaging register specify?",
        12001, 34000,
    ),
    _Domain(
        "dairy_culture", "DCU",
        ("Lactobacillus", "Streptococcus", "Bifidobacterium", "Lactococcus", "Leuconostoc", "Pediococcus", "Propionibacterium", "Geotrichum"),
        ("Enterococcus", "Brevibacterium", "Debaryomyces", "Kluyveromyces", "Penicillium", "Rhizopus", "Saccharomyces", "Weissella"),
        ("6 hours", "8 hours", "10 hours", "12 hours", "14 hours", "16 hours", "18 hours", "20 hours"),
        ("7 hours", "9 hours", "11 hours", "13 hours", "15 hours", "17 hours", "19 hours", "21 hours"),
        "Dairy culture batch {code} uses starter {first} and incubation time {second}.",
        "Fermentation register {code} names culture {first}; maturation duration is {second}.",
        "Which starter is used in this dairy culture batch?",
        "What incubation time is used in this dairy culture batch?",
        "What culture does this fermentation register name?",
        "What maturation duration is listed in this fermentation register?",
        12101, 35000,
    ),
    _Domain(
        "solar_array", "SA",
        ("monocrystalline", "polycrystalline", "thin-film", "PERC", "TOPCon", "HJT", "perovskite", "CIGS"),
        ("IBC", "CdTe", "tandem perovskite", "organic PV", "amorphous silicon", "bifacial mono", "shingled-cell", "multi-junction"),
        ("tilt 12 deg", "tilt 18 deg", "tilt 24 deg", "tilt 30 deg", "tilt 36 deg", "tilt 42 deg", "tilt 48 deg", "tilt 54 deg"),
        ("tilt 15 deg", "tilt 21 deg", "tilt 27 deg", "tilt 33 deg", "tilt 39 deg", "tilt 45 deg", "tilt 51 deg", "tilt 57 deg"),
        "Solar array record {code} lists module type {first} and mounting {second}.",
        "Photovoltaic ledger {code} identifies cell technology {first}; rack setting is {second}.",
        "Which module type is listed in this solar array record?",
        "What mounting tilt is listed in this solar array record?",
        "What cell technology does this photovoltaic ledger identify?",
        "What rack setting is recorded in this photovoltaic ledger?",
        12201, 36000,
    ),
    _Domain(
        "baggage_system", "BG",
        ("belt Alpha", "belt Bravo", "belt Charlie", "belt Delta", "belt Echo", "belt Foxtrot", "belt Golf", "belt Hotel"),
        ("belt India", "belt Juliet", "belt Kilo", "belt Lima", "belt Mike", "belt November", "belt Oscar", "belt Papa"),
        ("limit 18 kg", "limit 20 kg", "limit 22 kg", "limit 24 kg", "limit 26 kg", "limit 28 kg", "limit 30 kg", "limit 32 kg"),
        ("limit 19 kg", "limit 21 kg", "limit 23 kg", "limit 25 kg", "limit 27 kg", "limit 29 kg", "limit 31 kg", "limit 33 kg"),
        "Baggage system ticket {code} assigns {first} with weight {second}.",
        "Luggage routing card {code} sends items to {first}; accepted mass is {second}.",
        "Which belt is assigned by this baggage system ticket?",
        "What weight limit is assigned by this baggage system ticket?",
        "Where does this luggage routing card send the items?",
        "What accepted mass is listed on this luggage routing card?",
        12301, 37000,
    ),
    _Domain(
        "forest_plot", "FP",
        ("Douglas fir", "white pine", "red cedar", "silver birch", "black spruce", "sugar maple", "red oak", "aspen"),
        ("hemlock", "lodgepole pine", "yellow cedar", "paper birch", "Sitka spruce", "bigleaf maple", "white oak", "cottonwood"),
        ("plot A17", "plot B23", "plot C31", "plot D46", "plot E52", "plot F68", "plot G74", "plot H89"),
        ("plot J14", "plot K27", "plot L35", "plot M41", "plot N59", "plot P63", "plot Q78", "plot R86"),
        "Forest inventory {code} records species {first} in {second}.",
        "Silviculture ledger {code} names tree type {first}; sampling location is {second}.",
        "Which species is recorded in this forest inventory?",
        "Which plot is recorded in this forest inventory?",
        "What tree type does this silviculture ledger name?",
        "What sampling location is listed in this silviculture ledger?",
        12401, 38000,
    ),
    _Domain(
        "imaging_series", "IM",
        ("T1-weighted", "T2-weighted", "FLAIR", "DWI", "SWI", "TOF", "STIR", "PD-weighted"),
        ("ASL", "DTI", "MRA", "MRCP", "fMRI", "UTE", "DCE", "CINE"),
        ("slice 1.0 mm", "slice 1.5 mm", "slice 2.0 mm", "slice 2.5 mm", "slice 3.0 mm", "slice 3.5 mm", "slice 4.0 mm", "slice 4.5 mm"),
        ("slice 1.2 mm", "slice 1.7 mm", "slice 2.2 mm", "slice 2.7 mm", "slice 3.2 mm", "slice 3.7 mm", "slice 4.2 mm", "slice 4.7 mm"),
        "Imaging series {code} uses sequence {first} and thickness {second}.",
        "Radiology protocol {code} specifies acquisition {first}; section thickness is {second}.",
        "Which sequence is used in this imaging series?",
        "What slice thickness is used in this imaging series?",
        "What acquisition does this radiology protocol specify?",
        "What section thickness is recorded in this radiology protocol?",
        12501, 39000,
    ),
    _Domain(
        "semiconductor_step", "SC",
        ("argon", "nitrogen", "silane", "ammonia", "oxygen", "hydrogen", "chlorine", "helium"),
        ("neon", "xenon", "phosphine", "diborane", "nitrous oxide", "fluorine", "hydrogen bromide", "tetrafluoromethane"),
        ("2.4 Torr", "3.1 Torr", "3.8 Torr", "4.5 Torr", "5.2 Torr", "5.9 Torr", "6.6 Torr", "7.3 Torr"),
        ("2.7 Torr", "3.4 Torr", "4.1 Torr", "4.8 Torr", "5.5 Torr", "6.2 Torr", "6.9 Torr", "7.6 Torr"),
        "Semiconductor step {code} uses process gas {first} at chamber pressure {second}.",
        "Fab recipe {code} identifies reactant {first}; process pressure is {second}.",
        "Which process gas is used in this semiconductor step?",
        "What chamber pressure is used in this semiconductor step?",
        "What reactant does this fab recipe identify?",
        "What process pressure is listed in this fab recipe?",
        12601, 40000,
    ),
    _Domain(
        "stage_audio", "AU",
        ("cardioid", "supercardioid", "hypercardioid", "omnidirectional", "figure-eight", "shotgun", "boundary", "lavaliere"),
        ("subcardioid", "hemispherical", "parabolic", "contact", "stereo pair", "mid-side", "ambisonic", "headset"),
        ("gain 12 dB", "gain 16 dB", "gain 20 dB", "gain 24 dB", "gain 28 dB", "gain 32 dB", "gain 36 dB", "gain 40 dB"),
        ("gain 14 dB", "gain 18 dB", "gain 22 dB", "gain 26 dB", "gain 30 dB", "gain 34 dB", "gain 38 dB", "gain 42 dB"),
        "Stage audio cue {code} uses microphone pattern {first} and preamp {second}.",
        "Soundcheck sheet {code} records pickup mode {first}; input setting is {second}.",
        "Which microphone pattern is used in this stage audio cue?",
        "What preamp gain is used in this stage audio cue?",
        "What pickup mode does this soundcheck sheet record?",
        "What input setting is listed in this soundcheck sheet?",
        12701, 41000,
    ),
)


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
    aliases = tuple(f"{value} is the matching semantic value" for _, value in rows)
    ids = tuple(f"{case_id}__opaque_{i:02d}" for i in range(4))
    ga = next(i for i, (kind, _) in enumerate(rows) if kind == "a")
    gb = next(i for i, (kind, _) in enumerate(rows) if kind == "b")
    return texts, aliases, ids, ga, gb


def _build_case(spec: _Domain, index: int, split: Split) -> S6PairCase:
    first_values = spec.train_first if split == "train" else spec.dev_first
    second_values = spec.train_second if split == "train" else spec.dev_second
    first = first_values[index % len(first_values)]
    second = second_values[(index * 3 + 1) % len(second_values)]
    other_first = first_values[(index + 3) % len(first_values)]
    other_second = second_values[(index * 5 + 2) % len(second_values)]
    offset = index + (spec.dev_offset if split == "dev" else 0)
    case_id = f"{split}-s6-{spec.name}-{index:03d}"

    state_template = spec.state_train if split == "train" else spec.state_dev
    qa = spec.qa_train if split == "train" else spec.qa_dev
    qb = spec.qb_train if split == "train" else spec.qb_dev
    state = state_template.format(
        code=f"{spec.prefix}{offset:05d}",
        first=first,
        second=second,
    )
    texts, aliases, ids, ga, gb = _shuffle(
        case_id=case_id,
        first=first,
        second=second,
        distractor_first=other_first,
        distractor_second=other_second,
        seed=spec.seed + offset,
    )
    return S6PairCase(
        case_id,
        split,
        spec.name,
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


def generate_s6_pairs(split: Split) -> tuple[S6PairCase, ...]:
    count = 64 if split == "train" else 16
    rows = []
    for spec in _DOMAINS:
        rows.extend(_build_case(spec, i, split) for i in range(count))
    return tuple(rows)


def validate_s6_partitions(
    train: tuple[S6PairCase, ...],
    dev: tuple[S6PairCase, ...],
) -> None:
    if len(train) != 768 or len(dev) != 192:
        raise RuntimeError("Hira v1 S6 partition size changed")
    if any(row.language != "en" for row in (*train, *dev)):
        raise RuntimeError("Hira v1 S6-A must remain English-only")

    expected_domains = {spec.name for spec in _DOMAINS}
    if {row.domain for row in train} != expected_domains:
        raise RuntimeError("Hira v1 S6 TRAIN domain set changed")
    if {row.domain for row in dev} != expected_domains:
        raise RuntimeError("Hira v1 S6 DEV domain set changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S6 duplicate case IDs")
        for row in rows:
            if len(row.option_texts) != 4 or len(row.option_aliases) != 4:
                raise RuntimeError("Hira v1 S6 requires four options with aliases")
            if len(row.option_ids) != 4 or len(set(row.option_ids)) != 4:
                raise RuntimeError("Hira v1 S6 option IDs must be unique K=4")
            if row.gold_a == row.gold_b:
                raise RuntimeError("Hira v1 S6 paired golds must differ")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("Hira v1 S6 gold index out of range")

    train_states = {row.state for row in train}
    dev_states = {row.state for row in dev}
    if train_states & dev_states:
        raise RuntimeError("Hira v1 S6 TRAIN/DEV state overlap")

    train_q = {q for row in train for q in (row.question_a, row.question_b)}
    dev_q = {q for row in dev for q in (row.question_a, row.question_b)}
    if train_q & dev_q:
        raise RuntimeError("Hira v1 S6 TRAIN/DEV question overlap")


__all__ = ["S6PairCase", "generate_s6_pairs", "validate_s6_partitions"]
