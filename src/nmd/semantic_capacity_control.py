from __future__ import annotations

import math
import random
from typing import Sequence

from .contracts import LogicalOption
from .semantic_encoder_adaptation import (
    AdaptationCase,
    dev_selection_key,
    evaluate_encoder,
    train_candidate,
)


TRAIN_K_COUNTS = {8: 96, 16: 96, 32: 80, 64: 64, 128: 48, 255: 32}
DEV_K_COUNTS = {8: 24, 16: 24, 32: 24, 64: 24, 128: 24, 255: 24}
CONFIRM_K_COUNTS = {32: 48, 64: 48, 128: 48, 255: 48}

TRAIN_SEED = 91001
DEV_SEED = 92002
CONFIRM_SEED = 93003
GLOBAL_SEED = 271
EPOCHS = 3
TOP_N = 2
LR = 1e-5
MAX_LENGTH = 256

MODEL_SPECS = {
    "a13": {
        "model_id": "microsoft/xtremedistil-l6-h256-uncased",
        "revision": "4226d9e4d2c08703e5cb0491b479bfc6a1607181",
        "weight_file": "model.safetensors",
        "weight_sha256": "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880",
        "hidden_size": 256,
    },
    "a22": {
        "model_id": "microsoft/xtremedistil-l6-h384-uncased",
        "revision": "359df7d52613d4edc15647e6d65e0d87200eb747",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "38bd5f8a7d1b7045de8fee25bfac1777edf5a2ec8cd3399b21bde917b0278e23",
        "hidden_size": 384,
    },
}

TRAIN_TEMPLATES = (
    "w5e-train-manifest",
    "w5e-train-index",
    "w5e-train-catalog",
    "w5e-train-register",
)
DEV_TEMPLATES = (
    "w5e-dev-sheet",
    "w5e-dev-entry",
)
CONFIRM_TEMPLATES = (
    "w5e-confirm-brief",
    "w5e-confirm-panel",
    "w5e-confirm-file",
)

TRAIN_GEMS = (
    "agate","alexandrite","amethyst","aquamarine","beryl","carnelian","citrine",
    "chrysoberyl","diamond","emerald","garnet","iolite","jasper","lapis","malachite",
    "moonstone","kunzite","onyx","opal","larimar","peridot","spinel","sapphire","topaz",
)
TRAIN_BIRDS = (
    "albatross","canary","avocet","crow","bittern","bobolink","finch","flamingo",
    "bunting","curlew","kingfisher","lark","magpie","oriole","osprey","owl","parrot",
    "pelican","penguin","godwit","robin","sparrow","stork","toucan",
)
TRAIN_FLAVORS = (
    "bitter","buttery","citrusy","earthy","floral","fruity","garlicky","herbal",
    "malty","minty","nutty","peppery","roasted","salty","savory","smoky",
    "sour","spicy","sweet","tangy","tart","toasty","umami","vanilla",
)
TRAIN_SHAPES = (
    "arc","cardioid","crescent","deltoid","cylinder","diamondshape","dodecagon","helix",
    "heptagon","kite","octagon","oval","pentagon","nonagon","pyramid","rectangle",
    "rhombus","ring","torus","spiral","trefoil","star","trapezoid","wedge",
)

CONFIRM_FLOWERS = (
    "aster","azalea","begonia","camellia","carnation","daffodil","dahlia","daisy",
    "freesia","gardenia","hibiscus","hyacinth","anemone","jasmine","lavender","buttercup",
    "chrysanthemum","marigold","crocus","peony","poppy","foxglove","geranium","zinnia",
)
CONFIRM_VESSELS = (
    "catboat","brig","dhow","caravel","clipper","corvette","dinghy","dory",
    "pinnace","frigate","galleon","junk","ketch","launch","liner","lugger",
    "raft","schooner","skiff","sloop","tanker","trawler","tugboat","wherry",
)
CONFIRM_SOUNDS = (
    "buzz","chime","clang","crackle","drone","echo","hiss","hum","jingle",
    "knock","murmur","peal","ping","rattle","ringing","roar","rustle","screech",
    "snap","thud","tinkle","whirr","whistle","whoosh",
)
CONFIRM_TERRAINS = (
    "badland","butte","escarpment","floodplain","grassland","estuary","fjord","glacier",
    "gorge","heath","hillock","isthmus","headland","karst","moor","oasis",
    "polder","peninsula","saltflat","swamp","savanna","steppe","volcano","watershed",
)


def all_w5e_vocab() -> set[str]:
    groups = (
        TRAIN_GEMS,
        TRAIN_BIRDS,
        TRAIN_FLAVORS,
        TRAIN_SHAPES,
        CONFIRM_FLOWERS,
        CONFIRM_VESSELS,
        CONFIRM_SOUNDS,
        CONFIRM_TERRAINS,
    )
    return {item for group in groups for item in group}


def _vocab(split: str):
    if split in {"train", "dev"}:
        return TRAIN_GEMS, TRAIN_BIRDS, TRAIN_FLAVORS, TRAIN_SHAPES
    if split == "confirm":
        return CONFIRM_FLOWERS, CONFIRM_VESSELS, CONFIRM_SOUNDS, CONFIRM_TERRAINS
    raise ValueError(f"unknown W5e split: {split}")


def _signature_text(signature: tuple[str, str, str, str], split: str) -> str:
    a, b, c, d = signature
    if split in {"train", "dev"}:
        return f"gem {a}; bird {b}; flavor {c}; shape {d}"
    return f"flower {a}; vessel {b}; sound {c}; terrain {d}"


def _render(
    signature: tuple[str, str, str, str],
    template_id: str,
) -> tuple[str, str]:
    a, b, c, d = signature
    if template_id == "w5e-train-manifest":
        return (
            f"Manifest fields: gem {a}; bird {b}; flavor {c}; shape {d}.",
            "Which route agrees with the complete manifest?",
        )
    if template_id == "w5e-train-index":
        return (
            f"Index lists bird={b}, shape={d}, gem={a}, flavor={c}.",
            "Choose the route preserving every indexed attribute.",
        )
    if template_id == "w5e-train-catalog":
        return (
            f"Catalog row: {c} flavor; {a} gem; {b} bird; {d} shape.",
            "Which candidate reconstructs the catalog row exactly?",
        )
    if template_id == "w5e-train-register":
        return (
            f"Register links shape {d} with flavor {c}, bird {b}, gem {a}.",
            "Identify the fully matching registered route.",
        )
    if template_id == "w5e-dev-sheet":
        return (
            f"Sheet -> gem {a}; shape {d}; flavor {c}; bird {b}.",
            "Which route exactly matches the sheet?",
        )
    if template_id == "w5e-dev-entry":
        return (
            f"Entry contains bird {b}, flavor {c}, gem {a}, shape {d}.",
            "Select the candidate consistent with all entry fields.",
        )
    if template_id == "w5e-confirm-brief":
        return (
            f"Brief fields: flower {a}; vessel {b}; sound {c}; terrain {d}.",
            "Which route agrees with every brief field?",
        )
    if template_id == "w5e-confirm-panel":
        return (
            f"Panel records terrain {d}, sound {c}, flower {a}, vessel {b}.",
            "Select the route matching the whole panel.",
        )
    if template_id == "w5e-confirm-file":
        return (
            f"File -> vessel {b}; terrain {d}; flower {a}; sound {c}.",
            "Which candidate reconstructs all file attributes?",
        )
    raise ValueError(f"unknown W5e template: {template_id}")


def _random_signature(pools, rng: random.Random) -> tuple[str, str, str, str]:
    return tuple(rng.choice(pool) for pool in pools)  # type: ignore[return-value]


def _neighbors(
    target: tuple[str, str, str, str],
    pools,
    changes: int,
) -> list[tuple[str, str, str, str]]:
    if changes not in {1, 2}:
        raise ValueError("changes must be 1 or 2")
    out: list[tuple[str, str, str, str]] = []
    if changes == 1:
        for index in range(4):
            for value in pools[index]:
                if value == target[index]:
                    continue
                row = list(target)
                row[index] = value
                out.append(tuple(row))
        return out
    for i in range(4):
        for j in range(i + 1, 4):
            for vi in pools[i]:
                if vi == target[i]:
                    continue
                for vj in pools[j]:
                    if vj == target[j]:
                        continue
                    row = list(target)
                    row[i] = vi
                    row[j] = vj
                    out.append(tuple(row))
    return out


def generate_capacity_case(
    *,
    split: str,
    k: int,
    case_index: int,
    rng: random.Random,
    template_id: str,
) -> AdaptationCase:
    if not 2 <= k <= 255:
        raise ValueError("K must be in [2,255]")
    pools = _vocab(split)
    target = _random_signature(pools, rng)
    used = {target}
    distractors: list[tuple[str, str, str, str]] = []

    one = _neighbors(target, pools, 1)
    two = _neighbors(target, pools, 2)
    rng.shuffle(one)
    rng.shuffle(two)
    one_target = int(math.floor(0.50 * (k - 1)))
    two_target = int(math.floor(0.30 * (k - 1)))
    for row in one[: min(one_target, len(one))]:
        if row not in used:
            used.add(row)
            distractors.append(row)
    desired = min(one_target, len(one)) + min(two_target, len(two))
    for row in two:
        if len(distractors) >= desired:
            break
        if row not in used:
            used.add(row)
            distractors.append(row)
    while len(distractors) < k - 1:
        row = _random_signature(pools, rng)
        if row in used:
            continue
        used.add(row)
        distractors.append(row)

    signatures = [target, *distractors[: k - 1]]
    rng.shuffle(signatures)
    gold_index = signatures.index(target)
    options = tuple(
        LogicalOption(
            option_id=f"route-{index:03d}",
            criterion_text=_signature_text(signature, split),
        )
        for index, signature in enumerate(signatures)
    )
    state, question = _render(target, template_id)
    return AdaptationCase(
        case_id=f"w5e-{split}-{k}-{case_index:04d}",
        split=split,
        template_id=template_id,
        k=k,
        state_text=state,
        question_text=question,
        options=options,
        gold_index=gold_index,
    )


def generate_capacity_authority(split: str) -> list[AdaptationCase]:
    if split == "train":
        counts, seed, templates = TRAIN_K_COUNTS, TRAIN_SEED, TRAIN_TEMPLATES
    elif split == "dev":
        counts, seed, templates = DEV_K_COUNTS, DEV_SEED, DEV_TEMPLATES
    elif split == "confirm":
        counts, seed, templates = CONFIRM_K_COUNTS, CONFIRM_SEED, CONFIRM_TEMPLATES
    else:
        raise ValueError(f"unknown W5e split: {split}")
    rng = random.Random(seed)
    rows: list[AdaptationCase] = []
    index = 0
    for k in sorted(counts):
        for _ in range(counts[k]):
            rows.append(
                generate_capacity_case(
                    split=split,
                    k=k,
                    case_index=index,
                    rng=rng,
                    template_id=templates[index % len(templates)],
                )
            )
            index += 1
    return rows


def capacity_gates(
    a22_adapted: dict[str, object],
    a13_adapted: dict[str, object],
) -> dict[str, bool]:
    a22_per = a22_adapted["per_k"]
    a13_per = a13_adapted["per_k"]
    return {
        "overall_accuracy": float(a22_adapted["accuracy"]) >= 0.60,
        "k128_accuracy": float(a22_per["128"]["accuracy"]) >= 0.40,
        "k255_accuracy": float(a22_per["255"]["accuracy"]) >= 0.30,
        "k255_top5": float(a22_per["255"]["top5_recall"]) >= 0.70,
        "probability_mass": (
            float(a22_adapted["probability_mass_max_error"]) <= 1e-6
        ),
        "overall_capacity_gain": (
            float(a22_adapted["accuracy"])
            - float(a13_adapted["accuracy"])
            >= 0.10
        ),
        "mrr_capacity_gain": (
            float(a22_adapted["mrr"])
            - float(a13_adapted["mrr"])
            >= 0.05
        ),
        "k128_capacity_gain": (
            float(a22_per["128"]["accuracy"])
            > float(a13_per["128"]["accuracy"])
        ),
        "k255_capacity_gain": (
            float(a22_per["255"]["accuracy"])
            > float(a13_per["255"]["accuracy"])
        ),
    }


def capacity_verdict(
    a22_adapted: dict[str, object],
    a13_adapted: dict[str, object],
) -> tuple[str, dict[str, bool]]:
    gates = capacity_gates(a22_adapted, a13_adapted)
    competence_names = (
        "overall_accuracy",
        "k128_accuracy",
        "k255_accuracy",
        "k255_top5",
        "probability_mass",
    )
    capacity_names = (
        "overall_capacity_gain",
        "mrr_capacity_gain",
        "k128_capacity_gain",
        "k255_capacity_gain",
    )
    competence = all(gates[name] for name in competence_names)
    capacity = all(gates[name] for name in capacity_names)
    if competence and capacity:
        return "A22_CAPACITY_RESCUE", gates
    if competence:
        return "A22_CAPACITY_AMBIGUOUS", gates
    return "A22_CAPACITY_NO_RESCUE", gates
