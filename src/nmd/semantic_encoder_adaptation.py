from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contracts import LogicalOption
from .semantic import HFAutoSemanticEncoder


TRAIN_K_COUNTS = {8: 96, 16: 96, 32: 80, 64: 64, 128: 48, 255: 32}
DEV_K_COUNTS = {8: 24, 16: 24, 32: 24, 64: 24, 128: 24, 255: 24}
CONFIRM_K_COUNTS = {32: 48, 64: 48, 128: 48, 255: 48}

TRAIN_SEED = 81001
DEV_SEED = 82002
CONFIRM_SEED = 83003
GLOBAL_SEED = 171
EPOCHS = 3
LOGIT_SCALE = 10.0

TRAIN_TEMPLATES = (
    "w5d-train-card",
    "w5d-train-log",
    "w5d-train-record",
    "w5d-train-note",
)
DEV_TEMPLATES = ("w5d-dev-ledger", "w5d-dev-ticket")
CONFIRM_TEMPLATES = (
    "w5d-confirm-form",
    "w5d-confirm-dossier",
    "w5d-confirm-slip",
)

TRAIN_METALS = (
    "antimony","molybdenum","niobium","tantalum","vanadium","zirconium","hafnium","rhenium","ruthenium","selenium","tellurium","yttrium","scandium","thallium","indium","germanium","cesium","rubidium","strontium","barium","beryllium","neodymium","praseodymium","samarium",
)
TRAIN_INSTRUMENTS = (
    "sitar","dulcimer","lute","koto","shamisen","erhu","guzheng","theremin","ocarina","recorder","euphonium","bugle","celesta","clavichord","harpsichord","zither","balalaika","oud","didgeridoo","kalimba","panpipe","bagpipe","castanet","gong",
)
TRAIN_WEATHER = (
    "misty","sunny","cloudy","windy","stormy","rainy","snowy","humid",
    "dry","foggy","breezy","icy","mild","hot","cold","thunderous",
    "drizzly","hazy","frosty","tropical","arid","chilly","blustery","overcast",
)
TRAIN_TOOLS = (
    "adze","awl","pliers","rasp","sander","mallet","level","ruler","reamer","clamp","file","shovel","rake","trowel","auger","crowbar","screwdriver","spanner","anvil","hatchet","shears","gimlet","caliper","drawknife",
)

CONFIRM_FRUITS = (
    "dragonfruit","durian","rambutan","mangosteen","starfruit","breadfruit","tamarind","jackfruit","cherry","mango","papaya","guava","orange","grapefruit","apricot","nectarine","kiwi","banana","coconut","lychee","persimmon","pomegranate","quince","date",
)
CONFIRM_VEHICLES = (
    "sedan","coupe","van","truck","scooter","bicycle","hovercraft","monorail","funicular","gondola","tractor","bulldozer","subway","locomotive","catamaran","helicopter","yacht","airship","taxi","rickshaw","snowmobile","motorcycle","seaplane","kayak",
)
CONFIRM_TEXTURES = (
    "smooth","rough","silky","grainy","fuzzy","glossy","matte","sticky",
    "slippery","crisp","soft","hard","brittle","rubbery","velvety","waxy",
    "leathery","spongy","flaky","powdery","fibrous","porous","dense","coarse",
)
CONFIRM_PLACES = (
    "lighthouse","vineyard","dockyard","citadel","desert","meadow","bakery","temple","fortress","marina","village","rotunda","boathouse","windmill","arboretum","planetarium","atrium","gazebo","basilica","catacomb","refinery","granary","apiary","aviary",
)


@dataclass(frozen=True)
class AdaptationCase:
    case_id: str
    split: str
    template_id: str
    k: int
    state_text: str
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int


def all_w5d_vocab() -> set[str]:
    groups = (
        TRAIN_METALS, TRAIN_INSTRUMENTS, TRAIN_WEATHER, TRAIN_TOOLS,
        CONFIRM_FRUITS, CONFIRM_VEHICLES, CONFIRM_TEXTURES, CONFIRM_PLACES,
    )
    return {item for group in groups for item in group}


def _vocab(split: str):
    if split in {"train", "dev"}:
        return TRAIN_METALS, TRAIN_INSTRUMENTS, TRAIN_WEATHER, TRAIN_TOOLS
    if split == "confirm":
        return CONFIRM_FRUITS, CONFIRM_VEHICLES, CONFIRM_TEXTURES, CONFIRM_PLACES
    raise ValueError(f"unknown W5d split: {split}")


def _signature_text(signature: tuple[str, str, str, str], split: str) -> str:
    a, b, c, d = signature
    if split in {"train", "dev"}:
        return f"metal {a}; instrument {b}; weather {c}; tool {d}"
    return f"fruit {a}; vehicle {b}; texture {c}; place {d}"


def _render(
    signature: tuple[str, str, str, str],
    template_id: str,
) -> tuple[str, str]:
    a, b, c, d = signature
    if template_id == "w5d-train-card":
        return (
            f"Card fields: metal {a}; instrument {b}; weather {c}; tool {d}.",
            "Which route matches every card field?",
        )
    if template_id == "w5d-train-log":
        return (
            f"Log says tool={d}, weather={c}, metal={a}, instrument={b}.",
            "Select the route consistent with the whole log.",
        )
    if template_id == "w5d-train-record":
        return (
            f"Record: {b} instrument; {a} metal; {d} tool; {c} weather.",
            "Which candidate reconstructs this record exactly?",
        )
    if template_id == "w5d-train-note":
        return (
            f"Note links metal {a} with tool {d}; weather {c}; instrument {b}.",
            "Identify the option preserving all note attributes.",
        )
    if template_id == "w5d-dev-ledger":
        return (
            f"Ledger -> weather {c}; tool {d}; instrument {b}; metal {a}.",
            "Which route exactly agrees with the ledger?",
        )
    if template_id == "w5d-dev-ticket":
        return (
            f"Ticket contains instrument {b}, metal {a}, weather {c}, tool {d}.",
            "Choose the fully matching route.",
        )
    if template_id == "w5d-confirm-form":
        return (
            f"Form fields: fruit {a}; vehicle {b}; texture {c}; place {d}.",
            "Which route agrees with every form field?",
        )
    if template_id == "w5d-confirm-dossier":
        return (
            f"Dossier records place {d}, texture {c}, fruit {a}, vehicle {b}.",
            "Select the candidate that exactly matches the dossier.",
        )
    if template_id == "w5d-confirm-slip":
        return (
            f"Slip -> vehicle {b}; fruit {a}; place {d}; texture {c}.",
            "Which route reconstructs all slip attributes?",
        )
    raise ValueError(f"unknown W5d template: {template_id}")


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


def generate_adaptation_case(
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
        case_id=f"w5d-{split}-{k}-{case_index:04d}",
        split=split,
        template_id=template_id,
        k=k,
        state_text=state,
        question_text=question,
        options=options,
        gold_index=gold_index,
    )


def generate_adaptation_authority(split: str) -> list[AdaptationCase]:
    if split == "train":
        counts, seed, templates = TRAIN_K_COUNTS, TRAIN_SEED, TRAIN_TEMPLATES
    elif split == "dev":
        counts, seed, templates = DEV_K_COUNTS, DEV_SEED, DEV_TEMPLATES
    elif split == "confirm":
        counts, seed, templates = CONFIRM_K_COUNTS, CONFIRM_SEED, CONFIRM_TEMPLATES
    else:
        raise ValueError(f"unknown W5d split: {split}")
    rng = random.Random(seed)
    rows: list[AdaptationCase] = []
    index = 0
    for k in sorted(counts):
        for _ in range(counts[k]):
            rows.append(
                generate_adaptation_case(
                    split=split,
                    k=k,
                    case_index=index,
                    rng=rng,
                    template_id=templates[index % len(templates)],
                )
            )
            index += 1
    return rows


def discover_transformer_layers(model: nn.Module) -> list[nn.Module]:
    candidates = [
        getattr(getattr(model, "encoder", None), "layer", None),
        getattr(getattr(model, "transformer", None), "layer", None),
    ]
    prefix = getattr(model, "base_model_prefix", "")
    if prefix and hasattr(model, prefix):
        base = getattr(model, prefix)
        candidates.extend([
            getattr(getattr(base, "encoder", None), "layer", None),
            getattr(getattr(base, "transformer", None), "layer", None),
        ])
    for value in candidates:
        if value is None:
            continue
        try:
            layers = list(value)
        except TypeError:
            continue
        if layers:
            return layers
    raise RuntimeError(
        "unable to discover transformer block list for controlled A13 adaptation"
    )


def configure_top_layer_adaptation(
    encoder: HFAutoSemanticEncoder,
    *,
    top_n: int,
) -> dict[str, object]:
    if top_n not in {1, 2}:
        raise ValueError("W5d top_n must be 1 or 2")
    model = encoder.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = discover_transformer_layers(model)
    if len(layers) < top_n:
        raise RuntimeError(
            f"encoder has only {len(layers)} transformer blocks"
        )
    selected = layers[-top_n:]
    for layer in selected:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)

    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in model.parameters())
    if not trainable_names or trainable_count <= 0:
        raise RuntimeError("W5d configured zero trainable parameters")
    return {
        "layer_count": len(layers),
        "top_n": top_n,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": total_count - trainable_count,
        "total_parameter_count": total_count,
    }


def adaptation_state_dict(encoder: HFAutoSemanticEncoder) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in encoder.model.state_dict().items()
        if any(
            name == trainable_name
            for trainable_name, parameter in encoder.model.named_parameters()
            if parameter.requires_grad
        )
    }


def load_adaptation_state(
    encoder: HFAutoSemanticEncoder,
    state: dict[str, Tensor],
    *,
    top_n: int,
) -> dict[str, object]:
    info = configure_top_layer_adaptation(encoder, top_n=top_n)
    expected = set(info["trainable_parameter_names"])
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise RuntimeError(
            f"adaptation checkpoint parameter mismatch; missing={missing}, extra={extra}"
        )
    current = encoder.model.state_dict()
    for name, tensor in state.items():
        current[name] = tensor.to(dtype=current[name].dtype)
    encoder.model.load_state_dict(current, strict=True)
    return info


def semantic_logits(
    encoder: HFAutoSemanticEncoder,
    case: AdaptationCase,
) -> Tensor:
    texts = [
        case.state_text,
        case.question_text,
        *[option.criterion_text for option in case.options],
    ]
    batch = encoder.encode_texts(texts)
    pooled = batch.pooled_embeddings
    context = F.normalize(pooled[0] + pooled[1], dim=-1)
    options = F.normalize(pooled[2:], dim=-1)
    return torch.einsum("d,kd->k", context, options) * LOGIT_SCALE


def hard_brier(probabilities: Tensor, gold_index: int) -> Tensor:
    onehot = torch.zeros_like(probabilities)
    onehot[gold_index] = 1.0
    return ((probabilities - onehot) ** 2).sum()


def adaptation_loss(
    encoder: HFAutoSemanticEncoder,
    case: AdaptationCase,
) -> Tensor:
    logits = semantic_logits(encoder, case)
    gold = torch.tensor([case.gold_index], device=logits.device)
    ce = F.cross_entropy(logits.unsqueeze(0), gold)
    probabilities = torch.softmax(logits, dim=-1)
    return ce + 0.1 * hard_brier(probabilities, case.gold_index)


@torch.inference_mode()
def evaluate_encoder(
    encoder: HFAutoSemanticEncoder,
    cases: Sequence[AdaptationCase],
) -> dict[str, object]:
    encoder.eval()
    correct = 0
    top5 = 0
    reciprocal: list[float] = []
    brier: list[float] = []
    mass: list[float] = []
    per_k: dict[int, dict] = {}

    for case in cases:
        logits = semantic_logits(encoder, case)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu()
        gold = case.gold_index
        order = torch.argsort(probabilities, descending=True)
        pred = int(order[0])
        is_correct = int(pred == gold)
        is_top5 = int(gold in set(order[:5].tolist()))
        rank = int((order == gold).nonzero(as_tuple=False)[0].item()) + 1
        err = abs(float(probabilities.sum()) - 1.0)

        correct += is_correct
        top5 += is_top5
        reciprocal.append(1.0 / rank)
        brier.append(float(hard_brier(probabilities, gold)))
        mass.append(err)

        slot = per_k.setdefault(
            case.k,
            {"n": 0, "correct": 0, "top5": 0, "rr": [], "mass": []},
        )
        slot["n"] += 1
        slot["correct"] += is_correct
        slot["top5"] += is_top5
        slot["rr"].append(1.0 / rank)
        slot["mass"].append(err)

    n = len(cases)
    if n == 0:
        raise ValueError("W5d evaluator requires cases")
    return {
        "case_count": n,
        "accuracy": correct / n,
        "top5_recall": top5 / n,
        "mrr": sum(reciprocal) / n,
        "hard_brier": sum(brier) / n,
        "probability_mass_max_error": max(mass),
        "per_k": {
            str(k): {
                "n": slot["n"],
                "accuracy": slot["correct"] / slot["n"],
                "top5_recall": slot["top5"] / slot["n"],
                "mrr": sum(slot["rr"]) / slot["n"],
                "probability_mass_max_error": max(slot["mass"]),
            }
            for k, slot in sorted(per_k.items())
        },
    }


def dev_selection_key(metrics: dict[str, object], epoch: int) -> tuple:
    per = metrics["per_k"]
    return (
        -float(metrics["accuracy"]),
        -float(per["128"]["accuracy"]),
        -float(per["255"]["accuracy"]),
        -float(metrics["top5_recall"]),
        -float(metrics["mrr"]),
        float(metrics["hard_brier"]),
        int(epoch),
    )


def train_candidate(
    encoder: HFAutoSemanticEncoder,
    train_cases: Sequence[AdaptationCase],
    dev_cases: Sequence[AdaptationCase],
    *,
    top_n: int,
    lr: float,
    epochs: int = EPOCHS,
    seed: int = GLOBAL_SEED,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object], dict[str, object]]:
    info = configure_top_layer_adaptation(encoder, top_n=top_n)
    optimizer = torch.optim.AdamW(
        [p for p in encoder.model.parameters() if p.requires_grad],
        lr=float(lr),
        weight_decay=0.01,
    )
    best_key = None
    best_metrics = None
    best_state = None
    history: list[dict[str, object]] = []

    for epoch in range(1, int(epochs) + 1):
        encoder.train()
        order = list(range(len(train_cases)))
        random.Random(int(seed) + epoch).shuffle(order)
        loss_sum = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = adaptation_loss(encoder, train_cases[index])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())

        metrics = evaluate_encoder(encoder, dev_cases)
        metrics["epoch"] = epoch
        metrics["mean_train_case_loss"] = loss_sum / len(train_cases)
        history.append(metrics)
        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_state = adaptation_state_dict(encoder)

    if best_state is None or best_metrics is None:
        raise RuntimeError("W5d candidate produced no checkpoint")
    load_adaptation_state(encoder, best_state, top_n=top_n)
    return history, best_state, best_metrics, info


def competence_gates(
    adapted: dict[str, object],
    frozen: dict[str, object],
) -> dict[str, bool]:
    adapted_per = adapted["per_k"]
    frozen_per = frozen["per_k"]
    return {
        "overall_accuracy": float(adapted["accuracy"]) >= 0.60,
        "k128_accuracy": float(adapted_per["128"]["accuracy"]) >= 0.40,
        "k255_accuracy": float(adapted_per["255"]["accuracy"]) >= 0.30,
        "k255_top5": float(adapted_per["255"]["top5_recall"]) >= 0.70,
        "probability_mass": float(adapted["probability_mass_max_error"]) <= 1e-6,
        "overall_gain_vs_frozen": (
            float(adapted["accuracy"]) - float(frozen["accuracy"]) >= 0.20
        ),
        "k128_gain_vs_frozen": (
            float(adapted_per["128"]["accuracy"])
            > float(frozen_per["128"]["accuracy"])
        ),
        "k255_gain_vs_frozen": (
            float(adapted_per["255"]["accuracy"])
            > float(frozen_per["255"]["accuracy"])
        ),
    }


def confirm_verdict(
    adapted: dict[str, object],
    frozen: dict[str, object],
) -> tuple[str, dict[str, bool]]:
    gates = competence_gates(adapted, frozen)
    if all(gates.values()):
        return "A13_ADAPTATION_PASS", gates
    return "A13_ADAPTATION_FAIL_CAPACITY_TRIGGER", gates
