from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .contracts import LogicalOption
from .hira import HIRACore
from .runtime import NolaneHira, PRIMITIVE_TO_ID, RelationMode


TRAIN_K_COUNTS = {8: 128, 16: 112, 32: 96, 64: 80, 128: 64, 255: 32}
DEV_K_COUNTS = {8: 32, 16: 32, 32: 32, 64: 32, 128: 16, 255: 16}
CONFIRM_K_COUNTS = {32: 48, 64: 48, 128: 48, 255: 48}

TRAIN_SEED = 61001
DEV_SEED = 62002
CONFIRM_SEED = 63003
GLOBAL_SEED = 101
EPOCHS = 8
FORCED_BUDGET = 255
D_MODEL = 256

RELATION_MODES: tuple[RelationMode, ...] = (
    "pooled",
    "option_tokens",
    "state_tokens",
    "dual_tokens",
)

TRAIN_TEMPLATES = (
    "train-record",
    "train-order",
    "train-fields",
    "train-report",
)
DEV_TEMPLATES = ("dev-note", "dev-ticket")
CONFIRM_TEMPLATES = (
    "confirm-dossier",
    "confirm-transcript",
    "confirm-card",
)

TRAIN_TAGS = (
    "crimson","scarlet","vermilion","ochre","saffron","indigo","violet","teal",
    "cyan","magenta","ivory","charcoal","azure","coral","lime","navy",
    "maroon","turquoise","beige","khaki","umber","lilac","jade","rose",
)
TRAIN_ROLES = (
    "architect","chef","clerk","dentist","farmer","guard","judge","mechanic",
    "nurse","plumber","sailor","tailor","teacher","vendor","writer","courier",
    "ranger","porter","broker","curator","analyst","driver","inspector","gardener",
    "electrician","carpenter","scientist","pharmacist","librarian","conductor",
    "photographer","translator",
)
TRAIN_ACTIONS = (
    "align","assemble","audit","balance","classify","clean","collect","compare",
    "count","cut","deliver","examine","fasten","grade","label","load",
    "mark","measure","package","repair","route","scan","stow","sort",
    "stack","test","trace","verify","weigh","inspect","catalog","dispatch",
)
TRAIN_SITES = (
    "arcade","avenue","bridge","courtyard","depot","garden","gate","kiosk",
    "market","pier","plaza","station","terminal","terrace","theater","tower",
    "tunnel","yard","museum","school","clinic","factory","garage","park",
    "port","square","warehouse","campus","stadium","orchard","checkpoint","boulevard",
)

CONFIRM_TAGS = (
    "acacia","birch","cedar","elm","fir","hazel","juniper","maple",
    "oak","pine","willow","yew","bamboo","cactus","fern","iris",
    "lily","moss","orchid","palm","reed","spruce","tulip","lotus",
)
CONFIRM_ROLES = (
    "astronaut","barber","captain","coach","dancer","engineer","florist","geologist",
    "historian","jeweler","lawyer","medic","miner","researcher","sculptor","surgeon",
    "technician","waiter","zoologist","chemist","diver","editor","musician","surveyor",
    "therapist","beekeeper","blacksmith","brewer","composer","optician","reporter","welder",
)
CONFIRM_ACTIONS = (
    "calibrate","estimate","gauge","monitor","observe","record","sample","survey",
    "tally","time","track","validate","quantify","map","chart","probe",
    "benchmark","document","index","log","profile","review","score","screen",
)
CONFIRM_SITES = (
    "airport","archive","barn","bunker","cabin","courthouse","greenhouse","hospital",
    "hotel","laboratory","mill","observatory","pavilion","restaurant","shelter","shipyard",
    "arena","embassy","fort","gym","hangar","mall","monastery","palace",
    "prison","ranch","university","zoo","aquarium","chapel","foundry","terminal_annex",
)


@dataclass(frozen=True)
class TokenRoutingCase:
    case_id: str
    split: str
    template_id: str
    k: int
    state_text: str
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int


def _vocab(split: str):
    if split in {"train", "dev"}:
        return TRAIN_TAGS, TRAIN_ROLES, TRAIN_ACTIONS, TRAIN_SITES
    if split == "confirm":
        return CONFIRM_TAGS, CONFIRM_ROLES, CONFIRM_ACTIONS, CONFIRM_SITES
    raise ValueError(f"unknown split: {split}")


def _signature_text(signature: tuple[str, str, str, str]) -> str:
    a, b, c, d = signature
    return f"color {a}; role {b}; operation {c}; site {d}"


def _render_state(
    signature: tuple[str, str, str, str],
    template_id: str,
) -> tuple[str, str]:
    a, b, c, d = signature
    if template_id == "train-record":
        state = f"Record: color={a}; role={b}; operation={c}; site={d}."
        question = "Which route matches every field in the record?"
    elif template_id == "train-order":
        state = f"Send the {b} to {d} to {c}; use the {a} color code."
        question = "Which route satisfies the full order?"
    elif template_id == "train-fields":
        state = json.dumps(
            {"site": d, "operation": c, "role": b, "color": a},
            sort_keys=True,
            separators=(",", ":"),
        )
        question = "Select the route with the same four fields."
    elif template_id == "train-report":
        state = f"The report names {d} as the site. The {b} must {c}. Code color: {a}."
        question = "Which candidate route agrees with the report?"
    elif template_id == "dev-note":
        state = f"Note for {d}: assign {b}; required operation is {c}; identifying color is {a}."
        question = "Identify the candidate consistent with the note."
    elif template_id == "dev-ticket":
        state = f"Ticket [{a}] requests {c} by the {b} at {d}."
        question = "Which route reconstructs this ticket exactly?"
    elif template_id == "confirm-dossier":
        state = f"Dossier says site {d}, specialist {b}, task {c}, marker {a}."
        question = "Which route agrees with the entire dossier?"
    elif template_id == "confirm-transcript":
        state = f"At {d}, the {b} is instructed to {c}; the reference marker is {a}."
        question = "Select the signature consistent with the transcript."
    elif template_id == "confirm-card":
        state = json.dumps(
            {"marker": a, "specialist": b, "task": c, "place": d},
            sort_keys=False,
            separators=(",", ":"),
        )
        question = "Which route corresponds to this card?"
    else:
        raise ValueError(f"unknown template: {template_id}")
    return state, question


def _random_signature(
    pools,
    rng: random.Random,
) -> tuple[str, str, str, str]:
    return tuple(rng.choice(pool) for pool in pools)


def _neighbors(
    target: tuple[str, str, str, str],
    pools,
    changes: int,
) -> list[tuple[str, str, str, str]]:
    if changes not in {1, 2}:
        raise ValueError("changes must be 1 or 2")
    out: list[tuple[str, str, str, str]] = []
    if changes == 1:
        for i in range(4):
            for value in pools[i]:
                if value == target[i]:
                    continue
                row = list(target)
                row[i] = value
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


def generate_token_case(
    *,
    split: str,
    k: int,
    case_index: int,
    rng: random.Random,
    template_id: str,
) -> TokenRoutingCase:
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
    for row in two:
        if len(distractors) >= min(one_target, len(one)) + min(two_target, len(two)):
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
            criterion_text=_signature_text(signature),
        )
        for index, signature in enumerate(signatures)
    )
    state, question = _render_state(target, template_id)
    return TokenRoutingCase(
        case_id=f"w5b-{split}-{k}-{case_index:04d}",
        split=split,
        template_id=template_id,
        k=k,
        state_text=state,
        question_text=question,
        options=options,
        gold_index=gold_index,
    )


def generate_token_authority(split: str) -> list[TokenRoutingCase]:
    if split == "train":
        counts, seed, templates = TRAIN_K_COUNTS, TRAIN_SEED, TRAIN_TEMPLATES
    elif split == "dev":
        counts, seed, templates = DEV_K_COUNTS, DEV_SEED, DEV_TEMPLATES
    elif split == "confirm":
        counts, seed, templates = CONFIRM_K_COUNTS, CONFIRM_SEED, CONFIRM_TEMPLATES
    else:
        raise ValueError(f"unknown split: {split}")

    rng = random.Random(seed)
    rows: list[TokenRoutingCase] = []
    index = 0
    for k in sorted(counts):
        for _ in range(counts[k]):
            template = templates[index % len(templates)]
            rows.append(
                generate_token_case(
                    split=split,
                    k=k,
                    case_index=index,
                    rng=rng,
                    template_id=template,
                )
            )
            index += 1
    return rows


@torch.inference_mode()
def compile_token_cache(
    model: NolaneHira,
    cases: Sequence[TokenRoutingCase],
) -> dict:
    model.eval()
    before = model.state_encode_calls
    rows = []
    for case in cases:
        memory = model.compile_state(case.state_text, segment_tokens=32)
        qbatch = model.encoder.encode_texts([case.question_text])
        criteria = [option.criterion_text for option in case.options]
        obatch = model.encoder.encode_texts(criteria)
        option_embeddings = F.normalize(obatch.pooled_embeddings, dim=-1)

        rows.append({
            "case_id": case.case_id,
            "split": case.split,
            "template_id": case.template_id,
            "k": case.k,
            "gold_index": case.gold_index,
            "state_segments": memory.segment_embeddings.detach().cpu().to(torch.float16),
            "state_tokens": memory.token_embeddings.detach().cpu().to(torch.float16),
            "question_embedding": qbatch.pooled_embeddings[0].detach().cpu().to(torch.float16),
            "option_embeddings": option_embeddings.detach().cpu().to(torch.float16),
            "option_tokens": obatch.token_embeddings.detach().cpu().to(torch.float16),
            "option_token_mask": obatch.attention_mask.detach().cpu().bool(),
        })

    state_calls = model.state_encode_calls - before
    if state_calls != len(cases):
        raise RuntimeError("W5b cache violated state-once semantics")
    cache = {
        "schema_version": "r8-w5b-token-cache-v1",
        "case_count": len(rows),
        "state_encode_calls": state_calls,
        "state_encode_calls_per_case": state_calls / max(1, len(rows)),
        "cases": rows,
    }
    validate_token_cache(cache)
    return cache


def validate_token_cache(cache: dict, *, expected_split: str | None = None) -> None:
    if cache.get("schema_version") != "r8-w5b-token-cache-v1":
        raise ValueError("unexpected W5b cache schema")
    cases = cache.get("cases")
    if not isinstance(cases, list) or len(cases) != cache.get("case_count"):
        raise ValueError("W5b cache case count mismatch")
    seen = set()
    for case in cases:
        case_id = case.get("case_id")
        if not case_id or case_id in seen:
            raise ValueError("invalid or duplicate W5b case id")
        seen.add(case_id)
        if expected_split is not None and case.get("split") != expected_split:
            raise ValueError("W5b cache split mismatch")
        k = int(case["k"])
        if not 2 <= k <= 255:
            raise ValueError("invalid W5b cached K")
        if case["state_segments"].ndim != 2 or case["state_segments"].shape[-1] != D_MODEL:
            raise ValueError("state_segments must be [S,256]")
        if case["state_tokens"].ndim != 2 or case["state_tokens"].shape[-1] != D_MODEL:
            raise ValueError("state_tokens must be [T,256]")
        if case["question_embedding"].shape != (D_MODEL,):
            raise ValueError("question_embedding must be [256]")
        if case["option_embeddings"].shape != (k, D_MODEL):
            raise ValueError("option_embeddings must be [K,256]")
        option_tokens = case["option_tokens"]
        option_mask = case["option_token_mask"]
        if option_tokens.ndim != 3 or option_tokens.shape[0] != k or option_tokens.shape[-1] != D_MODEL:
            raise ValueError("option_tokens must be [K,T,256]")
        if option_mask.shape != option_tokens.shape[:2] or option_mask.dtype != torch.bool:
            raise ValueError("option_token_mask shape/type mismatch")
        if not 0 <= int(case["gold_index"]) < k:
            raise ValueError("gold index out of range")


def save_token_cache(cache: dict, path) -> None:
    validate_token_cache(cache)
    torch.save(cache, path)


def load_token_cache(path) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    validate_token_cache(cache)
    return cache


def _forward_cached(
    hira: HIRACore,
    case: dict,
    *,
    relation_mode: RelationMode,
):
    if relation_mode not in RELATION_MODES:
        raise ValueError(f"invalid W5b relation mode: {relation_mode}")
    use_state_tokens = relation_mode in {"state_tokens", "dual_tokens"}
    use_option_tokens = relation_mode in {"option_tokens", "dual_tokens"}

    state = (
        case["state_tokens"]
        if use_state_tokens
        else case["state_segments"]
    ).float().unsqueeze(0)
    question = case["question_embedding"].float().unsqueeze(0)
    options = case["option_embeddings"].float().unsqueeze(0)
    qtype = torch.tensor([PRIMITIVE_TO_ID["choice"]], dtype=torch.long)

    option_tokens = None
    option_token_mask = None
    if use_option_tokens:
        option_tokens = case["option_tokens"].float().unsqueeze(0)
        option_token_mask = case["option_token_mask"].unsqueeze(0)

    return hira(
        question,
        state,
        options,
        qtype,
        option_tokens=option_tokens,
        option_token_mask=option_token_mask,
        forced_budget=FORCED_BUDGET,
        adaptive_budget=False,
    )


def _hard_brier(p: Tensor, gold_index: int) -> Tensor:
    onehot = torch.zeros_like(p)
    onehot[..., gold_index] = 1.0
    return ((p - onehot) ** 2).sum(-1).mean()


def token_case_loss(
    hira: HIRACore,
    case: dict,
    *,
    relation_mode: RelationMode,
) -> Tensor:
    out = _forward_cached(hira, case, relation_mode=relation_mode)
    gold_index = int(case["gold_index"])
    gold = torch.tensor([gold_index], dtype=torch.long)
    ce = torch.nn.functional.cross_entropy(out.logits, gold)
    return ce + 0.1 * _hard_brier(out.probabilities, gold_index)


@torch.inference_mode()
def evaluate_token_cache(
    hira: HIRACore,
    cache: dict,
    *,
    relation_mode: RelationMode,
) -> dict:
    validate_token_cache(cache)
    hira.eval()
    correct = 0
    top5 = 0
    reciprocal = []
    brier = []
    mass = []
    budgets = []
    per_k: dict[int, dict] = {}

    for case in cache["cases"]:
        out = _forward_cached(hira, case, relation_mode=relation_mode)
        p = out.probabilities[0].detach().cpu()
        gold = int(case["gold_index"])
        order = torch.argsort(p, descending=True)
        pred = int(order[0])
        is_correct = int(pred == gold)
        is_top5 = int(gold in set(order[:5].tolist()))
        rank = int((order == gold).nonzero(as_tuple=False)[0].item()) + 1

        correct += is_correct
        top5 += is_top5
        reciprocal.append(1.0 / rank)
        onehot = torch.zeros_like(p)
        onehot[gold] = 1.0
        brier.append(float(((p - onehot) ** 2).sum()))
        mass.append(abs(float(p.sum()) - 1.0))
        budget = int(out.candidate_budget.item())
        budgets.append(budget)

        k = int(case["k"])
        slot = per_k.setdefault(
            k,
            {"n": 0, "correct": 0, "top5": 0, "rr": [], "budgets": [], "mass": []},
        )
        slot["n"] += 1
        slot["correct"] += is_correct
        slot["top5"] += is_top5
        slot["rr"].append(1.0 / rank)
        slot["budgets"].append(budget)
        slot["mass"].append(mass[-1])

    n = len(cache["cases"])
    out_per_k = {}
    for k, slot in sorted(per_k.items()):
        out_per_k[str(k)] = {
            "n": slot["n"],
            "accuracy": slot["correct"] / slot["n"],
            "top5_recall": slot["top5"] / slot["n"],
            "mrr": sum(slot["rr"]) / slot["n"],
            "candidate_budget_min": min(slot["budgets"]),
            "candidate_budget_max": max(slot["budgets"]),
            "probability_mass_max_error": max(slot["mass"]),
        }

    return {
        "case_count": n,
        "accuracy": correct / n,
        "top5_recall": top5 / n,
        "mrr": sum(reciprocal) / n,
        "hard_brier": sum(brier) / n,
        "probability_mass_max_error": max(mass),
        "candidate_budget_min": min(budgets),
        "candidate_budget_max": max(budgets),
        "per_k": out_per_k,
        "relation_mode": relation_mode,
    }


def dev_selection_key(metrics: dict, epoch: int) -> tuple:
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


def train_token_candidate(
    hira: HIRACore,
    train_cache: dict,
    dev_cache: dict,
    *,
    relation_mode: RelationMode,
    lr: float = 3e-4,
    epochs: int = EPOCHS,
    seed: int = GLOBAL_SEED,
) -> tuple[list[dict], dict[str, Tensor], dict]:
    validate_token_cache(train_cache, expected_split="train")
    validate_token_cache(dev_cache, expected_split="dev")
    optimizer = torch.optim.AdamW(
        hira.parameters(),
        lr=float(lr),
        weight_decay=0.01,
    )
    history = []
    best_key = None
    best_state = None
    best_metrics = None
    cases = train_cache["cases"]

    for epoch in range(1, int(epochs) + 1):
        hira.train()
        order = list(range(len(cases)))
        random.Random(int(seed) + epoch).shuffle(order)
        loss_sum = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = token_case_loss(
                hira,
                cases[index],
                relation_mode=relation_mode,
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())

        metrics = evaluate_token_cache(
            hira,
            dev_cache,
            relation_mode=relation_mode,
        )
        metrics["epoch"] = epoch
        metrics["mean_train_case_loss"] = loss_sum / len(cases)
        history.append(metrics)
        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in hira.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("W5b candidate produced no checkpoint")
    hira.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics
