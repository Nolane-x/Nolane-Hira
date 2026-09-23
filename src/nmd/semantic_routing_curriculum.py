from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
from typing import Sequence

import torch
from torch import Tensor

from .contracts import LogicalOption
from .hira import HIRACore
from .losses import LossWeights, typed_decision_loss
from .runtime import NolaneHira, PRIMITIVE_TO_ID


TRAIN_K_COUNTS = {8: 160, 16: 160, 32: 128, 64: 96, 128: 64, 255: 32}
DEV_K_COUNTS = {8: 32, 16: 32, 32: 32, 64: 32, 128: 16, 255: 16}
CONFIRM_K_COUNTS = {32: 64, 64: 64, 128: 64, 255: 64}

TRAIN_SEED = 51001
DEV_SEED = 52002
CONFIRM_SEED = 53003

GLOBAL_SEED = 91
EPOCHS = 6
FORCED_BUDGET = 255
D_MODEL = 256

TRAIN_TEMPLATES = ("train-json", "train-active", "train-bundle", "train-two-clause")
DEV_TEMPLATES = ("dev-dispatch", "dev-manifest")
CONFIRM_TEMPLATES = ("confirm-ledger", "confirm-brief", "confirm-card")

TRAIN_MATERIALS = (
    "basalt","brass","ceramic","cobalt","copper","granite","graphite","iron",
    "marble","nickel","obsidian","quartz","silver","slate","steel","tin",
    "titanium","zinc","amber","bronze","clay","flint","mica","pearl",
)
TRAIN_ENTITIES = (
    "badger","beaver","crane","dolphin","eagle","falcon","fox","heron",
    "ibis","lynx","otter","panda","raven","seal","tiger","yak",
    "chisel","drill","hammer","lathe","plane","saw","wrench","vise",
    "barge","bus","canoe","cart","ferry","glider","tram","wagon",
)
TRAIN_ACTIONS = (
    "advance","circle","climb","cross","descend","dock","enter","follow",
    "halt","lift","move","pass","return","rotate","signal","turn",
    "wait","weave","approach","depart",
)
TRAIN_LOCATIONS = (
    "basin","canyon","cliff","delta","dune","forest","grove","harbor",
    "hill","lagoon","marsh","mesa","plain","quarry","ravine","reef",
    "ridge","shore","tundra","valley","wharf","woodland","plateau","inlet",
)

CONFIRM_SYMBOLS = (
    "comet","eclipse","galaxy","meteor","nebula","orbit","planet","quasar",
    "rain","sleet","snow","storm","thunder","wind","frost","mist",
    "circle","cone","cube","ellipse","hexagon","prism","sphere","triangle",
)
CONFIRM_ITEMS = (
    "banjo","cello","flute","harp","oboe","piano","trumpet","violin",
    "apple","bread","cocoa","fig","lemon","melon","olive","peach",
    "rice","soup","tea","wheat","yogurt","berry","plum","pear",
)
CONFIRM_ACTIONS = (
    "braid","carve","fold","glaze","knit","mold","paint","press",
    "sew","shape","stitch","weave","wrap","etch","polish","bind",
)
CONFIRM_LOCATIONS = (
    "alcove","attic","cellar","foyer","gallery","kitchen","lobby","loft",
    "office","pantry","parlor","studio","workshop","library","hallway","room",
    "balcony","closet","nursery","vault","lounge","chamber","vestibule","den",
)


@dataclass(frozen=True)
class RoutingCase:
    case_id: str
    split: str
    template_id: str
    k: int
    state_text: str
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int


def _signature_text(signature: tuple[str, str, str, str]) -> str:
    a, b, c, d = signature
    return f"marker={a}; entity={b}; action={c}; location={d}"


def _render_state(
    signature: tuple[str, str, str, str],
    template_id: str,
) -> tuple[str, str]:
    a, b, c, d = signature
    if template_id == "train-json":
        state = json.dumps(
            {"marker": a, "entity": b, "action": c, "location": d},
            sort_keys=True,
            separators=(",", ":"),
        )
        q = "Which route signature matches all requested fields?"
    elif template_id == "train-active":
        state = f"The active route uses {a}, with {b} set to {c} at the {d}."
        q = "Which option describes the active route?"
    elif template_id == "train-bundle":
        state = f"Route attributes: marker {a}; entity {b}; action {c}; location {d}."
        q = "Select the option with the same route attributes."
    elif template_id == "train-two-clause":
        state = f"Use marker {a} and entity {b}. The required action is {c} in {d}."
        q = "Which route option satisfies the description?"
    elif template_id == "dev-dispatch":
        state = f"Dispatch record says {b} must {c} at {d}, under the {a} marker."
        q = "Identify the route signature consistent with the dispatch record."
    elif template_id == "dev-manifest":
        state = f"Manifest: [{d}] [{c}] [{b}] [{a}]."
        q = "Which candidate reconstructs the manifest attributes?"
    elif template_id == "confirm-ledger":
        state = f"Ledger entry -> place {d}; operation {c}; item {b}; symbol {a}."
        q = "Which signature agrees with every ledger field?"
    elif template_id == "confirm-brief":
        state = f"Brief: at {d}, {b} performs {c}; the identifying symbol is {a}."
        q = "Select the fully matching signature."
    elif template_id == "confirm-card":
        state = json.dumps(
            {"where": d, "do": c, "item": b, "symbol": a},
            sort_keys=False,
            separators=(",", ":"),
        )
        q = "Which option corresponds to this routing card?"
    else:
        raise ValueError(f"unknown template_id: {template_id}")
    return state, q


def _vocab(split: str):
    if split in {"train", "dev"}:
        return (
            TRAIN_MATERIALS,
            TRAIN_ENTITIES,
            TRAIN_ACTIONS,
            TRAIN_LOCATIONS,
        )
    if split == "confirm":
        return (
            CONFIRM_SYMBOLS,
            CONFIRM_ITEMS,
            CONFIRM_ACTIONS,
            CONFIRM_LOCATIONS,
        )
    raise ValueError(f"unknown split: {split}")


def _enumerate_neighbors(
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


def _random_signature(pools, rng: random.Random) -> tuple[str, str, str, str]:
    return tuple(rng.choice(pool) for pool in pools)  # type: ignore[return-value]


def generate_routing_case(
    *,
    split: str,
    case_index: int,
    k: int,
) -> RoutingCase:
    if k < 2 or k > 255:
        raise ValueError("K must be in [2,255]")
    if split == "train":
        seed = TRAIN_SEED + case_index * 1009 + k
        templates = TRAIN_TEMPLATES
    elif split == "dev":
        seed = DEV_SEED + case_index * 1009 + k
        templates = DEV_TEMPLATES
    elif split == "confirm":
        seed = CONFIRM_SEED + case_index * 1009 + k
        templates = CONFIRM_TEMPLATES
    else:
        raise ValueError(f"unknown split: {split}")

    rng = random.Random(seed)
    pools = _vocab(split)
    target = _random_signature(pools, rng)
    template_id = templates[case_index % len(templates)]
    state_text, question_text = _render_state(target, template_id)

    distractor_count = k - 1
    target_one = int(distractor_count * 0.50)
    target_two = int(distractor_count * 0.30)

    one_neighbors = _enumerate_neighbors(target, pools, 1)
    two_neighbors = _enumerate_neighbors(target, pools, 2)
    rng.shuffle(one_neighbors)
    rng.shuffle(two_neighbors)

    one_count = min(target_one, len(one_neighbors))
    remaining_after_one = distractor_count - one_count
    two_count = min(target_two, len(two_neighbors), remaining_after_one)

    signatures: set[tuple[str, str, str, str]] = {target}
    distractors: list[tuple[str, str, str, str]] = []

    for candidate in one_neighbors[:one_count]:
        signatures.add(candidate)
        distractors.append(candidate)
    for candidate in two_neighbors[:two_count]:
        if candidate not in signatures:
            signatures.add(candidate)
            distractors.append(candidate)

    attempts = 0
    while len(distractors) < distractor_count:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("unable to generate unique random distractors")
        candidate = _random_signature(pools, rng)
        if candidate not in signatures:
            signatures.add(candidate)
            distractors.append(candidate)

    gold_index = rng.randrange(k)
    rows = list(distractors)
    rows.insert(gold_index, target)
    options = tuple(
        LogicalOption(
            option_id=f"route-{i:03d}",
            criterion_text=_signature_text(signature),
        )
        for i, signature in enumerate(rows)
    )
    return RoutingCase(
        case_id=f"{split}-k{k:03d}-{case_index:04d}",
        split=split,
        template_id=template_id,
        k=k,
        state_text=state_text,
        question_text=question_text,
        options=options,
        gold_index=gold_index,
    )


def generate_authority(split: str) -> list[RoutingCase]:
    counts = {
        "train": TRAIN_K_COUNTS,
        "dev": DEV_K_COUNTS,
        "confirm": CONFIRM_K_COUNTS,
    }[split]
    rows: list[RoutingCase] = []
    offset = 0
    for k, count in counts.items():
        for local_index in range(count):
            rows.append(
                generate_routing_case(
                    split=split,
                    case_index=offset + local_index,
                    k=k,
                )
            )
        offset += count
    return rows


@torch.inference_mode()
def compile_routing_cache(
    model: NolaneHira,
    cases: Sequence[RoutingCase],
) -> dict:
    model.eval()
    before = model.state_encode_calls
    rows = []
    for case in cases:
        memory = model.compile_state(case.state_text, segment_tokens=32)
        schema, receipt = model.compile_schema(
            primitive="choice",
            question_text=case.question_text,
            options=case.options,
            use_cache=False,
        )
        rows.append({
            "case_id": case.case_id,
            "split": case.split,
            "template_id": case.template_id,
            "k": case.k,
            "gold_index": case.gold_index,
            "state_segments": memory.segment_embeddings.detach().cpu().to(torch.float16),
            "question_embedding": schema.question_embedding.detach().cpu().to(torch.float16),
            "option_embeddings": schema.option_embeddings.detach().cpu().to(torch.float16),
            "schema_hash": receipt.schema_hash,
        })
    state_calls = model.state_encode_calls - before
    if state_calls != len(cases):
        raise RuntimeError("routing cache violated state-once semantics")
    return {
        "schema_version": "r8-w5a-routing-cache-v1",
        "case_count": len(rows),
        "state_encode_calls": state_calls,
        "state_encode_calls_per_case": state_calls / max(1, len(rows)),
        "cases": rows,
    }


def validate_routing_cache(cache: dict, *, expected_split: str | None = None) -> None:
    if cache.get("schema_version") != "r8-w5a-routing-cache-v1":
        raise ValueError("unexpected routing cache schema")
    cases = cache.get("cases")
    if not isinstance(cases, list) or len(cases) != cache.get("case_count"):
        raise ValueError("routing cache case count mismatch")
    ids=set()
    for case in cases:
        if case["case_id"] in ids:
            raise ValueError("duplicate routing case id")
        ids.add(case["case_id"])
        if expected_split is not None and case["split"] != expected_split:
            raise ValueError("routing cache split mismatch")
        k=int(case["k"])
        if not 2 <= k <= 255:
            raise ValueError("invalid cached K")
        if case["state_segments"].ndim != 2 or case["state_segments"].shape[-1] != D_MODEL:
            raise ValueError("state segments must be [S,256]")
        if case["question_embedding"].shape != (D_MODEL,):
            raise ValueError("question embedding must be [256]")
        if case["option_embeddings"].shape != (k,D_MODEL):
            raise ValueError("option embeddings must be [K,256]")
        if not 0 <= int(case["gold_index"]) < k:
            raise ValueError("gold index out of range")


def save_routing_cache(cache: dict, path) -> None:
    validate_routing_cache(cache)
    torch.save(cache, path)


def load_routing_cache(path) -> dict:
    cache=torch.load(path,map_location="cpu",weights_only=True)
    validate_routing_cache(cache)
    return cache


def _forward_cached(hira: HIRACore, case: dict):
    q=case["question_embedding"].float().unsqueeze(0)
    seg=case["state_segments"].float().unsqueeze(0)
    opt=case["option_embeddings"].float().unsqueeze(0)
    qtype=torch.tensor([PRIMITIVE_TO_ID["choice"]],dtype=torch.long)
    return hira(
        q,seg,opt,qtype,
        forced_budget=FORCED_BUDGET,
        adaptive_budget=False,
    )


def _hard_brier(p: Tensor, gold_index: int) -> Tensor:
    onehot=torch.zeros_like(p)
    onehot[...,gold_index]=1.0
    return ((p-onehot)**2).sum(-1).mean()


def routing_case_loss(hira: HIRACore, case: dict) -> Tensor:
    out=_forward_cached(hira,case)
    logits=out.logits
    gold=torch.tensor([int(case["gold_index"])],dtype=torch.long)
    ce=torch.nn.functional.cross_entropy(logits,gold)
    brier=_hard_brier(out.probabilities,int(case["gold_index"]))
    return ce + 0.1*brier


@torch.inference_mode()
def evaluate_routing_cache(hira: HIRACore, cache: dict) -> dict:
    validate_routing_cache(cache)
    hira.eval()
    cases=cache["cases"]
    correct=0
    top5=0
    rr=[]
    brier=[]
    mass=[]
    budgets=[]
    per_k={}
    for case in cases:
        out=_forward_cached(hira,case)
        p=out.probabilities[0].detach().cpu()
        gold=int(case["gold_index"])
        order=torch.argsort(p,descending=True)
        pred=int(order[0])
        correct += int(pred==gold)
        top5 += int(gold in set(order[:5].tolist()))
        rank=int((order==gold).nonzero(as_tuple=False)[0].item())+1
        rr.append(1.0/rank)
        onehot=torch.zeros_like(p); onehot[gold]=1.0
        brier.append(float(((p-onehot)**2).sum()))
        mass.append(abs(float(p.sum())-1.0))
        budgets.append(int(out.candidate_budget.item()))
        k=int(case["k"])
        slot=per_k.setdefault(
            k,
            {
                "n":0,
                "correct":0,
                "top5":0,
                "rr":[],
                "budgets":[],
                "mass_errors":[],
            },
        )
        slot["n"]+=1
        slot["correct"]+=int(pred==gold)
        slot["top5"]+=int(gold in set(order[:5].tolist()))
        slot["rr"].append(1.0/rank)
        slot["budgets"].append(int(out.candidate_budget.item()))
        slot["mass_errors"].append(abs(float(p.sum())-1.0))
    per_k_metrics={
        str(k):{
            "n":v["n"],
            "accuracy":v["correct"]/v["n"],
            "top5_recall":v["top5"]/v["n"],
            "mrr":sum(v["rr"])/v["n"],
            "candidate_budget_min":min(v["budgets"]),
            "candidate_budget_max":max(v["budgets"]),
            "probability_mass_max_error":max(v["mass_errors"]),
        }
        for k,v in sorted(per_k.items())
    }
    return {
        "case_count":len(cases),
        "accuracy":correct/len(cases),
        "top5_recall":top5/len(cases),
        "mrr":sum(rr)/len(rr),
        "hard_brier":sum(brier)/len(brier),
        "probability_mass_max_error":max(mass),
        "candidate_budget_min":min(budgets),
        "candidate_budget_max":max(budgets),
        "per_k":per_k_metrics,
    }


def dev_selection_key(metrics: dict, epoch: int) -> tuple:
    per=metrics["per_k"]
    return (
        -float(metrics["accuracy"]),
        -float(per["128"]["accuracy"]),
        -float(per["255"]["accuracy"]),
        -float(metrics["mrr"]),
        float(metrics["hard_brier"]),
        int(epoch),
    )


def train_candidate(
    hira: HIRACore,
    train_cache: dict,
    dev_cache: dict,
    *,
    lr: float,
    epochs: int = EPOCHS,
    seed: int = GLOBAL_SEED,
) -> tuple[list[dict], dict[str,Tensor], dict]:
    validate_routing_cache(train_cache,expected_split="train")
    validate_routing_cache(dev_cache,expected_split="dev")
    optimizer=torch.optim.AdamW(hira.parameters(),lr=lr,weight_decay=0.01)
    history=[]
    best_key=None
    best_state=None
    best_metrics=None
    cases=train_cache["cases"]
    for epoch in range(1,epochs+1):
        hira.train()
        order=list(range(len(cases)))
        random.Random(seed+epoch).shuffle(order)
        loss_sum=0.0
        for idx in order:
            optimizer.zero_grad(set_to_none=True)
            loss=routing_case_loss(hira,cases[idx])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        metrics=evaluate_routing_cache(hira,dev_cache)
        metrics["epoch"]=epoch
        metrics["mean_train_case_loss"]=loss_sum/len(cases)
        history.append(metrics)
        key=dev_selection_key(metrics,epoch)
        if best_key is None or key<best_key:
            best_key=key
            best_metrics=dict(metrics)
            best_state={k:v.detach().cpu().clone() for k,v in hira.state_dict().items()}
    if best_state is None or best_metrics is None:
        raise RuntimeError("candidate produced no checkpoint")
    hira.load_state_dict(best_state,strict=True)
    return history,best_state,best_metrics
