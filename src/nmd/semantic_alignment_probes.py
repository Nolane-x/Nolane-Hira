from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contracts import LogicalOption
from .runtime import NolaneHira


ProbeKind = Literal["pooled_cosine", "token_max", "bilinear", "pair_mlp"]

TRAIN_K_COUNTS = {8: 160, 16: 160, 32: 144, 64: 128, 128: 112, 255: 64}
DEV_K_COUNTS = {8: 32, 16: 32, 32: 32, 64: 32, 128: 32, 255: 32}
CONFIRM_K_COUNTS = {32: 64, 64: 64, 128: 64, 255: 64}

TRAIN_SEED = 71001
DEV_SEED = 72002
CONFIRM_SEED = 73003
GLOBAL_SEED = 131
EPOCHS = 8
D_MODEL = 256
D_ALIGN = 128

TRAIN_TEMPLATES = (
    "train-envelope",
    "train-message",
    "train-index",
    "train-briefing",
)
DEV_TEMPLATES = ("dev-summary", "dev-register")
CONFIRM_TEMPLATES = (
    "confirm-memo",
    "confirm-sheet",
    "confirm-entry",
)

TRAIN_REGIONS = (
    "norway","sweden","finland","denmark","iceland","estonia","latvia","lithuania",
    "spain","portugal","france","germany","italy","greece","poland","austria",
    "belgium","ireland","switzerland","croatia","serbia","romania","bulgaria","hungary",
)
TRAIN_SPORTS = (
    "tennis","cricket","rugby","hockey","soccer","boxing","rowing","fencing",
    "cycling","skiing","skating","archery","golf","baseball","volleyball","basketball",
    "surfing","wrestling","karate","judo","squash","badminton","handball","lacrosse",
)
TRAIN_TRAITS = (
    "calm","eager","proud","shy","brave","gentle","loyal","honest",
    "patient","curious","cheerful","serious","quiet","bold","clever","humble",
    "joyful","kind","lively","polite","rapid","steady","vivid","warm",
)
TRAIN_OBJECTS = (
    "sofa","desk","bench","stool","couch","cabinet","dresser","wardrobe",
    "shelf","table","chair","bed","cradle","hammock","mirror","carpet",
    "curtain","lamp","lantern","pillow","mattress","chest","cupboard","bookcase",
)

CONFIRM_BODY = (
    "ankle","elbow","wrist","knee","shoulder","spine","kidney","liver",
    "lung","heart","stomach","thumb","finger","heel","hip","jaw",
    "neck","rib","skull","tooth","tongue","vein","muscle","nerve",
)
CONFIRM_SUBJECTS = (
    "algebra","biology","physics","chemistry","geography","economics","literature","philosophy",
    "geometry","calculus","ecology","genetics","botany","zoology","linguistics","sociology",
    "psychology","statistics","astronomy","geology","ethics","logic","anthropology","politics",
)
CONFIRM_CLOTHING = (
    "jacket","shirt","pants","skirt","scarf","glove","boot","coat",
    "hat","belt","sock","dress","sweater","vest","blouse","jeans",
    "helmet","sandal","tie","cap","robe","sleeve","uniform","cloak",
)
CONFIRM_COMPUTING = (
    "keyboard","mouse","monitor","router","modem","server","printer","scanner",
    "camera","speaker","microphone","tablet","laptop","desktop","switch","firewall",
    "gateway","browser","console","terminal","sensor","adapter","controller","display",
)


@dataclass(frozen=True)
class AlignmentCase:
    case_id: str
    split: str
    template_id: str
    k: int
    state_text: str
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int


class BilinearAlignmentProbe(nn.Module):
    def __init__(self, d_model: int = D_MODEL, d_align: int = D_ALIGN):
        super().__init__()
        self.context_proj = nn.Linear(d_model * 2, d_align, bias=False)
        self.option_proj = nn.Linear(d_model, d_align, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(
        self,
        state_global: Tensor,
        question: Tensor,
        options: Tensor,
    ) -> Tensor:
        context = torch.cat([state_global, question], dim=-1)
        c = F.normalize(self.context_proj(context), dim=-1)
        o = F.normalize(self.option_proj(options), dim=-1)
        scale = self.log_scale.exp().clamp(0.1, 100.0)
        return torch.einsum("bd,bkd->bk", c, o) * scale


class PairMLPAlignmentProbe(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        d_align: int = D_ALIGN,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_align),
            nn.GELU(),
        )
        self.option_proj = nn.Sequential(
            nn.Linear(d_model, d_align),
            nn.GELU(),
        )
        self.score = nn.Sequential(
            nn.Linear(d_align * 4, d_align),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_align, 1),
        )

    def forward(
        self,
        state_global: Tensor,
        question: Tensor,
        options: Tensor,
    ) -> Tensor:
        context = self.context_proj(
            torch.cat([state_global, question], dim=-1)
        )
        option = self.option_proj(options)
        c = context[:, None, :].expand_as(option)
        features = torch.cat(
            [c, option, c * option, (c - option).abs()],
            dim=-1,
        )
        return self.score(features).squeeze(-1)


def probe_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _vocab(split: str):
    if split in {"train", "dev"}:
        return TRAIN_REGIONS, TRAIN_SPORTS, TRAIN_TRAITS, TRAIN_OBJECTS
    if split == "confirm":
        return CONFIRM_BODY, CONFIRM_SUBJECTS, CONFIRM_CLOTHING, CONFIRM_COMPUTING
    raise ValueError(f"unknown split: {split}")


def all_w5c_vocab() -> set[str]:
    groups = (
        TRAIN_REGIONS, TRAIN_SPORTS, TRAIN_TRAITS, TRAIN_OBJECTS,
        CONFIRM_BODY, CONFIRM_SUBJECTS, CONFIRM_CLOTHING, CONFIRM_COMPUTING,
    )
    return {item for group in groups for item in group}


def _signature_text(signature: tuple[str, str, str, str]) -> str:
    a, b, c, d = signature
    return f"region {a}; sport {b}; trait {c}; object {d}"


def _render_state(
    signature: tuple[str, str, str, str],
    template_id: str,
) -> tuple[str, str]:
    a, b, c, d = signature
    if template_id == "train-envelope":
        state = f"Envelope: region {a}; sport {b}; trait {c}; object {d}."
        question = "Which route matches every field in the envelope?"
    elif template_id == "train-message":
        state = f"Message from {a}: use {b}; noted trait {c}; assigned object {d}."
        question = "Select the route consistent with the whole message."
    elif template_id == "train-index":
        state = f"Index entry [{d}] [{b}] [{a}] [{c}]."
        question = "Which route reconstructs the indexed attributes?"
    elif template_id == "train-briefing":
        state = f"Briefing names {a} and {b}. The descriptor is {c}; object is {d}."
        question = "Which candidate route agrees with the briefing?"
    elif template_id == "dev-summary":
        state = f"Summary: object {d}, trait {c}, sport {b}, region {a}."
        question = "Identify the option that preserves all summary attributes."
    elif template_id == "dev-register":
        state = f"Register places {b} with {d}; region={a}; trait={c}."
        question = "Which route exactly matches the register?"
    elif template_id == "confirm-memo":
        state = f"Memo fields: body {a}; subject {b}; clothing {c}; device {d}."
        question = "Which route agrees with every memo field?"
    elif template_id == "confirm-sheet":
        state = f"Sheet records device {d}, clothing {c}, body {a}, subject {b}."
        question = "Select the fully matching route."
    elif template_id == "confirm-entry":
        state = f"Entry -> subject {b}; body {a}; device {d}; clothing {c}."
        question = "Which candidate reconstructs the entry exactly?"
    else:
        raise ValueError(f"unknown template: {template_id}")
    return state, question


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


def generate_alignment_case(
    *,
    split: str,
    k: int,
    case_index: int,
    rng: random.Random,
    template_id: str,
) -> AlignmentCase:
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
            criterion_text=_signature_text(signature),
        )
        for index, signature in enumerate(signatures)
    )
    state, question = _render_state(target, template_id)
    return AlignmentCase(
        case_id=f"w5c-{split}-{k}-{case_index:04d}",
        split=split,
        template_id=template_id,
        k=k,
        state_text=state,
        question_text=question,
        options=options,
        gold_index=gold_index,
    )


def generate_alignment_authority(split: str) -> list[AlignmentCase]:
    if split == "train":
        counts, seed, templates = TRAIN_K_COUNTS, TRAIN_SEED, TRAIN_TEMPLATES
    elif split == "dev":
        counts, seed, templates = DEV_K_COUNTS, DEV_SEED, DEV_TEMPLATES
    elif split == "confirm":
        counts, seed, templates = CONFIRM_K_COUNTS, CONFIRM_SEED, CONFIRM_TEMPLATES
    else:
        raise ValueError(f"unknown split: {split}")
    rng = random.Random(seed)
    rows: list[AlignmentCase] = []
    index = 0
    for k in sorted(counts):
        for _ in range(counts[k]):
            template = templates[index % len(templates)]
            rows.append(
                generate_alignment_case(
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
def compile_alignment_cache(
    model: NolaneHira,
    cases: Sequence[AlignmentCase],
) -> dict:
    model.eval()
    before = model.state_encode_calls
    rows = []
    for case in cases:
        memory = model.compile_state(case.state_text, segment_tokens=32)
        qbatch = model.encoder.encode_texts([case.question_text])
        obatch = model.encoder.encode_texts(
            [option.criterion_text for option in case.options]
        )
        rows.append({
            "case_id": case.case_id,
            "split": case.split,
            "template_id": case.template_id,
            "k": case.k,
            "gold_index": case.gold_index,
            "state_global": memory.global_embedding.detach().cpu().to(torch.float16),
            "state_tokens": memory.token_embeddings.detach().cpu().to(torch.float16),
            "question_embedding": qbatch.pooled_embeddings[0].detach().cpu().to(torch.float16),
            "option_embeddings": F.normalize(
                obatch.pooled_embeddings, dim=-1
            ).detach().cpu().to(torch.float16),
            "option_tokens": obatch.token_embeddings.detach().cpu().to(torch.float16),
            "option_token_mask": obatch.attention_mask.detach().cpu().bool(),
        })
    state_calls = model.state_encode_calls - before
    if state_calls != len(cases):
        raise RuntimeError("W5c cache violated state-once semantics")
    cache = {
        "schema_version": "r8-w5c-alignment-cache-v1",
        "case_count": len(rows),
        "state_encode_calls": state_calls,
        "state_encode_calls_per_case": state_calls / max(1, len(rows)),
        "cases": rows,
    }
    validate_alignment_cache(cache)
    return cache


def validate_alignment_cache(
    cache: dict,
    *,
    expected_split: str | None = None,
) -> None:
    if cache.get("schema_version") != "r8-w5c-alignment-cache-v1":
        raise ValueError("unexpected W5c cache schema")
    cases = cache.get("cases")
    if not isinstance(cases, list) or len(cases) != cache.get("case_count"):
        raise ValueError("W5c cache case count mismatch")
    seen = set()
    for case in cases:
        case_id = case.get("case_id")
        if not case_id or case_id in seen:
            raise ValueError("invalid or duplicate W5c case id")
        seen.add(case_id)
        if expected_split is not None and case.get("split") != expected_split:
            raise ValueError("W5c cache split mismatch")
        k = int(case["k"])
        if not 2 <= k <= 255:
            raise ValueError("invalid W5c K")
        if case["state_global"].shape != (D_MODEL,):
            raise ValueError("state_global must be [256]")
        if case["state_tokens"].ndim != 2 or case["state_tokens"].shape[-1] != D_MODEL:
            raise ValueError("state_tokens must be [T,256]")
        if case["question_embedding"].shape != (D_MODEL,):
            raise ValueError("question_embedding must be [256]")
        if case["option_embeddings"].shape != (k, D_MODEL):
            raise ValueError("option_embeddings must be [K,256]")
        option_tokens = case["option_tokens"]
        mask = case["option_token_mask"]
        if option_tokens.ndim != 3 or option_tokens.shape[0] != k or option_tokens.shape[-1] != D_MODEL:
            raise ValueError("option_tokens must be [K,T,256]")
        if mask.shape != option_tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("option_token_mask mismatch")
        if not 0 <= int(case["gold_index"]) < k:
            raise ValueError("gold index out of range")


def save_alignment_cache(cache: dict, path) -> None:
    validate_alignment_cache(cache)
    torch.save(cache, path)


def load_alignment_cache(path) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    validate_alignment_cache(cache)
    return cache


def _pooled_cosine_logits(case: dict) -> Tensor:
    state = F.normalize(case["state_global"].float(), dim=-1)
    question = F.normalize(case["question_embedding"].float(), dim=-1)
    context = F.normalize(state + question, dim=-1)
    options = F.normalize(case["option_embeddings"].float(), dim=-1)
    return torch.einsum("d,kd->k", context, options) * 10.0


def _token_max_logits(case: dict) -> Tensor:
    state = F.normalize(case["state_tokens"].float(), dim=-1)
    option = F.normalize(case["option_tokens"].float(), dim=-1)
    mask = case["option_token_mask"]
    similarity = torch.einsum("sd,ktd->kst", state, option)
    best = similarity.max(dim=1).values
    best = best.masked_fill(~mask, 0.0)
    denom = mask.sum(-1).clamp_min(1).to(best.dtype)
    return (best.sum(-1) / denom) * 10.0


def _learned_logits(model: nn.Module, case: dict) -> Tensor:
    return model(
        case["state_global"].float().unsqueeze(0),
        case["question_embedding"].float().unsqueeze(0),
        case["option_embeddings"].float().unsqueeze(0),
    )[0]


def probe_logits(
    probe: ProbeKind,
    case: dict,
    *,
    model: nn.Module | None = None,
) -> Tensor:
    if probe == "pooled_cosine":
        return _pooled_cosine_logits(case)
    if probe == "token_max":
        return _token_max_logits(case)
    if probe in {"bilinear", "pair_mlp"}:
        if model is None:
            raise ValueError("learned probe requires model")
        return _learned_logits(model, case)
    raise ValueError(f"unknown probe: {probe}")


def _hard_brier(probabilities: Tensor, gold_index: int) -> Tensor:
    onehot = torch.zeros_like(probabilities)
    onehot[gold_index] = 1.0
    return ((probabilities - onehot) ** 2).sum()


def learned_probe_loss(
    model: nn.Module,
    case: dict,
) -> Tensor:
    logits = _learned_logits(model, case)
    gold = int(case["gold_index"])
    ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([gold]))
    probabilities = torch.softmax(logits, dim=-1)
    return ce + 0.1 * _hard_brier(probabilities, gold)


@torch.inference_mode()
def evaluate_probe(
    probe: ProbeKind,
    cache: dict,
    *,
    model: nn.Module | None = None,
) -> dict:
    validate_alignment_cache(cache)
    if model is not None:
        model.eval()
    correct = 0
    top5 = 0
    reciprocal: list[float] = []
    brier: list[float] = []
    mass: list[float] = []
    per_k: dict[int, dict] = {}

    for case in cache["cases"]:
        logits = probe_logits(probe, case, model=model)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu()
        gold = int(case["gold_index"])
        order = torch.argsort(probabilities, descending=True)
        pred = int(order[0])
        is_correct = int(pred == gold)
        is_top5 = int(gold in set(order[:5].tolist()))
        rank = int((order == gold).nonzero(as_tuple=False)[0].item()) + 1
        onehot = torch.zeros_like(probabilities)
        onehot[gold] = 1.0
        err = abs(float(probabilities.sum()) - 1.0)

        correct += is_correct
        top5 += is_top5
        reciprocal.append(1.0 / rank)
        brier.append(float(((probabilities - onehot) ** 2).sum()))
        mass.append(err)

        k = int(case["k"])
        slot = per_k.setdefault(
            k,
            {"n": 0, "correct": 0, "top5": 0, "rr": [], "mass": []},
        )
        slot["n"] += 1
        slot["correct"] += is_correct
        slot["top5"] += is_top5
        slot["rr"].append(1.0 / rank)
        slot["mass"].append(err)

    n = len(cache["cases"])
    out_per_k = {
        str(k): {
            "n": slot["n"],
            "accuracy": slot["correct"] / slot["n"],
            "top5_recall": slot["top5"] / slot["n"],
            "mrr": sum(slot["rr"]) / slot["n"],
            "probability_mass_max_error": max(slot["mass"]),
        }
        for k, slot in sorted(per_k.items())
    }
    return {
        "probe": probe,
        "case_count": n,
        "accuracy": correct / n,
        "top5_recall": top5 / n,
        "mrr": sum(reciprocal) / n,
        "hard_brier": sum(brier) / n,
        "probability_mass_max_error": max(mass),
        "per_k": out_per_k,
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


def train_learned_probe(
    model: nn.Module,
    train_cache: dict,
    dev_cache: dict,
    *,
    probe: Literal["bilinear", "pair_mlp"],
    lr: float = 3e-4,
    epochs: int = EPOCHS,
    seed: int = GLOBAL_SEED,
) -> tuple[list[dict], dict[str, Tensor], dict]:
    validate_alignment_cache(train_cache, expected_split="train")
    validate_alignment_cache(dev_cache, expected_split="dev")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=0.01,
    )
    history = []
    best_key = None
    best_state = None
    best_metrics = None
    cases = train_cache["cases"]

    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = list(range(len(cases)))
        random.Random(int(seed) + epoch).shuffle(order)
        loss_sum = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = learned_probe_loss(model, cases[index])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())

        metrics = evaluate_probe(probe, dev_cache, model=model)
        metrics["epoch"] = epoch
        metrics["mean_train_case_loss"] = loss_sum / len(cases)
        history.append(metrics)
        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("W5c learned probe produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics


def competence_gate(metrics: dict) -> dict[str, bool]:
    per = metrics["per_k"]
    return {
        "overall_accuracy": float(metrics["accuracy"]) >= 0.60,
        "k128_accuracy": float(per["128"]["accuracy"]) >= 0.40,
        "k255_accuracy": float(per["255"]["accuracy"]) >= 0.30,
        "k255_top5": float(per["255"]["top5_recall"]) >= 0.70,
        "probability_mass": float(metrics["probability_mass_max_error"]) <= 1e-6,
    }


def classify_confirm(probe_metrics: dict[str, dict]) -> tuple[str, dict[str, dict[str, bool]]]:
    gates = {
        name: competence_gate(metrics)
        for name, metrics in probe_metrics.items()
    }
    passes = {
        name: all(result.values())
        for name, result in gates.items()
    }
    if passes.get("pooled_cosine"):
        verdict = "DIRECT_POOLED_SIGNAL"
    elif passes.get("token_max"):
        verdict = "DIRECT_TOKEN_SIGNAL"
    elif passes.get("bilinear"):
        verdict = "LINEAR_RECOVERABLE_SIGNAL"
    elif passes.get("pair_mlp"):
        verdict = "NONLINEAR_RECOVERABLE_SIGNAL"
    else:
        verdict = "A13_PROBE_FAIL"
    return verdict, gates
