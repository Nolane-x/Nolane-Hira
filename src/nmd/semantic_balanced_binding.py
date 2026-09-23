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


TRAIN_K_COUNTS = {8: 128, 16: 112, 32: 96, 64: 80, 128: 64, 255: 32}
DEV_K_COUNTS = {8: 32, 16: 32, 32: 32, 64: 32, 128: 24, 255: 24}
CONFIRM_K_COUNTS = {32: 48, 64: 48, 128: 48, 255: 48}

TRAIN_SEED = 131001
DEV_SEED = 132002
CONFIRM_SEED = 133003
GLOBAL_SEED = 503
EPOCHS = 6
LR = 3e-4
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256
MIN_COVERAGE_WEIGHT = 0.5
SALIENCE_THRESHOLD = 0.5
INITIAL_LOGIT_SCALE = 10.0

CANDIDATES = (
    "idf-maxsim-proj128",
    "idf-competitive-proj128",
    "idf-greedy-unique-proj128",
)

TRAIN_TEMPLATES = (
    "w5h-train-ledger",
    "w5h-train-dossier",
    "w5h-train-card",
    "w5h-train-index",
)
DEV_TEMPLATES = (
    "w5h-dev-placard",
    "w5h-dev-memo",
)
CONFIRM_TEMPLATES = (
    "w5h-confirm-registry",
    "w5h-confirm-capsule",
    "w5h-confirm-bulletin",
)

TRAIN_BIRDS = (
    "albatross","canary","falcon","heron","ibis","kingfisher","lark","magpie",
    "osprey","partridge","quail","raven","sparrow","toucan","vulture","warbler",
    "woodpecker","crane","cuckoo","egret","finch","gull","hoopoe","jay",
)
TRAIN_MINERALS = (
    "agate","beryl","calcite","dolomite","fluorite","garnet","hematite","jadeite",
    "kyanite","lazurite","magnetite","obsidian","olivine","pyrite","quartz","rutile",
    "serpentine","topaz","turquoise","zircon","malachite","azurite","corundum","feldspar",
)
TRAIN_RIVERS = (
    "amazon","danube","euphrates","ganges","mekong","nile","rhine","seine",
    "thames","volga","yangtze","zambezi","orinoco","parana","congo","indus",
    "loire","rhone","tagus","tigris","yenisei","lena","amur","murray",
)
TRAIN_FABRICS = (
    "brocade","calico","canvas","cashmere","chiffon","corduroy","damask","denim",
    "flannel","gabardine","georgette","gingham","jersey","linen","muslin","organza",
    "satin","suede","taffeta","tweed","velvet","voile","seersucker","poplin",
)

CONFIRM_FISH = (
    "anchovy","barracuda","bonito","carp","catfish","cod","flounder","grouper",
    "haddock","halibut","herring","mackerel","marlin","perch","pike","salmon",
    "sardine","snapper","sturgeon","swordfish","tilapia","trout","tuna","turbot",
)
CONFIRM_FLOWERS = (
    "amaryllis","anemone","aster","azalea","begonia","camellia","carnation","chrysanthemum",
    "crocus","daffodil","dahlia","foxglove","geranium","hibiscus","hyacinth","iris",
    "jasmine","lilac","marigold","peony","petunia","primrose","tulip","zinnia",
)
CONFIRM_MOUNTAINS = (
    "aconcagua","annapurna","denali","elbrus","everest","fuji","kilimanjaro","matterhorn",
    "olympus","vesuvius","etna","rainier","shasta","whitney","kosciuszko","ararat",
    "damavand","montblanc","makalu","manaslu","lhotse","dhaulagiri","chimborazo","cotopaxi",
)
CONFIRM_INSTRUMENTS = (
    "accordion","bagpipe","banjo","bassoon","cello","clarinet","dulcimer","flute",
    "harmonica","harpsichord","mandolin","oboe","piccolo","saxophone","sitar","trombone",
    "trumpet","tuba","ukulele","viola","violin","xylophone","zither","marimba",
)


@dataclass(frozen=True)
class BindingCase:
    case_id: str
    split: str
    template_id: str
    k: int
    state_text: str
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int


def all_w5h_vocab() -> set[str]:
    groups = (
        TRAIN_BIRDS,
        TRAIN_MINERALS,
        TRAIN_RIVERS,
        TRAIN_FABRICS,
        CONFIRM_FISH,
        CONFIRM_FLOWERS,
        CONFIRM_MOUNTAINS,
        CONFIRM_INSTRUMENTS,
    )
    return {item for group in groups for item in group}


def _vocab(split: str):
    if split in {"train", "dev"}:
        return (
            TRAIN_BIRDS,
            TRAIN_MINERALS,
            TRAIN_RIVERS,
            TRAIN_FABRICS,
        )
    if split == "confirm":
        return (
            CONFIRM_FISH,
            CONFIRM_FLOWERS,
            CONFIRM_MOUNTAINS,
            CONFIRM_INSTRUMENTS,
        )
    raise ValueError(f"unknown W5h split: {split}")


def _signature_text(signature: tuple[str, str, str, str], split: str) -> str:
    a, b, c, d = signature
    if split in {"train", "dev"}:
        return f"bird {a}; mineral {b}; river {c}; fabric {d}"
    return f"fish {a}; flower {b}; mountain {c}; instrument {d}"


def _render(
    signature: tuple[str, str, str, str],
    template_id: str,
) -> tuple[str, str]:
    a, b, c, d = signature
    if template_id == "w5h-train-ledger":
        return (
            f"Ledger fields: bird {a}; mineral {b}; river {c}; fabric {d}.",
            "Which candidate preserves every ledger field?",
        )
    if template_id == "w5h-train-dossier":
        return (
            f"Dossier records river={c}, fabric={d}, bird={a}, mineral={b}.",
            "Select the candidate matching the complete dossier.",
        )
    if template_id == "w5h-train-card":
        return (
            f"Card links {a} with {b}; its river is {c} and fabric is {d}.",
            "Which candidate agrees with all four card attributes?",
        )
    if template_id == "w5h-train-index":
        return (
            f"Index -> fabric {d}; bird {a}; mineral {b}; river {c}.",
            "Identify the fully matching indexed candidate.",
        )
    if template_id == "w5h-dev-placard":
        return (
            f"Placard shows mineral {b}, river {c}, fabric {d}, bird {a}.",
            "Which candidate reconstructs the placard exactly?",
        )
    if template_id == "w5h-dev-memo":
        return (
            f"Memo pairs fabric={d} with river={c}; mineral={b}; bird={a}.",
            "Choose the candidate consistent with every memo field.",
        )
    if template_id == "w5h-confirm-registry":
        return (
            f"Registry fields: fish {a}; flower {b}; mountain {c}; instrument {d}.",
            "Which candidate matches the registry in every field?",
        )
    if template_id == "w5h-confirm-capsule":
        return (
            f"Capsule -> instrument {d}; fish {a}; mountain {c}; flower {b}.",
            "Select the candidate preserving the entire capsule signature.",
        )
    if template_id == "w5h-confirm-bulletin":
        return (
            f"Bulletin links {b} to {a}; its instrument is {d} and mountain is {c}.",
            "Which candidate reconstructs all bulletin attributes?",
        )
    raise ValueError(f"unknown W5h template: {template_id}")


def _random_signature(pools, rng: random.Random) -> tuple[str, str, str, str]:
    return tuple(rng.choice(pool) for pool in pools)  # type: ignore[return-value]


def _neighbors(
    target: tuple[str, str, str, str],
    pools,
    changes: int,
) -> list[tuple[str, str, str, str]]:
    if changes not in {1, 2}:
        raise ValueError("changes must be 1 or 2")
    rows: list[tuple[str, str, str, str]] = []
    if changes == 1:
        for i in range(4):
            for value in pools[i]:
                if value == target[i]:
                    continue
                row = list(target)
                row[i] = value
                rows.append(tuple(row))
        return rows
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
                    rows.append(tuple(row))
    return rows


def generate_binding_case(
    *,
    split: str,
    k: int,
    case_index: int,
    rng: random.Random,
    template_id: str,
) -> BindingCase:
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
            option_id=f"candidate-{index:03d}",
            criterion_text=_signature_text(signature, split),
        )
        for index, signature in enumerate(signatures)
    )
    state, question = _render(target, template_id)
    return BindingCase(
        case_id=f"w5h-{split}-{k}-{case_index:04d}",
        split=split,
        template_id=template_id,
        k=k,
        state_text=state,
        question_text=question,
        options=options,
        gold_index=gold_index,
    )


def generate_binding_authority(split: str) -> list[BindingCase]:
    if split == "train":
        counts, seed, templates = TRAIN_K_COUNTS, TRAIN_SEED, TRAIN_TEMPLATES
    elif split == "dev":
        counts, seed, templates = DEV_K_COUNTS, DEV_SEED, DEV_TEMPLATES
    elif split == "confirm":
        counts, seed, templates = CONFIRM_K_COUNTS, CONFIRM_SEED, CONFIRM_TEMPLATES
    else:
        raise ValueError(f"unknown W5h split: {split}")
    rng = random.Random(seed)
    rows: list[BindingCase] = []
    index = 0
    for k in sorted(counts):
        for _ in range(counts[k]):
            rows.append(
                generate_binding_case(
                    split=split,
                    k=k,
                    case_index=index,
                    rng=rng,
                    template_id=templates[index % len(templates)],
                )
            )
            index += 1
    return rows


def _content_mask(attention_mask: Tensor) -> Tensor:
    mask = attention_mask.bool().clone()
    if mask.ndim != 2:
        raise ValueError("attention mask must be [B,T]")
    for row in range(mask.shape[0]):
        valid = mask[row].nonzero(as_tuple=False).flatten()
        if valid.numel() <= 2:
            raise ValueError("text must contain content tokens beyond boundary tokens")
        mask[row, valid[0]] = False
        mask[row, valid[-1]] = False
    return mask


def _option_salience(input_ids: Tensor, option_mask: Tensor) -> Tensor:
    if input_ids.shape != option_mask.shape:
        raise ValueError("input_ids/option_mask shape mismatch")
    k = input_ids.shape[0]
    df: dict[int, int] = {}
    for row in range(k):
        ids = set(input_ids[row][option_mask[row]].tolist())
        for token_id in ids:
            df[int(token_id)] = df.get(int(token_id), 0) + 1
    weights = torch.zeros_like(input_ids, dtype=torch.float32)
    for row in range(k):
        valid = option_mask[row]
        ids = input_ids[row]
        for pos in valid.nonzero(as_tuple=False).flatten().tolist():
            token_id = int(ids[pos])
            weights[row, pos] = math.log((k + 1) / (df[token_id] + 1)) + 1.0
        mean = weights[row][valid].mean()
        if not torch.isfinite(mean) or float(mean) <= 0.0:
            raise ValueError("invalid option salience mean")
        weights[row][valid] /= mean
    return weights


@torch.inference_mode()
def compile_binding_cache(
    encoder: HFAutoSemanticEncoder,
    cases: Sequence[BindingCase],
) -> dict:
    encoder.eval()
    rows = []
    encoder_calls = 0
    for case in cases:
        texts = [
            case.state_text,
            case.question_text,
            *[option.criterion_text for option in case.options],
        ]
        tokenized = encoder.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=encoder.max_length,
            return_tensors="pt",
        )
        batch = encoder.encode_texts(texts)
        encoder_calls += 1
        content = _content_mask(batch.attention_mask.detach().cpu())
        ids = tokenized["input_ids"].detach().cpu()
        option_ids = ids[2:]
        option_mask = content[2:]
        salience = _option_salience(option_ids, option_mask)
        rows.append({
            "case_id": case.case_id,
            "split": case.split,
            "template_id": case.template_id,
            "k": case.k,
            "gold_index": case.gold_index,
            "state_tokens": batch.token_embeddings[0].detach().cpu().to(torch.float16),
            "state_mask": content[0],
            "question_tokens": batch.token_embeddings[1].detach().cpu().to(torch.float16),
            "question_mask": content[1],
            "option_tokens": batch.token_embeddings[2:].detach().cpu().to(torch.float16),
            "option_mask": option_mask,
            "option_input_ids": option_ids,
            "option_salience": salience,
            "state_pooled": batch.pooled_embeddings[0].detach().cpu().to(torch.float16),
            "question_pooled": batch.pooled_embeddings[1].detach().cpu().to(torch.float16),
            "option_pooled": batch.pooled_embeddings[2:].detach().cpu().to(torch.float16),
        })
    cache = {
        "schema_version": "r8-w5h-balanced-binding-cache-v1",
        "case_count": len(rows),
        "encoder_calls": encoder_calls,
        "state_text_encodes": len(rows),
        "state_text_encodes_per_case": len(rows) / max(1, len(rows)),
        "cases": rows,
    }
    validate_binding_cache(cache)
    return cache


def validate_binding_cache(
    cache: dict,
    *,
    expected_split: str | None = None,
) -> None:
    if cache.get("schema_version") != "r8-w5h-balanced-binding-cache-v1":
        raise ValueError("unexpected W5h cache schema")
    cases = cache.get("cases")
    if not isinstance(cases, list) or len(cases) != cache.get("case_count"):
        raise ValueError("W5h cache case count mismatch")
    ids: set[str] = set()
    for case in cases:
        case_id = str(case["case_id"])
        if not case_id or case_id in ids:
            raise ValueError("duplicate/invalid W5h case id")
        ids.add(case_id)
        if expected_split is not None and case["split"] != expected_split:
            raise ValueError("W5h cache split mismatch")
        k = int(case["k"])
        if not 2 <= k <= 255:
            raise ValueError("invalid W5h K")
        if not 0 <= int(case["gold_index"]) < k:
            raise ValueError("W5h gold index out of range")
        for token_key, mask_key in (
            ("state_tokens", "state_mask"),
            ("question_tokens", "question_mask"),
        ):
            tokens = case[token_key]
            mask = case[mask_key]
            if tokens.ndim != 2 or tokens.shape[-1] != 256:
                raise ValueError(f"{token_key} must be [T,256]")
            if mask.shape != tokens.shape[:1] or mask.dtype != torch.bool:
                raise ValueError(f"{mask_key} mismatch")
            if int(mask.sum()) < 1:
                raise ValueError(f"{mask_key} has no content tokens")
        options = case["option_tokens"]
        option_mask = case["option_mask"]
        input_ids = case["option_input_ids"]
        salience = case["option_salience"]
        if options.ndim != 3 or options.shape[0] != k or options.shape[-1] != 256:
            raise ValueError("option_tokens must be [K,T,256]")
        if option_mask.shape != options.shape[:2] or option_mask.dtype != torch.bool:
            raise ValueError("option_mask mismatch")
        if input_ids.shape != option_mask.shape or input_ids.dtype != torch.long:
            raise ValueError("option_input_ids mismatch")
        if salience.shape != option_mask.shape or not torch.is_floating_point(salience):
            raise ValueError("option_salience mismatch")
        if (option_mask.sum(-1) < 1).any():
            raise ValueError("each option requires content tokens")
        if not torch.isfinite(salience).all():
            raise ValueError("non-finite option salience")
        for row in range(k):
            mean = salience[row][option_mask[row]].mean()
            if abs(float(mean) - 1.0) > 1e-5:
                raise ValueError("valid option salience must have mean one")
            if (salience[row][~option_mask[row]] != 0).any():
                raise ValueError("masked option salience must be zero")
        if case["state_pooled"].shape != (256,):
            raise ValueError("state_pooled must be [256]")
        if case["question_pooled"].shape != (256,):
            raise ValueError("question_pooled must be [256]")
        if case["option_pooled"].shape != (k, 256):
            raise ValueError("option_pooled must be [K,256]")


def save_binding_cache(cache: dict, path) -> None:
    validate_binding_cache(cache)
    torch.save(cache, path)


def load_binding_cache(path) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    validate_binding_cache(cache)
    return cache


class BalancedBindingMatcher(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        if mode not in CANDIDATES:
            raise ValueError(f"unknown W5h candidate: {mode}")
        self.mode = mode
        self.projection = nn.Linear(256, 128, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(INITIAL_LOGIT_SCALE)))

    @property
    def candidate_name(self) -> str:
        return self.mode

    def scale(self) -> Tensor:
        return self.log_scale.clamp(math.log(0.1), math.log(100.0)).exp()

    def _project(self, x: Tensor) -> Tensor:
        return F.normalize(self.projection(x), dim=-1)

    def _competitive_coverage(
        self,
        similarity: Tensor,
        option_mask: Tensor,
        context_mask: Tensor,
        salience: Tensor,
    ) -> Tensor:
        weights = salience * option_mask.to(salience.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        common = (similarity * weights[:, :, None]).sum(dim=1) / denom
        adjusted = similarity - common[:, None, :]
        adjusted = adjusted.masked_fill(
            ~context_mask[None, None, :],
            -1e4,
        )
        return adjusted.max(dim=-1).values

    def _greedy_unique_coverage(
        self,
        similarity: Tensor,
        option_mask: Tensor,
        context_mask: Tensor,
        salience: Tensor,
    ) -> Tensor:
        coverage = similarity.max(dim=-1).values.clone()
        context_positions = (
            context_mask.nonzero(as_tuple=False).flatten().tolist()
        )
        for row in range(similarity.shape[0]):
            option_positions = (
                option_mask[row].nonzero(as_tuple=False).flatten().tolist()
            )
            ordered = sorted(
                option_positions,
                key=lambda pos: (-float(salience[row, pos]), int(pos)),
            )
            unused = set(context_positions)
            for pos in ordered:
                if not unused:
                    break
                choices = sorted(unused)
                indices = torch.tensor(
                    choices,
                    dtype=torch.long,
                    device=similarity.device,
                )
                values = similarity[row, pos].index_select(0, indices)
                local = int(values.argmax().item())
                chosen = choices[local]
                coverage[row, pos] = similarity[row, pos, chosen]
                unused.remove(chosen)
        return coverage

    def forward_case(self, case: dict) -> Tensor:
        state = case["state_tokens"].float()
        question = case["question_tokens"].float()
        context = torch.cat([state, question], dim=0)
        context_mask = torch.cat(
            [case["state_mask"], case["question_mask"]],
            dim=0,
        ).bool()
        options = case["option_tokens"].float()
        option_mask = case["option_mask"].bool()
        salience = case["option_salience"].float()

        context = self._project(context)
        options = self._project(options)
        similarity = torch.einsum("ktd,cd->ktc", options, context)
        similarity = similarity.masked_fill(
            ~context_mask[None, None, :],
            -1e4,
        )

        if self.mode == "idf-maxsim-proj128":
            coverage = similarity.max(dim=-1).values
        elif self.mode == "idf-competitive-proj128":
            coverage = self._competitive_coverage(
                similarity,
                option_mask,
                context_mask,
                salience,
            )
        elif self.mode == "idf-greedy-unique-proj128":
            coverage = self._greedy_unique_coverage(
                similarity,
                option_mask,
                context_mask,
                salience,
            )
        else:
            raise AssertionError(self.mode)

        weights = salience * option_mask.to(salience.dtype)
        salient_mask = option_mask & (salience >= SALIENCE_THRESHOLD)
        empty = salient_mask.sum(-1) == 0
        if empty.any():
            salient_mask = salient_mask.clone()
            salient_mask[empty] = option_mask[empty]

        weighted_mean = (
            (coverage * weights).sum(-1)
            / weights.sum(-1).clamp_min(1e-8)
        )
        min_coverage = coverage.masked_fill(
            ~salient_mask,
            1e4,
        ).min(dim=-1).values
        score = weighted_mean + MIN_COVERAGE_WEIGHT * min_coverage
        return score * self.scale().to(score.device, score.dtype)


def _hard_brier(probabilities: Tensor, gold_index: int) -> Tensor:
    onehot = torch.zeros_like(probabilities)
    onehot[gold_index] = 1.0
    return ((probabilities - onehot) ** 2).sum()


def matcher_case_loss(matcher: BalancedBindingMatcher, case: dict) -> Tensor:
    logits = matcher.forward_case(case)
    gold = torch.tensor([int(case["gold_index"])], dtype=torch.long)
    ce = F.cross_entropy(logits.unsqueeze(0), gold)
    probabilities = torch.softmax(logits, dim=-1)
    return ce + 0.1 * _hard_brier(probabilities, int(case["gold_index"]))


def _accumulate_metrics(cases: Sequence[dict], logits_fn) -> dict[str, object]:
    correct = 0
    top5 = 0
    reciprocal: list[float] = []
    brier: list[float] = []
    mass: list[float] = []
    per_k: dict[int, dict] = {}
    for case in cases:
        logits = logits_fn(case)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu()
        gold = int(case["gold_index"])
        order = torch.argsort(probabilities, descending=True)
        pred = int(order[0])
        is_correct = int(pred == gold)
        is_top5 = int(gold in set(order[:5].tolist()))
        rank = int((order == gold).nonzero(as_tuple=False)[0].item()) + 1
        err = abs(float(probabilities.sum()) - 1.0)
        correct += is_correct
        top5 += is_top5
        reciprocal.append(1.0 / rank)
        brier.append(float(_hard_brier(probabilities, gold)))
        mass.append(err)
        slot = per_k.setdefault(
            int(case["k"]),
            {"n": 0, "correct": 0, "top5": 0, "rr": [], "mass": []},
        )
        slot["n"] += 1
        slot["correct"] += is_correct
        slot["top5"] += is_top5
        slot["rr"].append(1.0 / rank)
        slot["mass"].append(err)
    n = len(cases)
    if n == 0:
        raise ValueError("W5h evaluator requires cases")
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


@torch.inference_mode()
def evaluate_matcher(
    matcher: BalancedBindingMatcher,
    cache: dict,
) -> dict[str, object]:
    validate_binding_cache(cache)
    matcher.eval()
    return _accumulate_metrics(cache["cases"], matcher.forward_case)


@torch.inference_mode()
def evaluate_pooled_baseline(cache: dict) -> dict[str, object]:
    validate_binding_cache(cache)

    def logits_fn(case: dict) -> Tensor:
        state = case["state_pooled"].float()
        question = case["question_pooled"].float()
        options = case["option_pooled"].float()
        context = F.normalize(state + question, dim=-1)
        options = F.normalize(options, dim=-1)
        return torch.einsum("d,kd->k", context, options) * INITIAL_LOGIT_SCALE

    return _accumulate_metrics(cache["cases"], logits_fn)


def dev_selection_key(metrics: dict[str, object], epoch: int) -> tuple:
    per = metrics["per_k"]
    return (
        -float(metrics["accuracy"]),
        -float(per["255"]["top5_recall"]),
        -float(per["255"]["accuracy"]),
        -float(per["128"]["accuracy"]),
        -float(metrics["top5_recall"]),
        -float(metrics["mrr"]),
        float(metrics["hard_brier"]),
        int(epoch),
    )


def train_matcher(
    matcher: BalancedBindingMatcher,
    train_cache: dict,
    dev_cache: dict,
    *,
    epochs: int = EPOCHS,
    seed: int = GLOBAL_SEED,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object]]:
    validate_binding_cache(train_cache, expected_split="train")
    validate_binding_cache(dev_cache, expected_split="dev")
    optimizer = torch.optim.AdamW(
        matcher.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_metrics = None
    cases = train_cache["cases"]
    for epoch in range(1, epochs + 1):
        matcher.train()
        order = list(range(len(cases)))
        random.Random(seed + epoch).shuffle(order)
        loss_sum = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = matcher_case_loss(matcher, cases[index])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        metrics = evaluate_matcher(matcher, dev_cache)
        metrics["epoch"] = epoch
        metrics["mean_train_case_loss"] = loss_sum / len(cases)
        history.append(metrics)
        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in matcher.state_dict().items()
            }
    if best_state is None or best_metrics is None:
        raise RuntimeError("W5h candidate produced no checkpoint")
    matcher.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics


def competence_gates(
    selected: dict[str, object],
    control: dict[str, object],
) -> dict[str, bool]:
    sel = selected["per_k"]
    base = control["per_k"]
    return {
        "overall_accuracy": float(selected["accuracy"]) >= 0.60,
        "k128_accuracy": float(sel["128"]["accuracy"]) >= 0.40,
        "k255_accuracy": float(sel["255"]["accuracy"]) >= 0.30,
        "k255_top5": float(sel["255"]["top5_recall"]) >= 0.70,
        "probability_mass": float(selected["probability_mass_max_error"]) <= 1e-6,
        "overall_gain_vs_control": (
            float(selected["accuracy"]) - float(control["accuracy"]) >= 0.05
        ),
        "k255_top5_gain_vs_control": (
            float(sel["255"]["top5_recall"])
            - float(base["255"]["top5_recall"]) >= 0.05
        ),
    }


def confirm_verdict(
    selected: dict[str, object],
    control: dict[str, object],
) -> tuple[str, dict[str, bool]]:
    gates = competence_gates(selected, control)
    competence = (
        gates["overall_accuracy"]
        and gates["k128_accuracy"]
        and gates["k255_accuracy"]
        and gates["k255_top5"]
        and gates["probability_mass"]
    )
    mechanism = (
        gates["overall_gain_vs_control"]
        and gates["k255_top5_gain_vs_control"]
    )
    if competence and mechanism:
        return "BALANCED_BINDING_RESCUE", gates
    if competence and not mechanism:
        return "BALANCED_BINDING_CONTROL_ALREADY_RESCUES", gates
    overall_gain = float(selected["accuracy"]) - float(control["accuracy"])
    k255_top5_gain = (
        float(selected["per_k"]["255"]["top5_recall"])
        - float(control["per_k"]["255"]["top5_recall"])
    )
    mrr_gain = float(selected["mrr"]) - float(control["mrr"])
    if overall_gain >= 0.03 or k255_top5_gain >= 0.03 or mrr_gain >= 0.03:
        return "BALANCED_BINDING_PARTIAL", gates
    return "BALANCED_BINDING_FAIL", gates
