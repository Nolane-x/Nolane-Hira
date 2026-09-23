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

TRAIN_SEED = 101001
DEV_SEED = 102002
CONFIRM_SEED = 103003
GLOBAL_SEED = 311
EPOCHS = 6
LR = 3e-4
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256
MIN_COVERAGE_WEIGHT = 0.5
RAW_LOGIT_SCALE = 10.0

TRAIN_TEMPLATES = (
    "w5f-train-roster",
    "w5f-train-table",
    "w5f-train-note",
    "w5f-train-map",
)
DEV_TEMPLATES = (
    "w5f-dev-slip",
    "w5f-dev-grid",
)
CONFIRM_TEMPLATES = (
    "w5f-confirm-page",
    "w5f-confirm-tag",
    "w5f-confirm-report",
)

TRAIN_PROFESSIONS = (
    "architect","barber","brewer","carpenter","dentist","editor","farmer","florist",
    "geologist","jeweler","librarian","mason","nurse","optician","painter","plumber",
    "potter","ranger","sailor","tailor","teacher","weaver","welder","writer",
)
TRAIN_TREES = (
    "acacia","alder","aspen","birch","cedar","cypress","elm","fir",
    "hemlock","hickory","juniper","larch","maple","mulberry","oak","palm",
    "pine","poplar","redwood","spruce","sycamore","willow","yew","sequoia",
)
TRAIN_CURRENCIES = (
    "baht","dinar","dirham","dollar","dram","euro","forint","franc",
    "hryvnia","krona","krone","lari","leu","lira","manat","naira",
    "peso","pound","rand","rial","ringgit","rupee","shekel","won",
)
TRAIN_DANCES = (
    "ballet","bolero","cancan","conga","foxtrot","gavotte","jig","mambo",
    "minuet","polka","rumba","salsa","samba","tango","twist","waltz",
    "mazurka","flamenco","fandango","tarantella","hornpipe","quadrille","reel","jive",
)

CONFIRM_LANGUAGES = (
    "arabic","bengali","catalan","danish","dutch","estonian","finnish","greek",
    "hebrew","hindi","hungarian","icelandic","indonesian","irish","italian","japanese",
    "korean","latvian","malay","nepali","polish","romanian","swedish","thai",
)
CONFIRM_MAMMALS = (
    "alpaca","armadillo","bison","buffalo","camel","capybara","cheetah","cougar",
    "deer","ferret","gazelle","giraffe","hamster","horse","jaguar","kangaroo",
    "koala","lemur","moose","porcupine","rabbit","raccoon","wombat","zebra",
)
CONFIRM_SPICES = (
    "anise","basil","caraway","cardamom","cinnamon","clove","coriander","cumin",
    "dill","fennel","fenugreek","ginger","mace","mustard","nutmeg","paprika",
    "parsley","rosemary","saffron","sage","sumac","tarragon","turmeric","wasabi",
)
CONFIRM_STYLES = (
    "baroque","bauhaus","brutalist","byzantine","carolingian","classical","colonial","deco",
    "edwardian","georgian","gothic","mannerist","modernist","mughal","neoclassical","palladian",
    "postmodern","renaissance","roman","romanesque","rococo","tudor","vernacular","victorian",
)


@dataclass(frozen=True)
class LateInteractionCase:
    case_id: str
    split: str
    template_id: str
    k: int
    state_text: str
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int


def all_w5f_vocab() -> set[str]:
    groups = (
        TRAIN_PROFESSIONS,
        TRAIN_TREES,
        TRAIN_CURRENCIES,
        TRAIN_DANCES,
        CONFIRM_LANGUAGES,
        CONFIRM_MAMMALS,
        CONFIRM_SPICES,
        CONFIRM_STYLES,
    )
    return {item for group in groups for item in group}


def _vocab(split: str):
    if split in {"train", "dev"}:
        return (
            TRAIN_PROFESSIONS,
            TRAIN_TREES,
            TRAIN_CURRENCIES,
            TRAIN_DANCES,
        )
    if split == "confirm":
        return (
            CONFIRM_LANGUAGES,
            CONFIRM_MAMMALS,
            CONFIRM_SPICES,
            CONFIRM_STYLES,
        )
    raise ValueError(f"unknown W5f split: {split}")


def _signature_text(signature: tuple[str, str, str, str], split: str) -> str:
    a, b, c, d = signature
    if split in {"train", "dev"}:
        return f"profession {a}; tree {b}; currency {c}; dance {d}"
    return f"language {a}; mammal {b}; spice {c}; style {d}"


def _render(
    signature: tuple[str, str, str, str],
    template_id: str,
) -> tuple[str, str]:
    a, b, c, d = signature
    if template_id == "w5f-train-roster":
        return (
            f"Roster fields: profession {a}; tree {b}; currency {c}; dance {d}.",
            "Which route preserves all roster fields?",
        )
    if template_id == "w5f-train-table":
        return (
            f"Table row lists currency={c}, profession={a}, dance={d}, tree={b}.",
            "Select the candidate matching the complete table row.",
        )
    if template_id == "w5f-train-note":
        return (
            f"Note: the {a} entry uses {c}, references {b}, and is paired with {d}.",
            "Which route agrees with every note attribute?",
        )
    if template_id == "w5f-train-map":
        return (
            f"Mapping -> tree {b}; dance {d}; profession {a}; currency {c}.",
            "Identify the fully matching route.",
        )
    if template_id == "w5f-dev-slip":
        return (
            f"Slip records dance {d}, currency {c}, tree {b}, profession {a}.",
            "Which candidate reconstructs the slip?",
        )
    if template_id == "w5f-dev-grid":
        return (
            f"Grid cells: [{a}] [{b}] [{c}] [{d}].",
            "Choose the route consistent with all four grid cells.",
        )
    if template_id == "w5f-confirm-page":
        return (
            f"Page fields: language {a}; mammal {b}; spice {c}; style {d}.",
            "Which route matches the page exactly?",
        )
    if template_id == "w5f-confirm-tag":
        return (
            f"Tag -> style {d}; spice {c}; language {a}; mammal {b}.",
            "Select the candidate agreeing with every tagged field.",
        )
    if template_id == "w5f-confirm-report":
        return (
            f"Report links {b} with {a}; its spice is {c} and style is {d}.",
            "Which route reconstructs the complete report?",
        )
    raise ValueError(f"unknown W5f template: {template_id}")


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


def generate_late_interaction_case(
    *,
    split: str,
    k: int,
    case_index: int,
    rng: random.Random,
    template_id: str,
) -> LateInteractionCase:
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
    return LateInteractionCase(
        case_id=f"w5f-{split}-{k}-{case_index:04d}",
        split=split,
        template_id=template_id,
        k=k,
        state_text=state,
        question_text=question,
        options=options,
        gold_index=gold_index,
    )


def generate_late_interaction_authority(split: str) -> list[LateInteractionCase]:
    if split == "train":
        counts, seed, templates = TRAIN_K_COUNTS, TRAIN_SEED, TRAIN_TEMPLATES
    elif split == "dev":
        counts, seed, templates = DEV_K_COUNTS, DEV_SEED, DEV_TEMPLATES
    elif split == "confirm":
        counts, seed, templates = CONFIRM_K_COUNTS, CONFIRM_SEED, CONFIRM_TEMPLATES
    else:
        raise ValueError(f"unknown W5f split: {split}")
    rng = random.Random(seed)
    rows: list[LateInteractionCase] = []
    index = 0
    for k in sorted(counts):
        for _ in range(counts[k]):
            rows.append(
                generate_late_interaction_case(
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


@torch.inference_mode()
def compile_late_interaction_cache(
    encoder: HFAutoSemanticEncoder,
    cases: Sequence[LateInteractionCase],
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
        batch = encoder.encode_texts(texts)
        encoder_calls += 1
        content = _content_mask(batch.attention_mask.detach().cpu())
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
            "option_mask": content[2:],
            "state_pooled": batch.pooled_embeddings[0].detach().cpu().to(torch.float16),
            "question_pooled": batch.pooled_embeddings[1].detach().cpu().to(torch.float16),
            "option_pooled": batch.pooled_embeddings[2:].detach().cpu().to(torch.float16),
        })
    cache = {
        "schema_version": "r8-w5f-late-interaction-cache-v1",
        "case_count": len(rows),
        "encoder_calls": encoder_calls,
        "state_text_encodes": len(rows),
        "state_text_encodes_per_case": len(rows) / max(1, len(rows)),
        "cases": rows,
    }
    validate_late_interaction_cache(cache)
    return cache


def validate_late_interaction_cache(
    cache: dict,
    *,
    expected_split: str | None = None,
) -> None:
    if cache.get("schema_version") != "r8-w5f-late-interaction-cache-v1":
        raise ValueError("unexpected W5f cache schema")
    cases = cache.get("cases")
    if not isinstance(cases, list) or len(cases) != cache.get("case_count"):
        raise ValueError("W5f cache case count mismatch")
    ids: set[str] = set()
    for case in cases:
        if case["case_id"] in ids:
            raise ValueError("duplicate W5f case id")
        ids.add(case["case_id"])
        if expected_split is not None and case["split"] != expected_split:
            raise ValueError("W5f cache split mismatch")
        k = int(case["k"])
        if not 2 <= k <= 255:
            raise ValueError("invalid W5f K")
        if not 0 <= int(case["gold_index"]) < k:
            raise ValueError("W5f gold index out of range")
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
        if options.ndim != 3 or options.shape[0] != k or options.shape[-1] != 256:
            raise ValueError("option_tokens must be [K,T,256]")
        if option_mask.shape != options.shape[:2] or option_mask.dtype != torch.bool:
            raise ValueError("option_mask mismatch")
        if (option_mask.sum(-1) < 1).any():
            raise ValueError("each option requires content tokens")
        if case["state_pooled"].shape != (256,):
            raise ValueError("state_pooled must be [256]")
        if case["question_pooled"].shape != (256,):
            raise ValueError("question_pooled must be [256]")
        if case["option_pooled"].shape != (k, 256):
            raise ValueError("option_pooled must be [K,256]")


def save_late_interaction_cache(cache: dict, path) -> None:
    validate_late_interaction_cache(cache)
    torch.save(cache, path)


def load_late_interaction_cache(path) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    validate_late_interaction_cache(cache)
    return cache


class LateInteractionMatcher(nn.Module):
    def __init__(self, projection_dim: int | None):
        super().__init__()
        if projection_dim is not None and projection_dim not in {64, 128}:
            raise ValueError("projection_dim must be None, 64, or 128")
        self.projection_dim = projection_dim
        self.projection = (
            None
            if projection_dim is None
            else nn.Linear(256, projection_dim, bias=False)
        )
        if projection_dim is None:
            self.register_parameter("log_scale", None)
        else:
            self.log_scale = nn.Parameter(torch.tensor(math.log(RAW_LOGIT_SCALE)))

    @property
    def candidate_name(self) -> str:
        if self.projection_dim is None:
            return "raw-maxsim"
        return f"proj{self.projection_dim}-maxsim"

    def scale(self) -> Tensor:
        if self.log_scale is None:
            return torch.tensor(RAW_LOGIT_SCALE)
        return self.log_scale.clamp(math.log(0.1), math.log(100.0)).exp()

    def _project(self, x: Tensor) -> Tensor:
        if self.projection is not None:
            x = self.projection(x)
        return F.normalize(x, dim=-1)

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

        context = self._project(context)
        options = self._project(options)

        similarity = torch.einsum("ktd,cd->ktc", options, context)
        similarity = similarity.masked_fill(
            ~context_mask[None, None, :],
            -1e4,
        )
        coverage = similarity.max(dim=-1).values

        valid = option_mask.to(coverage.dtype)
        mean_coverage = (
            (coverage * valid).sum(-1)
            / valid.sum(-1).clamp_min(1.0)
        )
        min_coverage = coverage.masked_fill(
            ~option_mask,
            1e4,
        ).min(dim=-1).values
        score = mean_coverage + MIN_COVERAGE_WEIGHT * min_coverage
        return score * self.scale().to(score.device, score.dtype)


def _hard_brier(probabilities: Tensor, gold_index: int) -> Tensor:
    onehot = torch.zeros_like(probabilities)
    onehot[gold_index] = 1.0
    return ((probabilities - onehot) ** 2).sum()


def matcher_case_loss(matcher: LateInteractionMatcher, case: dict) -> Tensor:
    logits = matcher.forward_case(case)
    gold = torch.tensor([int(case["gold_index"])], dtype=torch.long)
    ce = F.cross_entropy(logits.unsqueeze(0), gold)
    probabilities = torch.softmax(logits, dim=-1)
    return ce + 0.1 * _hard_brier(probabilities, int(case["gold_index"]))


def _accumulate_metrics(
    cases: Sequence[dict],
    logits_fn,
) -> dict[str, object]:
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
        raise ValueError("W5f evaluator requires cases")
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
    matcher: LateInteractionMatcher,
    cache: dict,
) -> dict[str, object]:
    validate_late_interaction_cache(cache)
    matcher.eval()
    return _accumulate_metrics(
        cache["cases"],
        matcher.forward_case,
    )


@torch.inference_mode()
def evaluate_pooled_baseline(cache: dict) -> dict[str, object]:
    validate_late_interaction_cache(cache)

    def logits_fn(case: dict) -> Tensor:
        state = case["state_pooled"].float()
        question = case["question_pooled"].float()
        options = case["option_pooled"].float()
        context = F.normalize(state + question, dim=-1)
        options = F.normalize(options, dim=-1)
        return torch.einsum("d,kd->k", context, options) * RAW_LOGIT_SCALE

    return _accumulate_metrics(cache["cases"], logits_fn)


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


def train_projected_matcher(
    matcher: LateInteractionMatcher,
    train_cache: dict,
    dev_cache: dict,
    *,
    epochs: int = EPOCHS,
    seed: int = GLOBAL_SEED,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object]]:
    if matcher.projection is None:
        raise ValueError("raw matcher is evaluation-only")
    validate_late_interaction_cache(train_cache, expected_split="train")
    validate_late_interaction_cache(dev_cache, expected_split="dev")
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
        raise RuntimeError("W5f candidate produced no checkpoint")
    matcher.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics


def competence_gates(
    selected: dict[str, object],
    pooled: dict[str, object],
) -> dict[str, bool]:
    sel = selected["per_k"]
    base = pooled["per_k"]
    return {
        "overall_accuracy": float(selected["accuracy"]) >= 0.60,
        "k128_accuracy": float(sel["128"]["accuracy"]) >= 0.40,
        "k255_accuracy": float(sel["255"]["accuracy"]) >= 0.30,
        "k255_top5": float(sel["255"]["top5_recall"]) >= 0.70,
        "probability_mass": float(selected["probability_mass_max_error"]) <= 1e-6,
        "overall_gain_vs_pooled": (
            float(selected["accuracy"]) - float(pooled["accuracy"]) >= 0.30
        ),
        "k128_gain_vs_pooled": (
            float(sel["128"]["accuracy"]) > float(base["128"]["accuracy"])
        ),
        "k255_gain_vs_pooled": (
            float(sel["255"]["accuracy"]) > float(base["255"]["accuracy"])
        ),
    }


def confirm_verdict(
    selected: dict[str, object],
    pooled: dict[str, object],
) -> tuple[str, dict[str, bool]]:
    gates = competence_gates(selected, pooled)
    if all(gates.values()):
        return "LATE_INTERACTION_RESCUE", gates
    overall_gain = float(selected["accuracy"]) - float(pooled["accuracy"])
    mrr_gain = float(selected["mrr"]) - float(pooled["mrr"])
    if overall_gain >= 0.10 or mrr_gain >= 0.05:
        return "LATE_INTERACTION_PARTIAL", gates
    return "LATE_INTERACTION_FAIL", gates
