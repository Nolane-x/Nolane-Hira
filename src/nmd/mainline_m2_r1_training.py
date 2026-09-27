from __future__ import annotations

from copy import deepcopy
import random
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from .mainline import HiraV0Mainline
from .mainline_m2_authority import M2HighKSemanticCase
from .w33_coevidence_semantic import CoEvidenceSemanticScorer
from .w34_transfer_core import W34_CANDIDATE_PARAMETER_COUNT

M2_R1_CACHE_SCHEMA = "hira-v0-mainline-m2-r1-cache-v1"

M2_R1_CANDIDATES = (
    "ce-primary",
    "ce-replica",
    "ce-margin-primary",
    "ce-margin-replica",
)

M2_R1_SEEDS = {
    "ce-primary": 22031,
    "ce-replica": 22037,
    "ce-margin-primary": 22043,
    "ce-margin-replica": 22051,
}

M2_R1_EPOCHS = 8
M2_R1_LR = 2e-4
M2_R1_WEIGHT_DECAY = 0.01
M2_R1_GRAD_CLIP = 1.0
M2_R1_TEMPERATURE = 0.07
M2_R1_PAIR_MARGIN = 0.06
M2_R1_PAIR_MARGIN_WEIGHT = 0.25

M2_R1_PRIMARY_GATES = {
    64: {"top1": 0.70, "top5": 0.90, "mrr": 0.75},
    128: {"top1": 0.60, "top5": 0.85, "mrr": 0.65},
}
M2_R1_REPLICA_GATES = {
    64: {"top1": 0.65, "top5": 0.85, "mrr": 0.70},
    128: {"top1": 0.55, "top5": 0.80, "mrr": 0.60},
}

CANDIDATE_PREFIXES = (
    "state_adapter.",
    "schema_adapter.",
    "interaction_state.",
    "interaction_schema.",
    "composition_state.",
    "composition_schema.",
)


def candidate_family(candidate: str) -> str:
    if candidate not in M2_R1_CANDIDATES:
        raise ValueError(f"unknown M2-R1 candidate: {candidate}")
    return "ce-margin" if candidate.startswith("ce-margin") else "ce"


def candidate_seed(candidate: str) -> int:
    if candidate not in M2_R1_SEEDS:
        raise ValueError(f"unknown M2-R1 candidate: {candidate}")
    return int(M2_R1_SEEDS[candidate])


@torch.inference_mode()
def compile_m2_r1_cache(
    model: HiraV0Mainline,
    rows: Sequence[M2HighKSemanticCase],
    *,
    split: str,
) -> dict[str, object]:
    if not rows:
        raise ValueError("M2-R1 cache requires rows")
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M2-R1 cache requires frozen mainline")

    before = model.runtime.state_encode_calls
    cached: list[dict[str, object]] = []

    for row in rows:
        memory = model.runtime.compile_state(row.state_text, segment_tokens=32)
        if memory.content_token_embeddings is None:
            raise RuntimeError("M2-R1 cache requires state content tokens")

        schema, _ = model.runtime.compile_schema(
            primitive="choice",
            question_text=row.question_text,
            options=row.options,
            use_cache=False,
            include_token_artifacts=True,
        )
        required = (
            schema.option_view_token_embeddings,
            schema.option_view_token_mask,
            schema.option_view_mask,
        )
        if any(value is None for value in required):
            raise RuntimeError("M2-R1 cache schema token artifacts incomplete")

        cached.append(
            {
                "case_id": row.case_id,
                "split": split,
                "domain_id": row.domain_id,
                "k": int(row.k),
                "gold_index": int(row.gold_index),
                "option_ids": tuple(option.option_id for option in row.options),
                "state_tokens": (
                    memory.content_token_embeddings.detach().cpu().float()
                ),
                "option_view_tokens": (
                    schema.option_view_token_embeddings.detach().cpu().float()
                ),
                "option_view_token_mask": (
                    schema.option_view_token_mask.detach().cpu().bool()
                ),
                "option_view_mask": (
                    schema.option_view_mask.detach().cpu().bool()
                ),
            }
        )

    state_delta = model.runtime.state_encode_calls - before
    if state_delta != len(rows):
        raise RuntimeError("M2-R1 cache violated state-once compilation")

    cache = {
        "metadata": {
            "schema_version": M2_R1_CACHE_SCHEMA,
            "split": split,
            "case_count": len(cached),
            "state_encode_count": state_delta,
            "state_encodes_per_case": 1.0,
            "decision_core_frozen": True,
            "encoder_gradient_updates": False,
            "projection_gradient_updates": False,
        },
        "cases": cached,
    }
    validate_m2_r1_cache(cache, expected_split=split)
    return cache


def validate_m2_r1_cache(
    cache: Mapping[str, object],
    *,
    expected_split: str | None = None,
) -> None:
    metadata = cache.get("metadata")
    cases = cache.get("cases")
    if not isinstance(metadata, Mapping) or not isinstance(cases, list):
        raise ValueError("M2-R1 cache requires metadata/cases")
    if metadata.get("schema_version") != M2_R1_CACHE_SCHEMA:
        raise ValueError("unexpected M2-R1 cache schema")
    if expected_split is not None and metadata.get("split") != expected_split:
        raise ValueError("M2-R1 cache split mismatch")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("M2-R1 cache case count mismatch")
    if float(metadata.get("state_encodes_per_case", -1.0)) != 1.0:
        raise ValueError("M2-R1 cache state-once contract changed")
    if metadata.get("decision_core_frozen") is not True:
        raise ValueError("M2-R1 cache decision core must be frozen")

    seen = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("M2-R1 cache duplicate/invalid case id")
        seen.add(case_id)
        k = int(case.get("k", -1))
        gold = int(case.get("gold_index", -1))
        state = case.get("state_tokens")
        options = case.get("option_view_tokens")
        token_mask = case.get("option_view_token_mask")
        view_mask = case.get("option_view_mask")
        if not 2 <= k <= 255 or not 0 <= gold < k:
            raise ValueError("M2-R1 cache K/gold contract changed")
        if not isinstance(state, Tensor) or state.ndim != 2 or state.shape[-1] != 256:
            raise ValueError("M2-R1 state tokens must be [S,256]")
        if (
            not isinstance(options, Tensor)
            or options.ndim != 4
            or options.shape[0] != k
            or options.shape[-1] != 256
        ):
            raise ValueError("M2-R1 option views must be [K,V,T,256]")
        if (
            not isinstance(token_mask, Tensor)
            or token_mask.shape != options.shape[:3]
            or token_mask.dtype != torch.bool
        ):
            raise ValueError("M2-R1 option token mask mismatch")
        if (
            not isinstance(view_mask, Tensor)
            or view_mask.shape != options.shape[:2]
            or view_mask.dtype != torch.bool
        ):
            raise ValueError("M2-R1 option view mask mismatch")
        if not bool(torch.isfinite(state).all() and torch.isfinite(options).all()):
            raise ValueError("M2-R1 cache contains non-finite embeddings")


def _candidate_parameters(scorer: CoEvidenceSemanticScorer) -> list[nn.Parameter]:
    modules = (
        scorer.state_adapter,
        scorer.schema_adapter,
        scorer.interaction_state,
        scorer.interaction_schema,
        scorer.composition_state,
        scorer.composition_schema,
    )
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def clone_w34_for_rescue(
    frozen_w34: CoEvidenceSemanticScorer,
) -> CoEvidenceSemanticScorer:
    scorer = CoEvidenceSemanticScorer(
        d_model=frozen_w34.d_model,
        d_rel=frozen_w34.d_rel,
        rank=frozen_w34.rank,
    )
    scorer.load_state_dict(
        {
            name: value.detach().cpu().clone()
            for name, value in frozen_w34.state_dict().items()
        },
        strict=True,
    )
    scorer.projection.weight.requires_grad_(False)
    for prefix in CANDIDATE_PREFIXES:
        module_name = prefix[:-1]
        module = getattr(scorer, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    if scorer.candidate_parameter_count != W34_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("M2-R1 W34 candidate parameter count changed")
    if scorer.candidate_trainable_parameter_count != W34_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("M2-R1 must train exactly 8,192 parameters")
    if scorer.trainable_parameter_count != W34_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("M2-R1 unexpected trainable parameter surface")
    return scorer


def candidate_state_dict(
    scorer: CoEvidenceSemanticScorer,
) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in scorer.state_dict().items()
        if name.startswith(CANDIDATE_PREFIXES)
    }


def _case_logits(
    scorer: CoEvidenceSemanticScorer,
    case: Mapping[str, object],
) -> Tensor:
    state = case["state_tokens"].float()
    options = case["option_view_tokens"].float()
    token_mask = case["option_view_token_mask"].bool()
    view_mask = case["option_view_mask"].bool()
    return scorer(
        state_tokens=state.unsqueeze(0),
        state_mask=torch.ones(
            1,
            state.shape[0],
            dtype=torch.bool,
            device=state.device,
        ),
        option_view_tokens=options.unsqueeze(0),
        option_view_token_mask=token_mask.unsqueeze(0),
        option_view_mask=view_mask.unsqueeze(0),
    )[0]


@torch.inference_mode()
def evaluate_m2_r1_cache(
    cache: Mapping[str, object],
    scorer: CoEvidenceSemanticScorer,
) -> dict[str, object]:
    validate_m2_r1_cache(cache)
    scorer.eval()
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for case in cache["cases"]:
        grouped.setdefault(int(case["k"]), []).append(case)

    per_k: dict[str, dict[str, float]] = {}
    for k, rows in sorted(grouped.items()):
        correct = 0
        top5 = 0
        reciprocal = []
        ranks = []
        confidence = []
        gold_probability = []
        for case in rows:
            logits = _case_logits(scorer, case)
            probabilities = torch.softmax(logits, dim=-1)
            order = torch.argsort(probabilities, descending=True)
            gold = int(case["gold_index"])
            rank = int((order == gold).nonzero(as_tuple=False)[0]) + 1
            correct += int(rank == 1)
            top5 += int(rank <= min(5, k))
            reciprocal.append(1.0 / rank)
            ranks.append(float(rank))
            confidence.append(float(probabilities.max()))
            gold_probability.append(float(probabilities[gold]))
        n = len(rows)
        per_k[str(k)] = {
            "case_count": float(n),
            "top1": correct / n,
            "top5": top5 / n,
            "mrr": sum(reciprocal) / n,
            "mean_gold_rank": sum(ranks) / n,
            "mean_confidence": sum(confidence) / n,
            "mean_gold_probability": sum(gold_probability) / n,
        }
    return {"case_count": len(cache["cases"]), "per_k": per_k}


def _selection_key(metrics: Mapping[str, object], epoch: int) -> tuple[float, ...]:
    per_k = metrics["per_k"]
    k64 = per_k["64"]
    k128 = per_k["128"]
    return (
        float(k128["top1"]),
        float(k64["top1"]),
        float(k128["top5"]),
        float(k64["top5"]),
        float(k128["mrr"]),
        float(k64["mrr"]),
        -float(epoch),
    )


def _train_loss(
    logits: Tensor,
    gold: int,
    *,
    use_margin: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    ce = F.cross_entropy(
        (logits / M2_R1_TEMPERATURE).unsqueeze(0),
        torch.tensor([gold], dtype=torch.long, device=logits.device),
    )
    negative = logits.clone()
    negative[gold] = -torch.inf
    best_negative = negative.max()
    margin = F.relu(
        logits.new_tensor(M2_R1_PAIR_MARGIN)
        - (logits[gold] - best_negative)
    )
    total = ce + (M2_R1_PAIR_MARGIN_WEIGHT * margin if use_margin else 0.0)
    return total, ce, margin


def train_m2_r1_candidate(
    candidate: str,
    frozen_w34: CoEvidenceSemanticScorer,
    train_cache: Mapping[str, object],
    dev_cache: Mapping[str, object],
    *,
    epochs: int = M2_R1_EPOCHS,
) -> dict[str, object]:
    validate_m2_r1_cache(train_cache, expected_split="train")
    validate_m2_r1_cache(dev_cache, expected_split="dev")
    if candidate not in M2_R1_CANDIDATES:
        raise ValueError(f"unknown M2-R1 candidate: {candidate}")

    seed = candidate_seed(candidate)
    random.seed(seed)
    torch.manual_seed(seed)

    scorer = clone_w34_for_rescue(frozen_w34)
    parameters = _candidate_parameters(scorer)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=M2_R1_LR,
        weight_decay=M2_R1_WEIGHT_DECAY,
    )
    use_margin = candidate_family(candidate) == "ce-margin"

    best_key = None
    best_state = None
    best_epoch = None
    best_dev = None
    history = []
    cases = train_cache["cases"]

    if int(epochs) < 1:
        raise ValueError("M2-R1 epochs must be >=1")

    for epoch in range(1, int(epochs) + 1):
        scorer.train()
        order = list(range(len(cases)))
        random.Random(seed + epoch).shuffle(order)
        total_sum = 0.0
        ce_sum = 0.0
        margin_sum = 0.0

        for index in order:
            case = cases[index]
            optimizer.zero_grad(set_to_none=True)
            logits = _case_logits(scorer, case)
            total, ce, margin = _train_loss(
                logits,
                int(case["gold_index"]),
                use_margin=use_margin,
            )
            total.backward()
            if scorer.projection.weight.grad is not None:
                raise RuntimeError("M2-R1 frozen W28 projection received gradient")
            torch.nn.utils.clip_grad_norm_(parameters, M2_R1_GRAD_CLIP)
            optimizer.step()
            total_sum += float(total.detach())
            ce_sum += float(ce.detach())
            margin_sum += float(margin.detach())

        dev = evaluate_m2_r1_cache(dev_cache, scorer)
        key = _selection_key(dev, epoch)
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": total_sum / len(cases),
                "train_ce": ce_sum / len(cases),
                "train_pair_margin": margin_sum / len(cases),
                "dev": deepcopy(dev),
                "selection_key": list(key),
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = candidate_state_dict(scorer)
            best_epoch = epoch
            best_dev = deepcopy(dev)

    if best_state is None or best_epoch is None or best_dev is None:
        raise RuntimeError("M2-R1 candidate produced no selected checkpoint")
    return {
        "candidate": candidate,
        "family": candidate_family(candidate),
        "seed": seed,
        "parameter_count": W34_CANDIDATE_PARAMETER_COUNT,
        "selected_epoch": best_epoch,
        "selected_dev": best_dev,
        "selected_state_dict": best_state,
        "selection_key": list(_selection_key(best_dev, best_epoch)),
        "history": history,
    }


def _semantic_gate(
    evaluation: Mapping[str, object],
    thresholds: Mapping[int, Mapping[str, float]],
) -> dict[str, object]:
    per_k = evaluation["per_k"]
    result = {}
    passed = True
    for k, gate in thresholds.items():
        metrics = per_k[str(k)]
        semantic = {
            "top1": float(metrics["top1"]) >= float(gate["top1"]),
            "top5": float(metrics["top5"]) >= float(gate["top5"]),
            "mrr": float(metrics["mrr"]) >= float(gate["mrr"]),
        }
        k_pass = all(semantic.values())
        result[str(k)] = {
            "pass": k_pass,
            "semantic": semantic,
            "thresholds": dict(gate),
        }
        passed = passed and k_pass
    return {"pass": passed, "per_k": result}


def primary_cached_gate(evaluation: Mapping[str, object]) -> dict[str, object]:
    return _semantic_gate(evaluation, M2_R1_PRIMARY_GATES)


def replica_cached_gate(evaluation: Mapping[str, object]) -> dict[str, object]:
    return _semantic_gate(evaluation, M2_R1_REPLICA_GATES)



def install_m2_r1_candidate(
    model: HiraV0Mainline,
    state_dict: Mapping[str, Tensor],
) -> None:
    scorer = model.runtime.coevidence_symmetric_semantic_scorer
    if scorer is None:
        raise RuntimeError("M2-R1 mainline co-evidence scorer missing")
    scorer.load_candidate_state_dict(dict(state_dict), freeze=True)
    if scorer.trainable_parameter_count != 0:
        raise RuntimeError("M2-R1 installed scorer must be frozen")
    if scorer.candidate_parameter_count != W34_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("M2-R1 installed candidate capacity changed")


def _runtime_mechanics_gate(metrics: Mapping[str, object]) -> dict[str, bool]:
    return {
        "probability_mass": float(metrics["probability_mass_max_error"]) <= 1e-6,
        "permutation": (
            float(metrics["permutation_max_error"]) <= 2e-6
            and float(metrics["selected_option_invariant_rate"]) == 1.0
        ),
        "full_k": float(metrics["full_k_rate"]) == 1.0,
        "state_once": float(metrics["state_once_rate"]) == 1.0,
        "relation_off": float(metrics["relation_delta_max"]) == 0.0,
        "finite": float(metrics["finite_rate"]) == 1.0,
    }


def m2_r1_runtime_gate(
    evaluation: Mapping[str, object],
    *,
    replica: bool = False,
) -> dict[str, object]:
    thresholds = M2_R1_REPLICA_GATES if replica else M2_R1_PRIMARY_GATES
    per_k = evaluation["per_k"]
    result: dict[str, object] = {}
    passed = True
    for k, gate in thresholds.items():
        metrics = per_k[str(k)]
        mechanics = _runtime_mechanics_gate(metrics)
        semantic = {
            "top1": float(metrics["top1"]) >= float(gate["top1"]),
            "top5": float(metrics["top5"]) >= float(gate["top5"]),
            "mrr": float(metrics["mrr"]) >= float(gate["mrr"]),
        }
        k_pass = all(mechanics.values()) and all(semantic.values())
        result[str(k)] = {
            "pass": k_pass,
            "mechanics": mechanics,
            "semantic": semantic,
            "thresholds": dict(gate),
        }
        passed = passed and k_pass
    return {"pass": passed, "per_k": result}

def select_m2_r1_family(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_name = {str(row["candidate"]): row for row in results}
    expected = set(M2_R1_CANDIDATES)
    if set(by_name) != expected:
        raise ValueError("M2-R1 candidate tournament is incomplete")

    primaries = [by_name["ce-primary"], by_name["ce-margin-primary"]]
    selected_primary = max(
        primaries,
        key=lambda row: tuple(float(x) for x in row["selection_key"]),
    )
    family = str(selected_primary["family"])
    selected_replica = by_name[f"{family}-replica"]

    primary_gate = primary_cached_gate(selected_primary["selected_dev"])
    replica_gate = replica_cached_gate(selected_replica["selected_dev"])
    return {
        "family": family,
        "primary_candidate": selected_primary["candidate"],
        "replica_candidate": selected_replica["candidate"],
        "primary": selected_primary,
        "replica": selected_replica,
        "primary_cached_gate": primary_gate,
        "replica_cached_gate": replica_gate,
        "cached_dev_qualified": bool(primary_gate["pass"] and replica_gate["pass"]),
    }


__all__ = [
    "CANDIDATE_PREFIXES",
    "M2_R1_CACHE_SCHEMA",
    "M2_R1_CANDIDATES",
    "M2_R1_EPOCHS",
    "M2_R1_GRAD_CLIP",
    "M2_R1_LR",
    "M2_R1_PAIR_MARGIN",
    "M2_R1_PAIR_MARGIN_WEIGHT",
    "M2_R1_PRIMARY_GATES",
    "M2_R1_REPLICA_GATES",
    "M2_R1_SEEDS",
    "M2_R1_TEMPERATURE",
    "M2_R1_WEIGHT_DECAY",
    "candidate_family",
    "candidate_seed",
    "candidate_state_dict",
    "clone_w34_for_rescue",
    "compile_m2_r1_cache",
    "evaluate_m2_r1_cache",
    "install_m2_r1_candidate",
    "m2_r1_runtime_gate",
    "primary_cached_gate",
    "replica_cached_gate",
    "select_m2_r1_family",
    "train_m2_r1_candidate",
    "validate_m2_r1_cache",
]
