from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import math
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .hira import HIRACore
from .losses import LossWeights, typed_decision_loss
from .runtime import NolaneHira, PRIMITIVE_TO_ID
from .typed_decisions import TypedDecisionCase


W3_DEV_PER_WORKFLOW = 60
W3_TRAIN_PER_WORKFLOW = 240
W3_EXPECTED_WORKFLOW_CASES = 300
W3_EXPECTED_DECISIONS_PER_CASE = 5
W3_HIDDEN_SIZE = 256
W3_GLOBAL_SEED = 71
W3_EPOCHS = 8


def _case_hash(case_id: str) -> str:
    return sha256(
        ("R8-W3-DEV:" + case_id).encode("utf-8")
    ).hexdigest()


def hash_case_ids(cases: Sequence[TypedDecisionCase]) -> str:
    payload = "\n".join(
        sorted(case.case_id for case in cases)
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def split_w3_train_dev(
    cases: Sequence[TypedDecisionCase],
) -> tuple[list[TypedDecisionCase], list[TypedDecisionCase]]:
    by_workflow: dict[str, list[TypedDecisionCase]] = defaultdict(list)
    for case in cases:
        by_workflow[case.workflow].append(case)

    train: list[TypedDecisionCase] = []
    dev: list[TypedDecisionCase] = []
    for workflow in sorted(by_workflow):
        rows = by_workflow[workflow]
        if len(rows) != W3_EXPECTED_WORKFLOW_CASES:
            raise ValueError(
                f"{workflow} must contain exactly "
                f"{W3_EXPECTED_WORKFLOW_CASES} train cases"
            )
        ordered = sorted(
            rows,
            key=lambda case: (_case_hash(case.case_id), case.case_id),
        )
        dev.extend(ordered[:W3_DEV_PER_WORKFLOW])
        train.extend(ordered[W3_DEV_PER_WORKFLOW:])

    if len(train) != W3_TRAIN_PER_WORKFLOW * len(by_workflow):
        raise RuntimeError("W3 train split count mismatch")
    if len(dev) != W3_DEV_PER_WORKFLOW * len(by_workflow):
        raise RuntimeError("W3 dev split count mismatch")
    if {case.case_id for case in train} & {case.case_id for case in dev}:
        raise RuntimeError("W3 train/dev overlap")
    return train, dev


@torch.inference_mode()
def compile_w3_feature_cache(
    model: NolaneHira,
    cases: Sequence[TypedDecisionCase],
    *,
    segment_tokens: int = 32,
) -> dict:
    model.eval()
    train, dev = split_w3_train_dev(cases)
    role_by_id = {
        **{case.case_id: "train" for case in train},
        **{case.case_id: "dev" for case in dev},
    }

    before = model.state_encode_calls
    cached_cases: list[dict] = []
    for case in sorted(cases, key=lambda item: item.case_id):
        memory = model.compile_state(
            case.state_text,
            segment_tokens=segment_tokens,
        )
        decisions: list[dict] = []
        for decision in case.decisions:
            schema, receipt = model.compile_schema(
                primitive=decision.primitive,
                question_text=decision.question_text,
                options=decision.options,
                use_cache=True,
            )
            values = [
                option.value
                for option in decision.options
            ]
            if decision.primitive == "score":
                if any(value is None for value in values):
                    raise RuntimeError(
                        "score cache requires explicit option values"
                    )
                score_support = torch.tensor(
                    values,
                    dtype=torch.float32,
                )
            else:
                score_support = torch.empty(
                    0,
                    dtype=torch.float32,
                )
            decisions.append(
                {
                    "question_id": decision.question_id,
                    "primitive": decision.primitive,
                    "question_embedding": (
                        schema.question_embedding
                        .detach()
                        .cpu()
                        .to(torch.float16)
                    ),
                    "option_embeddings": (
                        schema.option_embeddings
                        .detach()
                        .cpu()
                        .to(torch.float16)
                    ),
                    "schema_hash": receipt.schema_hash,
                    "gold_index": int(decision.gold_index),
                    "gold_probabilities": torch.tensor(
                        decision.gold_probabilities,
                        dtype=torch.float32,
                    ),
                    "gold_score": (
                        None
                        if decision.gold_score is None
                        else float(decision.gold_score)
                    ),
                    "score_support": score_support,
                }
            )
        cached_cases.append(
            {
                "case_id": case.case_id,
                "workflow": case.workflow,
                "role": role_by_id[case.case_id],
                "state_segments": (
                    memory.segment_embeddings
                    .detach()
                    .cpu()
                    .to(torch.float16)
                ),
                "decisions": decisions,
            }
        )

    state_calls = model.state_encode_calls - before
    if state_calls != len(cases):
        raise RuntimeError(
            "feature cache violated one-state-encode-per-case authority"
        )

    result = {
        "metadata": {
            "schema_version": "r8-w3-typed-feature-cache-v1",
            "case_count": len(cases),
            "train_case_count": len(train),
            "dev_case_count": len(dev),
            "decision_count": sum(
                len(case.decisions) for case in cases
            ),
            "state_encode_calls": state_calls,
            "state_encode_calls_per_case": state_calls / len(cases),
            "train_case_id_sha256": hash_case_ids(train),
            "dev_case_id_sha256": hash_case_ids(dev),
            "segment_tokens": int(segment_tokens),
        },
        "cases": cached_cases,
    }
    validate_w3_feature_cache(result)
    return result


def validate_w3_feature_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W3 cache must be a dict")
    metadata = cache.get("metadata")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(cases, list):
        raise ValueError("W3 cache requires metadata and cases")
    if metadata.get("schema_version") != "r8-w3-typed-feature-cache-v1":
        raise ValueError("unexpected W3 cache schema")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W3 cache case count mismatch")
    if not cases:
        raise ValueError("W3 cache is empty")

    roles = {"train": 0, "dev": 0}
    seen_ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen_ids:
            raise ValueError("invalid or duplicate W3 case id")
        seen_ids.add(case_id)
        role = case.get("role")
        if role not in roles:
            raise ValueError("invalid W3 cache role")
        roles[role] += 1
        segments = case.get("state_segments")
        if not isinstance(segments, Tensor):
            raise ValueError("cached state_segments must be a tensor")
        if segments.ndim != 2 or segments.shape[-1] != W3_HIDDEN_SIZE:
            raise ValueError(
                "cached state_segments must be [S,256]"
            )
        decisions = case.get("decisions")
        if (
            not isinstance(decisions, list)
            or len(decisions) != W3_EXPECTED_DECISIONS_PER_CASE
        ):
            raise ValueError(
                "each W3 case must contain exactly five decisions"
            )
        for decision in decisions:
            primitive = decision.get("primitive")
            if primitive not in PRIMITIVE_TO_ID:
                raise ValueError("invalid cached primitive")
            question = decision.get("question_embedding")
            options = decision.get("option_embeddings")
            gold_probs = decision.get("gold_probabilities")
            if (
                not isinstance(question, Tensor)
                or question.shape != (W3_HIDDEN_SIZE,)
            ):
                raise ValueError("question embedding must be [256]")
            if (
                not isinstance(options, Tensor)
                or options.ndim != 2
                or options.shape[-1] != W3_HIDDEN_SIZE
                or options.shape[0] < 2
            ):
                raise ValueError("option embeddings must be [K,256]")
            if (
                not isinstance(gold_probs, Tensor)
                or gold_probs.shape != (options.shape[0],)
            ):
                raise ValueError("gold probabilities must be [K]")
            if not torch.isfinite(gold_probs).all():
                raise ValueError("non-finite cached gold probabilities")
            if abs(float(gold_probs.sum()) - 1.0) > 5e-4:
                raise ValueError("cached gold probability mass mismatch")
            gold_index = int(decision.get("gold_index", -1))
            if not 0 <= gold_index < options.shape[0]:
                raise ValueError("cached gold index out of range")
            support = decision.get("score_support")
            if primitive == "score":
                if (
                    not isinstance(support, Tensor)
                    or support.shape != (options.shape[0],)
                ):
                    raise ValueError(
                        "score support must match option count"
                    )

    if roles["train"] != int(metadata.get("train_case_count", -1)):
        raise ValueError("W3 cached train count mismatch")
    if roles["dev"] != int(metadata.get("dev_case_count", -1)):
        raise ValueError("W3 cached dev count mismatch")


def save_w3_feature_cache(cache: dict, path: str | Path) -> Path:
    validate_w3_feature_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w3_feature_cache(path: str | Path) -> dict:
    cache = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    validate_w3_feature_cache(cache)
    return cache


def cache_role_cases(cache: dict, role: str) -> list[dict]:
    validate_w3_feature_cache(cache)
    if role not in {"train", "dev"}:
        raise ValueError("role must be train or dev")
    return [
        case
        for case in cache["cases"]
        if case["role"] == role
    ]


def forward_w3_cached_decision(
    hira: HIRACore,
    case: dict,
    decision: dict,
):
    segments = case["state_segments"].float().unsqueeze(0)
    question = decision["question_embedding"].float().unsqueeze(0)
    options = decision["option_embeddings"].float().unsqueeze(0)
    qtype = torch.tensor(
        [PRIMITIVE_TO_ID[decision["primitive"]]],
        dtype=torch.long,
    )
    return hira(
        question,
        segments,
        options,
        qtype,
        forced_budget=options.shape[1],
        adaptive_budget=False,
    )


def loss_w3_cached_case(
    hira: HIRACore,
    case: dict,
    *,
    weights: LossWeights,
) -> Tensor:
    losses: list[Tensor] = []
    for decision in case["decisions"]:
        out = forward_w3_cached_decision(
            hira,
            case,
            decision,
        )
        logits = out.logits
        gold = torch.tensor(
            [int(decision["gold_index"])],
            dtype=torch.long,
        )
        teacher = decision[
            "gold_probabilities"
        ].float().unsqueeze(0)
        gold_score = None
        support = None
        if decision["primitive"] == "score":
            gold_score = torch.tensor(
                [float(decision["gold_score"])],
                dtype=logits.dtype,
            )
            support = decision["score_support"].float()
        loss, _ = typed_decision_loss(
            logits,
            gold,
            teacher_probs=teacher,
            gold_score=gold_score,
            score_support=support,
            weights=weights,
        )
        losses.append(loss)
    return torch.stack(losses).mean()


def _ece15(
    confidence: list[float],
    correct: list[float],
) -> float:
    result = 0.0
    n = len(confidence)
    if n == 0:
        return float("nan")
    for index in range(15):
        lo = index / 15
        hi = (index + 1) / 15
        selected = [
            i
            for i, value in enumerate(confidence)
            if value > lo and value <= hi
        ]
        if not selected:
            continue
        mean_conf = sum(
            confidence[i] for i in selected
        ) / len(selected)
        mean_correct = sum(
            correct[i] for i in selected
        ) / len(selected)
        result += (
            len(selected)
            / n
            * abs(mean_conf - mean_correct)
        )
    return result


@torch.inference_mode()
def evaluate_w3_cached_cases(
    hira: HIRACore,
    cases: Iterable[dict],
) -> dict[str, object]:
    hira.eval()
    case_count = 0
    decision_count = 0
    correct_total = 0
    primitive_counts = {"choice": 0, "score": 0, "noul": 0}
    primitive_correct = {"choice": 0, "score": 0, "noul": 0}
    soft_accuracy_sum = 0.0
    hard_brier_sum = 0.0
    soft_brier_sum = 0.0
    nll_sum = 0.0
    kl_sum = 0.0
    confidence: list[float] = []
    correct: list[float] = []
    score_errors: list[float] = []
    max_mass_error = 0.0

    for case in cases:
        case_count += 1
        for decision in case["decisions"]:
            out = forward_w3_cached_decision(
                hira,
                case,
                decision,
            )
            p = out.probabilities[0].detach().cpu().to(torch.float64)
            gold = decision[
                "gold_probabilities"
            ].detach().cpu().to(torch.float64)
            gold_index = int(decision["gold_index"])
            predicted = int(p.argmax().item())
            is_correct = float(predicted == gold_index)

            decision_count += 1
            correct_total += int(is_correct)
            primitive = decision["primitive"]
            primitive_counts[primitive] += 1
            primitive_correct[primitive] += int(is_correct)

            soft_accuracy_sum += float((p * gold).sum())
            onehot = torch.zeros_like(p)
            onehot[gold_index] = 1.0
            hard_brier_sum += float(((p - onehot) ** 2).sum())
            soft_brier_sum += float(((p - gold) ** 2).sum())
            nll_sum += -math.log(max(float(p[gold_index]), 1e-12))
            ratio = gold / p.clamp_min(1e-12)
            kl_sum += float(
                (
                    gold
                    * torch.log(ratio.clamp(min=1e-12, max=1e4))
                ).sum()
            )
            confidence.append(float(p.max()))
            correct.append(is_correct)
            max_mass_error = max(
                max_mass_error,
                abs(float(p.sum()) - 1.0),
            )

            if primitive == "score":
                support = decision["score_support"].to(
                    dtype=p.dtype,
                )
                predicted_score = float((p * support).sum())
                score_errors.append(
                    abs(
                        predicted_score
                        - float(decision["gold_score"])
                    )
                )

    if case_count == 0 or decision_count == 0:
        raise ValueError("W3 evaluator requires cases")

    primitive_accuracy = {
        primitive: (
            primitive_correct[primitive] / primitive_counts[primitive]
            if primitive_counts[primitive]
            else None
        )
        for primitive in primitive_counts
    }
    score_mae = (
        sum(score_errors) / len(score_errors)
        if score_errors
        else None
    )
    return {
        "case_count": case_count,
        "decision_count": decision_count,
        "accuracy": correct_total / decision_count,
        "primitive_counts": primitive_counts,
        "primitive_accuracy": primitive_accuracy,
        "soft_accuracy": soft_accuracy_sum / decision_count,
        "hard_brier": hard_brier_sum / decision_count,
        "soft_brier": soft_brier_sum / decision_count,
        "nll": nll_sum / decision_count,
        "kl_gold_to_prediction": kl_sum / decision_count,
        "ece": _ece15(confidence, correct),
        "score_mae": score_mae,
        "score_count": len(score_errors),
        "probability_mass_max_error": max_mass_error,
        "cached_feature_evaluation": True,
        "state_encode_calls_during_eval": 0,
        "source_state_encode_calls_per_case": 1.0,
        "decisions_per_source_state_encode": decision_count / case_count,
    }


def dev_selection_key(
    metrics: dict[str, object],
    *,
    epoch: int,
) -> tuple:
    score_mae = metrics.get("score_mae")
    return (
        -float(metrics["accuracy"]),
        float(metrics["hard_brier"]),
        -float(metrics["soft_accuracy"]),
        float("inf") if score_mae is None else float(score_mae),
        float(metrics["ece"]),
        int(epoch),
    )


def train_w3_candidate(
    hira: HIRACore,
    train_cases: Sequence[dict],
    dev_cases: Sequence[dict],
    *,
    weights: LossWeights,
    lr: float,
    epochs: int = W3_EPOCHS,
    seed: int = W3_GLOBAL_SEED,
    weight_decay: float = 0.01,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object]]:
    optimizer = torch.optim.AdamW(
        hira.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_metrics = None

    for epoch in range(1, int(epochs) + 1):
        hira.train()
        order = list(range(len(train_cases)))
        random.Random(int(seed) + epoch).shuffle(order)
        loss_sum = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_w3_cached_case(
                hira,
                train_cases[index],
                weights=weights,
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())

        metrics = evaluate_w3_cached_cases(
            hira,
            dev_cases,
        )
        metrics["epoch"] = epoch
        metrics["mean_train_case_loss"] = (
            loss_sum / max(1, len(train_cases))
        )
        history.append(metrics)

        key = dev_selection_key(metrics, epoch=epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in hira.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("W3 candidate produced no checkpoint")
    hira.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics
