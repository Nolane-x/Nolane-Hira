from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .competitive import CompetitiveCoarseScorer
from .hira import HIRACore, count_parameters
from .losses import LossWeights, typed_decision_loss
from .runtime import NolaneHira, PRIMITIVE_TO_ID
from .typed_competitive_authority import AuthorityCase


CANDIDATES = (
    "legacy-w3-joint",
    "competitive-w5i-joint",
    "competitive-w5i-scorer-only",
)
COMPETITIVE_CANDIDATES = (
    "competitive-w5i-joint",
    "competitive-w5i-scorer-only",
)
GLOBAL_SEED = 809
EPOCHS = 6
LR = 3e-4
WEIGHT_DECAY = 0.01
HIRA_PARAMETER_COUNT = 422_159
SCORER_PARAMETER_COUNT = 32_769
JOINT_PARAMETER_COUNT = HIRA_PARAMETER_COUNT + SCORER_PARAMETER_COUNT

LOSS_WEIGHTS = LossWeights(
    hard_ce=1.0,
    teacher_kl=0.5,
    brier=0.1,
    soft_brier=0.5,
    ordinal_mae=0.2,
)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_case_ids(cases: Sequence[AuthorityCase]) -> str:
    payload = "\n".join(
        sorted(case.typed.case_id for case in cases)
    ).encode("utf-8")
    return sha256(payload).hexdigest()


@torch.inference_mode()
def compile_w6b_cache(
    model: NolaneHira,
    cases: Sequence[AuthorityCase],
    *,
    expected_split: str,
) -> dict:
    model.eval()
    before = model.state_encode_calls
    rows: list[dict] = []

    for authority_case in cases:
        if authority_case.split != expected_split:
            raise ValueError("W6b authority split mismatch")
        typed = authority_case.typed
        memory = model.compile_state(
            typed.state_text,
            segment_tokens=32,
        )
        if memory.content_token_embeddings is None:
            raise RuntimeError(
                "W6b cache requires state content token embeddings"
            )

        decisions: list[dict] = []
        for decision in typed.decisions:
            schema, receipt = model.compile_schema(
                primitive=decision.primitive,
                question_text=decision.question_text,
                options=decision.options,
                use_cache=False,
                include_token_artifacts=True,
            )
            required = (
                schema.question_token_embeddings,
                schema.question_content_token_mask,
                schema.option_token_embeddings,
                schema.option_token_ids,
                schema.option_content_token_mask,
            )
            if any(value is None for value in required):
                raise RuntimeError(
                    "W6b schema token artifacts are incomplete"
                )

            raw_values = [option.value for option in decision.options]
            if decision.primitive == "score":
                if any(value is None for value in raw_values):
                    raise RuntimeError(
                        "W6b score requires explicit numeric option values"
                    )
                score_support = torch.tensor(
                    raw_values,
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
                    "schema_hash": receipt.schema_hash,
                    "question_embedding": (
                        schema.question_embedding.detach().cpu().to(torch.float16)
                    ),
                    "option_embeddings": (
                        schema.option_embeddings.detach().cpu().to(torch.float16)
                    ),
                    "question_tokens": (
                        schema.question_token_embeddings.detach().cpu().to(
                            torch.float16
                        )
                    ),
                    "question_content_mask": (
                        schema.question_content_token_mask.detach().cpu().bool()
                    ),
                    "option_tokens": (
                        schema.option_token_embeddings.detach().cpu().to(
                            torch.float16
                        )
                    ),
                    "option_token_ids": (
                        schema.option_token_ids.detach().cpu().long()
                    ),
                    "option_content_mask": (
                        schema.option_content_token_mask.detach().cpu().bool()
                    ),
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

        rows.append(
            {
                "case_id": typed.case_id,
                "split": authority_case.split,
                "template_id": authority_case.template_id,
                "diagnosis_k": int(authority_case.diagnosis_k),
                "severity": int(authority_case.severity),
                "confidence": authority_case.confidence,
                "state_segments": (
                    memory.segment_embeddings.detach().cpu().to(torch.float16)
                ),
                "state_content_tokens": (
                    memory.content_token_embeddings.detach().cpu().to(
                        torch.float16
                    )
                ),
                "decisions": decisions,
            }
        )

    state_calls = model.state_encode_calls - before
    if state_calls != len(cases):
        raise RuntimeError(
            "W6b cache violated one-state-encode-per-case contract"
        )

    cache = {
        "metadata": {
            "schema_version": "r8-w6b-typed-reliability-cache-v1",
            "split": expected_split,
            "case_count": len(rows),
            "decision_count": sum(
                len(row["decisions"]) for row in rows
            ),
            "case_id_sha256": hash_case_ids(cases),
            "state_encode_calls": state_calls,
            "state_encode_calls_per_case": (
                state_calls / max(1, len(rows))
            ),
            "confirm_exposed": expected_split == "confirm",
        },
        "cases": rows,
    }
    validate_w6b_cache(cache, expected_split=expected_split)
    return cache


def validate_w6b_cache(
    cache: dict,
    *,
    expected_split: str | None = None,
) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W6b cache must be a dict")
    metadata = cache.get("metadata")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(cases, list):
        raise ValueError("W6b cache requires metadata and cases")
    if metadata.get("schema_version") != "r8-w6b-typed-reliability-cache-v1":
        raise ValueError("unexpected W6b cache schema")
    split = metadata.get("split")
    if expected_split is not None and split != expected_split:
        raise ValueError("W6b cache split mismatch")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W6b cache case count mismatch")
    if float(metadata.get("state_encode_calls_per_case", -1.0)) != 1.0:
        raise ValueError("W6b cache must encode state exactly once per case")

    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("invalid or duplicate W6b case id")
        seen.add(case_id)
        if case.get("split") != split:
            raise ValueError("W6b case/cache split mismatch")
        k = int(case.get("diagnosis_k", -1))
        if k not in {8, 16, 32, 64}:
            raise ValueError("invalid W6b diagnosis K")

        segments = case.get("state_segments")
        state_tokens = case.get("state_content_tokens")
        if (
            not isinstance(segments, Tensor)
            or segments.ndim != 2
            or segments.shape[-1] != 256
        ):
            raise ValueError("W6b state_segments must be [S,256]")
        if (
            not isinstance(state_tokens, Tensor)
            or state_tokens.ndim != 2
            or state_tokens.shape[-1] != 256
            or state_tokens.shape[0] < 1
        ):
            raise ValueError(
                "W6b state_content_tokens must be [T,256]"
            )

        decisions = case.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != 5:
            raise ValueError("W6b requires exactly five decisions per case")
        if [decision.get("question_id") for decision in decisions] != [
            "diagnosis",
            "response",
            "needs_review",
            "risk",
            "urgency",
        ]:
            raise ValueError("unexpected W6b decision order")

        for decision in decisions:
            primitive = decision.get("primitive")
            if primitive not in PRIMITIVE_TO_ID:
                raise ValueError("invalid W6b primitive")
            question = decision.get("question_embedding")
            options = decision.get("option_embeddings")
            question_tokens = decision.get("question_tokens")
            question_mask = decision.get("question_content_mask")
            option_tokens = decision.get("option_tokens")
            option_ids = decision.get("option_token_ids")
            option_mask = decision.get("option_content_mask")
            gold_probs = decision.get("gold_probabilities")

            if not isinstance(question, Tensor) or question.shape != (256,):
                raise ValueError("W6b question_embedding must be [256]")
            if (
                not isinstance(options, Tensor)
                or options.ndim != 2
                or options.shape[-1] != 256
                or options.shape[0] < 2
            ):
                raise ValueError("W6b option_embeddings must be [K,256]")
            if (
                not isinstance(question_tokens, Tensor)
                or question_tokens.ndim != 2
                or question_tokens.shape[-1] != 256
            ):
                raise ValueError("W6b question_tokens must be [T,256]")
            if (
                not isinstance(question_mask, Tensor)
                or question_mask.shape != question_tokens.shape[:1]
                or question_mask.dtype != torch.bool
                or int(question_mask.sum()) < 1
            ):
                raise ValueError("W6b question content mask mismatch")
            if (
                not isinstance(option_tokens, Tensor)
                or option_tokens.ndim != 3
                or option_tokens.shape[0] != options.shape[0]
                or option_tokens.shape[-1] != 256
            ):
                raise ValueError("W6b option_tokens must be [K,T,256]")
            if (
                not isinstance(option_ids, Tensor)
                or option_ids.shape != option_tokens.shape[:2]
                or option_ids.dtype != torch.long
            ):
                raise ValueError("W6b option token IDs mismatch")
            if (
                not isinstance(option_mask, Tensor)
                or option_mask.shape != option_tokens.shape[:2]
                or option_mask.dtype != torch.bool
                or (option_mask.sum(-1) < 1).any()
            ):
                raise ValueError("W6b option content mask mismatch")
            if (
                not isinstance(gold_probs, Tensor)
                or gold_probs.shape != (options.shape[0],)
                or not torch.isfinite(gold_probs).all()
                or abs(float(gold_probs.sum()) - 1.0) > 1e-6
            ):
                raise ValueError("W6b gold probability contract failed")
            gold = int(decision.get("gold_index", -1))
            if not 0 <= gold < options.shape[0]:
                raise ValueError("W6b gold index out of range")
            if int(gold_probs.argmax()) != gold:
                raise ValueError("W6b hard/soft gold mismatch")

            support = decision.get("score_support")
            if primitive == "score":
                if (
                    not isinstance(support, Tensor)
                    or support.shape != (options.shape[0],)
                ):
                    raise ValueError("W6b score support mismatch")
                if decision.get("gold_score") is None:
                    raise ValueError("W6b score gold_score missing")


def save_w6b_cache(cache: dict, path: str | Path) -> Path:
    validate_w6b_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w6b_cache(
    path: str | Path,
    *,
    expected_split: str | None = None,
) -> dict:
    cache = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    validate_w6b_cache(cache, expected_split=expected_split)
    return cache


def _cached_forward(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer | None,
    case: dict,
    decision: dict,
    *,
    competitive: bool,
):
    state_segments = case["state_segments"].float().unsqueeze(0)
    question = decision["question_embedding"].float().unsqueeze(0)
    options = decision["option_embeddings"].float().unsqueeze(0)
    qtype = torch.tensor(
        [PRIMITIVE_TO_ID[decision["primitive"]]],
        dtype=torch.long,
    )
    coarse_override = None
    if competitive:
        if scorer is None:
            raise ValueError("competitive cached forward requires scorer")
        state_tokens = case["state_content_tokens"].float().unsqueeze(0)
        question_tokens = decision["question_tokens"].float().unsqueeze(0)
        question_mask = decision[
            "question_content_mask"
        ].bool().unsqueeze(0)
        option_tokens = decision["option_tokens"].float().unsqueeze(0)
        option_ids = decision["option_token_ids"].long().unsqueeze(0)
        option_mask = decision["option_content_mask"].bool().unsqueeze(0)
        coarse_override = scorer(
            state_tokens=state_tokens,
            state_mask=torch.ones(
                1,
                state_tokens.shape[1],
                dtype=torch.bool,
            ),
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_tokens=option_tokens,
            option_token_ids=option_ids,
            option_mask=option_mask,
        )

    return hira(
        question,
        state_segments,
        options,
        qtype,
        coarse_override=coarse_override,
        forced_budget=options.shape[1],
        adaptive_budget=False,
    )


def _loss_cached_case(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer | None,
    case: dict,
    *,
    competitive: bool,
) -> Tensor:
    losses: list[Tensor] = []
    for decision in case["decisions"]:
        out = _cached_forward(
            hira,
            scorer,
            case,
            decision,
            competitive=competitive,
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
            weights=LOSS_WEIGHTS,
        )
        losses.append(loss)
    return torch.stack(losses).mean()


def _ece15(
    confidence: list[float],
    targets: list[float],
) -> float:
    if not confidence:
        return float("nan")
    total = len(confidence)
    result = 0.0
    for index in range(15):
        lo = index / 15
        hi = (index + 1) / 15
        selected = [
            row
            for row, value in enumerate(confidence)
            if value > lo and value <= hi
        ]
        if not selected:
            continue
        mean_conf = sum(confidence[row] for row in selected) / len(selected)
        mean_target = (
            sum(targets[row] for row in selected) / len(selected)
        )
        result += (
            len(selected) / total * abs(mean_conf - mean_target)
        )
    return result


@torch.inference_mode()
def evaluate_w6b_cases(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer | None,
    cases: Iterable[dict],
    *,
    competitive: bool,
) -> dict[str, object]:
    hira.eval()
    if scorer is not None:
        scorer.eval()

    case_count = 0
    decision_count = 0
    correct_total = 0
    primitive_counts = {"choice": 0, "score": 0, "noul": 0}
    primitive_correct = {"choice": 0, "score": 0, "noul": 0}
    diagnosis = {
        8: {"n": 0, "correct": 0},
        16: {"n": 0, "correct": 0},
        32: {"n": 0, "correct": 0},
        64: {"n": 0, "correct": 0},
    }
    soft_accuracy_sum = 0.0
    hard_brier_sum = 0.0
    soft_brier_sum = 0.0
    nll_sum = 0.0
    kl_sum = 0.0
    confidence_values: list[float] = []
    correctness_values: list[float] = []
    soft_target_values: list[float] = []
    score_errors: list[float] = []
    max_mass_error = 0.0

    for case in cases:
        case_count += 1
        for decision in case["decisions"]:
            out = _cached_forward(
                hira,
                scorer,
                case,
                decision,
                competitive=competitive,
            )
            p = out.probabilities[0].detach().cpu().to(torch.float64)
            gold = decision[
                "gold_probabilities"
            ].detach().cpu().to(torch.float64)
            gold_index = int(decision["gold_index"])
            predicted = int(p.argmax())
            is_correct = float(predicted == gold_index)

            decision_count += 1
            correct_total += int(is_correct)
            primitive = decision["primitive"]
            primitive_counts[primitive] += 1
            primitive_correct[primitive] += int(is_correct)

            if decision["question_id"] == "diagnosis":
                slot = diagnosis[int(case["diagnosis_k"])]
                slot["n"] += 1
                slot["correct"] += int(is_correct)

            soft_accuracy_sum += float((p * gold).sum())
            onehot = torch.zeros_like(p)
            onehot[gold_index] = 1.0
            hard_brier_sum += float(((p - onehot) ** 2).sum())
            soft_brier_sum += float(((p - gold) ** 2).sum())
            nll_sum += -math.log(
                max(float(p[gold_index]), 1e-12)
            )
            ratio = gold / p.clamp_min(1e-12)
            kl_sum += float(
                (
                    gold
                    * torch.log(ratio.clamp(min=1e-12, max=1e4))
                ).sum()
            )
            confidence_values.append(float(p.max()))
            correctness_values.append(is_correct)
            soft_target_values.append(float(gold[predicted]))
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
        raise ValueError("W6b evaluator requires cases")

    primitive_accuracy = {
        primitive: primitive_correct[primitive] / primitive_counts[primitive]
        for primitive in primitive_counts
    }
    diagnosis_per_k = {
        str(k): {
            "n": slot["n"],
            "accuracy": slot["correct"] / slot["n"],
        }
        for k, slot in diagnosis.items()
        if slot["n"]
    }
    return {
        "case_count": case_count,
        "decision_count": decision_count,
        "accuracy": correct_total / decision_count,
        "primitive_counts": primitive_counts,
        "primitive_accuracy": primitive_accuracy,
        "diagnosis_per_k": diagnosis_per_k,
        "soft_accuracy": soft_accuracy_sum / decision_count,
        "hard_brier": hard_brier_sum / decision_count,
        "soft_brier": soft_brier_sum / decision_count,
        "nll": nll_sum / decision_count,
        "kl_gold_to_prediction": kl_sum / decision_count,
        "ece": _ece15(confidence_values, correctness_values),
        "raw_ece": _ece15(confidence_values, correctness_values),
        "soft_ece": _ece15(confidence_values, soft_target_values),
        "score_mae": sum(score_errors) / len(score_errors),
        "score_count": len(score_errors),
        "probability_mass_max_error": max_mass_error,
        "source_state_encodes_per_case": 1.0,
        "cached_feature_evaluation": True,
    }


def dev_selection_key(
    metrics: dict[str, object],
    epoch: int,
) -> tuple:
    return (
        -float(metrics["accuracy"]),
        float(metrics["hard_brier"]),
        -float(metrics["soft_accuracy"]),
        -float(metrics["diagnosis_per_k"]["64"]["accuracy"]),
        float(metrics["score_mae"]),
        float(metrics["soft_ece"]),
        int(epoch),
    )


def configure_candidate_trainability(
    candidate: str,
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer | None,
) -> tuple[bool, list[Tensor]]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown W6b candidate: {candidate}")
    competitive = candidate != "legacy-w3-joint"
    if competitive and scorer is None:
        raise ValueError("competitive candidate requires scorer")
    if not competitive and scorer is not None:
        raise ValueError("legacy candidate must not attach scorer")

    if candidate == "competitive-w5i-scorer-only":
        for parameter in hira.parameters():
            parameter.requires_grad_(False)
    else:
        for parameter in hira.parameters():
            parameter.requires_grad_(True)
    if scorer is not None:
        for parameter in scorer.parameters():
            parameter.requires_grad_(True)

    parameters = [
        parameter
        for parameter in (
            list(hira.parameters())
            + ([] if scorer is None else list(scorer.parameters()))
        )
        if parameter.requires_grad
    ]
    return competitive, parameters


def train_w6b_candidate(
    candidate: str,
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer | None,
    train_cases: Sequence[dict],
    dev_cases: Sequence[dict],
    *,
    epochs: int = EPOCHS,
    seed: int = GLOBAL_SEED,
) -> tuple[
    list[dict[str, object]],
    dict[str, Tensor],
    dict[str, Tensor] | None,
    dict[str, object],
]:
    competitive, parameters = configure_candidate_trainability(
        candidate,
        hira,
        scorer,
    )
    if not parameters:
        raise RuntimeError("W6b candidate has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history: list[dict[str, object]] = []
    best_key = None
    best_hira = None
    best_scorer = None
    best_metrics = None

    for epoch in range(1, int(epochs) + 1):
        if candidate == "competitive-w5i-scorer-only":
            hira.eval()
        else:
            hira.train()
        if scorer is not None:
            scorer.train()

        order = list(range(len(train_cases)))
        random.Random(int(seed) + epoch).shuffle(order)
        loss_sum = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_cached_case(
                hira,
                scorer,
                train_cases[index],
                competitive=competitive,
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())

        metrics = evaluate_w6b_cases(
            hira,
            scorer,
            dev_cases,
            competitive=competitive,
        )
        metrics["epoch"] = epoch
        metrics["mean_train_case_loss"] = (
            loss_sum / max(1, len(train_cases))
        )
        history.append(metrics)
        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_hira = {
                name: tensor.detach().cpu().clone()
                for name, tensor in hira.state_dict().items()
            }
            best_scorer = (
                None
                if scorer is None
                else {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in scorer.state_dict().items()
                }
            )

    if best_hira is None or best_metrics is None:
        raise RuntimeError("W6b candidate produced no checkpoint")
    hira.load_state_dict(best_hira, strict=True)
    if scorer is not None:
        if best_scorer is None:
            raise RuntimeError("W6b scorer checkpoint missing")
        scorer.load_state_dict(best_scorer, strict=True)

    return history, best_hira, best_scorer, best_metrics


def absolute_production_gates(metrics: dict[str, object]) -> dict[str, bool]:
    primitive = metrics["primitive_accuracy"]
    return {
        "overall_accuracy": float(metrics["accuracy"]) >= 0.65,
        "choice_accuracy": float(primitive["choice"]) >= 0.65,
        "noul_accuracy": float(primitive["noul"]) >= 0.70,
        "score_accuracy": float(primitive["score"]) >= 0.60,
        "diagnosis_k64_accuracy": (
            float(metrics["diagnosis_per_k"]["64"]["accuracy"]) >= 0.55
        ),
        "hard_brier": float(metrics["hard_brier"]) <= 0.50,
        "soft_ece": float(metrics["soft_ece"]) <= 0.15,
        "score_mae": float(metrics["score_mae"]) <= 0.55,
        "probability_mass": (
            float(metrics["probability_mass_max_error"]) <= 1e-6
        ),
        "state_once": (
            float(metrics["source_state_encodes_per_case"]) == 1.0
        ),
    }


def mechanism_gates(
    selected: dict[str, object],
    legacy: dict[str, object],
) -> dict[str, bool]:
    return {
        "overall_accuracy_gain": (
            float(selected["accuracy"]) - float(legacy["accuracy"]) >= 0.03
        ),
        "diagnosis_k64_gain": (
            float(selected["diagnosis_per_k"]["64"]["accuracy"])
            - float(legacy["diagnosis_per_k"]["64"]["accuracy"])
            >= 0.05
        ),
        "hard_brier_nonregression": (
            float(selected["hard_brier"])
            <= float(legacy["hard_brier"]) + 0.02
        ),
        "soft_ece_nonregression": (
            float(selected["soft_ece"])
            <= float(legacy["soft_ece"]) + 0.02
        ),
        "score_mae_nonregression": (
            float(selected["score_mae"])
            <= float(legacy["score_mae"]) + 0.05
        ),
    }


def confirm_verdict(
    selected: dict[str, object],
    legacy: dict[str, object],
) -> tuple[str, dict[str, bool], dict[str, bool]]:
    absolute = absolute_production_gates(selected)
    control_absolute = absolute_production_gates(legacy)
    mechanism = mechanism_gates(selected, legacy)

    if all(absolute.values()) and all(mechanism.values()):
        return "PRODUCTION_COMPETITIVE_RESCUE", absolute, mechanism
    if all(control_absolute.values()) and not all(mechanism.values()):
        return (
            "PRODUCTION_LEGACY_ALREADY_STRONG",
            absolute,
            mechanism,
        )

    overall_gain = float(selected["accuracy"]) - float(legacy["accuracy"])
    k64_gain = (
        float(selected["diagnosis_per_k"]["64"]["accuracy"])
        - float(legacy["diagnosis_per_k"]["64"]["accuracy"])
    )
    brier_improvement = (
        float(legacy["hard_brier"]) - float(selected["hard_brier"])
    )
    if (
        overall_gain >= 0.02
        or k64_gain >= 0.03
        or brier_improvement >= 0.02
    ):
        return "PRODUCTION_COMPETITIVE_PARTIAL", absolute, mechanism
    return "PRODUCTION_COMPETITIVE_FAIL", absolute, mechanism
