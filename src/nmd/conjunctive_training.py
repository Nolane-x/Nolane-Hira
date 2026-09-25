from __future__ import annotations

import math
import random
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .competitive import CompetitiveCoarseScorer
from .conjunctive_coarse import (
    CONJUNCTIVE_TOTAL_PARAMETER_COUNT,
    ConjunctiveEvidenceOutput,
    ConjunctiveEvidenceScorer,
)
from .hira import HIRACore
from .losses import typed_decision_loss
from .runtime import PRIMITIVE_TO_ID
from .typed_competitive_cache import (
    LOSS_WEIGHTS,
    SCORER_PARAMETER_COUNT,
    _ece15,
)


CANDIDATES = (
    "frozen-w6e-control",
    "freeform-retune-control",
    "conjunctive-primary",
    "conjunctive-replica",
)

EPOCHS = 6
LR = 3e-4
WEIGHT_DECAY = 0.01
PAIR_MARGIN_WEIGHT = 0.25
PAIR_MARGIN = 0.20
FACTOR_LOSS_WEIGHT = 0.50

SEEDS = {
    "frozen-w6e-control": 0,
    "freeform-retune-control": 1901,
    "conjunctive-primary": 1907,
    "conjunctive-replica": 1913,
}


def candidate_seed(candidate: str) -> int:
    if candidate not in SEEDS:
        raise ValueError(f"unknown W7 candidate: {candidate}")
    return int(SEEDS[candidate])


def expected_trainable_parameters(candidate: str) -> int:
    if candidate == "frozen-w6e-control":
        return 0
    if candidate == "freeform-retune-control":
        return SCORER_PARAMETER_COUNT
    if candidate in {"conjunctive-primary", "conjunctive-replica"}:
        return CONJUNCTIVE_TOTAL_PARAMETER_COUNT
    raise ValueError(f"unknown W7 candidate: {candidate}")


def configure_trainability(
    candidate: str,
    hira: HIRACore,
    scorer: nn.Module,
) -> list[Tensor]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown W7 candidate: {candidate}")
    for parameter in hira.parameters():
        parameter.requires_grad_(False)

    if candidate == "frozen-w6e-control":
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
        return []

    if candidate == "freeform-retune-control":
        if not isinstance(scorer, CompetitiveCoarseScorer):
            raise ValueError("W7 freeform control requires CompetitiveCoarseScorer")
    else:
        if not isinstance(scorer, ConjunctiveEvidenceScorer):
            raise ValueError("W7 conjunctive path requires ConjunctiveEvidenceScorer")

    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    parameters = [parameter for parameter in scorer.parameters() if parameter.requires_grad]
    count = sum(parameter.numel() for parameter in parameters)
    expected = expected_trainable_parameters(candidate)
    if count != expected:
        raise RuntimeError(
            f"W7 trainable parameter contract changed: {count} != {expected}"
        )
    return parameters


def _scorer_inputs(case: dict, decision: dict) -> dict[str, Tensor]:
    state_tokens = case["state_content_tokens"].float().unsqueeze(0)
    return {
        "state_tokens": state_tokens,
        "state_mask": torch.ones(
            1,
            state_tokens.shape[1],
            dtype=torch.bool,
        ),
        "question_tokens": decision["question_tokens"].float().unsqueeze(0),
        "question_mask": decision["question_content_mask"].bool().unsqueeze(0),
        "option_tokens": decision["option_tokens"].float().unsqueeze(0),
        "option_token_ids": decision["option_token_ids"].long().unsqueeze(0),
        "option_mask": decision["option_content_mask"].bool().unsqueeze(0),
    }


def coarse_logits(
    scorer: nn.Module,
    case: dict,
    decision: dict,
) -> tuple[Tensor, ConjunctiveEvidenceOutput | None]:
    inputs = _scorer_inputs(case, decision)
    if (
        isinstance(scorer, ConjunctiveEvidenceScorer)
        and decision["question_id"] == "diagnosis"
    ):
        aux = scorer.forward_with_factors(
            **inputs,
            factor_tokens=decision["factor_tokens"].float().unsqueeze(0),
            factor_token_mask=(
                decision["factor_token_mask"].bool().unsqueeze(0)
            ),
            factor_present_mask=(
                decision["factor_present_mask"].bool().unsqueeze(0)
            ),
        )
        return aux.logits, aux
    logits = scorer(**inputs)
    return logits, None


def cached_forward(
    hira: HIRACore,
    scorer: nn.Module,
    case: dict,
    decision: dict,
):
    coarse, aux = coarse_logits(scorer, case, decision)
    state_segments = case["state_segments"].float().unsqueeze(0)
    question = decision["question_embedding"].float().unsqueeze(0)
    options = decision["option_embeddings"].float().unsqueeze(0)
    qtype = torch.tensor(
        [PRIMITIVE_TO_ID[decision["primitive"]]],
        dtype=torch.long,
    )
    out = hira(
        question,
        state_segments,
        options,
        qtype,
        coarse_override=coarse,
        forced_budget=options.shape[1],
        adaptive_budget=False,
    )
    return out, aux


def typed_case_loss(
    hira: HIRACore,
    scorer: nn.Module,
    case: dict,
) -> Tensor:
    losses: list[Tensor] = []
    for decision in case["decisions"]:
        out, _ = cached_forward(hira, scorer, case, decision)
        logits = out.logits
        gold = torch.tensor(
            [int(decision["gold_index"])],
            dtype=torch.long,
        )
        teacher = decision["gold_probabilities"].float().unsqueeze(0)
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


def one_field_pair_margin_loss(
    scorer: nn.Module,
    case: dict,
) -> Tensor:
    decision = case["decisions"][0]
    labels = decision.get("one_field_negative_indices")
    if not isinstance(labels, dict):
        raise ValueError("W7 one-field labels missing")
    logits, _ = coarse_logits(scorer, case, decision)
    logits = logits[0]
    gold_index = int(decision["gold_index"])
    gold = logits[gold_index]
    losses: list[Tensor] = []
    for role in range(4):
        negatives = labels.get(role, labels.get(str(role), []))
        for index in negatives:
            margin = gold - logits[int(index)]
            losses.append(F.relu(PAIR_MARGIN - margin))
    if not losses:
        raise ValueError("W7 pair-margin loss requires one-field negatives")
    return torch.stack(losses).mean()


def balanced_factor_evidence_loss(
    scorer: ConjunctiveEvidenceScorer,
    case: dict,
) -> Tensor:
    decision = case["decisions"][0]
    _, aux = coarse_logits(scorer, case, decision)
    if aux is None:
        raise RuntimeError("W7 factor loss requires conjunctive output")
    logits = aux.factor_logits[0]
    targets = decision["factor_target_mask"].to(
        device=logits.device,
        dtype=logits.dtype,
    )
    if logits.shape != targets.shape:
        raise ValueError("W7 factor logit/target shape mismatch")

    role_losses: list[Tensor] = []
    for role in range(logits.shape[1]):
        role_logits = logits[:, role]
        role_targets = targets[:, role]
        positive = role_targets > 0.5
        negative = ~positive
        if not bool(positive.any()) or not bool(negative.any()):
            raise ValueError("W7 factor loss requires both classes per role")
        pos_loss = F.binary_cross_entropy_with_logits(
            role_logits[positive],
            torch.ones_like(role_logits[positive]),
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            role_logits[negative],
            torch.zeros_like(role_logits[negative]),
        )
        role_losses.append(0.5 * (pos_loss + neg_loss))
    return torch.stack(role_losses).mean()


def _rank_metrics(logits: Tensor, gold_index: int) -> tuple[int, float, bool]:
    order = sorted(
        range(int(logits.shape[0])),
        key=lambda index: (-float(logits[index]), index),
    )
    rank = order.index(int(gold_index)) + 1
    return rank, 1.0 / rank, int(gold_index) in order[:5]


def _pair_batch(
    hira: HIRACore,
    scorer: nn.Module,
    case: dict,
) -> dict[str, object]:
    decision = case["decisions"][0]
    k = int(decision["option_embeddings"].shape[0])
    gold_index = int(decision["gold_index"])
    negatives = [index for index in range(k) if index != gold_index]
    n = len(negatives)

    state_tokens = case["state_content_tokens"].float()
    question_tokens = decision["question_tokens"].float()
    question_mask = decision["question_content_mask"].bool()
    option_tokens = decision["option_tokens"].float()
    option_ids = decision["option_token_ids"].long()
    option_mask = decision["option_content_mask"].bool()

    pair_indices = torch.tensor(
        [[gold_index, negative] for negative in negatives],
        dtype=torch.long,
    )
    pair_option_tokens = option_tokens[pair_indices]
    pair_option_ids = option_ids[pair_indices]
    pair_option_mask = option_mask[pair_indices]

    scorer_inputs = {
        "state_tokens": state_tokens.unsqueeze(0).expand(n, -1, -1),
        "state_mask": torch.ones(
            n,
            state_tokens.shape[0],
            dtype=torch.bool,
        ),
        "question_tokens": question_tokens.unsqueeze(0).expand(n, -1, -1),
        "question_mask": question_mask.unsqueeze(0).expand(n, -1),
        "option_tokens": pair_option_tokens,
        "option_token_ids": pair_option_ids,
        "option_mask": pair_option_mask,
    }

    if isinstance(scorer, ConjunctiveEvidenceScorer):
        factor_tokens = decision["factor_tokens"].float()
        factor_mask = decision["factor_token_mask"].bool()
        factor_present = decision["factor_present_mask"].bool()
        aux = scorer.forward_with_factors(
            **scorer_inputs,
            factor_tokens=factor_tokens[pair_indices],
            factor_token_mask=factor_mask[pair_indices],
            factor_present_mask=factor_present[pair_indices],
        )
        pair_coarse = aux.logits
    else:
        pair_coarse = scorer(**scorer_inputs)

    state_segments = case["state_segments"].float()
    question = decision["question_embedding"].float()
    options = decision["option_embeddings"].float()
    qtype = torch.full(
        (n,),
        PRIMITIVE_TO_ID[decision["primitive"]],
        dtype=torch.long,
    )
    pair_out = hira(
        question.unsqueeze(0).expand(n, -1),
        state_segments.unsqueeze(0).expand(n, -1, -1),
        options[pair_indices],
        qtype,
        coarse_override=pair_coarse,
        forced_budget=2,
        adaptive_budget=False,
    )
    coarse_margin = pair_coarse[:, 0] - pair_coarse[:, 1]
    final_margin = pair_out.logits[:, 0] - pair_out.logits[:, 1]

    one_field = set()
    labels = decision.get("one_field_negative_indices", {})
    for role in range(4):
        one_field.update(
            int(index)
            for index in labels.get(role, labels.get(str(role), []))
        )

    one_field_positions = [
        row for row, negative in enumerate(negatives)
        if negative in one_field
    ]
    if not one_field_positions:
        raise ValueError("W7 K2 evaluator requires one-field negatives")

    return {
        "negative_indices": negatives,
        "coarse_margin": coarse_margin.detach().cpu(),
        "final_margin": final_margin.detach().cpu(),
        "coarse_loss_count": int((coarse_margin <= 0).sum()),
        "final_loss_count": int((final_margin <= 0).sum()),
        "coarse_win_rate": float((coarse_margin > 0).float().mean()),
        "final_win_rate": float((final_margin > 0).float().mean()),
        "one_field_coarse_accuracy": float(
            (coarse_margin[one_field_positions] > 0).float().mean()
        ),
        "one_field_final_accuracy": float(
            (final_margin[one_field_positions] > 0).float().mean()
        ),
    }


@torch.inference_mode()
def evaluate_w7(
    hira: HIRACore,
    scorer: nn.Module,
    cases: Sequence[dict],
) -> dict[str, object]:
    hira.eval()
    scorer.eval()

    decision_count = 0
    correct_total = 0
    primitive_counts = {"choice": 0, "score": 0, "noul": 0}
    primitive_correct = {"choice": 0, "score": 0, "noul": 0}
    diagnosis = {
        k: {
            "n": 0,
            "final_correct": 0,
            "coarse_correct": 0,
            "final_top5": 0,
            "coarse_top5": 0,
            "final_mrr": 0.0,
            "coarse_mrr": 0.0,
        }
        for k in (8, 16, 32, 64)
    }

    hard_brier = 0.0
    soft_brier = 0.0
    confidence_values: list[float] = []
    correctness_values: list[float] = []
    soft_target_values: list[float] = []
    score_errors: list[float] = []
    max_mass_error = 0.0

    pair_coarse_wins: list[float] = []
    pair_final_wins: list[float] = []
    one_field_coarse: list[float] = []
    one_field_final: list[float] = []
    k64_pair_losses_coarse: list[int] = []
    k64_pair_losses_final: list[int] = []

    factor_balanced_losses: list[float] = []
    factor_positive_correct = 0
    factor_positive_total = 0
    factor_negative_correct = 0
    factor_negative_total = 0
    gold_min_factor_prob: list[float] = []
    winner_min_factor_prob: list[float] = []
    residual_magnitudes: list[float] = []

    for case in cases:
        diagnosis_pair = _pair_batch(hira, scorer, case)
        pair_coarse_wins.append(float(diagnosis_pair["coarse_win_rate"]))
        pair_final_wins.append(float(diagnosis_pair["final_win_rate"]))
        one_field_coarse.append(
            float(diagnosis_pair["one_field_coarse_accuracy"])
        )
        one_field_final.append(
            float(diagnosis_pair["one_field_final_accuracy"])
        )
        if int(case["diagnosis_k"]) == 64:
            k64_pair_losses_coarse.append(
                int(diagnosis_pair["coarse_loss_count"])
            )
            k64_pair_losses_final.append(
                int(diagnosis_pair["final_loss_count"])
            )

        for decision in case["decisions"]:
            out, aux = cached_forward(hira, scorer, case, decision)
            p = out.probabilities[0].detach().cpu().to(torch.float64)
            gold_probs = decision["gold_probabilities"].detach().cpu().to(
                torch.float64
            )
            gold_index = int(decision["gold_index"])
            predicted = int(p.argmax())
            is_correct = int(predicted == gold_index)

            decision_count += 1
            correct_total += is_correct
            primitive = decision["primitive"]
            primitive_counts[primitive] += 1
            primitive_correct[primitive] += is_correct

            onehot = torch.zeros_like(p)
            onehot[gold_index] = 1.0
            hard_brier += float(((p - onehot) ** 2).sum())
            soft_brier += float(((p - gold_probs) ** 2).sum())
            confidence_values.append(float(p.max()))
            correctness_values.append(float(is_correct))
            soft_target_values.append(float(gold_probs[predicted]))
            max_mass_error = max(
                max_mass_error,
                abs(float(p.sum()) - 1.0),
            )

            if primitive == "score":
                support = decision["score_support"].to(dtype=p.dtype)
                predicted_score = float((p * support).sum())
                score_errors.append(
                    abs(predicted_score - float(decision["gold_score"]))
                )

            if decision["question_id"] == "diagnosis":
                k = int(case["diagnosis_k"])
                slot = diagnosis[k]
                coarse = out.coarse_logits[0].detach().cpu()
                final = out.logits[0].detach().cpu()
                coarse_rank, coarse_mrr, coarse_top5 = _rank_metrics(
                    coarse,
                    gold_index,
                )
                final_rank, final_mrr, final_top5 = _rank_metrics(
                    final,
                    gold_index,
                )
                slot["n"] += 1
                slot["coarse_correct"] += int(coarse_rank == 1)
                slot["final_correct"] += int(final_rank == 1)
                slot["coarse_top5"] += int(coarse_top5)
                slot["final_top5"] += int(final_top5)
                slot["coarse_mrr"] += coarse_mrr
                slot["final_mrr"] += final_mrr

                if aux is not None:
                    targets = decision["factor_target_mask"].bool()
                    probs = aux.factor_probabilities[0].detach().cpu()
                    logits = aux.factor_logits[0]
                    role_losses = []
                    for role in range(4):
                        target = targets[:, role]
                        pos = target
                        neg = ~target
                        pos_logits = logits[:, role][pos]
                        neg_logits = logits[:, role][neg]
                        if pos_logits.numel() and neg_logits.numel():
                            role_losses.append(
                                0.5 * (
                                    F.binary_cross_entropy_with_logits(
                                        pos_logits,
                                        torch.ones_like(pos_logits),
                                    )
                                    + F.binary_cross_entropy_with_logits(
                                        neg_logits,
                                        torch.zeros_like(neg_logits),
                                    )
                                )
                            )
                        factor_positive_correct += int(
                            (probs[:, role][pos] >= 0.5).sum()
                        )
                        factor_positive_total += int(pos.sum())
                        factor_negative_correct += int(
                            (probs[:, role][neg] < 0.5).sum()
                        )
                        factor_negative_total += int(neg.sum())
                    if role_losses:
                        factor_balanced_losses.append(
                            float(torch.stack(role_losses).mean().detach())
                        )
                    gold_min_factor_prob.append(
                        float(probs[gold_index].min())
                    )
                    winner_min_factor_prob.append(
                        float(probs[predicted].min())
                    )
                    residual_magnitudes.append(
                        float(
                            aux.conjunction_residual[0]
                            .abs()
                            .mean()
                            .detach()
                            .cpu()
                        )
                    )

    if decision_count == 0:
        raise ValueError("W7 evaluator requires cases")

    primitive_accuracy = {
        key: primitive_correct[key] / primitive_counts[key]
        for key in primitive_counts
    }
    diagnosis_per_k = {}
    for k, slot in diagnosis.items():
        if not slot["n"]:
            continue
        n = slot["n"]
        diagnosis_per_k[str(k)] = {
            "n": n,
            "accuracy": slot["final_correct"] / n,
            "final_top1": slot["final_correct"] / n,
            "final_top5": slot["final_top5"] / n,
            "final_mrr": slot["final_mrr"] / n,
            "coarse_top1": slot["coarse_correct"] / n,
            "coarse_top5": slot["coarse_top5"] / n,
            "coarse_mrr": slot["coarse_mrr"] / n,
        }

    result: dict[str, object] = {
        "case_count": len(cases),
        "decision_count": decision_count,
        "accuracy": correct_total / decision_count,
        "primitive_accuracy": primitive_accuracy,
        "diagnosis_per_k": diagnosis_per_k,
        "hard_brier": hard_brier / decision_count,
        "soft_brier": soft_brier / decision_count,
        "raw_ece": _ece15(confidence_values, correctness_values),
        "soft_ece": _ece15(confidence_values, soft_target_values),
        "score_mae": sum(score_errors) / len(score_errors),
        "probability_mass_max_error": max_mass_error,
        "source_state_encodes_per_case": 1.0,
        "k2_pair_coarse_win_rate": (
            sum(pair_coarse_wins) / len(pair_coarse_wins)
        ),
        "k2_pair_final_win_rate": (
            sum(pair_final_wins) / len(pair_final_wins)
        ),
        "one_field_k2_coarse_accuracy": (
            sum(one_field_coarse) / len(one_field_coarse)
        ),
        "one_field_k2_final_accuracy": (
            sum(one_field_final) / len(one_field_final)
        ),
        "mean_k64_pair_loss_count_coarse": (
            sum(k64_pair_losses_coarse) / len(k64_pair_losses_coarse)
        ),
        "mean_k64_pair_loss_count_final": (
            sum(k64_pair_losses_final) / len(k64_pair_losses_final)
        ),
    }

    if factor_balanced_losses:
        result["factor_balanced_bce"] = (
            sum(factor_balanced_losses) / len(factor_balanced_losses)
        )
        result["factor_positive_accuracy"] = (
            factor_positive_correct / factor_positive_total
        )
        result["factor_negative_accuracy"] = (
            factor_negative_correct / factor_negative_total
        )
        result["gold_min_factor_probability"] = (
            sum(gold_min_factor_prob) / len(gold_min_factor_prob)
        )
        result["winner_min_factor_probability"] = (
            sum(winner_min_factor_prob) / len(winner_min_factor_prob)
        )
        result["mean_conjunction_residual_magnitude"] = (
            sum(residual_magnitudes) / len(residual_magnitudes)
        )
    return result


def dev_selection_key(
    metrics: dict[str, object],
    epoch: int,
) -> tuple:
    diagnosis = metrics["diagnosis_per_k"]
    primitive = metrics["primitive_accuracy"]
    return (
        -float(diagnosis["64"]["final_top1"]),
        -float(diagnosis["32"]["final_top1"]),
        -float(primitive["choice"]),
        -float(metrics["accuracy"]),
        float(metrics["mean_k64_pair_loss_count_final"]),
        float(metrics["hard_brier"]),
        int(epoch),
    )


def train_w7_candidate(
    candidate: str,
    hira: HIRACore,
    scorer: nn.Module,
    train_cases: Sequence[dict],
    dev_cases: Sequence[dict],
    *,
    epochs: int = EPOCHS,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object]]:
    if candidate == "frozen-w6e-control":
        raise ValueError("frozen W7 control is not trainable")

    parameters = configure_trainability(candidate, hira, scorer)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    seed = candidate_seed(candidate)

    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_metrics = None

    for epoch in range(1, int(epochs) + 1):
        hira.eval()
        scorer.train()
        order = list(range(len(train_cases)))
        random.Random(seed + epoch).shuffle(order)

        typed_sum = 0.0
        pair_sum = 0.0
        factor_sum = 0.0
        total_sum = 0.0

        for index in order:
            case = train_cases[index]
            optimizer.zero_grad(set_to_none=True)
            typed_loss = typed_case_loss(hira, scorer, case)
            pair_loss = one_field_pair_margin_loss(scorer, case)
            loss = typed_loss + PAIR_MARGIN_WEIGHT * pair_loss

            factor_loss = None
            if isinstance(scorer, ConjunctiveEvidenceScorer):
                factor_loss = balanced_factor_evidence_loss(scorer, case)
                loss = loss + FACTOR_LOSS_WEIGHT * factor_loss

            loss.backward()
            optimizer.step()

            typed_sum += float(typed_loss.detach())
            pair_sum += float(pair_loss.detach())
            if factor_loss is not None:
                factor_sum += float(factor_loss.detach())
            total_sum += float(loss.detach())

        metrics = evaluate_w7(hira, scorer, dev_cases)
        metrics["epoch"] = epoch
        metrics["mean_train_typed_loss"] = typed_sum / len(train_cases)
        metrics["mean_train_pair_loss"] = pair_sum / len(train_cases)
        metrics["mean_train_factor_loss"] = (
            None
            if not isinstance(scorer, ConjunctiveEvidenceScorer)
            else factor_sum / len(train_cases)
        )
        metrics["mean_train_total_loss"] = total_sum / len(train_cases)
        history.append(metrics)

        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in scorer.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("W7 candidate produced no DEV checkpoint")
    scorer.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics
