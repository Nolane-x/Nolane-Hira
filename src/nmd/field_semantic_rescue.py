from __future__ import annotations

import random
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .competitive import (
    MIN_COVERAGE_WEIGHT,
    SALIENCE_THRESHOLD,
    CompetitiveCoarseScorer,
    candidate_relative_idf,
)
from .hira import HIRACore
from .typed_competitive_cache import _loss_cached_case, evaluate_w6b_cases

ADAPTER_D_REL = 128
ADAPTER_BOTTLENECK = 32
ADAPTER_PARAMETER_COUNT = 8192
SCORER_PARAMETER_COUNT = 32769
EPOCHS = 6
LR = 3e-4
WEIGHT_DECAY = 0.01
PAIR_LOSS_WEIGHT = 0.5
GLOBAL_SEED = 1601

CANDIDATES = (
    "frozen-joint-control",
    "projection-retune-control",
    "semantic-residual-adapter",
)


class SemanticResidualAdapter(nn.Module):
    """Tiny identity-initialized correction to projected token geometry."""

    def __init__(
        self,
        d_rel: int = ADAPTER_D_REL,
        bottleneck: int = ADAPTER_BOTTLENECK,
    ):
        super().__init__()
        self.d_rel = int(d_rel)
        self.bottleneck = int(bottleneck)
        self.down = nn.Linear(self.d_rel, self.bottleneck, bias=False)
        self.up = nn.Linear(self.bottleneck, self.d_rel, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, projected: Tensor) -> Tensor:
        if projected.shape[-1] != self.d_rel:
            raise ValueError("semantic adapter dimension mismatch")
        return projected + self.up(F.gelu(self.down(projected)))


class SemanticAdaptedCompetitiveScorer(nn.Module):
    """Research-only scorer with a shared semantic residual adapter.

    The frozen production projection and logit scale remain owned by base.
    With a zero-initialized up projection this starts functionally identical
    to the frozen scorer up to floating-point roundoff.
    """

    def __init__(
        self,
        base: CompetitiveCoarseScorer,
        adapter: SemanticResidualAdapter | None = None,
    ):
        super().__init__()
        if base.d_rel != ADAPTER_D_REL:
            raise ValueError("W6h requires the frozen 128-d relation space")
        self.base = base
        self.adapter = adapter or SemanticResidualAdapter(d_rel=base.d_rel)
        self.d_model = base.d_model
        self.d_rel = base.d_rel

    def scale(self) -> Tensor:
        return self.base.scale()

    def _project(self, x: Tensor) -> Tensor:
        raw = self.base.projection(x)
        return F.normalize(self.adapter(raw), dim=-1)

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_tokens: Tensor,
        option_token_ids: Tensor,
        option_mask: Tensor,
    ) -> Tensor:
        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.d_model:
            raise ValueError("state_tokens must be [B,S,D]")
        if question_tokens.ndim != 3 or question_tokens.shape[-1] != self.d_model:
            raise ValueError("question_tokens must be [B,Q,D]")
        if option_tokens.ndim != 4 or option_tokens.shape[-1] != self.d_model:
            raise ValueError("option_tokens must be [B,K,T,D]")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("state_mask mismatch")
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("question_mask mismatch")
        if option_mask.shape != option_tokens.shape[:3] or option_mask.dtype != torch.bool:
            raise ValueError("option_mask mismatch")
        if option_token_ids.shape != option_mask.shape or option_token_ids.dtype != torch.long:
            raise ValueError("option token IDs mismatch")
        if option_tokens.shape[1] < 2:
            raise ValueError("competitive coarse requires at least two options")

        context = torch.cat([state_tokens, question_tokens], dim=1)
        context_mask = torch.cat([state_mask, question_mask], dim=1)
        projected_context = self._project(context)
        projected_options = self._project(option_tokens)

        similarity = torch.einsum(
            "bktd,bcd->bktc",
            projected_options,
            projected_context,
        )
        similarity = similarity.masked_fill(
            ~context_mask[:, None, None, :],
            -1e4,
        )

        salience = candidate_relative_idf(option_token_ids, option_mask)
        salience = salience.to(
            device=similarity.device,
            dtype=similarity.dtype,
        )
        weights = salience * option_mask.to(salience.dtype)
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-8)

        common = (similarity * weights[..., None]).sum(dim=2) / denom
        adjusted = similarity - common[:, :, None, :]
        adjusted = adjusted.masked_fill(
            ~context_mask[:, None, None, :],
            -1e4,
        )
        coverage = adjusted.max(dim=-1).values

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
        raw = weighted_mean + MIN_COVERAGE_WEIGHT * min_coverage
        logits = raw * self.scale().to(raw.device, raw.dtype)
        if not torch.isfinite(logits).all():
            raise ValueError("W6h semantic scorer produced non-finite logits")
        return logits


def adapter_parameter_count(adapter: nn.Module) -> int:
    return sum(parameter.numel() for parameter in adapter.parameters())


def _criterion_fields(text: str) -> tuple[str, ...]:
    fields = tuple(part.strip() for part in text.split(";"))
    if len(fields) != 4 or any(not field for field in fields):
        raise ValueError("W6h diagnosis criterion must contain four fields")
    return fields


def one_field_negative_indices(
    option_texts: Sequence[str],
    gold_index: int,
) -> dict[int, list[int]]:
    if not 0 <= int(gold_index) < len(option_texts):
        raise ValueError("gold index out of range")
    parsed = [_criterion_fields(text) for text in option_texts]
    gold = parsed[int(gold_index)]
    result = {index: [] for index in range(4)}
    for option_index, fields in enumerate(parsed):
        if option_index == int(gold_index):
            continue
        changed = [
            field_index
            for field_index, (left, right) in enumerate(zip(gold, fields))
            if left != right
        ]
        if len(changed) == 1:
            result[changed[0]].append(option_index)
    return result


def attach_one_field_pair_labels(
    cache: dict,
    authority_cases: Sequence,
) -> dict:
    if len(cache["cases"]) != len(authority_cases):
        raise ValueError("W6h cache/authority case count mismatch")
    for cached, authority in zip(cache["cases"], authority_cases):
        if cached["case_id"] != authority.typed.case_id:
            raise ValueError("W6h cache/authority order mismatch")
        diagnosis = authority.typed.decisions[0]
        labels = one_field_negative_indices(
            [option.criterion_text for option in diagnosis.options],
            diagnosis.gold_index,
        )
        cached["decisions"][0]["one_field_negative_indices"] = labels
    return cache


def _coarse_logits_cached(scorer: nn.Module, case: dict) -> Tensor:
    decision = case["decisions"][0]
    state_tokens = case["state_content_tokens"].float().unsqueeze(0)
    question_tokens = decision["question_tokens"].float().unsqueeze(0)
    option_tokens = decision["option_tokens"].float().unsqueeze(0)
    option_ids = decision["option_token_ids"].long().unsqueeze(0)
    option_mask = decision["option_content_mask"].bool().unsqueeze(0)
    return scorer(
        state_tokens=state_tokens,
        state_mask=torch.ones(
            1,
            state_tokens.shape[1],
            dtype=torch.bool,
        ),
        question_tokens=question_tokens,
        question_mask=decision["question_content_mask"].bool().unsqueeze(0),
        option_tokens=option_tokens,
        option_token_ids=option_ids,
        option_mask=option_mask,
    )[0]


def semantic_pair_loss(scorer: nn.Module, case: dict) -> Tensor:
    decision = case["decisions"][0]
    labels = decision.get("one_field_negative_indices")
    if not isinstance(labels, dict):
        raise ValueError("W6h pair labels missing")
    logits = _coarse_logits_cached(scorer, case)
    gold_index = int(decision["gold_index"])
    gold = logits[gold_index]
    losses: list[Tensor] = []
    for role in range(4):
        negatives = labels.get(role, labels.get(str(role), []))
        for index in negatives:
            losses.append(F.softplus(-(gold - logits[int(index)])))
    if not losses:
        raise ValueError("W6h requires at least one one-field negative")
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate_field_pairs(
    scorer: nn.Module,
    cases: Iterable[dict],
) -> dict[str, object]:
    scorer.eval()
    correct = 0
    total = 0
    role_correct = [0, 0, 0, 0]
    role_total = [0, 0, 0, 0]
    margins: list[float] = []
    for case in cases:
        decision = case["decisions"][0]
        labels = decision.get("one_field_negative_indices")
        if not isinstance(labels, dict):
            raise ValueError("W6h pair labels missing")
        logits = _coarse_logits_cached(scorer, case)
        gold_index = int(decision["gold_index"])
        gold = logits[gold_index]
        for role in range(4):
            negatives = labels.get(role, labels.get(str(role), []))
            for index in negatives:
                margin = float(gold - logits[int(index)])
                margins.append(margin)
                total += 1
                role_total[role] += 1
                if margin > 0:
                    correct += 1
                    role_correct[role] += 1
    if total == 0 or any(value == 0 for value in role_total):
        raise ValueError("W6h field-pair evaluator requires all four roles")
    return {
        "mean_pair_accuracy": correct / total,
        "role_pair_accuracy": {
            str(role): role_correct[role] / role_total[role]
            for role in range(4)
        },
        "mean_pair_margin": sum(margins) / len(margins),
        "pair_count": total,
    }


def evaluate_w6h(
    hira: HIRACore,
    scorer: nn.Module,
    cases: Sequence[dict],
) -> dict[str, object]:
    typed = evaluate_w6b_cases(
        hira,
        scorer,
        cases,
        competitive=True,
    )
    typed.update(evaluate_field_pairs(scorer, cases))
    return typed


def dev_selection_key(metrics: dict[str, object], epoch: int) -> tuple:
    primitive = metrics["primitive_accuracy"]
    return (
        -float(metrics["diagnosis_per_k"]["64"]["accuracy"]),
        -float(metrics["mean_pair_accuracy"]),
        -float(metrics["accuracy"]),
        -float(primitive["choice"]),
        -float(primitive["score"]),
        -float(primitive["noul"]),
        float(metrics["hard_brier"]),
        float(metrics["score_mae"]),
        int(epoch),
    )


def configure_trainability(
    candidate: str,
    hira: HIRACore,
    scorer: nn.Module,
) -> list[Tensor]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown W6h candidate: {candidate}")
    for parameter in hira.parameters():
        parameter.requires_grad_(False)

    if candidate == "frozen-joint-control":
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
        return []

    if candidate == "projection-retune-control":
        if not isinstance(scorer, CompetitiveCoarseScorer):
            raise ValueError("projection control requires base scorer")
        for parameter in scorer.parameters():
            parameter.requires_grad_(True)
        parameters = list(scorer.parameters())
        if sum(parameter.numel() for parameter in parameters) != SCORER_PARAMETER_COUNT:
            raise RuntimeError("W6h projection-control parameter contract changed")
        return parameters

    if not isinstance(scorer, SemanticAdaptedCompetitiveScorer):
        raise ValueError("adapter path requires SemanticAdaptedCompetitiveScorer")
    for parameter in scorer.base.parameters():
        parameter.requires_grad_(False)
    for parameter in scorer.adapter.parameters():
        parameter.requires_grad_(True)
    parameters = list(scorer.adapter.parameters())
    if sum(parameter.numel() for parameter in parameters) != ADAPTER_PARAMETER_COUNT:
        raise RuntimeError("W6h adapter parameter contract changed")
    return parameters


def train_w6h_candidate(
    candidate: str,
    hira: HIRACore,
    scorer: nn.Module,
    train_cases: Sequence[dict],
    dev_cases: Sequence[dict],
    *,
    epochs: int = EPOCHS,
    seed: int = GLOBAL_SEED,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object]]:
    if candidate == "frozen-joint-control":
        raise ValueError("frozen W6h control is not trainable")
    parameters = configure_trainability(candidate, hira, scorer)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_metrics = None

    for epoch in range(1, int(epochs) + 1):
        hira.eval()
        scorer.train()
        order = list(range(len(train_cases)))
        random.Random(int(seed) + epoch).shuffle(order)
        loss_sum = 0.0
        pair_sum = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            typed_loss = _loss_cached_case(
                hira,
                scorer,
                train_cases[index],
                competitive=True,
            )
            pair_loss = semantic_pair_loss(scorer, train_cases[index])
            loss = typed_loss + PAIR_LOSS_WEIGHT * pair_loss
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            pair_sum += float(pair_loss.detach())

        metrics = evaluate_w6h(hira, scorer, dev_cases)
        metrics["epoch"] = epoch
        metrics["mean_train_case_loss"] = loss_sum / max(1, len(train_cases))
        metrics["mean_train_pair_loss"] = pair_sum / max(1, len(train_cases))
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
        raise RuntimeError("W6h candidate produced no frozen checkpoint")
    scorer.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics


def absolute_gates(metrics: dict[str, object]) -> dict[str, bool]:
    primitive = metrics["primitive_accuracy"]
    roles = metrics["role_pair_accuracy"]
    return {
        "overall_accuracy": float(metrics["accuracy"]) >= 0.65,
        "choice_accuracy": float(primitive["choice"]) >= 0.65,
        "score_accuracy": float(primitive["score"]) >= 0.60,
        "noul_accuracy": float(primitive["noul"]) >= 0.65,
        "diagnosis_k64_accuracy": (
            float(metrics["diagnosis_per_k"]["64"]["accuracy"]) >= 0.55
        ),
        "mean_field_pair_accuracy": float(metrics["mean_pair_accuracy"]) >= 0.80,
        "every_role_pair_accuracy": min(float(value) for value in roles.values()) >= 0.70,
        "probability_mass": float(metrics["probability_mass_max_error"]) <= 1e-6,
        "state_once": float(metrics["source_state_encodes_per_case"]) == 1.0,
    }


def causal_gates(
    adapter: dict[str, object],
    frozen: dict[str, object],
    projection: dict[str, object],
) -> dict[str, bool]:
    return {
        "k64_gain_vs_frozen": (
            float(adapter["diagnosis_per_k"]["64"]["accuracy"])
            - float(frozen["diagnosis_per_k"]["64"]["accuracy"])
            >= 0.10
        ),
        "field_pair_gain_vs_frozen": (
            float(adapter["mean_pair_accuracy"])
            - float(frozen["mean_pair_accuracy"])
            >= 0.10
        ),
        "overall_nonregression_vs_frozen": (
            float(adapter["accuracy"]) >= float(frozen["accuracy"]) - 0.02
        ),
        "k64_comparable_to_projection": (
            float(adapter["diagnosis_per_k"]["64"]["accuracy"])
            >= float(projection["diagnosis_per_k"]["64"]["accuracy"]) - 0.03
        ),
        "field_pair_comparable_to_projection": (
            float(adapter["mean_pair_accuracy"])
            >= float(projection["mean_pair_accuracy"]) - 0.03
        ),
    }


def rescue_verdict(
    *,
    confirm_y: dict[str, dict[str, object]],
    confirm_z: dict[str, dict[str, object]],
) -> tuple[str, dict[str, object]]:
    domains = {"Y": confirm_y, "Z": confirm_z}
    full: dict[str, bool] = {}
    absolute: dict[str, dict[str, bool]] = {}
    causal: dict[str, dict[str, bool]] = {}
    material: dict[str, bool] = {}

    for domain, paths in domains.items():
        adapter = paths["semantic-residual-adapter"]
        frozen = paths["frozen-joint-control"]
        projection = paths["projection-retune-control"]
        absolute[domain] = absolute_gates(adapter)
        causal[domain] = causal_gates(adapter, frozen, projection)
        full[domain] = all(absolute[domain].values()) and all(causal[domain].values())
        material[domain] = (
            float(adapter["diagnosis_per_k"]["64"]["accuracy"])
            - float(frozen["diagnosis_per_k"]["64"]["accuracy"])
            >= 0.05
            and float(adapter["mean_pair_accuracy"])
            - float(frozen["mean_pair_accuracy"])
            >= 0.05
        )

    if all(full.values()):
        verdict = "FIELD_SEMANTIC_RESCUE"
    elif sum(full.values()) == 1 or all(material.values()):
        verdict = "FIELD_SEMANTIC_PARTIAL"
    else:
        verdict = "FIELD_SEMANTIC_FAIL"

    return verdict, {
        "full_pass": full,
        "absolute_gates": absolute,
        "causal_gates": causal,
        "material_signal": material,
    }
