from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .anchor_preserving_residual import (
    AnchorPreservingResidualMixer,
    MAX_SOURCE_FRACTION,
    count_anchor_residual_parameters,
)
from .competitive import CompetitiveCoarseScorer
from .hira import HIRACore
from .interface_decomposition_eval import _q0_symmetric
from .losses import typed_decision_loss
from .typed_competitive_cache import LOSS_WEIGHTS


CANDIDATES = (
    "production-frozen-control",
    "multiview-anchor-control",
    "unbounded-residual-control",
    "bounded-residual-primary",
    "bounded-residual-replica",
)
TRAINABLE_CANDIDATES = CANDIDATES[2:]

EPOCHS = 8
LR = 0.03
WEIGHT_DECAY = 0.0

SEEDS = {
    "production-frozen-control": 0,
    "multiview-anchor-control": 0,
    "unbounded-residual-control": 2303,
    "bounded-residual-primary": 2311,
    "bounded-residual-replica": 2323,
}


def candidate_seed(candidate: str) -> int:
    if candidate not in SEEDS:
        raise ValueError(f"unknown W15 candidate: {candidate}")
    return int(SEEDS[candidate])


def candidate_bounded(candidate: str) -> bool:
    if candidate == "unbounded-residual-control":
        return False
    if candidate in {"bounded-residual-primary", "bounded-residual-replica"}:
        return True
    raise ValueError(f"W15 candidate has no residual mixer: {candidate}")


def configure_trainability(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    mixer: AnchorPreservingResidualMixer,
) -> list[Tensor]:
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    for parameter in mixer.parameters():
        parameter.requires_grad_(True)
    params = [p for p in mixer.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 6:
        raise RuntimeError("W15 residual parameter budget must be exactly six")
    return params


def _rank_metrics(logits: Tensor, gold_index: int) -> tuple[int, float, bool]:
    values = logits.detach().cpu().float()
    gold = float(values[gold_index])
    better = int((values > gold).sum())
    tied_before = sum(
        1 for index in range(gold_index)
        if float(values[index]) == gold
    )
    rank = better + tied_before + 1
    return rank, 1.0 / rank, rank <= min(5, values.numel())


def _margin(logits: Tensor, gold_index: int) -> float:
    values = logits.detach().cpu().float()
    others = torch.cat(
        [values[:gold_index], values[gold_index + 1 :]]
    )
    return float(values[gold_index]) - float(others.max())


def _ece15(confidence: list[float], targets: list[float]) -> float:
    if not confidence:
        return float("nan")
    result = 0.0
    total = len(confidence)
    for index in range(15):
        lo = index / 15
        hi = (index + 1) / 15
        chosen = [
            row
            for row, value in enumerate(confidence)
            if value > lo and value <= hi
        ]
        if not chosen:
            continue
        mean_conf = sum(confidence[row] for row in chosen) / len(chosen)
        mean_target = sum(targets[row] for row in chosen) / len(chosen)
        result += len(chosen) / total * abs(mean_conf - mean_target)
    return result


@torch.no_grad()
def decision_components(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    case: dict,
    decision: dict,
) -> dict[str, Tensor]:
    state_tokens = case["state_content_tokens"].float()
    projection = scorer.projection.weight.detach().float()

    anchors: list[Tensor] = []
    for view_id in ("D0", "D1", "D2"):
        view = decision["views"][view_id]
        anchors.append(
            _q0_symmetric(
                state_tokens,
                view["option_tokens"].float(),
                view["option_content_mask"].bool(),
                projection,
            )
        )

    raw_anchor = torch.stack(anchors, dim=0).mean(dim=0)
    scale = scorer.scale().detach().float()
    anchor_logits = raw_anchor * scale
    d0_anchor_logits = anchors[0] * scale

    d0 = decision["views"]["D0"]
    state = state_tokens.unsqueeze(0)
    competitive_logits = scorer(
        state_tokens=state,
        state_mask=torch.ones(
            1,
            state.shape[1],
            dtype=torch.bool,
        ),
        question_tokens=d0["question_tokens"].float().unsqueeze(0),
        question_mask=d0["question_content_mask"].bool().unsqueeze(0),
        option_tokens=d0["option_tokens"].float().unsqueeze(0),
        option_token_ids=d0["option_token_ids"].long().unsqueeze(0),
        option_mask=d0["option_content_mask"].bool().unsqueeze(0),
    )[0]

    qtype = torch.tensor([int(decision["qtype"])], dtype=torch.long)
    state_segments = case["state_segments"].float().unsqueeze(0)
    question = d0["question_embedding"].float().unsqueeze(0)
    options = d0["option_embeddings"].float().unsqueeze(0)

    anchor_hira = hira(
        question,
        state_segments,
        options,
        qtype,
        coarse_override=anchor_logits.unsqueeze(0),
        forced_budget=options.shape[1],
        adaptive_budget=False,
    )
    production_hira = hira(
        question,
        state_segments,
        options,
        qtype,
        coarse_override=competitive_logits.unsqueeze(0),
        forced_budget=options.shape[1],
        adaptive_budget=False,
    )

    return {
        "anchor_logits": anchor_logits,
        "d0_anchor_logits": d0_anchor_logits,
        "competitive_logits": competitive_logits,
        "relation_delta": anchor_hira.relation_delta[0].detach().float(),
        "production_logits": production_hira.logits[0].detach().float(),
        "option_mask": torch.ones(
            anchor_logits.shape[0],
            dtype=torch.bool,
        ),
    }


def candidate_logits(
    candidate: str,
    components: dict[str, Tensor],
    decision: dict,
    mixer: AnchorPreservingResidualMixer | None,
):
    if candidate == "production-frozen-control":
        return components["production_logits"], None
    if candidate == "multiview-anchor-control":
        return components["anchor_logits"], None
    if candidate not in TRAINABLE_CANDIDATES or mixer is None:
        raise ValueError("W15 residual candidate requires a mixer")

    qtype = torch.tensor([int(decision["qtype"])], dtype=torch.long)
    out = mixer(
        anchor_logits=components["anchor_logits"].unsqueeze(0),
        d0_anchor_logits=components["d0_anchor_logits"].unsqueeze(0),
        competitive_logits=components["competitive_logits"].unsqueeze(0),
        relation_delta=components["relation_delta"].unsqueeze(0),
        qtype=qtype,
        option_mask=components["option_mask"].unsqueeze(0),
    )
    return out.logits[0], out


def _loss_for_case(
    candidate: str,
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    mixer: AnchorPreservingResidualMixer,
    case: dict,
) -> Tensor:
    losses: list[Tensor] = []
    for decision in case["decisions"]:
        components = decision_components(hira, scorer, case, decision)
        logits, _ = candidate_logits(
            candidate,
            components,
            decision,
            mixer,
        )
        gold = torch.tensor([int(decision["gold_index"])], dtype=torch.long)
        teacher = decision["gold_probabilities"].float().unsqueeze(0)
        score = None
        support = None
        if decision["primitive"] == "score":
            score = torch.tensor(
                [float(decision["gold_score"])],
                dtype=logits.dtype,
            )
            support = decision["score_support"].float()
        loss, _ = typed_decision_loss(
            logits.unsqueeze(0),
            gold,
            teacher_probs=teacher,
            gold_score=score,
            score_support=support,
            weights=LOSS_WEIGHTS,
        )
        losses.append(loss)
    return torch.stack(losses).mean()


def _beta_receipt(
    mixer: AnchorPreservingResidualMixer | None,
) -> dict[str, list[float]] | None:
    if mixer is None:
        return None
    return {
        "competitive": [
            float(value)
            for value in mixer.beta_competitive().detach().cpu()
        ],
        "relation": [
            float(value)
            for value in mixer.beta_relation().detach().cpu()
        ],
    }


@torch.inference_mode()
def evaluate_w15(
    candidate: str,
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    cases: Iterable[dict],
    *,
    mixer: AnchorPreservingResidualMixer | None = None,
) -> dict[str, object]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown W15 candidate: {candidate}")
    hira.eval()
    scorer.eval()
    if mixer is not None:
        mixer.eval()

    case_count = 0
    decision_count = 0
    correct_total = 0
    primitive_n = {"choice": 0, "noul": 0, "score": 0}
    primitive_correct = {"choice": 0, "noul": 0, "score": 0}
    non_diag_n = 0
    non_diag_correct = 0

    diagnosis = {
        4: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0, "margin": 0.0},
        8: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0, "margin": 0.0},
        16: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0, "margin": 0.0},
    }
    anchor_diag = {
        k: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0}
        for k in (4, 8, 16)
    }
    production_diag = {
        k: {"n": 0, "correct": 0, "top5": 0, "mrr": 0.0}
        for k in (4, 8, 16)
    }
    anchor_correct_final_wrong = {k: 0 for k in (4, 8, 16)}
    anchor_wrong_final_correct = {k: 0 for k in (4, 8, 16)}
    anchor_correct_count = {k: 0 for k in (4, 8, 16)}
    anchor_wrong_count = {k: 0 for k in (4, 8, 16)}

    hard_brier_sum = 0.0
    soft_brier_sum = 0.0
    confidence_values: list[float] = []
    correctness_values: list[float] = []
    soft_target_values: list[float] = []
    score_errors: list[float] = []
    max_mass_error = 0.0

    raw_comp_abs: list[float] = []
    raw_rel_abs: list[float] = []
    corr_comp_abs: list[float] = []
    corr_rel_abs: list[float] = []
    correction_spread_ratio: list[float] = []
    saturation_count = 0
    saturation_total = 0
    bound_violation = 0.0

    for case in cases:
        case_count += 1
        for decision in case["decisions"]:
            components = decision_components(hira, scorer, case, decision)
            logits, residual_out = candidate_logits(
                candidate,
                components,
                decision,
                mixer,
            )
            p = torch.softmax(logits.double(), dim=-1)
            teacher = decision["gold_probabilities"].double()
            gold = int(decision["gold_index"])
            pred = int(p.argmax())
            correct = pred == gold

            decision_count += 1
            correct_total += int(correct)
            primitive = str(decision["primitive"])
            primitive_n[primitive] += 1
            primitive_correct[primitive] += int(correct)
            if decision["question_id"] != "diagnosis":
                non_diag_n += 1
                non_diag_correct += int(correct)

            hard = torch.zeros_like(p)
            hard[gold] = 1.0
            hard_brier_sum += float(((p - hard) ** 2).sum())
            soft_brier_sum += float(((p - teacher) ** 2).sum())
            confidence_values.append(float(p[pred]))
            correctness_values.append(float(correct))
            soft_target_values.append(float(teacher[pred]))
            max_mass_error = max(
                max_mass_error,
                abs(float(p.sum()) - 1.0),
            )

            if primitive == "score":
                support = decision["score_support"].double()
                expected = float((p * support).sum())
                score_errors.append(
                    abs(expected - float(decision["gold_score"]))
                )

            if decision["question_id"] == "diagnosis":
                k = int(case["diagnosis_k"])
                rank, mrr, top5 = _rank_metrics(logits, gold)
                slot = diagnosis[k]
                slot["n"] += 1
                slot["correct"] += int(rank == 1)
                slot["top5"] += int(top5)
                slot["mrr"] += mrr
                slot["margin"] += _margin(logits, gold)

                for name, source, table in (
                    ("anchor", components["anchor_logits"], anchor_diag),
                    (
                        "production",
                        components["production_logits"],
                        production_diag,
                    ),
                ):
                    del name
                    s_rank, s_mrr, s_top5 = _rank_metrics(source, gold)
                    s = table[k]
                    s["n"] += 1
                    s["correct"] += int(s_rank == 1)
                    s["top5"] += int(s_top5)
                    s["mrr"] += s_mrr

                anchor_correct = (
                    int(components["anchor_logits"].argmax()) == gold
                )
                final_correct = rank == 1
                if anchor_correct:
                    anchor_correct_count[k] += 1
                    anchor_correct_final_wrong[k] += int(not final_correct)
                else:
                    anchor_wrong_count[k] += 1
                    anchor_wrong_final_correct[k] += int(final_correct)

            if residual_out is not None:
                raw_c = residual_out.competitive_residual[0]
                raw_r = residual_out.relation_residual[0]
                corr_c = residual_out.competitive_correction[0]
                corr_r = residual_out.relation_correction[0]
                spread = float(residual_out.anchor_spread[0])
                raw_comp_abs.extend(float(x) for x in raw_c.abs())
                raw_rel_abs.extend(float(x) for x in raw_r.abs())
                corr_comp_abs.extend(float(x) for x in corr_c.abs())
                corr_rel_abs.extend(float(x) for x in corr_r.abs())
                correction_spread_ratio.extend(
                    float(x)
                    for x in ((corr_c + corr_r).abs() / spread)
                )
                if residual_out.bounded:
                    saturation_count += int(
                        ((raw_c / spread).abs() >= 2.0).sum()
                    )
                    saturation_count += int(
                        ((raw_r / spread).abs() >= 2.0).sum()
                    )
                    saturation_total += 2 * int(raw_c.numel())
                    cap = 2.0 * MAX_SOURCE_FRACTION * spread + 1e-6
                    bound_violation = max(
                        bound_violation,
                        float((corr_c + corr_r).abs().max()) - cap,
                    )

    def primitive_accuracy(name: str) -> float:
        return primitive_correct[name] / max(1, primitive_n[name])

    def summarize_diag(table: dict[int, dict]) -> dict[str, object]:
        out = {}
        for k, slot in table.items():
            n = int(slot["n"])
            row = {
                "n": n,
                "top1": slot["correct"] / max(1, n),
                "top5": slot["top5"] / max(1, n),
                "mrr": slot["mrr"] / max(1, n),
            }
            if "margin" in slot:
                row["mean_margin"] = slot["margin"] / max(1, n)
            out[str(k)] = row
        return out

    transitions = {}
    for k in (4, 8, 16):
        correct_n = anchor_correct_count[k]
        wrong_n = anchor_wrong_count[k]
        transitions[str(k)] = {
            "anchor_correct_count": correct_n,
            "anchor_wrong_count": wrong_n,
            "anchor_correct_damage_count": anchor_correct_final_wrong[k],
            "anchor_wrong_rescue_count": anchor_wrong_final_correct[k],
            "anchor_correct_damage_rate": (
                anchor_correct_final_wrong[k] / max(1, correct_n)
            ),
            "anchor_correct_retention": (
                1.0 - anchor_correct_final_wrong[k] / max(1, correct_n)
            ),
            "anchor_wrong_rescue_rate": (
                anchor_wrong_final_correct[k] / max(1, wrong_n)
            ),
        }

    return {
        "candidate": candidate,
        "case_count": case_count,
        "decision_count": decision_count,
        "accuracy": correct_total / max(1, decision_count),
        "primitive_accuracy": {
            name: primitive_accuracy(name)
            for name in ("choice", "noul", "score")
        },
        "non_diagnosis_accuracy": (
            non_diag_correct / max(1, non_diag_n)
        ),
        "hard_brier": hard_brier_sum / max(1, decision_count),
        "soft_brier": soft_brier_sum / max(1, decision_count),
        "raw_ece": _ece15(confidence_values, correctness_values),
        "soft_ece": _ece15(confidence_values, soft_target_values),
        "score_mae": (
            sum(score_errors) / max(1, len(score_errors))
        ),
        "probability_mass_max_error": max_mass_error,
        "source_state_encodes_per_case": 1.0,
        "diagnosis_per_k": summarize_diag(diagnosis),
        "anchor_diagnosis_per_k": summarize_diag(anchor_diag),
        "production_diagnosis_per_k": summarize_diag(production_diag),
        "anchor_transitions_per_k": transitions,
        "beta": _beta_receipt(mixer),
        "residual_diagnostics": {
            "mean_abs_raw_competitive": (
                sum(raw_comp_abs) / max(1, len(raw_comp_abs))
            ),
            "max_abs_raw_competitive": max(raw_comp_abs, default=0.0),
            "mean_abs_raw_relation": (
                sum(raw_rel_abs) / max(1, len(raw_rel_abs))
            ),
            "max_abs_raw_relation": max(raw_rel_abs, default=0.0),
            "mean_abs_competitive_correction": (
                sum(corr_comp_abs) / max(1, len(corr_comp_abs))
            ),
            "max_abs_competitive_correction": max(
                corr_comp_abs,
                default=0.0,
            ),
            "mean_abs_relation_correction": (
                sum(corr_rel_abs) / max(1, len(corr_rel_abs))
            ),
            "max_abs_relation_correction": max(
                corr_rel_abs,
                default=0.0,
            ),
            "mean_total_correction_to_spread": (
                sum(correction_spread_ratio)
                / max(1, len(correction_spread_ratio))
            ),
            "max_total_correction_to_spread": max(
                correction_spread_ratio,
                default=0.0,
            ),
            "tanh_saturation_rate": (
                saturation_count / max(1, saturation_total)
            ),
            "max_bound_violation": max(0.0, bound_violation),
        },
    }


def dev_selection_key(
    metrics: dict[str, object],
    epoch: int,
) -> tuple:
    diag = metrics["diagnosis_per_k"]
    trans = metrics["anchor_transitions_per_k"]
    return (
        -float(diag["16"]["top1"]),
        -float(trans["16"]["anchor_correct_retention"]),
        -float(metrics["accuracy"]),
        -float(diag["4"]["top1"]),
        float(metrics["score_mae"]),
        float(metrics["soft_ece"]),
        int(epoch),
    )


def train_w15_candidate(
    candidate: str,
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    mixer: AnchorPreservingResidualMixer,
    train_cases: Sequence[dict],
    dev_cases: Sequence[dict],
    *,
    epochs: int = EPOCHS,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object]]:
    if candidate not in TRAINABLE_CANDIDATES:
        raise ValueError("W15 train function requires residual candidate")
    if mixer.bounded != candidate_bounded(candidate):
        raise ValueError("W15 candidate/mixer bound mismatch")
    params = configure_trainability(hira, scorer, mixer)
    if count_anchor_residual_parameters(mixer) != 6:
        raise RuntimeError("W15 mixer parameter count changed")

    optimizer = torch.optim.AdamW(
        params,
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
        scorer.eval()
        mixer.train()
        order = list(range(len(train_cases)))
        random.Random(seed + epoch).shuffle(order)
        total_loss = 0.0

        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_case(
                candidate,
                hira,
                scorer,
                mixer,
                train_cases[index],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += float(loss.detach())

        metrics = evaluate_w15(
            candidate,
            hira,
            scorer,
            dev_cases,
            mixer=mixer,
        )
        metrics["epoch"] = epoch
        metrics["mean_train_total_loss"] = total_loss / len(train_cases)
        history.append(metrics)
        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in mixer.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("W15 candidate produced no DEV checkpoint")
    mixer.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics


def _metric(metrics: dict[str, object], path: tuple[str, ...]) -> float:
    value = metrics
    for key in path:
        value = value[key]  # type: ignore[index]
    return float(value)


def primary_gates(
    primary: dict[str, object],
    production: dict[str, object],
    anchor: dict[str, object],
) -> dict[str, bool]:
    p = primary["primitive_accuracy"]
    pd = primary["diagnosis_per_k"]
    prod = production["diagnosis_per_k"]
    anc = anchor["diagnosis_per_k"]
    trans = primary["anchor_transitions_per_k"]
    return {
        "overall": float(primary["accuracy"]) >= 0.85,
        "choice": float(p["choice"]) >= 0.85,
        "noul": float(p["noul"]) >= 0.80,
        "score": float(p["score"]) >= 0.80,
        "k4": float(pd["4"]["top1"]) >= 0.88,
        "k8": float(pd["8"]["top1"]) >= 0.82,
        "k16": float(pd["16"]["top1"]) >= 0.72,
        "retention_k4": float(
            trans["4"]["anchor_correct_retention"]
        ) >= 0.95,
        "retention_k16": float(
            trans["16"]["anchor_correct_retention"]
        ) >= 0.90,
        "probability_mass": float(
            primary["probability_mass_max_error"]
        ) <= 1e-6,
        "state_once": float(
            primary["source_state_encodes_per_case"]
        ) == 1.0,
        "overall_gain_vs_production": (
            float(primary["accuracy"]) - float(production["accuracy"]) >= 0.08
        ),
        "k4_gain_vs_production": (
            float(pd["4"]["top1"]) - float(prod["4"]["top1"]) >= 0.12
        ),
        "k8_gain_vs_production": (
            float(pd["8"]["top1"]) - float(prod["8"]["top1"]) >= 0.15
        ),
        "k16_gain_vs_production": (
            float(pd["16"]["top1"]) - float(prod["16"]["top1"]) >= 0.20
        ),
        "preserve_anchor_k4": float(pd["4"]["top1"]) >= float(
            anc["4"]["top1"]
        ) - 0.03,
        "preserve_anchor_k8": float(pd["8"]["top1"]) >= float(
            anc["8"]["top1"]
        ) - 0.04,
        "preserve_anchor_k16": float(pd["16"]["top1"]) >= float(
            anc["16"]["top1"]
        ) - 0.05,
        "non_diag_residual_value": float(
            primary["non_diagnosis_accuracy"]
        ) >= float(anchor["non_diagnosis_accuracy"]) + 0.02,
        "k16_anchor_wrong_rescue": float(
            trans["16"]["anchor_wrong_rescue_rate"]
        ) >= 0.08,
        "k16_anchor_damage": float(
            trans["16"]["anchor_correct_damage_rate"]
        ) <= 0.10,
    }


def replica_gates(
    replica: dict[str, object],
    anchor: dict[str, object],
) -> dict[str, bool]:
    del anchor
    d = replica["diagnosis_per_k"]
    t = replica["anchor_transitions_per_k"]
    a = replica["anchor_diagnosis_per_k"]
    return {
        "overall": float(replica["accuracy"]) >= 0.82,
        "k16": float(d["16"]["top1"]) >= 0.65,
        "retention_k16": float(
            t["16"]["anchor_correct_retention"]
        ) >= 0.85,
        "k16_vs_anchor": float(d["16"]["top1"]) >= float(
            a["16"]["top1"]
        ) - 0.08,
        "probability_mass": float(
            replica["probability_mass_max_error"]
        ) <= 1e-6,
        "state_once": float(replica["source_state_encodes_per_case"]) == 1.0,
    }


def bound_causal_gates(
    bounded: dict[str, object],
    unbounded: dict[str, object],
) -> dict[str, bool]:
    bd = bounded["diagnosis_per_k"]
    ud = unbounded["diagnosis_per_k"]
    bt = bounded["anchor_transitions_per_k"]
    ut = unbounded["anchor_transitions_per_k"]
    retention_gain = float(
        bt["16"]["anchor_correct_retention"]
    ) - float(ut["16"]["anchor_correct_retention"])
    k16_gain = float(bd["16"]["top1"]) - float(ud["16"]["top1"])
    return {
        "retention_or_k16": retention_gain >= 0.05 or k16_gain >= 0.05,
        "overall_nonregression": float(bounded["accuracy"]) >= float(
            unbounded["accuracy"]
        ) - 0.02,
        "non_diag_nonregression": float(
            bounded["non_diagnosis_accuracy"]
        ) >= float(unbounded["non_diagnosis_accuracy"]) - 0.02,
    }


def _anchor_only_domain(
    anchor: dict[str, object],
    production: dict[str, object],
    bounded: dict[str, object],
) -> bool:
    ad = anchor["diagnosis_per_k"]
    pd = production["diagnosis_per_k"]
    primary_g = primary_gates(bounded, production, anchor)
    residual_value = (
        primary_g["non_diag_residual_value"]
        and primary_g["k16_anchor_wrong_rescue"]
        and primary_g["k16_anchor_damage"]
    )
    return (
        float(anchor["accuracy"]) >= 0.82
        and float(ad["4"]["top1"]) >= 0.85
        and float(ad["16"]["top1"]) >= 0.68
        and float(ad["16"]["top1"]) >= float(pd["16"]["top1"]) + 0.20
        and (
            not residual_value
            or float(bounded["accuracy"]) < float(anchor["accuracy"]) - 0.02
        )
    )


def _partial_domain(
    bounded: dict[str, object],
    production: dict[str, object],
    anchor: dict[str, object],
) -> bool:
    bd = bounded["diagnosis_per_k"]
    pd = production["diagnosis_per_k"]
    ad = anchor["diagnosis_per_k"]
    bt = bounded["anchor_transitions_per_k"]
    return (
        float(bd["16"]["top1"]) >= float(pd["16"]["top1"]) + 0.15
        and float(bd["16"]["top1"]) >= float(ad["16"]["top1"]) - 0.08
        and float(bt["16"]["anchor_correct_retention"]) >= 0.82
        and float(bounded["accuracy"]) >= float(production["accuracy"]) + 0.05
    )


def w15_verdict(
    ce: dict[str, dict[str, object]],
    cf: dict[str, dict[str, object]],
) -> tuple[str, dict[str, object]]:
    domains = {"CE": ce, "CF": cf}
    details: dict[str, object] = {
        "primary": {},
        "replica": {},
        "bound_causal": {},
        "anchor_only": {},
        "partial": {},
    }
    full_primary = True
    full_replica = True
    full_bound = True
    full_anchor_only = True
    full_partial = True
    unbounded_primary = True

    for domain, rows in domains.items():
        production = rows["production-frozen-control"]
        anchor = rows["multiview-anchor-control"]
        unbounded = rows["unbounded-residual-control"]
        bounded = rows["bounded-residual-primary"]
        replica = rows["bounded-residual-replica"]

        pg = primary_gates(bounded, production, anchor)
        rg = replica_gates(replica, anchor)
        bg = bound_causal_gates(bounded, unbounded)
        ug = primary_gates(unbounded, production, anchor)
        anchor_only = _anchor_only_domain(anchor, production, bounded)
        partial = _partial_domain(bounded, production, anchor)

        details["primary"][domain] = pg
        details["replica"][domain] = rg
        details["bound_causal"][domain] = bg
        details["anchor_only"][domain] = anchor_only
        details["partial"][domain] = partial

        full_primary = full_primary and all(pg.values())
        full_replica = full_replica and all(rg.values())
        full_bound = full_bound and all(bg.values())
        full_anchor_only = full_anchor_only and anchor_only
        full_partial = full_partial and partial
        # For no-bounding-causal verdict, unbounded must independently meet
        # primary competence + anchor preservation. The frozen production
        # gains and residual-value conditions remain part of its primary
        # gate by design.
        unbounded_primary = unbounded_primary and all(ug.values())

    if full_primary and full_replica and full_bound:
        verdict = "BOUNDED_ANCHOR_RESIDUAL_RESCUE"
    elif full_primary and full_replica and (not full_bound) and unbounded_primary:
        verdict = "ANCHOR_RESIDUAL_RESCUE_NO_BOUNDING_CAUSAL"
    elif full_anchor_only:
        verdict = "ANCHOR_ONLY_PRODUCTION_REDESIGN"
    elif full_partial:
        verdict = "ANCHOR_PRESERVING_PARTIAL"
    else:
        verdict = "ANCHOR_PRESERVING_FAIL"

    details.update(
        {
            "full_primary": full_primary,
            "full_replica": full_replica,
            "full_bound_causal": full_bound,
            "full_anchor_only": full_anchor_only,
            "full_partial": full_partial,
            "unbounded_primary": unbounded_primary,
        }
    )
    return verdict, details


__all__ = [
    "CANDIDATES",
    "TRAINABLE_CANDIDATES",
    "EPOCHS",
    "LR",
    "WEIGHT_DECAY",
    "SEEDS",
    "bound_causal_gates",
    "candidate_bounded",
    "candidate_logits",
    "candidate_seed",
    "configure_trainability",
    "decision_components",
    "dev_selection_key",
    "evaluate_w15",
    "primary_gates",
    "replica_gates",
    "train_w15_candidate",
    "w15_verdict",
]
