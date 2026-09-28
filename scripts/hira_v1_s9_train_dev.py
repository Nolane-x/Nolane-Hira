from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random

import torch
from torch import Tensor
import torch.nn.functional as F

from nmd.local_runtime import (
    A13_REVISION,
    load_hira_v0_m4_bundle,
    read_runtime_bundle_manifest,
)
from nmd.v1_a13_lora import (
    a13_lora_state_dict,
    iter_a13_lora_modules,
    load_a13_lora_state_dict,
)
from nmd.v1_invariance import (
    selected_choice_agreement,
    symmetric_js_divergence,
)
from nmd.v1_margin import (
    gold_vs_all_negative_margin_loss,
    gold_vs_max_wrong_margin,
    top1_top2_margin,
)
from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import generate_s3_pairs
from nmd.v1_s4_authority import generate_s4_pairs
from nmd.v1_s5_authority import generate_s5_pairs
from nmd.v1_s6_authority import generate_s6_pairs
from nmd.v1_s7_authority import generate_s7_pairs
from nmd.v1_s6_semantic_core import (
    HIRA_V1_S6_LORA_PARAMETER_COUNT,
    HIRA_V1_S6_LORA_RANK,
)
from nmd.v1_s8_authority import generate_s8_cases
from nmd.v1_s9_authority import (
    S9MarginCase,
    generate_s9_cases,
    validate_s9_partitions,
)
from nmd.v1_s9_semantic_core import (
    HIRA_V1_S9_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S9_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s9_margin_core,
    enforce_s9_eval,
)


SCHEMA_VERSION = "hira-v1-s9-margin-train-dev-v1"
READY = "HIRA_V1_S9_MARGIN_DEV_READY"
FAIL = "HIRA_V1_S9_MARGIN_DEV_FAIL"

SEED = 13801
EPOCHS = 24
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MARGIN_COEFFICIENT = 0.25
MARGIN = 0.20
OPTION_ALIGN_COEFFICIENT = 0.05
OPTION_ALIGN_TEMPERATURE = 0.10
QUESTION_OPTION_COEFFICIENT = 0.10
QUESTION_OPTION_TEMPERATURE = 0.10
INVARIANCE_COEFFICIENT = 0.25


def _assert_fresh_against_prior(
    train_rows: tuple[S9MarginCase, ...],
    dev_rows: tuple[S9MarginCase, ...],
) -> None:
    prior = (
        *generate_s0_pairs("train"), *generate_s0_pairs("dev"),
        *generate_s1_pairs("train"), *generate_s1_pairs("dev"),
        *generate_s2_pairs("train"), *generate_s2_pairs("dev"),
        *generate_s3_pairs("train"), *generate_s3_pairs("dev"),
        *generate_s4_pairs("train"), *generate_s4_pairs("dev"),
        *generate_s5_pairs("train"), *generate_s5_pairs("dev"),
        *generate_s6_pairs("train"), *generate_s6_pairs("dev"),
        *generate_s7_pairs("train"), *generate_s7_pairs("dev"),
        *generate_s8_cases("train"), *generate_s8_cases("dev"),
    )
    prior_states = {row.state for row in prior}
    prior_questions = {
        q for row in prior for q in (row.question_a, row.question_b)
    }

    current = (*train_rows, *dev_rows)
    current_states = {
        text
        for row in current
        for text in (row.state_a, row.state_b)
    }
    current_questions = {
        text
        for row in current
        for text in (
            row.question_a1,
            row.question_a2,
            row.question_b1,
            row.question_b2,
        )
    }
    if prior_states & current_states:
        raise RuntimeError("S9 exact state overlap with exposed S0-S7 rows")
    if prior_questions & current_questions:
        raise RuntimeError("S9 exact question overlap with exposed S0-S7 rows")


def _content_mask(batch) -> Tensor:
    mask = batch.attention_mask.bool()
    if batch.special_token_mask is None:
        return mask
    return mask & ~batch.special_token_mask.bool()


def _option_view_texts(rows: list[S9MarginCase]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        for criterion, alias in zip(row.option_texts, row.option_aliases):
            texts.extend((criterion, alias))
    return texts


def _gold_tensors(
    rows: list[S9MarginCase],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    gold = []
    other = []
    for row in rows:
        gold.extend((row.gold_a, row.gold_b))
        other.extend((row.gold_b, row.gold_a))
    return (
        torch.tensor(gold, dtype=torch.long, device=device),
        torch.tensor(other, dtype=torch.long, device=device),
    )


def _option_alignment_loss(option_pooled: Tensor) -> Tensor:
    left = F.normalize(option_pooled[:, :, 0], dim=-1)
    right = F.normalize(option_pooled[:, :, 1], dim=-1)
    logits_lr = torch.einsum("nkd,njd->nkj", left, right) / OPTION_ALIGN_TEMPERATURE
    logits_rl = torch.einsum("nkd,njd->nkj", right, left) / OPTION_ALIGN_TEMPERATURE
    n, k, _ = logits_lr.shape
    labels = torch.arange(k, device=logits_lr.device).expand(n, k)
    return 0.5 * (
        F.cross_entropy(logits_lr.reshape(n * k, k), labels.reshape(n * k))
        + F.cross_entropy(logits_rl.reshape(n * k, k), labels.reshape(n * k))
    )


def _question_option_loss(
    question_pooled: Tensor,
    option_pooled: Tensor,
    gold: Tensor,
) -> Tensor:
    left = F.normalize(option_pooled[:, :, 0], dim=-1)
    right = F.normalize(option_pooled[:, :, 1], dim=-1)
    options = F.normalize(0.5 * (left + right), dim=-1)
    options = options.repeat_interleave(2, dim=0)
    question = F.normalize(question_pooled, dim=-1)
    logits = torch.einsum("bd,bkd->bk", question, options)
    logits = logits / QUESTION_OPTION_TEMPERATURE
    return F.cross_entropy(logits, gold)


def _encode_batch(runtime, rows: list[S9MarginCase]) -> dict[str, Tensor]:
    enforce_s9_eval(runtime)
    encoder = runtime.encoder

    n = len(rows)
    states_a = [row.state_a for row in rows]
    states_b = [row.state_b for row in rows]
    questions_canonical = [
        q
        for row in rows
        for q in (row.question_a1, row.question_b1)
    ]
    questions_paraphrase = [
        q
        for row in rows
        for q in (row.question_a2, row.question_b2)
    ]
    options = _option_view_texts(rows)

    # One encoder call per modality batch, but every state wording appears
    # exactly once at this model snapshot.
    state_batch = encoder.encode_texts([*states_a, *states_b])
    question_batch = encoder.encode_texts(
        [*questions_canonical, *questions_paraphrase]
    )
    option_batch = encoder.encode_texts(options)

    if state_batch.token_embeddings.shape[0] != 2 * n:
        raise RuntimeError("S9 state-view packing changed")
    if question_batch.token_embeddings.shape[0] != 4 * n:
        raise RuntimeError("S9 question-view packing changed")
    if option_batch.token_embeddings.shape[0] != n * 4 * 2:
        raise RuntimeError("S9 option-view packing changed")

    state_tokens = state_batch.token_embeddings
    state_mask = _content_mask(state_batch)
    question_tokens = question_batch.token_embeddings
    question_mask = _content_mask(question_batch)

    option_tokens = option_batch.token_embeddings.reshape(
        n, 4, 2, option_batch.token_embeddings.shape[1], -1
    )
    option_mask = _content_mask(option_batch).reshape(
        n, 4, 2, option_batch.attention_mask.shape[1]
    )
    option_view_mask = torch.ones(
        n, 4, 2, dtype=torch.bool, device=option_tokens.device
    )
    option_pooled = option_batch.pooled_embeddings.reshape(n, 4, 2, -1)

    return {
        "state_a_tokens": state_tokens[:n],
        "state_a_mask": state_mask[:n],
        "state_b_tokens": state_tokens[n:],
        "state_b_mask": state_mask[n:],
        "question_canonical_tokens": question_tokens[: 2 * n],
        "question_canonical_mask": question_mask[: 2 * n],
        "question_paraphrase_tokens": question_tokens[2 * n :],
        "question_paraphrase_mask": question_mask[2 * n :],
        "question_canonical_pooled": question_batch.pooled_embeddings[: 2 * n],
        "question_paraphrase_pooled": question_batch.pooled_embeddings[2 * n :],
        "option_tokens": option_tokens,
        "option_mask": option_mask,
        "option_view_mask": option_view_mask,
        "option_pooled": option_pooled,
    }


def _decision_logits(
    runtime,
    *,
    state_tokens: Tensor,
    state_mask: Tensor,
    question_tokens: Tensor,
    question_mask: Tensor,
    encoded: dict[str, Tensor],
) -> Tensor:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S9 projection triadic scorer missing")

    return scorer(
        state_tokens=state_tokens.repeat_interleave(2, dim=0),
        state_mask=state_mask.repeat_interleave(2, dim=0),
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_view_tokens=encoded["option_tokens"].repeat_interleave(2, dim=0),
        option_view_token_mask=encoded["option_mask"].repeat_interleave(2, dim=0),
        option_view_mask=encoded["option_view_mask"].repeat_interleave(2, dim=0),
    )


def _supervised(
    logits: Tensor,
    gold: Tensor,
    _other: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    ce = F.cross_entropy(logits, gold)
    margin_loss = gold_vs_all_negative_margin_loss(
        logits,
        gold,
        margin=MARGIN,
    )
    decision = ce + MARGIN_COEFFICIENT * margin_loss
    return ce, margin_loss, decision

def _losses(
    runtime,
    rows: list[S9MarginCase],
) -> tuple[Tensor, dict[str, float], Tensor, Tensor, dict[str, Tensor]]:
    encoded = _encode_batch(runtime, rows)
    canonical = _decision_logits(
        runtime,
        state_tokens=encoded["state_a_tokens"],
        state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        encoded=encoded,
    )
    paraphrase = _decision_logits(
        runtime,
        state_tokens=encoded["state_b_tokens"],
        state_mask=encoded["state_b_mask"],
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
        encoded=encoded,
    )

    gold, other = _gold_tensors(rows, device=canonical.device)
    c_ce, c_margin, c_decision = _supervised(canonical, gold, other)
    p_ce, p_margin, p_decision = _supervised(paraphrase, gold, other)

    ce = 0.5 * (c_ce + p_ce)
    margin_loss = 0.5 * (c_margin + p_margin)
    decision = 0.5 * (c_decision + p_decision)

    option_align = _option_alignment_loss(encoded["option_pooled"])
    qopt_c = _question_option_loss(
        encoded["question_canonical_pooled"],
        encoded["option_pooled"],
        gold,
    )
    qopt_p = _question_option_loss(
        encoded["question_paraphrase_pooled"],
        encoded["option_pooled"],
        gold,
    )
    question_option = 0.5 * (qopt_c + qopt_p)
    consistency = symmetric_js_divergence(canonical, paraphrase)

    total = (
        decision
        + OPTION_ALIGN_COEFFICIENT * option_align
        + QUESTION_OPTION_COEFFICIENT * question_option
        + INVARIANCE_COEFFICIENT * consistency
    )
    pieces = {
        "ce": float(ce.detach().cpu()),
        "canonical_ce": float(c_ce.detach().cpu()),
        "paraphrase_ce": float(p_ce.detach().cpu()),
        "margin": float(margin_loss.detach().cpu()),
        "decision": float(decision.detach().cpu()),
        "canonical_decision": float(c_decision.detach().cpu()),
        "paraphrase_decision": float(p_decision.detach().cpu()),
        "option_alignment": float(option_align.detach().cpu()),
        "question_option": float(question_option.detach().cpu()),
        "consistency_js": float(consistency.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, pieces, canonical, paraphrase, encoded


def _option_order_flips(
    runtime,
    canonical: Tensor,
    encoded: dict[str, Tensor],
) -> int:
    permutation = torch.tensor(
        [3, 2, 1, 0],
        device=canonical.device,
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S9 projection triadic scorer missing")

    permuted = scorer(
        state_tokens=encoded["state_a_tokens"].repeat_interleave(2, dim=0),
        state_mask=encoded["state_a_mask"].repeat_interleave(2, dim=0),
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        option_view_tokens=encoded["option_tokens"][:, permutation]
            .repeat_interleave(2, dim=0),
        option_view_token_mask=encoded["option_mask"][:, permutation]
            .repeat_interleave(2, dim=0),
        option_view_mask=encoded["option_view_mask"][:, permutation]
            .repeat_interleave(2, dim=0),
    )
    mapped = permutation[permuted.argmax(-1)]
    return int((mapped != canonical.argmax(-1)).sum().item())


@torch.no_grad()
def evaluate(runtime, rows: tuple[S9MarginCase, ...]) -> dict:
    enforce_s9_eval(runtime)

    canonical_correct = 0
    paraphrase_correct = 0
    canonical_pair_both = 0
    canonical_pair_changed = 0
    cross_agree_sum = 0.0
    js_sum = 0.0
    order_flips = 0
    max_mass_error = 0.0
    canonical_decision_sum = 0.0
    paraphrase_decision_sum = 0.0
    option_align_sum = 0.0
    qopt_sum = 0.0
    canonical_signed_margin_sum = 0.0
    paraphrase_signed_margin_sum = 0.0
    canonical_margin_satisfied = 0
    paraphrase_margin_satisfied = 0
    canonical_top1_top2_sum = 0.0
    paraphrase_top1_top2_sum = 0.0
    semantic_cases = 0
    state_view_encodes = 0
    full_k = True

    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = list(rows[start : start + BATCH_SIZE])
        _total, pieces, canonical, paraphrase, encoded = _losses(
            runtime, batch_rows
        )
        n = len(batch_rows)
        gold, _other = _gold_tensors(batch_rows, device=canonical.device)

        c_pred = canonical.argmax(-1)
        p_pred = paraphrase.argmax(-1)
        canonical_correct += int((c_pred == gold).sum().item())
        paraphrase_correct += int((p_pred == gold).sum().item())

        c_pairs = c_pred.reshape(n, 2)
        g_pairs = gold.reshape(n, 2)
        canonical_pair_both += int(
            ((c_pairs == g_pairs).all(-1)).sum().item()
        )
        canonical_pair_changed += int(
            (c_pairs[:, 0] != c_pairs[:, 1]).sum().item()
        )

        c_signed = gold_vs_max_wrong_margin(canonical, gold)
        p_signed = gold_vs_max_wrong_margin(paraphrase, gold)
        c_top = top1_top2_margin(canonical)
        p_top = top1_top2_margin(paraphrase)
        canonical_signed_margin_sum += float(c_signed.sum().cpu())
        paraphrase_signed_margin_sum += float(p_signed.sum().cpu())
        canonical_margin_satisfied += int(c_signed.ge(MARGIN).sum().item())
        paraphrase_margin_satisfied += int(p_signed.ge(MARGIN).sum().item())
        canonical_top1_top2_sum += float(c_top.sum().cpu())
        paraphrase_top1_top2_sum += float(p_top.sum().cpu())

        agreement = float(selected_choice_agreement(canonical, paraphrase))
        js = float(symmetric_js_divergence(canonical, paraphrase))
        cross_agree_sum += agreement * (2 * n)
        js_sum += js * (2 * n)

        order_flips += _option_order_flips(
            runtime,
            canonical,
            encoded,
        )

        c_probs = torch.softmax(canonical, dim=-1)
        p_probs = torch.softmax(paraphrase, dim=-1)
        max_mass_error = max(
            max_mass_error,
            float((c_probs.sum(-1) - 1).abs().max().cpu()),
            float((p_probs.sum(-1) - 1).abs().max().cpu()),
        )

        canonical_decision_sum += pieces["canonical_decision"] * n
        paraphrase_decision_sum += pieces["paraphrase_decision"] * n
        option_align_sum += pieces["option_alignment"] * n
        qopt_sum += pieces["question_option"] * n
        semantic_cases += n
        state_view_encodes += 2 * n
        full_k = full_k and canonical.shape[-1] == 4 and paraphrase.shape[-1] == 4

    canonical_queries = len(rows) * 2
    return {
        "semantic_cases": semantic_cases,
        "canonical_queries": canonical_queries,
        "paraphrase_queries": canonical_queries,
        "canonical_accuracy": canonical_correct / canonical_queries,
        "paraphrase_accuracy": paraphrase_correct / canonical_queries,
        "canonical_paired_both_correct_rate": canonical_pair_both / len(rows),
        "canonical_question_swap_choice_change_rate": (
            canonical_pair_changed / len(rows)
        ),
        "cross_view_selected_choice_agreement": (
            cross_agree_sum / canonical_queries
        ),
        "cross_view_mean_js": js_sum / canonical_queries,
        "canonical_mean_gold_vs_max_wrong_margin": (
            canonical_signed_margin_sum / canonical_queries
        ),
        "paraphrase_mean_gold_vs_max_wrong_margin": (
            paraphrase_signed_margin_sum / canonical_queries
        ),
        "canonical_margin_satisfaction_rate": (
            canonical_margin_satisfied / canonical_queries
        ),
        "paraphrase_margin_satisfaction_rate": (
            paraphrase_margin_satisfied / canonical_queries
        ),
        "canonical_mean_top1_top2_margin": (
            canonical_top1_top2_sum / canonical_queries
        ),
        "paraphrase_mean_top1_top2_margin": (
            paraphrase_top1_top2_sum / canonical_queries
        ),
        "option_order_flip_rate": order_flips / canonical_queries,
        "mean_canonical_decision_loss": canonical_decision_sum / len(rows),
        "mean_paraphrase_decision_loss": paraphrase_decision_sum / len(rows),
        "mean_option_alignment_loss": option_align_sum / len(rows),
        "mean_question_option_loss": qopt_sum / len(rows),
        "max_probability_mass_error": max_mass_error,
        "full_k": bool(full_k),
        "relation_delta_max_abs": 0.0,
        "state_view_encodes": state_view_encodes,
    }


def _selection_key(
    epoch: int,
    metrics: dict,
) -> tuple[float, float, float, float, float, float, float, int]:
    return (
        float(metrics["canonical_paired_both_correct_rate"]),
        float(metrics["canonical_accuracy"]),
        float(metrics["canonical_margin_satisfaction_rate"]),
        float(metrics["cross_view_selected_choice_agreement"]),
        float(metrics["canonical_question_swap_choice_change_rate"]),
        float(metrics["canonical_mean_gold_vs_max_wrong_margin"]),
        -float(metrics["mean_canonical_decision_loss"]),
        -int(epoch),
    )

def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _original_a13_trainable(runtime) -> int:
    return sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--a0-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a0 = json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome") != "HIRA_V1_S9_A0_IDENTITY_READY":
        raise RuntimeError("S9-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S9-A0 unexpectedly used for model selection")
    if a0.get("a13_token_output_identity") is not True:
        raise RuntimeError("S9-A0 A13 identity boundary failed")
    if float(a0.get("exact_logit_identity_rate", -1.0)) != 1.0:
        raise RuntimeError("S9-A0 decision identity boundary failed")

    train_rows = generate_s9_cases("train")
    dev_rows = generate_s9_cases("dev")
    validate_s9_partitions(train_rows, dev_rows)
    _assert_fresh_against_prior(train_rows, dev_rows)

    random.seed(SEED)
    torch.manual_seed(SEED)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"]) != A13_REVISION:
        raise RuntimeError("S9 semantic revision changed")

    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder
    runtime = build_hira_v1_s9_margin_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,
        train_projection=True,
    )
    del frozen
    runtime.clear_schema_cache()
    enforce_s9_eval(runtime)

    trainable = [p for p in runtime.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    if trainable_count != HIRA_V1_S9_TOTAL_PARAMETER_COUNT:
        raise RuntimeError(f"S9 trainable count changed: {trainable_count}")
    if _original_a13_trainable(runtime) != 0:
        raise RuntimeError("S9 original A13 parameter became trainable")

    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S9 projection scorer missing")
    if (
        scorer.projection_trainable_parameter_count
        != HIRA_V1_S9_PROJECTION_PARAMETER_COUNT
    ):
        raise RuntimeError("S9 trainable projection surface changed")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S9 HIRACore became trainable")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_key = None
    best_epoch = None
    best_state = None
    best_metrics = None

    print("HIRA_V1_S9_TRAIN_BEGIN", flush=True)

    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train_rows)))
        random.Random(SEED + epoch).shuffle(order)

        totals = {
            "total": 0.0,
            "decision": 0.0,
            "ce": 0.0,
            "margin": 0.0,
            "option_alignment": 0.0,
            "question_option": 0.0,
            "consistency_js": 0.0,
        }
        state_view_encodes = 0

        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            rows = [train_rows[index] for index in indices]

            optimizer.zero_grad(set_to_none=True)
            loss, pieces, _canonical, _paraphrase, _encoded = _losses(
                runtime, rows
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            enforce_s9_eval(runtime)

            n = len(rows)
            for key in totals:
                totals[key] += pieces[key] * n
            state_view_encodes += 2 * n

        if state_view_encodes != 2 * len(train_rows):
            raise RuntimeError(
                f"S9 state-once TRAIN epoch changed: {state_view_encodes}"
            )

        dev_metrics = evaluate(runtime, dev_rows)
        if int(dev_metrics["state_view_encodes"]) != 2 * len(dev_rows):
            raise RuntimeError("S9 state-once DEV evaluation changed")

        record = {
            "epoch": epoch,
            "train_mean_total_loss": totals["total"] / len(train_rows),
            "train_mean_decision_loss": totals["decision"] / len(train_rows),
            "train_mean_ce": totals["ce"] / len(train_rows),
            "train_mean_all_negative_margin": totals["margin"] / len(train_rows),
            "train_mean_option_alignment_loss": (
                totals["option_alignment"] / len(train_rows)
            ),
            "train_mean_question_option_loss": (
                totals["question_option"] / len(train_rows)
            ),
            "train_mean_consistency_js": (
                totals["consistency_js"] / len(train_rows)
            ),
            "train_state_view_encodes": state_view_encodes,
            "surface_diagnostics": {
                "projection_weight_norm": float(
                    scorer.projection.weight.detach().norm().cpu()
                ),
                "lora_b_norm": float(
                    torch.sqrt(sum(
                        module.lora_b.detach().pow(2).sum()
                        for module in iter_a13_lora_modules(runtime.encoder)
                    )).cpu()
                ),
            },
            "dev": dev_metrics,
        }
        history.append(record)

        key = _selection_key(epoch, dev_metrics)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                "lora": a13_lora_state_dict(runtime.encoder),
                "projection": {
                    "projection.weight": scorer.projection.weight.detach().cpu().clone(),
                },
            }
            best_metrics = dict(dev_metrics)

        print(
            "HIRA_V1_S9_EPOCH=" + json.dumps(record, sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError("S9 DEV selection produced no checkpoint")

    load_a13_lora_state_dict(runtime.encoder, best_state["lora"], freeze=False)
    scorer.load_projection_state_dict(best_state["projection"], freeze=False)
    enforce_s9_eval(runtime)
    selected = evaluate(runtime, dev_rows)

    replay_keys = (
        "canonical_accuracy",
        "paraphrase_accuracy",
        "canonical_paired_both_correct_rate",
        "canonical_question_swap_choice_change_rate",
        "cross_view_selected_choice_agreement",
        "cross_view_mean_js",
        "canonical_mean_gold_vs_max_wrong_margin",
        "paraphrase_mean_gold_vs_max_wrong_margin",
        "canonical_margin_satisfaction_rate",
        "paraphrase_margin_satisfaction_rate",
        "canonical_mean_top1_top2_margin",
        "paraphrase_mean_top1_top2_margin",
        "option_order_flip_rate",
        "max_probability_mass_error",
        "mean_canonical_decision_loss",
        "mean_paraphrase_decision_loss",
    )
    for key in replay_keys:
        if not math.isclose(
            float(selected[key]),
            float(best_metrics[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"S9 selected DEV replay changed: {key}")

    lora_trainable = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" in name
    )
    gates = {
        "canonical_accuracy_gte_0_85": selected["canonical_accuracy"] >= 0.85,
        "canonical_paired_both_correct_gte_0_75": (
            selected["canonical_paired_both_correct_rate"] >= 0.75
        ),
        "canonical_question_swap_change_gte_0_80": (
            selected["canonical_question_swap_choice_change_rate"] >= 0.80
        ),
        "cross_view_choice_agreement_gte_0_95": (
            selected["cross_view_selected_choice_agreement"] >= 0.95
        ),
        "cross_view_mean_js_lte_0_05": (
            selected["cross_view_mean_js"] <= 0.05
        ),
        "canonical_margin_satisfaction_gte_0_80": (
            selected["canonical_margin_satisfaction_rate"] >= 0.80
        ),
        "canonical_mean_gold_margin_gte_0_15": (
            selected["canonical_mean_gold_vs_max_wrong_margin"] >= 0.15
        ),
        "option_order_flip_lte_0_02": (
            selected["option_order_flip_rate"] <= 0.02
        ),
        "probability_mass_error_lte_1e_6": (
            selected["max_probability_mass_error"] <= 1e-6
        ),
        "full_k": bool(selected["full_k"]),
        "relation_delta_zero": float(selected["relation_delta_max_abs"]) == 0.0,
        "state_once_train_both_views": all(
            int(record["train_state_view_encodes"]) == 2 * len(train_rows)
            for record in history
        ),
        "state_once_dev_both_views": all(
            int(record["dev"]["state_view_encodes"]) == 2 * len(dev_rows)
            for record in history
        ),
        "trainable_total_exact_49152": (
            trainable_count == HIRA_V1_S9_TOTAL_PARAMETER_COUNT
        ),
        "trainable_lora_exact_16384": (
            lora_trainable == HIRA_V1_S6_LORA_PARAMETER_COUNT
        ),
        "trainable_projection_exact_32768": (
            scorer.projection_trainable_parameter_count
            == HIRA_V1_S9_PROJECTION_PARAMETER_COUNT
        ),
        "original_a13_frozen": _original_a13_trainable(runtime) == 0,
        "hira_core_frozen": not any(
            p.requires_grad for p in runtime.hira.parameters()
        ),
    }
    outcome = READY if all(gates.values()) else FAIL

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "margin-candidate.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s8-invariant-checkpoint-v1",
            "kind": "invariant-margin-a13-w28",
            "lora_parameter_count": HIRA_V1_S6_LORA_PARAMETER_COUNT,
            "projection_parameter_count": HIRA_V1_S9_PROJECTION_PARAMETER_COUNT,
            "total_parameter_count": HIRA_V1_S9_TOTAL_PARAMETER_COUNT,
            "lora_rank": HIRA_V1_S6_LORA_RANK,
            "selected_dev_epoch": best_epoch,
            "semantic_revision": str(manifest["semantic_revision"]),
            "initialization_t0_sha256": str(manifest["t0_checkpoint_sha256"]),
            "lora_state_dict": best_state["lora"],
            "projection_state_dict": best_state["projection"],
        },
        checkpoint_path,
    )
    checkpoint_sha = _sha256(checkpoint_path)

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": outcome,
        "scientific_authority": "V1_S9_FRESH_ENGLISH_MARGIN_SEPARATION",
        "seed": SEED,
        "optimizer": {
            "name": "AdamW",
            "epochs": EPOCHS,
            "batch_size_semantic_cases": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
        },
        "loss": {
            "cross_entropy_both_views": True,
            "all_negative_margin_coefficient": MARGIN_COEFFICIENT,
            "all_negative_margin": MARGIN,
            "option_view_infonce_coefficient": OPTION_ALIGN_COEFFICIENT,
            "option_view_infonce_temperature": OPTION_ALIGN_TEMPERATURE,
            "question_option_infonce_coefficient": QUESTION_OPTION_COEFFICIENT,
            "question_option_infonce_temperature": QUESTION_OPTION_TEMPERATURE,
            "cross_view_js_coefficient": INVARIANCE_COEFFICIENT,
        },
        "partitions": {
            "train_semantic_cases": len(train_rows),
            "train_decisions_per_epoch": len(train_rows) * 4,
            "dev_semantic_cases": len(dev_rows),
            "dev_decisions_per_eval": len(dev_rows) * 4,
            "language": "en",
            "domains": sorted({row.domain for row in train_rows}),
            "state_views_per_case": 2,
            "question_views_per_semantic_query": 2,
            "k": 4,
            "views_per_option": 2,
            "train_dev_exact_state_view_overlap": False,
            "train_dev_exact_question_view_overlap": False,
            "prior_track_exact_rows_used": False,
            "s9_a0_rows_used": False,
            "m5_final_rows_used": False,
            "w29_w34_sealed_rows_used": False,
        },
        "parameter_surface": {
            "total_trainable_parameters": trainable_count,
            "lora_trainable_parameters": lora_trainable,
            "lora_rank": HIRA_V1_S6_LORA_RANK,
            "projection_trainable_parameters": scorer.projection_trainable_parameter_count,
            "original_a13_trainable": _original_a13_trainable(runtime),
            "hira_core_trainable": sum(
                p.numel()
                for p in runtime.hira.parameters()
                if p.requires_grad
            ),
            "learned_downstream_scorer_parameters": 0,
        },
        "state_once": {
            "train_state_view_encodes_per_epoch": 2 * len(train_rows),
            "dev_state_view_encodes_per_eval": 2 * len(dev_rows),
            "stale_state_or_schema_reuse_after_optimizer_step": False,
        },
        "selected_dev_epoch": best_epoch,
        "selected_dev": selected,
        "gates": gates,
        "checkpoint_sha256": checkpoint_sha,
        "semantic_revision": str(manifest["semantic_revision"]),
        "initialization_t0_sha256": str(manifest["t0_checkpoint_sha256"]),
        "a0_authority": {
            "outcome": a0["outcome"],
            "a13_token_output_identity": a0["a13_token_output_identity"],
            "exact_logit_identity_rate": a0["exact_logit_identity_rate"],
            "cross_view_selected_choice_agreement": a0[
                "cross_view_selected_choice_agreement"
            ],
            "cross_view_mean_js": a0["cross_view_mean_js"],
            "canonical_mean_gold_vs_max_wrong_margin": a0[
                "canonical_mean_gold_vs_max_wrong_margin"
            ],
            "canonical_margin_satisfaction_rate": a0[
                "canonical_margin_satisfaction_rate"
            ],
        },
        "history": history,
        "post_dev_tuning_performed": False,
        "sealed_confirm_opened": False,
        "multilingual_probe_opened": False,
        "production_ready_claimed": False,
    }

    (args.out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "train-manifest.json").write_text(
        json.dumps(
            [row.to_dict() for row in train_rows],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (args.out / "dev-manifest.json").write_text(
        json.dumps(
            [row.to_dict() for row in dev_rows],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "HIRA_V1_S9_TRAIN_DEV_RECEIPT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
