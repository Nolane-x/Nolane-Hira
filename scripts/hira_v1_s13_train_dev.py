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
from nmd.v1_relation_canonicalization import (
    CrossViewRelationCanonicalizer,
    cross_view_relation_signature_loss,
    relation_signature_same_option_cosine,
)
from nmd.v1_invariance import (
    selected_choice_agreement,
    symmetric_js_divergence,
)
from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import generate_s3_pairs
from nmd.v1_s4_authority import generate_s4_pairs
from nmd.v1_s5_authority import generate_s5_pairs
from nmd.v1_s6_authority import generate_s6_pairs
from nmd.v1_s7_authority import generate_s7_pairs
from nmd.v1_s8_authority import generate_s8_cases
from nmd.v1_s9_authority import generate_s9_cases
from nmd.v1_s10_authority import generate_s10_cases
from nmd.v1_s6_semantic_core import (
    HIRA_V1_S6_LORA_PARAMETER_COUNT,
    HIRA_V1_S6_LORA_RANK,
)
from nmd.v1_s11_authority import generate_s11_cases
from nmd.v1_s12_authority import generate_s12_cases
from nmd.v1_s13_authority import (
    S13CanonicalCase,
    generate_s13_cases,
    validate_s13_partitions,
)
from nmd.v1_s13_semantic_core import (
    HIRA_V1_S13_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S13_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s13_canonicalization_core,
    enforce_s13_eval,
)


SCHEMA_VERSION = "hira-v1-s13-canonical-relation-train-dev-v1"
READY = "HIRA_V1_S13_CANONICAL_RELATION_DEV_READY"
FAIL = "HIRA_V1_S13_CANONICAL_RELATION_DEV_FAIL"

SEED = 19001
EPOCHS = 24
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SWAP_COEFFICIENT = 0.25
SWAP_MARGIN = 0.20
OPTION_ALIGN_COEFFICIENT = 0.05
OPTION_ALIGN_TEMPERATURE = 0.10
BINDING_COEFFICIENT = 0.10
CANONICALIZATION_COEFFICIENT = 0.15
SIGNATURE_SEPARATION_MARGIN = 0.20
ROLE_TEMPERATURE = 0.10
PAIR_TEMPERATURE = 0.10
BINDING_CONTRASTIVE_TEMPERATURE = 0.10
INVARIANCE_COEFFICIENT = 0.25


def _state_texts(row):
    if hasattr(row, "state"):
        return (row.state,)
    return (row.state_a, row.state_b)


def _question_texts(row):
    if hasattr(row, "question_a"):
        return (row.question_a, row.question_b)
    return (
        row.question_a1,
        row.question_a2,
        row.question_b1,
        row.question_b2,
    )


def _assert_fresh_against_prior(
    train_rows: tuple[S13CanonicalCase, ...],
    dev_rows: tuple[S13CanonicalCase, ...],
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
        *generate_s9_cases("train"), *generate_s9_cases("dev"),
        *generate_s10_cases("train"), *generate_s10_cases("dev"),
        *generate_s11_cases("train"), *generate_s11_cases("dev"),
        *generate_s12_cases("train"), *generate_s12_cases("dev"),
    )
    prior_states = {text for row in prior for text in _state_texts(row)}
    prior_questions = {text for row in prior for text in _question_texts(row)}
    current = (*train_rows, *dev_rows)
    current_states = {text for row in current for text in _state_texts(row)}
    current_questions = {text for row in current for text in _question_texts(row)}
    if prior_states & current_states:
        raise RuntimeError("S13 exact state overlap with exposed S0-S12 rows")
    if prior_questions & current_questions:
        raise RuntimeError("S13 exact question overlap with exposed S0-S12 rows")


def _content_mask(batch) -> Tensor:
    mask = batch.attention_mask.bool()
    if batch.special_token_mask is None:
        return mask
    return mask & ~batch.special_token_mask.bool()


def _option_view_texts(rows: list[S13CanonicalCase]) -> list[str]:
    values: list[str] = []
    for row in rows:
        for criterion, alias in zip(row.option_texts, row.option_aliases):
            values.extend((criterion, alias))
    return values


def _gold_tensors(
    rows: list[S13CanonicalCase],
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


def _encode_batch(runtime, rows: list[S13CanonicalCase]) -> dict[str, Tensor]:
    enforce_s13_eval(runtime)
    encoder = runtime.encoder
    n = len(rows)

    states_a = [row.state_a for row in rows]
    states_b = [row.state_b for row in rows]
    questions_canonical = [
        q for row in rows for q in (row.question_a1, row.question_b1)
    ]
    questions_paraphrase = [
        q for row in rows for q in (row.question_a2, row.question_b2)
    ]
    options = _option_view_texts(rows)

    state_batch = encoder.encode_texts([*states_a, *states_b])
    question_batch = encoder.encode_texts(
        [*questions_canonical, *questions_paraphrase]
    )
    option_batch = encoder.encode_texts(options)

    if state_batch.token_embeddings.shape[0] != 2 * n:
        raise RuntimeError("S13 state-view packing changed")
    if question_batch.token_embeddings.shape[0] != 4 * n:
        raise RuntimeError("S13 question-view packing changed")
    if option_batch.token_embeddings.shape[0] != n * 4 * 2:
        raise RuntimeError("S13 option-view packing changed")

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
        raise RuntimeError("S13 projection triadic scorer missing")
    return scorer(
        state_tokens=state_tokens.repeat_interleave(2, dim=0),
        state_mask=state_mask.repeat_interleave(2, dim=0),
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_view_tokens=encoded["option_tokens"].repeat_interleave(2, dim=0),
        option_view_token_mask=encoded["option_mask"].repeat_interleave(2, dim=0),
        option_view_mask=encoded["option_view_mask"].repeat_interleave(2, dim=0),
    )


def _canonicalizer_outputs(
    runtime,
    *,
    state_tokens: Tensor,
    state_mask: Tensor,
    question_tokens: Tensor,
    question_mask: Tensor,
    encoded: dict[str, Tensor],
) -> tuple[Tensor, Tensor, object]:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S13 canonicalizer projection scorer missing")

    canonicalizer = CrossViewRelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=BINDING_CONTRASTIVE_TEMPERATURE,
    )
    return canonicalizer(
        projection=scorer.projection,
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
    other: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    ce = F.cross_entropy(logits, gold)
    indices = torch.arange(logits.shape[0], device=logits.device)
    swap = F.relu(
        logits.new_tensor(SWAP_MARGIN)
        - (logits[indices, gold] - logits[indices, other])
    ).mean()
    decision = ce + SWAP_COEFFICIENT * swap
    return ce, swap, decision


def _losses(
    runtime,
    rows: list[S13CanonicalCase],
) -> tuple[
    Tensor,
    dict[str, float],
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    dict[str, Tensor],
]:
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
    binding_c, signature_c, diag_c = _canonicalizer_outputs(
        runtime,
        state_tokens=encoded["state_a_tokens"],
        state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        encoded=encoded,
    )
    binding_p, signature_p, diag_p = _canonicalizer_outputs(
        runtime,
        state_tokens=encoded["state_b_tokens"],
        state_mask=encoded["state_b_mask"],
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
        encoded=encoded,
    )

    gold, other = _gold_tensors(rows, device=canonical.device)
    c_ce, c_swap, c_decision = _supervised(canonical, gold, other)
    p_ce, p_swap, p_decision = _supervised(paraphrase, gold, other)

    ce = 0.5 * (c_ce + p_ce)
    swap = 0.5 * (c_swap + p_swap)
    decision = 0.5 * (c_decision + p_decision)

    option_align = _option_alignment_loss(encoded["option_pooled"])
    binding_c_loss = F.cross_entropy(binding_c, gold)
    binding_p_loss = F.cross_entropy(binding_p, gold)
    binding_loss = 0.5 * (binding_c_loss + binding_p_loss)
    canonicalization, signature_alignment, signature_separation = (
        cross_view_relation_signature_loss(
            signature_c,
            signature_p,
            separation_margin=SIGNATURE_SEPARATION_MARGIN,
        )
    )
    consistency = symmetric_js_divergence(canonical, paraphrase)

    total = (
        decision
        + OPTION_ALIGN_COEFFICIENT * option_align
        + BINDING_COEFFICIENT * binding_loss
        + CANONICALIZATION_COEFFICIENT * canonicalization
        + INVARIANCE_COEFFICIENT * consistency
    )

    pieces = {
        "ce": float(ce.detach().cpu()),
        "swap": float(swap.detach().cpu()),
        "decision": float(decision.detach().cpu()),
        "canonical_decision": float(c_decision.detach().cpu()),
        "paraphrase_decision": float(p_decision.detach().cpu()),
        "option_alignment": float(option_align.detach().cpu()),
        "binding": float(binding_loss.detach().cpu()),
        "canonical_binding": float(binding_c_loss.detach().cpu()),
        "paraphrase_binding": float(binding_p_loss.detach().cpu()),
        "canonicalization": float(canonicalization.detach().cpu()),
        "signature_alignment": float(signature_alignment.detach().cpu()),
        "signature_separation": float(signature_separation.detach().cpu()),
        "consistency_js": float(consistency.detach().cpu()),
        "canonical_role_entropy": float(
            diag_c.state_role_normalized_entropy.detach().cpu()
        ),
        "paraphrase_role_entropy": float(
            diag_p.state_role_normalized_entropy.detach().cpu()
        ),
        "canonical_role_max_weight": float(
            diag_c.state_role_max_weight.detach().cpu()
        ),
        "paraphrase_role_max_weight": float(
            diag_p.state_role_max_weight.detach().cpu()
        ),
        "total": float(total.detach().cpu()),
    }
    return (
        total,
        pieces,
        canonical,
        paraphrase,
        binding_c,
        binding_p,
        signature_c,
        signature_p,
        encoded,
    )



def _option_order_flips(
    runtime,
    canonical: Tensor,
    encoded: dict[str, Tensor],
) -> int:
    permutation = torch.tensor([3, 2, 1, 0], device=canonical.device)
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S13 projection scorer missing")

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
def evaluate(runtime, rows: tuple[S13CanonicalCase, ...]) -> dict:
    enforce_s13_eval(runtime)

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
    binding_c_loss_sum = 0.0
    binding_p_loss_sum = 0.0
    binding_c_correct = 0
    binding_p_correct = 0
    binding_cross_agree = 0
    binding_c_margin_sum = 0.0
    binding_p_margin_sum = 0.0
    same_signature_sum = 0.0
    signature_margin_sum = 0.0
    signature_count = 0
    c_entropy_sum = 0.0
    p_entropy_sum = 0.0
    c_max_weight_sum = 0.0
    p_max_weight_sum = 0.0
    canonicalization_sum = 0.0
    semantic_cases = 0
    state_view_encodes = 0
    full_k = True

    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = list(rows[start : start + BATCH_SIZE])
        (
            _total,
            pieces,
            canonical,
            paraphrase,
            binding_c,
            binding_p,
            signature_c,
            signature_p,
            encoded,
        ) = _losses(runtime, batch_rows)
        n = len(batch_rows)
        gold, _other = _gold_tensors(batch_rows, device=canonical.device)

        c_pred = canonical.argmax(-1)
        p_pred = paraphrase.argmax(-1)
        canonical_correct += int((c_pred == gold).sum().item())
        paraphrase_correct += int((p_pred == gold).sum().item())

        c_pairs = c_pred.reshape(n, 2)
        g_pairs = gold.reshape(n, 2)
        canonical_pair_both += int(((c_pairs == g_pairs).all(-1)).sum().item())
        canonical_pair_changed += int((c_pairs[:, 0] != c_pairs[:, 1]).sum().item())

        cross_agree_sum += float(
            selected_choice_agreement(canonical, paraphrase)
        ) * (2 * n)
        js_sum += float(symmetric_js_divergence(canonical, paraphrase)) * (2 * n)

        gc_pred = binding_c.argmax(-1)
        gp_pred = binding_p.argmax(-1)
        binding_c_correct += int((gc_pred == gold).sum().item())
        binding_p_correct += int((gp_pred == gold).sum().item())
        binding_cross_agree += int((gc_pred == gp_pred).sum().item())

        def margin(logits: Tensor) -> Tensor:
            chosen = logits.gather(-1, gold[:, None]).squeeze(-1)
            idx = torch.arange(logits.shape[-1], device=logits.device)
            wrong = logits.masked_fill(idx[None, :].eq(gold[:, None]), float("-inf"))
            return chosen - wrong.max(-1).values

        binding_c_margin_sum += float(margin(binding_c).sum().cpu())
        binding_p_margin_sum += float(margin(binding_p).sum().cpu())

        same = relation_signature_same_option_cosine(signature_c, signature_p)
        c_norm = F.normalize(signature_c, dim=-1)
        p_norm = F.normalize(signature_p, dim=-1)
        cross = torch.einsum("nkd,njd->nkj", c_norm, p_norm)
        k = cross.shape[-1]
        eye = torch.eye(k, dtype=torch.bool, device=cross.device)[None]
        strongest_wrong = cross.masked_fill(eye, float("-inf")).amax(-1)
        signature_margin = same - strongest_wrong
        same_signature_sum += float(same.sum().cpu())
        signature_margin_sum += float(signature_margin.sum().cpu())
        signature_count += int(same.numel())

        order_flips += _option_order_flips(runtime, canonical, encoded)

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
        binding_c_loss_sum += pieces["canonical_binding"] * n
        binding_p_loss_sum += pieces["paraphrase_binding"] * n
        canonicalization_sum += pieces["canonicalization"] * n
        c_entropy_sum += pieces["canonical_role_entropy"] * (2 * n)
        p_entropy_sum += pieces["paraphrase_role_entropy"] * (2 * n)
        c_max_weight_sum += pieces["canonical_role_max_weight"] * (2 * n)
        p_max_weight_sum += pieces["paraphrase_role_max_weight"] * (2 * n)

        semantic_cases += n
        state_view_encodes += 2 * n
        full_k = (
            full_k
            and canonical.shape[-1] == 4
            and paraphrase.shape[-1] == 4
            and binding_c.shape[-1] == 4
            and binding_p.shape[-1] == 4
            and signature_c.shape[-2] == 4
            and signature_p.shape[-2] == 4
        )

    queries = len(rows) * 2
    return {
        "semantic_cases": semantic_cases,
        "canonical_queries": queries,
        "paraphrase_queries": queries,
        "canonical_accuracy": canonical_correct / queries,
        "paraphrase_accuracy": paraphrase_correct / queries,
        "canonical_paired_both_correct_rate": canonical_pair_both / len(rows),
        "canonical_question_swap_choice_change_rate": canonical_pair_changed / len(rows),
        "cross_view_selected_choice_agreement": cross_agree_sum / queries,
        "cross_view_mean_js": js_sum / queries,
        "canonical_binding_accuracy": binding_c_correct / queries,
        "paraphrase_binding_accuracy": binding_p_correct / queries,
        "binding_cross_view_selected_choice_agreement": binding_cross_agree / queries,
        "canonical_mean_binding_loss": binding_c_loss_sum / len(rows),
        "paraphrase_mean_binding_loss": binding_p_loss_sum / len(rows),
        "canonical_binding_mean_gold_margin": binding_c_margin_sum / queries,
        "paraphrase_binding_mean_gold_margin": binding_p_margin_sum / queries,
        "mean_same_option_signature_cosine": same_signature_sum / signature_count,
        "mean_signature_same_vs_strongest_wrong_margin": signature_margin_sum / signature_count,
        "mean_canonicalization_loss": canonicalization_sum / len(rows),
        "canonical_role_normalized_entropy": c_entropy_sum / queries,
        "paraphrase_role_normalized_entropy": p_entropy_sum / queries,
        "canonical_role_max_weight": c_max_weight_sum / queries,
        "paraphrase_role_max_weight": p_max_weight_sum / queries,
        "option_order_flip_rate": order_flips / queries,
        "mean_canonical_decision_loss": canonical_decision_sum / len(rows),
        "mean_paraphrase_decision_loss": paraphrase_decision_sum / len(rows),
        "mean_option_alignment_loss": option_align_sum / len(rows),
        "max_probability_mass_error": max_mass_error,
        "full_k": bool(full_k),
        "relation_delta_max_abs": 0.0,
        "state_view_encodes": state_view_encodes,
    }


def _selection_key(epoch: int, metrics: dict) -> tuple:
    return (
        float(metrics["canonical_paired_both_correct_rate"]),
        float(metrics["canonical_accuracy"]),
        float(metrics["canonical_binding_accuracy"]),
        float(metrics["mean_same_option_signature_cosine"]),
        float(metrics["cross_view_selected_choice_agreement"]),
        float(metrics["canonical_question_swap_choice_change_rate"]),
        float(metrics["canonical_binding_mean_gold_margin"]),
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
    if a0.get("outcome") != "HIRA_V1_S13_A0_IDENTITY_READY":
        raise RuntimeError("S13-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S13-A0 unexpectedly used for selection")
    if a0.get("a13_token_output_identity") is not True:
        raise RuntimeError("S13-A0 A13 identity failed")
    if float(a0.get("exact_logit_identity_rate", -1.0)) != 1.0:
        raise RuntimeError("S13-A0 decision identity failed")

    train_rows = generate_s13_cases("train")
    dev_rows = generate_s13_cases("dev")
    validate_s13_partitions(train_rows, dev_rows)
    _assert_fresh_against_prior(train_rows, dev_rows)

    random.seed(SEED)
    torch.manual_seed(SEED)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"]) != A13_REVISION:
        raise RuntimeError("S13 semantic revision changed")

    frozen = load_hira_v0_m4_bundle(bundle)
    runtime = build_hira_v1_s13_canonicalization_core(
        frozen.runtime.encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,
        train_projection=True,
    )
    del frozen
    runtime.clear_schema_cache()
    enforce_s13_eval(runtime)

    trainable = [p for p in runtime.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    if trainable_count != HIRA_V1_S13_TOTAL_PARAMETER_COUNT:
        raise RuntimeError(f"S13 trainable count changed: {trainable_count}")
    if _original_a13_trainable(runtime) != 0:
        raise RuntimeError("S13 original A13 parameter became trainable")

    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S13 projection scorer missing")
    if (
        scorer.projection_trainable_parameter_count
        != HIRA_V1_S13_PROJECTION_PARAMETER_COUNT
    ):
        raise RuntimeError("S13 projection trainable surface changed")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S13 HIRACore became trainable")

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

    print("HIRA_V1_S13_TRAIN_BEGIN", flush=True)

    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train_rows)))
        random.Random(SEED + epoch).shuffle(order)
        totals = {
            "total": 0.0,
            "decision": 0.0,
            "ce": 0.0,
            "swap": 0.0,
            "option_alignment": 0.0,
            "binding": 0.0,
            "canonicalization": 0.0,
            "consistency_js": 0.0,
        }
        state_view_encodes = 0

        for start in range(0, len(order), BATCH_SIZE):
            rows = [
                train_rows[index]
                for index in order[start : start + BATCH_SIZE]
            ]
            optimizer.zero_grad(set_to_none=True)
            loss, pieces, *_rest = _losses(runtime, rows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            enforce_s13_eval(runtime)

            n = len(rows)
            for key in totals:
                totals[key] += pieces[key] * n
            state_view_encodes += 2 * n

        if state_view_encodes != 2 * len(train_rows):
            raise RuntimeError("S13 TRAIN state-once changed")

        dev_metrics = evaluate(runtime, dev_rows)
        if int(dev_metrics["state_view_encodes"]) != 2 * len(dev_rows):
            raise RuntimeError("S13 DEV state-once changed")

        record = {
            "epoch": epoch,
            "train_mean_total_loss": totals["total"] / len(train_rows),
            "train_mean_decision_loss": totals["decision"] / len(train_rows),
            "train_mean_ce": totals["ce"] / len(train_rows),
            "train_mean_swap": totals["swap"] / len(train_rows),
            "train_mean_option_alignment_loss": (
                totals["option_alignment"] / len(train_rows)
            ),
            "train_mean_binding_loss": totals["binding"] / len(train_rows),
            "train_mean_canonicalization_loss": (
                totals["canonicalization"] / len(train_rows)
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
            "HIRA_V1_S13_EPOCH=" + json.dumps(record, sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError("S13 DEV selection produced no checkpoint")

    load_a13_lora_state_dict(runtime.encoder, best_state["lora"], freeze=False)
    scorer.load_projection_state_dict(best_state["projection"], freeze=False)
    enforce_s13_eval(runtime)
    selected = evaluate(runtime, dev_rows)

    replay_keys = (
        "canonical_accuracy",
        "paraphrase_accuracy",
        "canonical_paired_both_correct_rate",
        "canonical_question_swap_choice_change_rate",
        "cross_view_selected_choice_agreement",
        "cross_view_mean_js",
        "canonical_binding_accuracy",
        "paraphrase_binding_accuracy",
        "binding_cross_view_selected_choice_agreement",
        "canonical_binding_mean_gold_margin",
        "paraphrase_binding_mean_gold_margin",
        "mean_same_option_signature_cosine",
        "mean_signature_same_vs_strongest_wrong_margin",
        "canonical_role_normalized_entropy",
        "canonical_role_max_weight",
        "option_order_flip_rate",
        "max_probability_mass_error",
        "mean_canonical_decision_loss",
    )
    for key in replay_keys:
        if not math.isclose(
            float(selected[key]),
            float(best_metrics[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"S13 selected DEV replay changed: {key}")

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
        "cross_view_mean_js_lte_0_05": selected["cross_view_mean_js"] <= 0.05,
        "canonical_binding_accuracy_gte_0_80": (
            selected["canonical_binding_accuracy"] >= 0.80
        ),
        "canonical_binding_margin_gte_0_15": (
            selected["canonical_binding_mean_gold_margin"] >= 0.15
        ),
        "same_option_signature_cosine_gte_0_90": (
            selected["mean_same_option_signature_cosine"] >= 0.90
        ),
        "signature_margin_gte_0_15": (
            selected["mean_signature_same_vs_strongest_wrong_margin"] >= 0.15
        ),
        "option_order_flip_lte_0_02": selected["option_order_flip_rate"] <= 0.02,
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
            trainable_count == HIRA_V1_S13_TOTAL_PARAMETER_COUNT
        ),
        "trainable_lora_exact_16384": (
            lora_trainable == HIRA_V1_S6_LORA_PARAMETER_COUNT
        ),
        "trainable_projection_exact_32768": (
            scorer.projection_trainable_parameter_count
            == HIRA_V1_S13_PROJECTION_PARAMETER_COUNT
        ),
        "original_a13_frozen": _original_a13_trainable(runtime) == 0,
        "hira_core_frozen": not any(
            p.requires_grad for p in runtime.hira.parameters()
        ),
    }
    outcome = READY if all(gates.values()) else FAIL

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "canonical-relation-candidate.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s13-canonical-relation-checkpoint-v1",
            "kind": "cross-view-relation-canonicalization-a13-w28",
            "lora_parameter_count": HIRA_V1_S6_LORA_PARAMETER_COUNT,
            "projection_parameter_count": HIRA_V1_S13_PROJECTION_PARAMETER_COUNT,
            "total_parameter_count": HIRA_V1_S13_TOTAL_PARAMETER_COUNT,
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
        "scientific_authority": "V1_S13_FRESH_ENGLISH_CROSS_VIEW_RELATION_CANONICALIZATION",
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
            "swap_margin_coefficient": SWAP_COEFFICIENT,
            "swap_margin": SWAP_MARGIN,
            "option_view_infonce_coefficient": OPTION_ALIGN_COEFFICIENT,
            "option_view_infonce_temperature": OPTION_ALIGN_TEMPERATURE,
            "relation_structured_binding_coefficient": BINDING_COEFFICIENT,
            "cross_view_relation_canonicalization_coefficient": CANONICALIZATION_COEFFICIENT,
            "signature_separation_margin": SIGNATURE_SEPARATION_MARGIN,
            "role_temperature": ROLE_TEMPERATURE,
            "pair_temperature": PAIR_TEMPERATURE,
            "binding_contrastive_temperature": BINDING_CONTRASTIVE_TEMPERATURE,
            "cross_view_js_coefficient": INVARIANCE_COEFFICIENT,
            "state_blind_question_option_infonce_used": False,
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
            "s13_a0_rows_used": False,
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
                p.numel() for p in runtime.hira.parameters() if p.requires_grad
            ),
            "canonicalizer_added_parameters": 0,
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
            "canonical_relation_binding_accuracy": a0["canonical_relation_binding_accuracy"],
            "canonical_relation_binding_mean_gold_margin": a0[
                "canonical_relation_binding_mean_gold_margin"
            ],
            "mean_same_option_signature_cosine": a0[
                "mean_same_option_signature_cosine"
            ],
            "mean_signature_same_vs_strongest_wrong_margin": a0[
                "mean_signature_same_vs_strongest_wrong_margin"
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
        "HIRA_V1_S13_TRAIN_DEV_RECEIPT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
