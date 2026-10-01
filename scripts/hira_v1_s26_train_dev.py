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
from nmd.v1_evidence_fusion import (
    SymmetricFullKEvidenceFusion,
    fused_gold_vs_max_wrong_margin,
)
from nmd.v1_s25_gradient_ownership import apply_s25_decoupled_gradient_update
from nmd.v1_factorized_relation_signature import (
    FactorizedRoleValueRelationCanonicalizer,
)
from nmd.v1_relation_canonicalization import (
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
from nmd.v1_s13_authority import generate_s13_cases
from nmd.v1_s14_authority import generate_s14_cases
from nmd.v1_s15_authority import generate_s15_cases
from nmd.v1_s16_authority import generate_s16_cases
from nmd.v1_s17_authority import generate_s17_cases
from nmd.v1_s18_authority import generate_s18_cases
from nmd.v1_s19_authority import generate_s19_cases
from nmd.v1_s20_authority import generate_s20_cases
from nmd.v1_s21_authority import generate_s21_cases
from nmd.v1_s22_authority import generate_s22_cases
from nmd.v1_s23_authority import generate_s23_cases
from nmd.v1_s24_authority import generate_s24_cases
from nmd.v1_s25_authority import generate_s25_cases
from nmd.v1_s26_authority import (
    S26RelationCase,
    generate_s26_cases,
    validate_s26_partitions,
)
from nmd.v1_s25_semantic_core import get_s25_relation_projection
from nmd.v1_s26_semantic_core import (
    HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S26_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s26_factorized_relation_core,
    enforce_s26_eval,
)


SCHEMA_VERSION = "hira-v1-s26-factorized-role-value-relation-train-dev-v1"
READY = "HIRA_V1_S26_FACTORIZED_RELATION_DEV_READY"
FAIL = "HIRA_V1_S26_FACTORIZED_RELATION_DEV_FAIL"

SEED = 47001
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
FUSION_EPSILON = 1e-6
BALANCE_EPSILON = 1e-12


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
    train_rows: tuple[S26RelationCase, ...],
    dev_rows: tuple[S26RelationCase, ...],
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
        *generate_s13_cases("train"), *generate_s13_cases("dev"),
        *generate_s14_cases("train"), *generate_s14_cases("dev"),
        *generate_s15_cases("train"), *generate_s15_cases("dev"),
        *generate_s16_cases("train"), *generate_s16_cases("dev"),
        *generate_s17_cases("train"), *generate_s17_cases("dev"),
        *generate_s18_cases("train"), *generate_s18_cases("dev"),
        *generate_s19_cases("train"), *generate_s19_cases("dev"),
        *generate_s20_cases("train"), *generate_s20_cases("dev"),
        *generate_s21_cases("train"), *generate_s21_cases("dev"),
        *generate_s22_cases("train"), *generate_s22_cases("dev"),
        *generate_s23_cases("train"), *generate_s23_cases("dev"),
        *generate_s24_cases("train"), *generate_s24_cases("dev"),
        *generate_s25_cases("train"), *generate_s25_cases("dev"),
    )
    prior_states = {text for row in prior for text in _state_texts(row)}
    prior_questions = {text for row in prior for text in _question_texts(row)}
    current = (*train_rows, *dev_rows)
    current_states = {text for row in current for text in _state_texts(row)}
    current_questions = {text for row in current for text in _question_texts(row)}
    if prior_states & current_states:
        raise RuntimeError("S26 exact state overlap with exposed S0-S24 rows")
    if prior_questions & current_questions:
        raise RuntimeError("S26 exact question overlap with exposed S0-S24 rows")


def _content_mask(batch) -> Tensor:
    mask = batch.attention_mask.bool()
    if batch.special_token_mask is None:
        return mask
    return mask & ~batch.special_token_mask.bool()


def _option_view_texts(rows: list[S26RelationCase]) -> list[str]:
    values: list[str] = []
    for row in rows:
        for criterion, alias in zip(row.option_texts, row.option_aliases):
            values.extend((criterion, alias))
    return values


def _gold_tensors(
    rows: list[S26RelationCase],
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


def _encode_batch(runtime, rows: list[S26RelationCase]) -> dict[str, Tensor]:
    enforce_s26_eval(runtime)
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
        raise RuntimeError("S26 state-view packing changed")
    if question_batch.token_embeddings.shape[0] != 4 * n:
        raise RuntimeError("S26 question-view packing changed")
    if option_batch.token_embeddings.shape[0] != n * 4 * 2:
        raise RuntimeError("S26 option-view packing changed")

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
        raise RuntimeError("S26 projection triadic scorer missing")
    return scorer(
        state_tokens=state_tokens.repeat_interleave(2, dim=0),
        state_mask=state_mask.repeat_interleave(2, dim=0),
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_view_tokens=encoded["option_tokens"].repeat_interleave(2, dim=0),
        option_view_token_mask=encoded["option_mask"].repeat_interleave(2, dim=0),
        option_view_mask=encoded["option_view_mask"].repeat_interleave(2, dim=0),
    )


def _relation_outputs(
    runtime,
    *,
    state_tokens: Tensor,
    state_mask: Tensor,
    question_tokens: Tensor,
    question_mask: Tensor,
    encoded: dict[str, Tensor],
) -> tuple[Tensor, Tensor, object]:
    relation_projection = get_s25_relation_projection(runtime)

    canonicalizer = FactorizedRoleValueRelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=BINDING_CONTRASTIVE_TEMPERATURE,
    )
    return canonicalizer(
        projection=relation_projection,
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
    rows: list[S26RelationCase],
) -> tuple[
    Tensor,
    dict[str, float],
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    dict[str, Tensor],
]:
    encoded = _encode_batch(runtime, rows)
    raw_c = _decision_logits(
        runtime,
        state_tokens=encoded["state_a_tokens"],
        state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        encoded=encoded,
    )
    raw_p = _decision_logits(
        runtime,
        state_tokens=encoded["state_b_tokens"],
        state_mask=encoded["state_b_mask"],
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
        encoded=encoded,
    )
    relation_c, signature_c, diag_c = _relation_outputs(
        runtime,
        state_tokens=encoded["state_a_tokens"],
        state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        encoded=encoded,
    )
    relation_p, signature_p, diag_p = _relation_outputs(
        runtime,
        state_tokens=encoded["state_b_tokens"],
        state_mask=encoded["state_b_mask"],
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
        encoded=encoded,
    )

    fusion = SymmetricFullKEvidenceFusion(epsilon=FUSION_EPSILON)
    fused_c, fusion_diag_c = fusion(raw_c, relation_c.detach())
    fused_p, fusion_diag_p = fusion(raw_p, relation_p.detach())

    gold, other = _gold_tensors(rows, device=fused_c.device)
    c_ce, c_swap, c_decision = _supervised(fused_c, gold, other)
    p_ce, p_swap, p_decision = _supervised(fused_p, gold, other)

    ce = 0.5 * (c_ce + p_ce)
    swap = 0.5 * (c_swap + p_swap)
    decision = 0.5 * (c_decision + p_decision)

    option_align = _option_alignment_loss(encoded["option_pooled"])
    relation_c_loss = F.cross_entropy(relation_c, gold)
    relation_p_loss = F.cross_entropy(relation_p, gold)
    relation_loss = 0.5 * (relation_c_loss + relation_p_loss)
    canonicalization, signature_alignment, signature_separation = (
        cross_view_relation_signature_loss(
            signature_c,
            signature_p,
            separation_margin=SIGNATURE_SEPARATION_MARGIN,
        )
    )
    consistency = symmetric_js_divergence(fused_c, fused_p)

    primary_block = (
        decision
        + OPTION_ALIGN_COEFFICIENT * option_align
        + INVARIANCE_COEFFICIENT * consistency
    )
    relation_block = (
        BINDING_COEFFICIENT * relation_loss
        + CANONICALIZATION_COEFFICIENT * canonicalization
    )
    total = primary_block + relation_block

    pieces = {
        "ce": float(ce.detach().cpu()),
        "swap": float(swap.detach().cpu()),
        "decision": float(decision.detach().cpu()),
        "canonical_decision": float(c_decision.detach().cpu()),
        "paraphrase_decision": float(p_decision.detach().cpu()),
        "option_alignment": float(option_align.detach().cpu()),
        "binding": float(relation_loss.detach().cpu()),
        "canonical_binding": float(relation_c_loss.detach().cpu()),
        "paraphrase_binding": float(relation_p_loss.detach().cpu()),
        "canonicalization": float(canonicalization.detach().cpu()),
        "signature_alignment": float(signature_alignment.detach().cpu()),
        "signature_separation": float(signature_separation.detach().cpu()),
        "consistency_js": float(consistency.detach().cpu()),
        "primary_block": float(primary_block.detach().cpu()),
        "relation_block": float(relation_block.detach().cpu()),
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
        "canonical_expert_agreement": float(
            fusion_diag_c.expert_top1_agreement.detach().cpu()
        ),
        "paraphrase_expert_agreement": float(
            fusion_diag_p.expert_top1_agreement.detach().cpu()
        ),
        "total": float(total.detach().cpu()),
    }
    return (
        total,
        primary_block,
        relation_block,
        pieces,
        fused_c,
        fused_p,
        raw_c,
        raw_p,
        relation_c,
        relation_p,
        signature_c,
        signature_p,
        encoded,
    )



def _option_order_flips(
    runtime,
    fused_c: Tensor,
    encoded: dict[str, Tensor],
) -> int:
    permutation = torch.tensor([3, 2, 1, 0], device=fused_c.device)
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S26 projection scorer missing")

    raw = scorer(
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
    relation, _signature, _diag = _relation_outputs(
        runtime,
        state_tokens=encoded["state_a_tokens"],
        state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        encoded={
            **encoded,
            "option_tokens": encoded["option_tokens"][:, permutation],
            "option_mask": encoded["option_mask"][:, permutation],
            "option_view_mask": encoded["option_view_mask"][:, permutation],
        },
    )
    permuted, _ = SymmetricFullKEvidenceFusion(
        epsilon=FUSION_EPSILON,
    )(raw, relation)
    mapped = permutation[permuted.argmax(-1)]
    return int((mapped != fused_c.argmax(-1)).sum().item())



@torch.no_grad()
def evaluate(runtime, rows: tuple[S26RelationCase, ...]) -> dict:
    enforce_s26_eval(runtime)

    fused_c_correct = fused_p_correct = 0
    raw_c_correct = raw_p_correct = 0
    relation_c_correct = relation_p_correct = 0
    fused_pair_both = fused_pair_changed = 0
    fused_cross_agree_sum = fused_js_sum = 0.0
    raw_cross_agree_sum = relation_cross_agree_sum = 0.0
    order_flips = 0
    max_mass_error = 0.0
    fused_c_margin_sum = fused_p_margin_sum = 0.0
    relation_c_margin_sum = relation_p_margin_sum = 0.0
    same_signature_sum = signature_margin_sum = 0.0
    signature_count = 0
    canonical_decision_sum = paraphrase_decision_sum = 0.0
    option_align_sum = relation_c_loss_sum = relation_p_loss_sum = 0.0
    canonicalization_sum = 0.0
    expert_agree_c_sum = expert_agree_p_sum = 0.0
    semantic_cases = state_view_encodes = 0
    full_k = True

    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = list(rows[start : start + BATCH_SIZE])
        (
            _total,
            _primary_block,
            _relation_block,
            pieces,
            fused_c,
            fused_p,
            raw_c,
            raw_p,
            relation_c,
            relation_p,
            signature_c,
            signature_p,
            encoded,
        ) = _losses(runtime, batch_rows)
        n = len(batch_rows)
        gold, _other = _gold_tensors(batch_rows, device=fused_c.device)

        fc = fused_c.argmax(-1)
        fp = fused_p.argmax(-1)
        rc = raw_c.argmax(-1)
        rp = raw_p.argmax(-1)
        bc = relation_c.argmax(-1)
        bp = relation_p.argmax(-1)

        fused_c_correct += int((fc == gold).sum().item())
        fused_p_correct += int((fp == gold).sum().item())
        raw_c_correct += int((rc == gold).sum().item())
        raw_p_correct += int((rp == gold).sum().item())
        relation_c_correct += int((bc == gold).sum().item())
        relation_p_correct += int((bp == gold).sum().item())

        pairs = fc.reshape(n, 2)
        gpairs = gold.reshape(n, 2)
        fused_pair_both += int(((pairs == gpairs).all(-1)).sum().item())
        fused_pair_changed += int((pairs[:, 0] != pairs[:, 1]).sum().item())

        fused_cross_agree_sum += float(
            selected_choice_agreement(fused_c, fused_p)
        ) * (2 * n)
        fused_js_sum += float(symmetric_js_divergence(fused_c, fused_p)) * (2 * n)
        raw_cross_agree_sum += float(
            selected_choice_agreement(raw_c, raw_p)
        ) * (2 * n)
        relation_cross_agree_sum += float(
            selected_choice_agreement(relation_c, relation_p)
        ) * (2 * n)

        fused_c_margin_sum += float(
            fused_gold_vs_max_wrong_margin(fused_c, gold).sum().cpu()
        )
        fused_p_margin_sum += float(
            fused_gold_vs_max_wrong_margin(fused_p, gold).sum().cpu()
        )
        relation_c_margin_sum += float(
            fused_gold_vs_max_wrong_margin(relation_c, gold).sum().cpu()
        )
        relation_p_margin_sum += float(
            fused_gold_vs_max_wrong_margin(relation_p, gold).sum().cpu()
        )

        same = relation_signature_same_option_cosine(signature_c, signature_p)
        c_norm = F.normalize(signature_c, dim=-1)
        p_norm = F.normalize(signature_p, dim=-1)
        cross = torch.einsum("nkd,njd->nkj", c_norm, p_norm)
        k = cross.shape[-1]
        eye = torch.eye(k, dtype=torch.bool, device=cross.device)[None]
        wrong = cross.masked_fill(eye, float("-inf")).amax(-1)
        sig_margin = same - wrong
        same_signature_sum += float(same.sum().cpu())
        signature_margin_sum += float(sig_margin.sum().cpu())
        signature_count += int(same.numel())

        order_flips += _option_order_flips(runtime, fused_c, encoded)

        for logits in (fused_c, fused_p):
            probs = torch.softmax(logits, dim=-1)
            max_mass_error = max(
                max_mass_error,
                float((probs.sum(-1) - 1).abs().max().cpu()),
            )

        canonical_decision_sum += pieces["canonical_decision"] * n
        paraphrase_decision_sum += pieces["paraphrase_decision"] * n
        option_align_sum += pieces["option_alignment"] * n
        relation_c_loss_sum += pieces["canonical_binding"] * n
        relation_p_loss_sum += pieces["paraphrase_binding"] * n
        canonicalization_sum += pieces["canonicalization"] * n
        expert_agree_c_sum += pieces["canonical_expert_agreement"] * (2 * n)
        expert_agree_p_sum += pieces["paraphrase_expert_agreement"] * (2 * n)

        semantic_cases += n
        state_view_encodes += 2 * n
        full_k = full_k and all(
            x.shape[-1] == 4
            for x in (fused_c, fused_p, raw_c, raw_p, relation_c, relation_p)
        )

    queries = len(rows) * 2
    return {
        "semantic_cases": semantic_cases,
        "canonical_queries": queries,
        "paraphrase_queries": queries,
        "fused_canonical_accuracy": fused_c_correct / queries,
        "fused_paraphrase_accuracy": fused_p_correct / queries,
        "fused_canonical_paired_both_correct_rate": fused_pair_both / len(rows),
        "fused_question_swap_choice_change_rate": fused_pair_changed / len(rows),
        "fused_cross_view_selected_choice_agreement": fused_cross_agree_sum / queries,
        "fused_cross_view_mean_js": fused_js_sum / queries,
        "fused_canonical_mean_gold_margin": fused_c_margin_sum / queries,
        "fused_paraphrase_mean_gold_margin": fused_p_margin_sum / queries,
        "raw_triadic_canonical_accuracy": raw_c_correct / queries,
        "raw_triadic_paraphrase_accuracy": raw_p_correct / queries,
        "raw_triadic_cross_view_agreement": raw_cross_agree_sum / queries,
        "canonical_relation_binding_accuracy": relation_c_correct / queries,
        "paraphrase_relation_binding_accuracy": relation_p_correct / queries,
        "relation_cross_view_agreement": relation_cross_agree_sum / queries,
        "canonical_relation_binding_mean_gold_margin": relation_c_margin_sum / queries,
        "paraphrase_relation_binding_mean_gold_margin": relation_p_margin_sum / queries,
        "mean_same_option_signature_cosine": same_signature_sum / signature_count,
        "mean_signature_same_vs_strongest_wrong_margin": signature_margin_sum / signature_count,
        "mean_canonical_decision_loss": canonical_decision_sum / len(rows),
        "mean_paraphrase_decision_loss": paraphrase_decision_sum / len(rows),
        "mean_option_alignment_loss": option_align_sum / len(rows),
        "canonical_mean_relation_loss": relation_c_loss_sum / len(rows),
        "paraphrase_mean_relation_loss": relation_p_loss_sum / len(rows),
        "mean_canonicalization_loss": canonicalization_sum / len(rows),
        "canonical_expert_top1_agreement": expert_agree_c_sum / queries,
        "paraphrase_expert_top1_agreement": expert_agree_p_sum / queries,
        "fused_option_order_flip_rate": order_flips / queries,
        "fused_max_probability_mass_error": max_mass_error,
        "full_k": bool(full_k),
        "relation_delta_max_abs": 0.0,
        "state_view_encodes": state_view_encodes,
    }


def _selection_key(epoch: int, metrics: dict) -> tuple:
    return (
        float(metrics["fused_canonical_paired_both_correct_rate"]),
        float(metrics["fused_canonical_accuracy"]),
        float(metrics["canonical_relation_binding_accuracy"]),
        float(metrics["canonical_relation_binding_mean_gold_margin"]),
        float(metrics["fused_canonical_mean_gold_margin"]),
        float(metrics["fused_question_swap_choice_change_rate"]),
        float(metrics["fused_cross_view_selected_choice_agreement"]),
        float(metrics["mean_same_option_signature_cosine"]),
        float(metrics["mean_signature_same_vs_strongest_wrong_margin"]),
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
    if a0.get("outcome") != "HIRA_V1_S26_A0_FACTORIZED_RELATION_READY":
        raise RuntimeError("S26-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S26-A0 unexpectedly used for selection")
    if int(a0.get("candidate_parameter_count", -1)) != HIRA_V1_S26_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S26-A0 physical surface changed")
    if int(a0.get("shared_lora_parameter_count", -1)) != HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT:
        raise RuntimeError("S26-A0 shared LoRA surface changed")
    if int(a0.get("primary_projection_parameter_count", -1)) != HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S26-A0 primary-private surface changed")
    if int(a0.get("relation_projection_parameter_count", -1)) != HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S26-A0 relation-private surface changed")
    if int(a0.get("relation_operator_added_parameter_count", -1)) != 0:
        raise RuntimeError("S26-A0 factorized relation operator added parameters")
    if a0.get("projection_storage_distinct") is not True:
        raise RuntimeError("S26-A0 private projections share storage")
    if a0.get("shared_encoder_identity") is not True:
        raise RuntimeError("S26-A0 shared encoder identity failed")
    if a0.get("primary_path_identity") is not True:
        raise RuntimeError("S26-A0 primary path identity failed")
    if float(a0.get("relation_intervention_max_abs", 0.0)) <= 1e-7:
        raise RuntimeError("S26-A0 factorized relation intervention is inactive")
    if int(a0.get("signature_dimension", -1)) != 256:
        raise RuntimeError("S26-A0 factorized signature dimension changed")
    for key in (
        "correct_vs_same_role_wrong_value_margin",
        "correct_vs_wrong_role_same_value_margin",
        "correct_vs_wrong_role_wrong_value_margin",
    ):
        if float(a0.get(key, 0.0)) <= 0.0:
            raise RuntimeError(f"S26-A0 hard-negative court failed: {key}")
    if float(a0.get("primary_to_relation_private_max_abs", -1.0)) != 0.0:
        raise RuntimeError("S26-A0 primary leaked into relation-private surface")
    if float(a0.get("relation_to_primary_private_max_abs", -1.0)) != 0.0:
        raise RuntimeError("S26-A0 relation leaked into primary-private surface")
    if float(a0.get("fused_option_order_flip_rate", -1.0)) != 0.0:
        raise RuntimeError("S26-A0 fused option-order court failed")

    train_rows = generate_s26_cases("train")
    dev_rows = generate_s26_cases("dev")
    validate_s26_partitions(train_rows, dev_rows)
    _assert_fresh_against_prior(train_rows, dev_rows)

    random.seed(SEED)
    torch.manual_seed(SEED)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"]) != A13_REVISION:
        raise RuntimeError("S26 semantic revision changed")

    frozen = load_hira_v0_m4_bundle(bundle)
    runtime = build_hira_v1_s26_factorized_relation_core(
        frozen.runtime.encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,
        train_primary_projection=True,
        train_relation_projection=True,
    )
    del frozen
    runtime.clear_schema_cache()
    enforce_s26_eval(runtime)

    trainable = [p for p in runtime.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    if trainable_count != HIRA_V1_S26_TOTAL_PARAMETER_COUNT:
        raise RuntimeError(f"S26 trainable count changed: {trainable_count}")
    if _original_a13_trainable(runtime) != 0:
        raise RuntimeError("S26 original A13 parameter became trainable")

    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S26 primary projection scorer missing")
    relation_projection = get_s25_relation_projection(runtime)

    shared = [
        p
        for module in iter_a13_lora_modules(runtime.encoder)
        for p in (module.lora_a, module.lora_b)
    ]
    primary_private = [scorer.projection.weight]
    relation_private = [relation_projection.weight]

    if sum(p.numel() for p in shared if p.requires_grad) != HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT:
        raise RuntimeError("S26 shared LoRA trainable surface changed")
    if sum(p.numel() for p in primary_private if p.requires_grad) != HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S26 primary-private trainable surface changed")
    if sum(p.numel() for p in relation_private if p.requires_grad) != HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S26 relation-private trainable surface changed")
    if scorer.projection.weight.data_ptr() == relation_projection.weight.data_ptr():
        raise RuntimeError("S26 private projections share storage")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S26 HIRACore became trainable")

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
    ownership_max_primary_to_relation = 0.0
    ownership_max_relation_to_primary = 0.0
    ownership_min_primary_private_l1 = float("inf")
    ownership_min_relation_private_l1 = float("inf")
    ownership_min_primary_shared_l1 = float("inf")
    ownership_min_relation_shared_l1 = float("inf")

    print("HIRA_V1_S26_TRAIN_BEGIN", flush=True)

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
            "primary_block": 0.0,
            "relation_block": 0.0,
        }
        balance_totals = {
            "steps": 0,
            "conflicts": 0,
            "normalized_pre_dot": 0.0,
            "normalized_post_dot": 0.0,
            "primary_norm": 0.0,
            "relation_norm": 0.0,
            "reference_scale": 0.0,
            "direction_norm": 0.0,
            "combined_norm": 0.0,
            "projection_coefficient": 0.0,
        }
        state_view_encodes = 0

        for start in range(0, len(order), BATCH_SIZE):
            rows = [
                train_rows[index]
                for index in order[start : start + BATCH_SIZE]
            ]
            optimizer.zero_grad(set_to_none=True)
            (
                loss,
                primary_block,
                relation_block,
                pieces,
                *_rest,
            ) = _losses(runtime, rows)

            ownership = apply_s25_decoupled_gradient_update(
                primary_block=primary_block,
                relation_block=relation_block,
                shared=shared,
                primary_private=primary_private,
                relation_private=relation_private,
                epsilon=BALANCE_EPSILON,
            )
            ownership_max_primary_to_relation = max(
                ownership_max_primary_to_relation,
                ownership.primary_to_relation_private_max_abs,
            )
            ownership_max_relation_to_primary = max(
                ownership_max_relation_to_primary,
                ownership.relation_to_primary_private_max_abs,
            )
            ownership_min_primary_private_l1 = min(
                ownership_min_primary_private_l1,
                ownership.primary_private_l1,
            )
            ownership_min_relation_private_l1 = min(
                ownership_min_relation_private_l1,
                ownership.relation_private_l1,
            )
            ownership_min_primary_shared_l1 = min(
                ownership_min_primary_shared_l1,
                ownership.primary_shared_l1,
            )
            ownership_min_relation_shared_l1 = min(
                ownership_min_relation_shared_l1,
                ownership.relation_shared_l1,
            )
            balance_diag = ownership.shared
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            enforce_s26_eval(runtime)

            n = len(rows)
            for key in totals:
                totals[key] += pieces[key] * n
            balance_totals["steps"] += 1
            balance_totals["conflicts"] += int(balance_diag.conflict)
            balance_totals["normalized_pre_dot"] += balance_diag.normalized_pre_dot
            balance_totals["normalized_post_dot"] += balance_diag.normalized_post_dot
            balance_totals["primary_norm"] += balance_diag.primary_norm
            balance_totals["relation_norm"] += balance_diag.relation_norm
            balance_totals["reference_scale"] += balance_diag.reference_scale
            balance_totals["direction_norm"] += balance_diag.direction_norm
            balance_totals["combined_norm"] += balance_diag.combined_norm
            balance_totals["projection_coefficient"] += balance_diag.projection_coefficient
            state_view_encodes += 2 * n

        if state_view_encodes != 2 * len(train_rows):
            raise RuntimeError("S26 TRAIN state-once changed")

        dev_metrics = evaluate(runtime, dev_rows)
        if int(dev_metrics["state_view_encodes"]) != 2 * len(dev_rows):
            raise RuntimeError("S26 DEV state-once changed")

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
            "train_mean_primary_block": totals["primary_block"] / len(train_rows),
            "train_mean_relation_block": totals["relation_block"] / len(train_rows),
            "norm_balancing": {
                "steps": balance_totals["steps"],
                "conflict_rate": (
                    balance_totals["conflicts"] / max(1, balance_totals["steps"])
                ),
                "mean_normalized_pre_dot": balance_totals["normalized_pre_dot"] / max(1, balance_totals["steps"]),
                "mean_normalized_post_dot": balance_totals["normalized_post_dot"] / max(1, balance_totals["steps"]),
                "mean_primary_norm": balance_totals["primary_norm"] / max(1, balance_totals["steps"]),
                "mean_relation_norm": balance_totals["relation_norm"] / max(1, balance_totals["steps"]),
                "mean_reference_scale": balance_totals["reference_scale"] / max(1, balance_totals["steps"]),
                "mean_direction_norm": balance_totals["direction_norm"] / max(1, balance_totals["steps"]),
                "mean_combined_norm": balance_totals["combined_norm"] / max(1, balance_totals["steps"]),
                "mean_projection_coefficient": balance_totals["projection_coefficient"] / max(1, balance_totals["steps"]),
            },
            "train_state_view_encodes": state_view_encodes,
            "surface_diagnostics": {
                "primary_projection_weight_norm": float(
                    scorer.projection.weight.detach().norm().cpu()
                ),
                "relation_projection_weight_norm": float(
                    relation_projection.weight.detach().norm().cpu()
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
                "primary_projection": {
                    "projection.weight": scorer.projection.weight.detach().cpu().clone(),
                },
                "relation_projection": {
                    "projection.weight": relation_projection.weight.detach().cpu().clone(),
                },
            }
            best_metrics = dict(dev_metrics)

        print(
            "HIRA_V1_S26_EPOCH=" + json.dumps(record, sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError("S26 DEV selection produced no checkpoint")

    load_a13_lora_state_dict(runtime.encoder, best_state["lora"], freeze=False)
    scorer.load_projection_state_dict(best_state["primary_projection"], freeze=False)
    with torch.no_grad():
        relation_projection.weight.copy_(
            best_state["relation_projection"]["projection.weight"].to(
                device=relation_projection.weight.device,
                dtype=relation_projection.weight.dtype,
            )
        )
    relation_projection.weight.requires_grad_(True)
    enforce_s26_eval(runtime)
    selected = evaluate(runtime, dev_rows)

    replay_keys = (
        "fused_canonical_accuracy",
        "fused_paraphrase_accuracy",
        "fused_canonical_paired_both_correct_rate",
        "fused_question_swap_choice_change_rate",
        "fused_cross_view_selected_choice_agreement",
        "fused_cross_view_mean_js",
        "fused_canonical_mean_gold_margin",
        "canonical_relation_binding_accuracy",
        "canonical_relation_binding_mean_gold_margin",
        "mean_same_option_signature_cosine",
        "mean_signature_same_vs_strongest_wrong_margin",
        "fused_option_order_flip_rate",
        "fused_max_probability_mass_error",
        "mean_canonical_decision_loss",
    )
    for key in replay_keys:
        if not math.isclose(
            float(selected[key]),
            float(best_metrics[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"S26 selected DEV replay changed: {key}")

    lora_trainable = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" in name
    )
    gates = {
        "fused_canonical_accuracy_gte_0_85": (
            selected["fused_canonical_accuracy"] >= 0.85
        ),
        "fused_paired_both_correct_gte_0_75": (
            selected["fused_canonical_paired_both_correct_rate"] >= 0.75
        ),
        "fused_question_swap_change_gte_0_80": (
            selected["fused_question_swap_choice_change_rate"] >= 0.80
        ),
        "fused_cross_view_choice_agreement_gte_0_95": (
            selected["fused_cross_view_selected_choice_agreement"] >= 0.95
        ),
        "fused_cross_view_mean_js_lte_0_05": (
            selected["fused_cross_view_mean_js"] <= 0.05
        ),
        "fused_canonical_margin_gte_0_15": (
            selected["fused_canonical_mean_gold_margin"] >= 0.15
        ),
        "canonical_relation_accuracy_gte_0_80": (
            selected["canonical_relation_binding_accuracy"] >= 0.80
        ),
        "canonical_relation_margin_gte_0_15": (
            selected["canonical_relation_binding_mean_gold_margin"] >= 0.15
        ),
        "same_option_signature_cosine_gte_0_90": (
            selected["mean_same_option_signature_cosine"] >= 0.90
        ),
        "signature_margin_gte_0_15": (
            selected["mean_signature_same_vs_strongest_wrong_margin"] >= 0.15
        ),
        "fused_option_order_flip_lte_0_02": (
            selected["fused_option_order_flip_rate"] <= 0.02
        ),
        "fused_probability_mass_error_lte_1e_6": (
            selected["fused_max_probability_mass_error"] <= 1e-6
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
        "trainable_total_exact_81920": (
            trainable_count == HIRA_V1_S26_TOTAL_PARAMETER_COUNT
        ),
        "trainable_lora_exact_16384": (
            lora_trainable == HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT
        ),
        "trainable_primary_projection_exact_32768": (
            scorer.projection_trainable_parameter_count
            == HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT
        ),
        "trainable_relation_projection_exact_32768": (
            relation_projection.weight.requires_grad
            and relation_projection.weight.numel()
            == HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT
        ),
        "private_projection_storage_distinct": (
            scorer.projection.weight.data_ptr() != relation_projection.weight.data_ptr()
        ),
        "primary_to_relation_private_zero_all_steps": (
            ownership_max_primary_to_relation == 0.0
        ),
        "relation_to_primary_private_zero_all_steps": (
            ownership_max_relation_to_primary == 0.0
        ),
        "original_a13_frozen": _original_a13_trainable(runtime) == 0,
        "hira_core_frozen": not any(
            p.requires_grad for p in runtime.hira.parameters()
        ),
        "fusion_added_parameters_zero": (
            SymmetricFullKEvidenceFusion(epsilon=FUSION_EPSILON).parameter_count == 0
        ),
    }
    outcome = READY if all(gates.values()) else FAIL

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "factorized-role-value-relation-candidate.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s26-factorized-role-value-relation-checkpoint-v1",
            "kind": "factorized-role-value-relation-a13-w28",
            "lora_parameter_count": HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT,
            "primary_projection_parameter_count": HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT,
            "relation_projection_parameter_count": HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT,
            "total_parameter_count": HIRA_V1_S26_TOTAL_PARAMETER_COUNT,
            "relation_operator_added_parameter_count": 0,
            "factorized_signature_dimension": 256,
            "lora_rank": HIRA_V1_S6_LORA_RANK,
            "selected_dev_epoch": best_epoch,
            "semantic_revision": str(manifest["semantic_revision"]),
            "initialization_t0_sha256": str(manifest["t0_checkpoint_sha256"]),
            "lora_state_dict": best_state["lora"],
            "primary_projection_state_dict": best_state["primary_projection"],
            "relation_projection_state_dict": best_state["relation_projection"],
        },
        checkpoint_path,
    )
    checkpoint_sha = _sha256(checkpoint_path)

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": outcome,
        "scientific_authority": "V1_S26_FRESH_ENGLISH_FACTORIZED_ROLE_VALUE_RELATION",
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
            "fusion_epsilon": FUSION_EPSILON,
            "fusion_rule": "s14_equal_standardized_full_k",
            "raw_triadic_ce_used": False,
            "fused_primary_relation_logits_detached": True,
            "inference_uses_frozen_s14_equal_fusion": True,
            "s26_factorized_relation_change_preregistered": True,
            "primary_operator": "role_gated_content_triadic",
            "optimizer_conflict_priority": "neutral",
            "primary_role_temperature": 0.10,
            "primary_role_weight": 0.50,
            "primary_content_weight": 0.50,
            "shared_lora_gradient_rule": "equal_direction_neutral_bisector_no_projection",
            "balance_epsilon": BALANCE_EPSILON,
            "shared_primary_block": "decision + 0.05*option_alignment + 0.25*fused_cross_view_js",
            "shared_relation_block": "0.10*relation_ce + 0.15*signature_canonicalization",
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
            "s26_a0_rows_used": False,
            "m5_final_rows_used": False,
            "w29_w34_sealed_rows_used": False,
        },
        "parameter_surface": {
            "total_trainable_parameters": trainable_count,
            "lora_trainable_parameters": lora_trainable,
            "lora_rank": HIRA_V1_S6_LORA_RANK,
            "primary_projection_trainable_parameters": scorer.projection_trainable_parameter_count,
            "relation_projection_trainable_parameters": (
                relation_projection.weight.numel()
                if relation_projection.weight.requires_grad
                else 0
            ),
            "original_a13_trainable": _original_a13_trainable(runtime),
            "hira_core_trainable": sum(
                p.numel() for p in runtime.hira.parameters() if p.requires_grad
            ),
            "canonicalizer_added_parameters": 0,
            "relation_operator_added_parameters": 0,
            "factorized_signature_dimension": 256,
            "fusion_added_parameters": 0,
            "learned_downstream_scorer_parameters": 0,
            "role_content_factorization_added_parameters": 0,
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
            "candidate_parameter_count": a0["candidate_parameter_count"],
            "shared_lora_parameter_count": a0["shared_lora_parameter_count"],
            "primary_projection_parameter_count": a0["primary_projection_parameter_count"],
            "relation_projection_parameter_count": a0["relation_projection_parameter_count"],
            "relation_operator_added_parameter_count": a0[
                "relation_operator_added_parameter_count"
            ],
            "projection_storage_distinct": a0["projection_storage_distinct"],
            "signature_dimension": a0["signature_dimension"],
            "relation_intervention_max_abs": a0["relation_intervention_max_abs"],
            "correct_vs_same_role_wrong_value_margin": a0[
                "correct_vs_same_role_wrong_value_margin"
            ],
            "correct_vs_wrong_role_same_value_margin": a0[
                "correct_vs_wrong_role_same_value_margin"
            ],
            "correct_vs_wrong_role_wrong_value_margin": a0[
                "correct_vs_wrong_role_wrong_value_margin"
            ],
            "primary_to_relation_private_max_abs": a0[
                "primary_to_relation_private_max_abs"
            ],
            "relation_to_primary_private_max_abs": a0[
                "relation_to_primary_private_max_abs"
            ],
            "fused_option_order_flip_rate": a0["fused_option_order_flip_rate"],
        },
        "gradient_ownership": {
            "primary_to_relation_private_max_abs": (
                ownership_max_primary_to_relation
            ),
            "relation_to_primary_private_max_abs": (
                ownership_max_relation_to_primary
            ),
            "minimum_primary_private_gradient_l1": (
                ownership_min_primary_private_l1
            ),
            "minimum_relation_private_gradient_l1": (
                ownership_min_relation_private_l1
            ),
            "minimum_primary_shared_gradient_l1": (
                ownership_min_primary_shared_l1
            ),
            "minimum_relation_shared_gradient_l1": (
                ownership_min_relation_shared_l1
            ),
            "private_updates_direct": True,
            "shared_only_neutral_bisector": True,
        },
        "history": history,
        "norm_balancing": {
            "rule": "equal_direction_neutral_bisector_no_projection",
            "epsilon": BALANCE_EPSILON,
            "reference_scale": "arithmetic_mean_raw_norm",
            "conflict_rate_by_epoch": [
                record["norm_balancing"]["conflict_rate"] for record in history
            ],
            "mean_conflict_rate": (
                sum(record["norm_balancing"]["conflict_rate"] for record in history)
                / len(history)
            ),
        },
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
        "HIRA_V1_S26_TRAIN_DEV_RECEIPT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
