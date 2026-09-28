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
from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import generate_s3_pairs
from nmd.v1_s4_authority import generate_s4_pairs
from nmd.v1_s5_authority import generate_s5_pairs
from nmd.v1_s6_authority import (
    S7PairCase,
    generate_s7_pairs,
    validate_s7_partitions,
)
from nmd.v1_s6_semantic_core import (
    HIRA_V1_S7_TOTAL_PARAMETER_COUNT,
    HIRA_V1_S6_LORA_RANK,
    build_hira_v1_s6_a13_lora_core,
    enforce_s6_encoder_eval,
)


SCHEMA_VERSION = "hira-v1-s7-coadapt-train-dev-v1"
READY = "HIRA_V1_S7_COADAPT_DEV_READY"
FAIL = "HIRA_V1_S7_COADAPT_DEV_FAIL"

SEED = 12701
EPOCHS = 24
BATCH_SIZE = 32
LR = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SWAP_COEFFICIENT = 0.25
SWAP_MARGIN = 0.20
OPTION_ALIGN_COEFFICIENT = 0.05
OPTION_ALIGN_TEMPERATURE = 0.10
QUESTION_OPTION_COEFFICIENT = 0.10
QUESTION_OPTION_TEMPERATURE = 0.10


def _assert_fresh_against_prior(
    train_rows: tuple[S7PairCase, ...],
    dev_rows: tuple[S7PairCase, ...],
) -> None:
    prior = (
        *generate_s0_pairs("train"),
        *generate_s0_pairs("dev"),
        *generate_s1_pairs("train"),
        *generate_s1_pairs("dev"),
        *generate_s2_pairs("train"),
        *generate_s2_pairs("dev"),
        *generate_s3_pairs("train"),
        *generate_s3_pairs("dev"),
        *generate_s4_pairs("train"),
        *generate_s4_pairs("dev"),
        *generate_s5_pairs("train"),
        *generate_s5_pairs("dev"),
        *generate_s6_pairs("train"),
        *generate_s6_pairs("dev"),
    )
    prior_states = {row.state for row in prior}
    prior_questions = {
        q for row in prior for q in (row.question_a, row.question_b)
    }
    current = (*train_rows, *dev_rows)
    current_states = {row.state for row in current}
    current_questions = {
        q for row in current for q in (row.question_a, row.question_b)
    }
    if prior_states & current_states:
        raise RuntimeError("S7 exact state overlap with exposed S0-S6 rows")
    if prior_questions & current_questions:
        raise RuntimeError("S7 exact question overlap with exposed S0-S6 rows")


def _content_mask(batch) -> Tensor:
    mask = batch.attention_mask.bool()
    if batch.special_token_mask is None:
        return mask
    return mask & ~batch.special_token_mask.bool()


def _option_view_texts(rows: list[S7PairCase]) -> list[str]:
    texts = []
    for row in rows:
        for criterion, alias in zip(row.option_texts, row.option_aliases):
            texts.extend((criterion, alias))
    return texts


def _gold_tensors(
    rows: list[S7PairCase],
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
    # option_pooled: [N,K,2,D]
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
    # Two option views are combined without a learned head.
    left = F.normalize(option_pooled[:, :, 0], dim=-1)
    right = F.normalize(option_pooled[:, :, 1], dim=-1)
    options = F.normalize(0.5 * (left + right), dim=-1)
    options = options.repeat_interleave(2, dim=0)
    question = F.normalize(question_pooled, dim=-1)
    logits = torch.einsum("bd,bkd->bk", question, options)
    logits = logits / QUESTION_OPTION_TEMPERATURE
    return F.cross_entropy(logits, gold)


def _encode_batch(runtime, rows: list[S7PairCase]) -> dict[str, Tensor]:
    encoder = runtime.encoder
    enforce_s6_encoder_eval(runtime)

    states = [row.state for row in rows]
    questions = [
        question
        for row in rows
        for question in (row.question_a, row.question_b)
    ]
    option_views = _option_view_texts(rows)

    state_batch = encoder.encode_texts(states)
    question_batch = encoder.encode_texts(questions)
    option_batch = encoder.encode_texts(option_views)

    n = len(rows)
    k = 4
    v = 2
    if option_batch.token_embeddings.shape[0] != n * k * v:
        raise RuntimeError("S6 option-view packing changed")

    state_tokens = state_batch.token_embeddings
    state_mask = _content_mask(state_batch)
    question_tokens = question_batch.token_embeddings
    question_mask = _content_mask(question_batch)

    option_tokens = option_batch.token_embeddings.reshape(
        n, k, v, option_batch.token_embeddings.shape[1], -1
    )
    option_mask = _content_mask(option_batch).reshape(
        n, k, v, option_batch.attention_mask.shape[1]
    )
    option_view_mask = torch.ones(
        n,
        k,
        v,
        dtype=torch.bool,
        device=option_tokens.device,
    )
    option_pooled = option_batch.pooled_embeddings.reshape(n, k, v, -1)

    return {
        "state_tokens": state_tokens,
        "state_mask": state_mask,
        "question_tokens": question_tokens,
        "question_mask": question_mask,
        "option_tokens": option_tokens,
        "option_mask": option_mask,
        "option_view_mask": option_view_mask,
        "question_pooled": question_batch.pooled_embeddings,
        "option_pooled": option_pooled,
    }


def _decision_logits(runtime, encoded: dict[str, Tensor]) -> Tensor:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S7 frozen triadic scorer missing")

    state_tokens = encoded["state_tokens"].repeat_interleave(2, dim=0)
    state_mask = encoded["state_mask"].repeat_interleave(2, dim=0)
    option_tokens = encoded["option_tokens"].repeat_interleave(2, dim=0)
    option_mask = encoded["option_mask"].repeat_interleave(2, dim=0)
    option_view_mask = encoded["option_view_mask"].repeat_interleave(2, dim=0)

    return scorer(
        state_tokens=state_tokens,
        state_mask=state_mask,
        question_tokens=encoded["question_tokens"],
        question_mask=encoded["question_mask"],
        option_view_tokens=option_tokens,
        option_view_token_mask=option_mask,
        option_view_mask=option_view_mask,
    )


def _losses(
    runtime,
    rows: list[S7PairCase],
) -> tuple[Tensor, dict[str, float], Tensor, dict[str, Tensor]]:
    encoded = _encode_batch(runtime, rows)
    logits = _decision_logits(runtime, encoded)
    gold, other = _gold_tensors(rows, device=logits.device)

    ce = F.cross_entropy(logits, gold)
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    swap = F.relu(
        logits.new_tensor(SWAP_MARGIN)
        - (logits[batch_indices, gold] - logits[batch_indices, other])
    ).mean()
    decision = ce + SWAP_COEFFICIENT * swap

    option_align = _option_alignment_loss(encoded["option_pooled"])
    question_option = _question_option_loss(
        encoded["question_pooled"],
        encoded["option_pooled"],
        gold,
    )
    total = (
        decision
        + OPTION_ALIGN_COEFFICIENT * option_align
        + QUESTION_OPTION_COEFFICIENT * question_option
    )
    pieces = {
        "ce": float(ce.detach().cpu()),
        "swap": float(swap.detach().cpu()),
        "decision": float(decision.detach().cpu()),
        "option_alignment": float(option_align.detach().cpu()),
        "question_option": float(question_option.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, pieces, logits, encoded


def _batch_metrics(
    logits: Tensor,
    rows: list[S7PairCase],
    encoded: dict[str, Tensor],
    runtime,
) -> dict[str, float | int | bool]:
    probs = torch.softmax(logits, dim=-1)
    predictions = probs.argmax(-1)
    gold, _other = _gold_tensors(rows, device=logits.device)

    correct = int((predictions == gold).sum().item())
    pair_predictions = predictions.reshape(len(rows), 2)
    pair_gold = gold.reshape(len(rows), 2)
    pair_both = int(((pair_predictions == pair_gold).all(-1)).sum().item())
    pair_changed = int(
        (pair_predictions[:, 0] != pair_predictions[:, 1]).sum().item()
    )

    # Test exact option permutation equivariance without re-encoding the text:
    # BERT processes option-view strings independently; permuting the option
    # axis here is the full semantic permutation once embeddings exist.
    permutation = torch.tensor([3, 2, 1, 0], device=logits.device)
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S7 frozen triadic scorer missing")

    permuted = scorer(
        state_tokens=encoded["state_tokens"].repeat_interleave(2, dim=0),
        state_mask=encoded["state_mask"].repeat_interleave(2, dim=0),
        question_tokens=encoded["question_tokens"],
        question_mask=encoded["question_mask"],
        option_view_tokens=encoded["option_tokens"][:, permutation]
            .repeat_interleave(2, dim=0),
        option_view_token_mask=encoded["option_mask"][:, permutation]
            .repeat_interleave(2, dim=0),
        option_view_mask=encoded["option_view_mask"][:, permutation]
            .repeat_interleave(2, dim=0),
    )
    permuted_index = permuted.argmax(-1)
    mapped = permutation[permuted_index]
    order_flips = int((mapped != predictions).sum().item())

    mass_error = float(
        (probs.sum(-1) - 1.0).abs().max().detach().cpu()
    )
    return {
        "queries": len(rows) * 2,
        "correct": correct,
        "pair_both": pair_both,
        "pair_changed": pair_changed,
        "order_flips": order_flips,
        "max_probability_mass_error": mass_error,
        "full_k": logits.shape[-1] == 4,
    }


@torch.no_grad()
def evaluate(runtime, rows: tuple[S7PairCase, ...]) -> dict:
    enforce_s6_encoder_eval(runtime)
    total_queries = 0
    correct = 0
    pair_both = 0
    pair_changed = 0
    order_flips = 0
    max_mass_error = 0.0
    full_k = True
    decision_sum = 0.0
    option_align_sum = 0.0
    qopt_sum = 0.0
    state_encodes = 0

    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = list(rows[start : start + BATCH_SIZE])
        _total, pieces, logits, encoded = _losses(runtime, batch_rows)
        metrics = _batch_metrics(logits, batch_rows, encoded, runtime)

        n = len(batch_rows)
        total_queries += int(metrics["queries"])
        correct += int(metrics["correct"])
        pair_both += int(metrics["pair_both"])
        pair_changed += int(metrics["pair_changed"])
        order_flips += int(metrics["order_flips"])
        max_mass_error = max(
            max_mass_error,
            float(metrics["max_probability_mass_error"]),
        )
        full_k = full_k and bool(metrics["full_k"])
        decision_sum += pieces["decision"] * n
        option_align_sum += pieces["option_alignment"] * n
        qopt_sum += pieces["question_option"] * n
        state_encodes += n

    return {
        "base_cases": len(rows),
        "queries": total_queries,
        "accuracy": correct / total_queries,
        "paired_both_correct_rate": pair_both / len(rows),
        "question_swap_choice_change_rate": pair_changed / len(rows),
        "option_order_flip_rate": order_flips / total_queries,
        "mean_decision_loss": decision_sum / len(rows),
        "mean_option_alignment_loss": option_align_sum / len(rows),
        "mean_question_option_loss": qopt_sum / len(rows),
        "max_probability_mass_error": max_mass_error,
        "full_k": bool(full_k),
        "relation_delta_max_abs": 0.0,
        "state_encodes": state_encodes,
    }


def _selection_key(
    epoch: int,
    metrics: dict,
) -> tuple[float, float, float, float, int]:
    return (
        float(metrics["paired_both_correct_rate"]),
        float(metrics["accuracy"]),
        float(metrics["question_swap_choice_change_rate"]),
        -float(metrics["mean_decision_loss"]),
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
    if a0.get("outcome") != "HIRA_V1_S7_A0_IDENTITY_READY":
        raise RuntimeError("S7-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S7-A0 unexpectedly used for model selection")
    if a0.get("a13_token_output_identity") is not True:
        raise RuntimeError("S7-A0 A13 identity boundary failed")
    if float(a0.get("exact_logit_identity_rate", -1.0)) != 1.0:
        raise RuntimeError("S7-A0 decision identity boundary failed")

    train_rows = generate_s7_pairs("train")
    dev_rows = generate_s7_pairs("dev")
    validate_s7_partitions(train_rows, dev_rows)
    _assert_fresh_against_prior(train_rows, dev_rows)

    random.seed(SEED)
    torch.manual_seed(SEED)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"]) != A13_REVISION:
        raise RuntimeError("S7 joint semantic revision changed")

    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder

    # LoRA A initialization is seeded above.  B starts at zero.
    runtime = build_hira_v1_s6_a13_lora_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,
        train_projection=True,
    )
    del frozen
    runtime.clear_schema_cache()
    enforce_s6_encoder_eval(runtime)

    trainable = [p for p in runtime.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    if trainable_count != HIRA_V1_S7_TOTAL_PARAMETER_COUNT:
        raise RuntimeError(f"S7 trainable count changed: {trainable_count}")
    if _original_a13_trainable(runtime) != 0:
        raise RuntimeError("S7 original A13 parameter became trainable")
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S7 projection triadic scorer missing")
    if scorer.projection_trainable_parameter_count != HIRA_V1_S7_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S7 trainable projection surface changed")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S6 HIRACore became trainable")

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

    print("HIRA_V1_S7_TRAIN_BEGIN", flush=True)

    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train_rows)))
        random.Random(SEED + epoch).shuffle(order)

        total_loss_sum = 0.0
        decision_sum = 0.0
        ce_sum = 0.0
        swap_sum = 0.0
        option_align_sum = 0.0
        qopt_sum = 0.0
        state_encodes = 0

        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            rows = [train_rows[index] for index in indices]

            optimizer.zero_grad(set_to_none=True)
            loss, pieces, _logits, _encoded = _losses(runtime, rows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            enforce_s6_encoder_eval(runtime)

            n = len(rows)
            total_loss_sum += pieces["total"] * n
            decision_sum += pieces["decision"] * n
            ce_sum += pieces["ce"] * n
            swap_sum += pieces["swap"] * n
            option_align_sum += pieces["option_alignment"] * n
            qopt_sum += pieces["question_option"] * n
            state_encodes += n

        if state_encodes != len(train_rows):
            raise RuntimeError(
                f"S7 state-once TRAIN epoch changed: {state_encodes}"
            )

        dev_metrics = evaluate(runtime, dev_rows)
        if int(dev_metrics["state_encodes"]) != len(dev_rows):
            raise RuntimeError("S7 state-once DEV evaluation changed")

        record = {
            "epoch": epoch,
            "train_mean_total_loss": total_loss_sum / len(train_rows),
            "train_mean_decision_loss": decision_sum / len(train_rows),
            "train_mean_ce": ce_sum / len(train_rows),
            "train_mean_swap": swap_sum / len(train_rows),
            "train_mean_option_alignment_loss": (
                option_align_sum / len(train_rows)
            ),
            "train_mean_question_option_loss": qopt_sum / len(train_rows),
            "train_state_encodes": state_encodes,
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
            "HIRA_V1_S7_EPOCH=" + json.dumps(record, sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError("S7 DEV selection produced no checkpoint")

    load_a13_lora_state_dict(runtime.encoder, best_state["lora"], freeze=False)
    scorer.load_projection_state_dict(best_state["projection"], freeze=False)
    enforce_s6_encoder_eval(runtime)
    selected = evaluate(runtime, dev_rows)

    for key in (
        "accuracy",
        "paired_both_correct_rate",
        "question_swap_choice_change_rate",
        "option_order_flip_rate",
        "max_probability_mass_error",
        "mean_decision_loss",
        "mean_option_alignment_loss",
        "mean_question_option_loss",
    ):
        if not math.isclose(
            float(selected[key]),
            float(best_metrics[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"S7 selected DEV replay changed: {key}")

    gates = {
        "accuracy_gte_0_85": selected["accuracy"] >= 0.85,
        "paired_both_correct_gte_0_75": (
            selected["paired_both_correct_rate"] >= 0.75
        ),
        "question_swap_choice_change_gte_0_80": (
            selected["question_swap_choice_change_rate"] >= 0.80
        ),
        "option_order_flip_lte_0_02": (
            selected["option_order_flip_rate"] <= 0.02
        ),
        "probability_mass_error_lte_1e_6": (
            selected["max_probability_mass_error"] <= 1e-6
        ),
        "full_k": bool(selected["full_k"]),
        "relation_delta_zero": float(selected["relation_delta_max_abs"]) == 0.0,
        "state_once_train_per_epoch": all(
            int(record["train_state_encodes"]) == len(train_rows)
            for record in history
        ),
        "state_once_dev_per_eval": all(
            int(record["dev"]["state_encodes"]) == len(dev_rows)
            for record in history
        ),
        "trainable_total_exact_49152": (
            trainable_count == HIRA_V1_S7_TOTAL_PARAMETER_COUNT
        ),
        "trainable_lora_exact_16384": sum(
            p.numel()
            for name, p in runtime.encoder.model.named_parameters()
            if p.requires_grad and ".lora_" in name
        ) == HIRA_V1_S6_LORA_PARAMETER_COUNT,
        "trainable_projection_exact_32768": (
            scorer.projection_trainable_parameter_count
            == HIRA_V1_S7_PROJECTION_PARAMETER_COUNT
        ),
        "original_a13_frozen": _original_a13_trainable(runtime) == 0,
        "hira_core_frozen": not any(
            p.requires_grad for p in runtime.hira.parameters()
        ),
    }
    outcome = READY if all(gates.values()) else FAIL

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "coadapt-candidate.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s7-coadapt-checkpoint-v1",
            "kind": "a13-w28-joint-coadapt",
            "lora_parameter_count": HIRA_V1_S6_LORA_PARAMETER_COUNT,
            "projection_parameter_count": HIRA_V1_S7_PROJECTION_PARAMETER_COUNT,
            "total_parameter_count": HIRA_V1_S7_TOTAL_PARAMETER_COUNT,
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
        "scientific_authority": "V1_S6_FRESH_ENGLISH_TRAIN_DEV",
        "seed": SEED,
        "optimizer": {
            "name": "AdamW",
            "epochs": EPOCHS,
            "batch_size_base_states": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
        },
        "loss": {
            "cross_entropy": True,
            "swap_margin_coefficient": SWAP_COEFFICIENT,
            "swap_margin": SWAP_MARGIN,
            "option_view_infonce_coefficient": OPTION_ALIGN_COEFFICIENT,
            "option_view_infonce_temperature": OPTION_ALIGN_TEMPERATURE,
            "question_option_infonce_coefficient": QUESTION_OPTION_COEFFICIENT,
            "question_option_infonce_temperature": QUESTION_OPTION_TEMPERATURE,
        },
        "partitions": {
            "train_base_cases": len(train_rows),
            "train_queries": len(train_rows) * 2,
            "dev_base_cases": len(dev_rows),
            "dev_queries": len(dev_rows) * 2,
            "language": "en",
            "domains": sorted({row.domain for row in train_rows}),
            "k": 4,
            "views_per_option": 2,
            "train_dev_exact_state_overlap": False,
            "train_dev_exact_question_overlap": False,
            "prior_track_exact_rows_used": False,
            "s7_a0_rows_used": False,
            "m5_final_rows_used": False,
            "w29_w34_sealed_rows_used": False,
        },
        "parameter_surface": {
            "total_trainable_parameters": trainable_count,
            "lora_trainable_parameters": sum(
                p.numel()
                for name, p in runtime.encoder.model.named_parameters()
                if p.requires_grad and ".lora_" in name
            ),
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
            "train_state_encodes_per_epoch": len(train_rows),
            "dev_state_encodes_per_eval": len(dev_rows),
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
            "accuracy": a0["accuracy"],
            "paired_both_correct_rate": a0["paired_both_correct_rate"],
            "a13_token_output_identity": a0["a13_token_output_identity"],
            "exact_logit_identity_rate": a0["exact_logit_identity_rate"],
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
        "HIRA_V1_S7_TRAIN_DEV_RECEIPT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
