from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.contracts import CompiledSchema, LogicalOption, StateMemory
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import S3PairCase, generate_s3_pairs, validate_s3_partitions
from nmd.v1_s3_semantic_core import (
    HIRA_V1_S3_FACTOR_PARAMETER_COUNT,
    build_hira_v1_s3_triadic_core,
)
from hira_v1_s3_a0_triadic import cases as s3_a0_cases


SCHEMA_VERSION = "hira-v1-s3-triadic-train-dev-v1"
READY = "HIRA_V1_S3_TRIADIC_DEV_READY"
FAIL = "HIRA_V1_S3_TRIADIC_DEV_FAIL"

SEED = 8301
EPOCHS = 20
LR = 5e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SWAP_COEFFICIENT = 0.25
SWAP_MARGIN = 0.20


@dataclass
class PreparedPair:
    case: S3PairCase
    memory: StateMemory
    schema_a: CompiledSchema
    schema_b: CompiledSchema
    reverse_a: CompiledSchema | None = None
    reverse_b: CompiledSchema | None = None


def _options(case: S3PairCase, *, reverse: bool = False) -> tuple[LogicalOption, ...]:
    options = tuple(
        LogicalOption(option_id=option_id, criterion_text=text)
        for option_id, text in zip(case.option_ids, case.option_texts)
    )
    return tuple(reversed(options)) if reverse else options


def _gold_id(case: S3PairCase, which: str) -> str:
    index = case.gold_a if which == "a" else case.gold_b
    return case.option_ids[index]


def _gold_index(schema: CompiledSchema, option_id: str) -> int:
    return next(i for i, option in enumerate(schema.options) if option.option_id == option_id)


def _assert_fresh_against_prior(
    train_rows: tuple[S3PairCase, ...],
    dev_rows: tuple[S3PairCase, ...],
) -> None:
    prior = (
        *generate_s0_pairs("train"),
        *generate_s0_pairs("dev"),
        *generate_s1_pairs("train"),
        *generate_s1_pairs("dev"),
        *generate_s2_pairs("train"),
        *generate_s2_pairs("dev"),
    )
    current = (*train_rows, *dev_rows)
    prior_states = {row.state for row in prior}
    current_states = {row.state for row in current}
    if prior_states & current_states:
        raise RuntimeError("S3 exact state overlap with exposed S0/S1/S2 rows")

    prior_questions = {q for row in prior for q in (row.question_a, row.question_b)}
    current_questions = {q for row in current for q in (row.question_a, row.question_b)}
    if prior_questions & current_questions:
        raise RuntimeError("S3 exact question overlap with exposed S0/S1/S2 rows")

    a0 = s3_a0_cases()
    a0_states = {row.state for row in a0}
    a0_questions = {q for row in a0 for q in (row.question_a, row.question_b)}
    if a0_states & current_states:
        raise RuntimeError("S3 exact state overlap with exposed S3-A0 rows")
    if a0_questions & current_questions:
        raise RuntimeError("S3 exact question overlap with exposed S3-A0 rows")


@torch.no_grad()
def prepare_pairs(
    runtime,
    rows: tuple[S3PairCase, ...],
    *,
    reverse: bool,
) -> tuple[list[PreparedPair], int]:
    before = runtime.state_encode_calls
    prepared = []
    for case in rows:
        memory = runtime.compile_state(case.state)
        schema_a, _ = runtime.compile_schema(
            primitive="choice",
            question_text=case.question_a,
            options=_options(case),
            use_cache=False,
            include_token_artifacts=True,
        )
        schema_b, _ = runtime.compile_schema(
            primitive="choice",
            question_text=case.question_b,
            options=_options(case),
            use_cache=False,
            include_token_artifacts=True,
        )
        reverse_a = reverse_b = None
        if reverse:
            reverse_a, _ = runtime.compile_schema(
                primitive="choice",
                question_text=case.question_a,
                options=_options(case, reverse=True),
                use_cache=False,
                include_token_artifacts=True,
            )
            reverse_b, _ = runtime.compile_schema(
                primitive="choice",
                question_text=case.question_b,
                options=_options(case, reverse=True),
                use_cache=False,
                include_token_artifacts=True,
            )
        prepared.append(PreparedPair(case, memory, schema_a, schema_b, reverse_a, reverse_b))
    return prepared, runtime.state_encode_calls - before


def _forward(runtime, memory: StateMemory, schema: CompiledSchema):
    return runtime.forward_compiled(
        memory,
        schema,
        coarse_mode="triadic_cp",
        relation_refinement=False,
    )


def pair_loss(runtime, row: PreparedPair) -> tuple[torch.Tensor, dict[str, float]]:
    gold_a_id = _gold_id(row.case, "a")
    gold_b_id = _gold_id(row.case, "b")
    gold_a = _gold_index(row.schema_a, gold_a_id)
    gold_b = _gold_index(row.schema_b, gold_b_id)

    out_a = _forward(runtime, row.memory, row.schema_a)
    out_b = _forward(runtime, row.memory, row.schema_b)

    target_a = torch.tensor([gold_a], device=out_a.logits.device)
    target_b = torch.tensor([gold_b], device=out_b.logits.device)
    ce_a = F.cross_entropy(out_a.logits.unsqueeze(0), target_a)
    ce_b = F.cross_entropy(out_b.logits.unsqueeze(0), target_b)
    ce = 0.5 * (ce_a + ce_b)

    other_a = _gold_index(row.schema_a, gold_b_id)
    other_b = _gold_index(row.schema_b, gold_a_id)
    margin_a = F.relu(
        out_a.logits.new_tensor(SWAP_MARGIN)
        - (out_a.logits[gold_a] - out_a.logits[other_a])
    )
    margin_b = F.relu(
        out_b.logits.new_tensor(SWAP_MARGIN)
        - (out_b.logits[gold_b] - out_b.logits[other_b])
    )
    swap = 0.5 * (margin_a + margin_b)
    loss = ce + SWAP_COEFFICIENT * swap
    return loss, {
        "ce": float(ce.detach().cpu()),
        "swap": float(swap.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate(runtime, rows: list[PreparedPair]) -> dict:
    total = 0
    correct = 0
    pair_both = 0
    pair_changed = 0
    order_flips = 0
    order_queries = 0
    max_mass_error = 0.0
    relation_delta_max = 0.0
    full_k = True
    losses = []

    for row in rows:
        pair = []
        for schema, reverse_schema, gold_id in (
            (row.schema_a, row.reverse_a, _gold_id(row.case, "a")),
            (row.schema_b, row.reverse_b, _gold_id(row.case, "b")),
        ):
            out = _forward(runtime, row.memory, schema)
            selected = out.selected_option_id
            ok = selected == gold_id
            total += 1
            correct += int(ok)
            pair.append((selected, ok))

            max_mass_error = max(
                max_mass_error,
                abs(float(out.probabilities.sum().cpu()) - 1.0),
            )
            full_k = full_k and int(out.hira.candidate_budget.item()) == 4
            relation_delta_max = max(
                relation_delta_max,
                float(out.hira.relation_delta.detach().abs().max().cpu()),
            )
            if reverse_schema is not None:
                reverse_out = _forward(runtime, row.memory, reverse_schema)
                order_flips += int(reverse_out.selected_option_id != selected)
                order_queries += 1
                max_mass_error = max(
                    max_mass_error,
                    abs(float(reverse_out.probabilities.sum().cpu()) - 1.0),
                )

        pair_both += int(pair[0][1] and pair[1][1])
        pair_changed += int(pair[0][0] != pair[1][0])
        _, pieces = pair_loss(runtime, row)
        losses.append(pieces["loss"])

    return {
        "base_cases": len(rows),
        "queries": total,
        "accuracy": correct / total,
        "paired_both_correct_rate": pair_both / len(rows),
        "question_swap_choice_change_rate": pair_changed / len(rows),
        "option_order_flip_rate": 0.0 if order_queries == 0 else order_flips / order_queries,
        "mean_loss": sum(losses) / len(losses),
        "max_probability_mass_error": max_mass_error,
        "full_k": bool(full_k),
        "relation_delta_max_abs": relation_delta_max,
    }


def _selection_key(epoch: int, metrics: dict) -> tuple[float, float, float, float, int]:
    return (
        float(metrics["paired_both_correct_rate"]),
        float(metrics["accuracy"]),
        float(metrics["question_swap_choice_change_rate"]),
        -float(metrics["mean_loss"]),
        -int(epoch),
    )


def _factor_state(runtime) -> dict[str, torch.Tensor]:
    scorer = runtime.triadic_cp_scorer
    if scorer is None:
        raise RuntimeError("S3 triadic scorer missing")
    return {
        "state_factor.weight": scorer.state_factor.weight.detach().cpu().clone(),
        "question_factor.weight": scorer.question_factor.weight.detach().cpu().clone(),
        "option_factor.weight": scorer.option_factor.weight.detach().cpu().clone(),
    }


def _load_factor_state(runtime, state: dict[str, torch.Tensor]) -> None:
    scorer = runtime.triadic_cp_scorer
    if scorer is None:
        raise RuntimeError("S3 triadic scorer missing")
    scorer.load_factor_state_dict(state, freeze=False)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--a0-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a0 = json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome") != "HIRA_V1_S3_A0_PARAMETER_FREE_TRIADIC_READY":
        raise RuntimeError("S3-A0 authority is not qualified")
    if int(a0.get("added_parameter_count", -1)) != 0:
        raise RuntimeError("S3-A0 parameter budget changed")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S3-A0 unexpectedly used for model selection")

    train_rows = generate_s3_pairs("train")
    dev_rows = generate_s3_pairs("dev")
    validate_s3_partitions(train_rows, dev_rows)
    _assert_fresh_against_prior(train_rows, dev_rows)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)

    torch.manual_seed(SEED)
    runtime = build_hira_v1_s3_triadic_core(
        frozen.runtime.encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_factors=True,
    )
    del frozen

    trainable = [p for p in runtime.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    if trainable_count != HIRA_V1_S3_FACTOR_PARAMETER_COUNT:
        raise RuntimeError(f"S3 trainable count changed: {trainable_count}")

    scorer = runtime.triadic_cp_scorer
    if scorer is None:
        raise RuntimeError("S3 triadic scorer missing")
    if scorer.factor_parameter_count != HIRA_V1_S3_FACTOR_PARAMETER_COUNT:
        raise RuntimeError("S3 factor capacity changed")
    if scorer.projection.weight.requires_grad:
        raise RuntimeError("S3 W28 projection must remain frozen")

    runtime.eval()
    train_prepared, train_state_calls = prepare_pairs(runtime, train_rows, reverse=False)
    dev_prepared, dev_state_calls = prepare_pairs(runtime, dev_rows, reverse=True)
    if train_state_calls != 256 or dev_state_calls != 64:
        raise RuntimeError(
            f"S3 state-once preparation changed: {train_state_calls}/{dev_state_calls}"
        )

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    best_key = None
    best_epoch = None
    best_state = None
    best_metrics = None

    print("HIRA_V1_S3_TRAIN_BEGIN", flush=True)

    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train_prepared)))
        random.Random(SEED + epoch).shuffle(order)
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_swap = 0.0

        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss, pieces = pair_loss(runtime, train_prepared[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            epoch_loss += pieces["loss"]
            epoch_ce += pieces["ce"]
            epoch_swap += pieces["swap"]

        dev_metrics = evaluate(runtime, dev_prepared)
        record = {
            "epoch": epoch,
            "train_mean_loss": epoch_loss / len(order),
            "train_mean_ce": epoch_ce / len(order),
            "train_mean_swap": epoch_swap / len(order),
            "dev": dev_metrics,
        }
        history.append(record)
        key = _selection_key(epoch, dev_metrics)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = _factor_state(runtime)
            best_metrics = dict(dev_metrics)
        print("HIRA_V1_S3_EPOCH=" + json.dumps(record, sort_keys=True), flush=True)

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError("S3 DEV selection produced no checkpoint")

    _load_factor_state(runtime, best_state)
    selected = evaluate(runtime, dev_prepared)
    for key in (
        "accuracy",
        "paired_both_correct_rate",
        "question_swap_choice_change_rate",
        "option_order_flip_rate",
        "max_probability_mass_error",
        "mean_loss",
    ):
        if not math.isclose(
            float(selected[key]),
            float(best_metrics[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"S3 selected DEV replay changed: {key}")

    gates = {
        "accuracy_gte_0_85": selected["accuracy"] >= 0.85,
        "paired_both_correct_gte_0_75": selected["paired_both_correct_rate"] >= 0.75,
        "question_swap_choice_change_gte_0_80": selected["question_swap_choice_change_rate"] >= 0.80,
        "option_order_flip_lte_0_02": selected["option_order_flip_rate"] <= 0.02,
        "probability_mass_error_lte_1e_6": selected["max_probability_mass_error"] <= 1e-6,
        "full_k": bool(selected["full_k"]),
        "relation_delta_zero": float(selected["relation_delta_max_abs"]) == 0.0,
        "trainable_parameters_exact_12288": trainable_count == HIRA_V1_S3_FACTOR_PARAMETER_COUNT,
        "state_once_train": train_state_calls == 256,
        "state_once_dev": dev_state_calls == 64,
    }
    outcome = READY if all(gates.values()) else FAIL

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "triadic-candidate.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s3-triadic-checkpoint-v1",
            "kind": "triadic-cp-semantic",
            "factor_parameter_count": HIRA_V1_S3_FACTOR_PARAMETER_COUNT,
            "selected_dev_epoch": best_epoch,
            "t0_checkpoint_sha256": str(manifest["t0_checkpoint_sha256"]),
            "factor_state_dict": best_state,
        },
        checkpoint_path,
    )
    checkpoint_sha = _sha256(checkpoint_path)

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": outcome,
        "scientific_authority": "V1_S3_FRESH_ENGLISH_TRAIN_DEV",
        "seed": SEED,
        "optimizer": {
            "name": "AdamW",
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
        },
        "loss": {
            "cross_entropy": True,
            "swap_margin_coefficient": SWAP_COEFFICIENT,
            "swap_margin": SWAP_MARGIN,
        },
        "partitions": {
            "train_base_cases": 256,
            "train_queries": 512,
            "dev_base_cases": 64,
            "dev_queries": 128,
            "language": "en",
            "domains": [
                "geology_sample",
                "broadcast_station",
                "robot_inventory",
                "water_treatment",
            ],
            "k": 4,
            "train_dev_exact_state_overlap": False,
            "train_dev_exact_question_overlap": False,
            "s0_exact_rows_used": False,
            "s1_exact_rows_used": False,
            "s2_exact_rows_used": False,
            "s3_a0_rows_used": False,
            "m5_final_rows_used": False,
            "w29_w34_sealed_rows_used": False,
        },
        "parameter_surface": {
            "factor_trainable_parameters": trainable_count,
            "w28_projection_trainable": 0,
            "encoder_trainable": 0,
            "hira_core_trainable": 0,
            "w34_in_semantic_path": False,
        },
        "state_once": {
            "train_encode_calls": train_state_calls,
            "dev_encode_calls": dev_state_calls,
        },
        "selected_dev_epoch": best_epoch,
        "selected_dev": selected,
        "gates": gates,
        "checkpoint_sha256": checkpoint_sha,
        "a0_authority": {
            "outcome": a0["outcome"],
            "accuracy": a0["a0"]["accuracy"],
            "paired_both_correct_rate": a0["a0"]["paired_both_correct_rate"],
            "question_changes_logits_rate": a0["a0"]["question_changes_logits_rate"],
            "question_changes_choice_rate": a0["a0"]["question_changes_choice_rate"],
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
        json.dumps([row.to_dict() for row in train_rows], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "dev-manifest.json").write_text(
        json.dumps([row.to_dict() for row in dev_rows], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("HIRA_V1_S3_TRAIN_DEV_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
