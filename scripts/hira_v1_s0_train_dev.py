from __future__ import annotations

import argparse
from collections import defaultdict
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
from nmd.v1_s0_authority import S0PairCase, generate_s0_pairs, validate_s0_partitions
from nmd.v1_semantic_core import (
    HIRA_V1_S0_CANDIDATE_PARAMETER_COUNT,
    HIRA_V1_S0_QUERY_PARAMETER_COUNT,
    HIRA_V1_S0_QUERY_RANK,
    build_hira_v1_s0_query_core,
)


SCHEMA_VERSION = "hira-v1-s0-query-train-dev-v1"
READY = "HIRA_V1_S0_QUERY_REPAIR_DEV_READY"
FAIL = "HIRA_V1_S0_QUERY_REPAIR_DEV_FAIL"

SEED = 5101
EPOCHS = 12
LR = 5e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SWAP_COEFFICIENT = 0.20
SWAP_MARGIN = 0.15


@dataclass
class PreparedPair:
    case: S0PairCase
    memory: StateMemory
    schema_a: CompiledSchema
    schema_b: CompiledSchema
    reverse_a: CompiledSchema | None = None
    reverse_b: CompiledSchema | None = None


def _options(case: S0PairCase, *, reverse: bool = False) -> tuple[LogicalOption, ...]:
    values = tuple(
        LogicalOption(
            option_id=option_id,
            criterion_text=text,
        )
        for option_id, text in zip(case.option_ids, case.option_texts)
    )
    return tuple(reversed(values)) if reverse else values


def _gold_id(case: S0PairCase, which: str) -> str:
    index = case.gold_a if which == "a" else case.gold_b
    return case.option_ids[index]


def _gold_index(schema: CompiledSchema, gold_id: str) -> int:
    return next(
        index
        for index, option in enumerate(schema.options)
        if option.option_id == gold_id
    )


@torch.no_grad()
def prepare_pairs(runtime, rows: tuple[S0PairCase, ...], *, reverse: bool) -> tuple[list[PreparedPair], int]:
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
        prepared.append(
            PreparedPair(
                case=case,
                memory=memory,
                schema_a=schema_a,
                schema_b=schema_b,
                reverse_a=reverse_a,
                reverse_b=reverse_b,
            )
        )
    return prepared, runtime.state_encode_calls - before


def _forward(runtime, memory: StateMemory, schema: CompiledSchema):
    out = runtime.forward_compiled(
        memory,
        schema,
        coarse_mode="query_conditioned_coevidence",
        relation_refinement=False,
    )
    if int(out.hira.candidate_budget.item()) != len(schema.options):
        raise RuntimeError("Hira v1 S0 full-K changed")
    if not bool(out.hira.selected_mask.all()):
        raise RuntimeError("Hira v1 S0 selected mask is not full-K")
    if float(out.hira.relation_delta.detach().abs().max().cpu()) != 0.0:
        raise RuntimeError("Hira v1 S0 relation refinement changed")
    if not bool(torch.isfinite(out.logits).all()):
        raise RuntimeError("Hira v1 S0 produced non-finite logits")
    if not bool(torch.isfinite(out.probabilities).all()):
        raise RuntimeError("Hira v1 S0 produced non-finite probabilities")
    return out


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
    by_language_total = defaultdict(int)
    by_language_correct = defaultdict(int)
    pair_both_correct = 0
    pair_choice_changed = 0
    order_flips = 0
    order_queries = 0
    max_mass_error = 0.0
    losses = []
    full_k = True
    relation_delta_max = 0.0

    for row in rows:
        gold_a_id = _gold_id(row.case, "a")
        gold_b_id = _gold_id(row.case, "b")
        outs = []
        for which, schema, reverse_schema, gold_id in (
            ("a", row.schema_a, row.reverse_a, gold_a_id),
            ("b", row.schema_b, row.reverse_b, gold_b_id),
        ):
            out = _forward(runtime, row.memory, schema)
            selected = out.selected_option_id
            is_correct = selected == gold_id
            total += 1
            correct += int(is_correct)
            by_language_total[row.case.language] += 1
            by_language_correct[row.case.language] += int(is_correct)

            mass_error = abs(float(out.probabilities.sum().cpu()) - 1.0)
            max_mass_error = max(max_mass_error, mass_error)
            full_k = full_k and int(out.hira.candidate_budget.item()) == 4
            relation_delta_max = max(
                relation_delta_max,
                float(out.hira.relation_delta.detach().abs().max().cpu()),
            )

            reverse_selected = None
            if reverse_schema is not None:
                reverse_out = _forward(runtime, row.memory, reverse_schema)
                reverse_selected = reverse_out.selected_option_id
                order_flips += int(reverse_selected != selected)
                order_queries += 1
                max_mass_error = max(
                    max_mass_error,
                    abs(float(reverse_out.probabilities.sum().cpu()) - 1.0),
                )
            outs.append((selected, is_correct))

        pair_both_correct += int(outs[0][1] and outs[1][1])
        pair_choice_changed += int(outs[0][0] != outs[1][0])

        loss, pieces = pair_loss(runtime, row)
        losses.append(pieces["loss"])

    accuracy = correct / total
    en_accuracy = by_language_correct["en"] / by_language_total["en"]
    vi_accuracy = by_language_correct["vi"] / by_language_total["vi"]
    return {
        "base_cases": len(rows),
        "queries": total,
        "accuracy": accuracy,
        "en_accuracy": en_accuracy,
        "vi_accuracy": vi_accuracy,
        "worst_language_accuracy": min(en_accuracy, vi_accuracy),
        "paired_both_correct_rate": pair_both_correct / len(rows),
        "question_swap_choice_change_rate": pair_choice_changed / len(rows),
        "option_order_flip_rate": (
            0.0 if order_queries == 0 else order_flips / order_queries
        ),
        "mean_loss": sum(losses) / len(losses),
        "max_probability_mass_error": max_mass_error,
        "full_k": bool(full_k),
        "relation_delta_max_abs": relation_delta_max,
    }


def _selection_key(epoch: int, metrics: dict) -> tuple[float, float, float, float, int]:
    return (
        float(metrics["paired_both_correct_rate"]),
        float(metrics["accuracy"]),
        float(metrics["worst_language_accuracy"]),
        -float(metrics["mean_loss"]),
        -int(epoch),
    )


def _query_state_dict(runtime) -> dict[str, torch.Tensor]:
    scorer = runtime.query_conditioned_coevidence_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S0 query scorer missing")
    return {
        "query_basis.weight": scorer.query_basis.weight.detach().cpu().clone(),
        "query_state.weight": scorer.query_state.weight.detach().cpu().clone(),
        "query_schema.weight": scorer.query_schema.weight.detach().cpu().clone(),
    }


def _load_query_state(runtime, state: dict[str, torch.Tensor]) -> None:
    scorer = runtime.query_conditioned_coevidence_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S0 query scorer missing")
    own = scorer.state_dict()
    with torch.no_grad():
        for key, value in state.items():
            own[key].copy_(value.to(device=own[key].device, dtype=own[key].dtype))


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--parameter-free-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    parameter_free = json.loads(
        args.parameter_free_result.read_text(encoding="utf-8")
    )
    if parameter_free.get("outcome") != "HIRA_V1_S0_PARAMETER_FREE_QUERY_BASELINE_READY":
        raise RuntimeError("S0 parameter-free authority is not qualified")
    if int(parameter_free.get("added_parameter_count", -1)) != 0:
        raise RuntimeError("S0 parameter-free authority changed parameter budget")

    train_rows = generate_s0_pairs("train")
    dev_rows = generate_s0_pairs("dev")
    validate_s0_partitions(train_rows, dev_rows)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder

    t0_path = bundle / str(manifest["t0_checkpoint"])
    w34_path = bundle / str(manifest["transfer_checkpoint"])
    runtime = build_hira_v1_s0_query_core(
        encoder,
        t0_path,
        w34_path,
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        expected_w34_sha256=str(manifest["transfer_checkpoint_sha256"]),
        train_query_binding=True,
    )
    del frozen

    trainable = [
        parameter
        for parameter in runtime.parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count != HIRA_V1_S0_QUERY_PARAMETER_COUNT:
        raise RuntimeError(
            f"S0 trainable count changed: {trainable_count}"
        )
    if runtime.query_conditioned_coevidence_scorer is None:
        raise RuntimeError("S0 query-conditioned scorer missing")
    if (
        runtime.query_conditioned_coevidence_scorer.candidate_parameter_count
        != HIRA_V1_S0_CANDIDATE_PARAMETER_COUNT
    ):
        raise RuntimeError("S0 candidate capacity changed")

    # Keep the frozen encoder/core in eval mode. Gradients still flow through
    # the 3,072 query parameters because requires_grad remains enabled there.
    runtime.eval()

    train_prepared, train_state_calls = prepare_pairs(
        runtime,
        train_rows,
        reverse=False,
    )
    dev_prepared, dev_state_calls = prepare_pairs(
        runtime,
        dev_rows,
        reverse=True,
    )
    if train_state_calls != 256 or dev_state_calls != 64:
        raise RuntimeError(
            f"S0 state-once preparation changed: {train_state_calls}/{dev_state_calls}"
        )

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

    print("HIRA_V1_S0_TRAIN_BEGIN", flush=True)

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
            best_state = _query_state_dict(runtime)
            best_metrics = dict(dev_metrics)

        print(
            "HIRA_V1_S0_EPOCH="
            + json.dumps(record, sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError("S0 DEV selection produced no checkpoint")

    _load_query_state(runtime, best_state)
    selected_metrics = evaluate(runtime, dev_prepared)

    # Selection replay must be deterministic.
    for key in (
        "accuracy",
        "en_accuracy",
        "vi_accuracy",
        "paired_both_correct_rate",
        "question_swap_choice_change_rate",
        "option_order_flip_rate",
        "max_probability_mass_error",
    ):
        if not math.isclose(
            float(selected_metrics[key]),
            float(best_metrics[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"S0 selected DEV replay changed: {key}")

    gates = {
        "accuracy_gte_0_80": selected_metrics["accuracy"] >= 0.80,
        "en_accuracy_gte_0_70": selected_metrics["en_accuracy"] >= 0.70,
        "vi_accuracy_gte_0_70": selected_metrics["vi_accuracy"] >= 0.70,
        "paired_both_correct_gte_0_65": (
            selected_metrics["paired_both_correct_rate"] >= 0.65
        ),
        "question_swap_choice_change_gte_0_70": (
            selected_metrics["question_swap_choice_change_rate"] >= 0.70
        ),
        "option_order_flip_lte_0_02": (
            selected_metrics["option_order_flip_rate"] <= 0.02
        ),
        "probability_mass_error_lte_1e_6": (
            selected_metrics["max_probability_mass_error"] <= 1e-6
        ),
        "full_k": bool(selected_metrics["full_k"]),
        "relation_delta_zero": (
            float(selected_metrics["relation_delta_max_abs"]) == 0.0
        ),
        "trainable_parameters_exact_3072": (
            trainable_count == HIRA_V1_S0_QUERY_PARAMETER_COUNT
        ),
        "state_once_train": train_state_calls == 256,
        "state_once_dev": dev_state_calls == 64,
    }
    passed = all(gates.values())
    outcome = READY if passed else FAIL

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "query-candidate.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s0-query-checkpoint-v1",
            "kind": "query-conditioned-coevidence",
            "query_rank": HIRA_V1_S0_QUERY_RANK,
            "query_parameter_count": HIRA_V1_S0_QUERY_PARAMETER_COUNT,
            "candidate_parameter_count": HIRA_V1_S0_CANDIDATE_PARAMETER_COUNT,
            "selected_dev_epoch": best_epoch,
            "t0_checkpoint_sha256": str(manifest["t0_checkpoint_sha256"]),
            "w34_checkpoint_sha256": str(manifest["transfer_checkpoint_sha256"]),
            "query_state_dict": best_state,
        },
        checkpoint_path,
    )
    checkpoint_sha = _sha256(checkpoint_path)

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": outcome,
        "scientific_authority": "V1_S0_FRESH_TRAIN_DEV",
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
            "languages": ["en", "vi"],
            "k": 4,
            "train_dev_exact_state_overlap": False,
            "train_dev_exact_question_overlap": False,
            "s0_localization_rows_used": False,
            "m5_final_rows_used": False,
            "w29_w34_sealed_rows_used": False,
        },
        "parameter_surface": {
            "query_rank": HIRA_V1_S0_QUERY_RANK,
            "query_trainable_parameters": trainable_count,
            "candidate_parameters": HIRA_V1_S0_CANDIDATE_PARAMETER_COUNT,
            "w34_base_trainable": 0,
            "encoder_trainable": 0,
            "hira_core_trainable": 0,
        },
        "state_once": {
            "train_encode_calls": train_state_calls,
            "dev_encode_calls": dev_state_calls,
        },
        "selected_dev_epoch": best_epoch,
        "selected_dev": selected_metrics,
        "gates": gates,
        "checkpoint_sha256": checkpoint_sha,
        "parameter_free_authority": {
            "outcome": parameter_free["outcome"],
            "accuracy": parameter_free["accuracy"],
            "question_changes_logits_rate": parameter_free["question_changes_logits_rate"],
            "question_changes_choice_rate": parameter_free["question_changes_choice_rate"],
        },
        "history": history,
        "post_dev_tuning_performed": False,
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

    print("HIRA_V1_S0_TRAIN_DEV_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
