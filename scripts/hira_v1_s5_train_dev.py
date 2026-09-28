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
from nmd.v1_s3_authority import generate_s3_pairs
from nmd.v1_s4_authority import generate_s4_pairs
from nmd.v1_s5_authority import S5PairCase, generate_s5_pairs, validate_s5_partitions
from nmd.v1_s5_semantic_core import (
    HIRA_V1_S5_PROJECTION_PARAMETER_COUNT,
    build_hira_v1_s5_projection_core,
)


SCHEMA_VERSION = "hira-v1-s5-projection-train-dev-v1"
READY = "HIRA_V1_S5_PROJECTION_DEV_READY"
FAIL = "HIRA_V1_S5_PROJECTION_DEV_FAIL"

SEED = 10501
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SWAP_COEFFICIENT = 0.25
SWAP_MARGIN = 0.20
ALIGN_COEFFICIENT = 0.10
ALIGN_TEMPERATURE = 0.10


@dataclass
class PreparedPair:
    case: S5PairCase
    memory: StateMemory
    schema_a: CompiledSchema
    schema_b: CompiledSchema
    reverse_a: CompiledSchema | None = None
    reverse_b: CompiledSchema | None = None


def _options(case: S5PairCase, *, reverse: bool = False) -> tuple[LogicalOption, ...]:
    values = tuple(
        LogicalOption(
            option_id=option_id,
            criterion_text=text,
            aliases=(alias,),
        )
        for option_id, text, alias in zip(
            case.option_ids,
            case.option_texts,
            case.option_aliases,
        )
    )
    return tuple(reversed(values)) if reverse else values


def _gold_id(case: S5PairCase, which: str) -> str:
    index = case.gold_a if which == "a" else case.gold_b
    return case.option_ids[index]


def _gold_index(schema: CompiledSchema, gold_id: str) -> int:
    return next(
        index
        for index, option in enumerate(schema.options)
        if option.option_id == gold_id
    )


def _assert_fresh_against_prior(
    train_rows: tuple[S5PairCase, ...],
    dev_rows: tuple[S5PairCase, ...],
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
    )
    prior_states = {row.state for row in prior}
    prior_questions = {
        q
        for row in prior
        for q in (row.question_a, row.question_b)
    }
    current = (*train_rows, *dev_rows)
    current_states = {row.state for row in current}
    current_questions = {
        q
        for row in current
        for q in (row.question_a, row.question_b)
    }
    if prior_states & current_states:
        raise RuntimeError("S5 exact state overlap with exposed S0-S4 rows")
    if prior_questions & current_questions:
        raise RuntimeError("S5 exact question overlap with exposed S0-S4 rows")


@torch.no_grad()
def prepare_pairs(
    runtime,
    rows: tuple[S5PairCase, ...],
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

        for schema in (schema_a, schema_b):
            if schema.option_view_mask is None:
                raise RuntimeError(f"{case.case_id}: option views missing")
            if schema.option_view_mask.shape[1] < 2:
                raise RuntimeError(f"{case.case_id}: S5 requires >=2 views")
            if not bool(schema.option_view_mask[:, :2].all()):
                raise RuntimeError(f"{case.case_id}: two active views required")

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
                case,
                memory,
                schema_a,
                schema_b,
                reverse_a,
                reverse_b,
            )
        )
    return prepared, runtime.state_encode_calls - before


def _forward(runtime, memory: StateMemory, schema: CompiledSchema):
    return runtime.forward_compiled(
        memory,
        schema,
        coarse_mode="projection_triadic",
        relation_refinement=False,
    )


def _alignment_loss(runtime, schema: CompiledSchema) -> torch.Tensor:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S5 projection scorer missing")
    ov = schema.option_view_token_embeddings
    ovtm = schema.option_view_token_mask
    ovm = schema.option_view_mask
    if ov is None or ovtm is None or ovm is None:
        raise RuntimeError("S5 alignment requires option view artifacts")
    return scorer.option_view_alignment_loss(
        option_view_tokens=ov.unsqueeze(0),
        option_view_token_mask=ovtm.unsqueeze(0),
        option_view_mask=ovm.unsqueeze(0),
        temperature=ALIGN_TEMPERATURE,
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
    decision = ce + SWAP_COEFFICIENT * swap

    # The option set is identical for the paired questions, so compute one
    # alignment term from schema A only.
    alignment = _alignment_loss(runtime, row.schema_a)
    total = decision + ALIGN_COEFFICIENT * alignment

    return total, {
        "ce": float(ce.detach().cpu()),
        "swap": float(swap.detach().cpu()),
        "decision": float(decision.detach().cpu()),
        "alignment": float(alignment.detach().cpu()),
        "total": float(total.detach().cpu()),
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
    full_k = True
    relation_delta_max = 0.0
    decision_losses = []
    alignment_losses = []

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
                order_flips += int(
                    reverse_out.selected_option_id != selected
                )
                order_queries += 1
                max_mass_error = max(
                    max_mass_error,
                    abs(float(reverse_out.probabilities.sum().cpu()) - 1.0),
                )

        pair_both += int(pair[0][1] and pair[1][1])
        pair_changed += int(pair[0][0] != pair[1][0])

        _, pieces = pair_loss(runtime, row)
        decision_losses.append(pieces["decision"])
        alignment_losses.append(pieces["alignment"])

    return {
        "base_cases": len(rows),
        "queries": total,
        "accuracy": correct / total,
        "paired_both_correct_rate": pair_both / len(rows),
        "question_swap_choice_change_rate": pair_changed / len(rows),
        "option_order_flip_rate": (
            0.0 if order_queries == 0 else order_flips / order_queries
        ),
        "mean_decision_loss": sum(decision_losses) / len(decision_losses),
        "mean_alignment_loss": sum(alignment_losses) / len(alignment_losses),
        "max_probability_mass_error": max_mass_error,
        "full_k": bool(full_k),
        "relation_delta_max_abs": relation_delta_max,
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


def _projection_state(runtime) -> dict[str, torch.Tensor]:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S5 projection scorer missing")
    return {
        "projection.weight": scorer.projection.weight.detach().cpu().clone(),
    }


def _load_projection_state(
    runtime,
    state: dict[str, torch.Tensor],
) -> None:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S5 projection scorer missing")
    scorer.load_projection_state_dict(state, freeze=False)


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
    if a0.get("outcome") != "HIRA_V1_S5_A0_IDENTITY_READY":
        raise RuntimeError("S5-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S5-A0 unexpectedly used for model selection")
    if float(a0.get("exact_logit_identity_rate", -1.0)) != 1.0:
        raise RuntimeError("S5-A0 W28 identity boundary failed")

    train_rows = generate_s5_pairs("train")
    dev_rows = generate_s5_pairs("dev")
    validate_s5_partitions(train_rows, dev_rows)
    _assert_fresh_against_prior(train_rows, dev_rows)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder

    runtime = build_hira_v1_s5_projection_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_projection=True,
    )
    del frozen

    trainable = [p for p in runtime.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    if trainable_count != HIRA_V1_S5_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError(f"S5 trainable count changed: {trainable_count}")
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S5 projection scorer missing")
    if scorer.projection_parameter_count != HIRA_V1_S5_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S5 projection capacity changed")

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
    if train_state_calls != 512 or dev_state_calls != 128:
        raise RuntimeError(
            f"S5 state-once preparation changed: "
            f"{train_state_calls}/{dev_state_calls}"
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

    print("HIRA_V1_S5_TRAIN_BEGIN", flush=True)

    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train_prepared)))
        random.Random(SEED + epoch).shuffle(order)

        train_total = 0.0
        train_decision = 0.0
        train_alignment = 0.0
        train_ce = 0.0
        train_swap = 0.0

        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss, pieces = pair_loss(runtime, train_prepared[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()

            train_total += pieces["total"]
            train_decision += pieces["decision"]
            train_alignment += pieces["alignment"]
            train_ce += pieces["ce"]
            train_swap += pieces["swap"]

        dev_metrics = evaluate(runtime, dev_prepared)
        record = {
            "epoch": epoch,
            "train_mean_total_loss": train_total / len(order),
            "train_mean_decision_loss": train_decision / len(order),
            "train_mean_ce": train_ce / len(order),
            "train_mean_swap": train_swap / len(order),
            "train_mean_alignment_loss": train_alignment / len(order),
            "dev": dev_metrics,
        }
        history.append(record)

        key = _selection_key(epoch, dev_metrics)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = _projection_state(runtime)
            best_metrics = dict(dev_metrics)

        print(
            "HIRA_V1_S5_EPOCH=" + json.dumps(record, sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError("S5 DEV selection produced no checkpoint")

    _load_projection_state(runtime, best_state)
    selected = evaluate(runtime, dev_prepared)
    for key in (
        "accuracy",
        "paired_both_correct_rate",
        "question_swap_choice_change_rate",
        "option_order_flip_rate",
        "max_probability_mass_error",
        "mean_decision_loss",
        "mean_alignment_loss",
    ):
        if not math.isclose(
            float(selected[key]),
            float(best_metrics[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"S5 selected DEV replay changed: {key}")

    gates = {
        "accuracy_gte_0_85": selected["accuracy"] >= 0.85,
        "paired_both_correct_gte_0_75": (
            selected["paired_both_correct_rate"] >= 0.75
        ),
        "question_swap_choice_change_gte_0_80": (
            selected["question_swap_choice_change_rate"] >= 0.80
        ),
        "option_order_flip_lte_0_02": selected["option_order_flip_rate"] <= 0.02,
        "probability_mass_error_lte_1e_6": (
            selected["max_probability_mass_error"] <= 1e-6
        ),
        "full_k": bool(selected["full_k"]),
        "relation_delta_zero": float(selected["relation_delta_max_abs"]) == 0.0,
        "state_once_train": train_state_calls == 512,
        "state_once_dev": dev_state_calls == 128,
        "trainable_projection_exact_32768": (
            trainable_count == HIRA_V1_S5_PROJECTION_PARAMETER_COUNT
        ),
        "encoder_frozen": not any(
            p.requires_grad for p in runtime.encoder.parameters()
        ),
        "hira_core_frozen": not any(
            p.requires_grad for p in runtime.hira.parameters()
        ),
    }
    outcome = READY if all(gates.values()) else FAIL

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "projection-candidate.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s5-projection-checkpoint-v1",
            "kind": "semantic-projection-relearning",
            "projection_parameter_count": HIRA_V1_S5_PROJECTION_PARAMETER_COUNT,
            "selected_dev_epoch": best_epoch,
            "initialization_t0_sha256": str(manifest["t0_checkpoint_sha256"]),
            "projection_state_dict": best_state,
        },
        checkpoint_path,
    )
    checkpoint_sha = _sha256(checkpoint_path)

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": outcome,
        "scientific_authority": "V1_S5_FRESH_ENGLISH_TRAIN_DEV",
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
            "option_view_infonce_coefficient": ALIGN_COEFFICIENT,
            "option_view_infonce_temperature": ALIGN_TEMPERATURE,
        },
        "partitions": {
            "train_base_cases": 512,
            "train_queries": 1024,
            "dev_base_cases": 128,
            "dev_queries": 256,
            "language": "en",
            "domains": [
                "battery_pack",
                "weather_station",
                "textile_batch",
                "cargo_drone",
                "aquaculture_feed",
                "telescope_schedule",
                "ceramic_kiln",
                "data_center",
            ],
            "k": 4,
            "views_per_option": 2,
            "train_dev_exact_state_overlap": False,
            "train_dev_exact_question_overlap": False,
            "prior_track_exact_rows_used": False,
            "s5_a0_rows_used": False,
            "m5_final_rows_used": False,
            "w29_w34_sealed_rows_used": False,
        },
        "parameter_surface": {
            "projection_trainable_parameters": trainable_count,
            "encoder_trainable": sum(
                p.numel()
                for p in runtime.encoder.parameters()
                if p.requires_grad
            ),
            "hira_core_trainable": sum(
                p.numel()
                for p in runtime.hira.parameters()
                if p.requires_grad
            ),
            "learned_downstream_scorer_parameters": 0,
        },
        "state_once": {
            "train_encode_calls": train_state_calls,
            "dev_encode_calls": dev_state_calls,
        },
        "selected_dev_epoch": best_epoch,
        "selected_dev": selected,
        "gates": gates,
        "checkpoint_sha256": checkpoint_sha,
        "initialization_t0_sha256": str(manifest["t0_checkpoint_sha256"]),
        "a0_authority": {
            "outcome": a0["outcome"],
            "accuracy": a0["accuracy"],
            "paired_both_correct_rate": a0["paired_both_correct_rate"],
            "exact_logit_identity_rate": a0["exact_logit_identity_rate"],
            "mean_frozen_view_infonce": a0["mean_frozen_view_infonce"],
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
        "HIRA_V1_S5_TRAIN_DEV_RECEIPT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
