from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.hira import HIRACore
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import (
    W28_T0_CHECKPOINT_SHA256,
    load_rescued_projection_checkpoint,
)
from nmd.typed_competitive_cache import file_sha256
from nmd.w34_transfer_authority import (
    FACTOR_IDS,
    PARTITION_DOMAINS,
    PRIMITIVES,
    QUESTION_TEXT,
    compose_severity,
    factor_options,
    generate_w34_partition,
)
from nmd.w34_transfer_cache import compile_w34_cache
from nmd.w34_transfer_core import (
    W34_CANDIDATE_PARAMETER_COUNT,
    build_hira_v0_w34_core,
    load_w34_candidate_checkpoint,
)
from nmd.w34_transfer_eval import (
    build_baseline_scorer,
    build_coevidence_scorer,
    evaluate_w34_cache,
    quality_gate_pass,
    transfer_gate_summary,
)

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"W34 A13 weight SHA mismatch: {actual}")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=256,
    )


def _validate_t0(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w28-candidate-receipt-v1":
        raise RuntimeError("W34 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("W34 T0 identity mismatch")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W34 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W34 T0 checkpoint bytes changed")
    return checkpoint


def _validate_training(directory: Path) -> tuple[Path, dict[str, object]]:
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    checkpoint = directory / "candidate.pt"
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("W34 unexpected training receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("W34 training receipt did not pass")
    if receipt.get("qualification_outcome") != "W34_REFERENCE_QUALIFIED":
        raise RuntimeError("W34 training was not qualification-gated")
    if receipt.get("qualification_domains") != ["RA", "RB"]:
        raise RuntimeError("W34 qualification identity changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("W34 confirm leakage detected")
    if receipt.get("qualification_rows_used_for_training") is not False:
        raise RuntimeError("W34 qualification-row training leakage detected")
    for key in (
        "w33_rows_used",
        "w32_rows_used",
        "w31_rows_used",
        "w30_rows_used",
        "w29_rows_used",
        "older_authority_rows_used",
    ):
        if receipt.get(key) is not False:
            raise RuntimeError(f"W34 prior-wave row leakage detected: {key}")
    if receipt.get("projection_training_performed") is not False:
        raise RuntimeError("W34 T0 projection was trained")
    if int(receipt.get("candidate_parameter_count", -1)) != W34_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("W34 candidate parameter count changed")
    actual = file_sha256(checkpoint)
    if actual != receipt.get("checkpoint_sha256"):
        raise RuntimeError("W34 candidate checkpoint/receipt SHA mismatch")
    load_w34_candidate_checkpoint(checkpoint, expected_sha256=actual)
    return checkpoint, receipt


def _compile_runtime_schemas(model):
    schemas = {}
    reversed_choice = {}
    for domain in PARTITION_DOMAINS["confirm"]:
        for factor in FACTOR_IDS:
            options = factor_options(domain, factor)
            for primitive in PRIMITIVES:
                schema, _ = model.compile_schema(
                    primitive=primitive,
                    question_text=QUESTION_TEXT[primitive][factor],
                    options=options,
                    use_cache=True,
                    include_token_artifacts=True,
                )
                schemas[(domain, factor, primitive)] = schema
            reversed_schema, _ = model.compile_schema(
                primitive="choice",
                question_text=QUESTION_TEXT["choice"][factor],
                options=tuple(reversed(options)),
                use_cache=True,
                include_token_artifacts=True,
            )
            reversed_choice[(domain, factor)] = reversed_schema
    return schemas, reversed_choice


@torch.inference_mode()
def _runtime_rows(model, rows, schemas, reversed_choice, cache_predictions):
    output = []
    for case in rows:
        before = model.state_encode_calls
        memory = model.compile_state(case.state_text)
        predictions = {}
        relation_delta_max = 0.0
        mass_error = 0.0
        full_k = True
        order_invariant = True
        cache_consistent = True
        primitive_agreement = True

        for factor in FACTOR_IDS:
            predictions[factor] = {}
            for primitive in PRIMITIVES:
                schema = schemas[(case.domain_id, factor, primitive)]
                out = model.forward_compiled(
                    memory,
                    schema,
                    coarse_mode="coevidence_symmetric_semantic",
                    relation_refinement=False,
                )
                selected = schema.options[int(out.probabilities.argmax())]
                pred = int(float(selected.value))
                predictions[factor][primitive] = pred
                relation_delta_max = max(
                    relation_delta_max,
                    float(out.hira.relation_delta.abs().max()),
                )
                mass_error = max(
                    mass_error,
                    abs(float(out.probabilities.sum()) - 1.0),
                )
                full_k = full_k and (
                    int(out.hira.candidate_budget.item()) == len(schema.options)
                    and bool(out.hira.selected_mask.all())
                )

            values = tuple(predictions[factor][p] for p in PRIMITIVES)
            primitive_agreement = primitive_agreement and len(set(values)) == 1

            cache_pred = int(
                cache_predictions[case.case_id]["factors"][factor]["pred"]
            )
            cache_consistent = cache_consistent and (
                predictions[factor]["choice"] == cache_pred
            )

            reversed_schema = reversed_choice[(case.domain_id, factor)]
            reversed_out = model.forward_compiled(
                memory,
                reversed_schema,
                coarse_mode="coevidence_symmetric_semantic",
                relation_refinement=False,
            )
            reversed_selected = reversed_schema.options[
                int(reversed_out.probabilities.argmax())
            ]
            order_invariant = order_invariant and (
                int(float(reversed_selected.value))
                == predictions[factor]["choice"]
            )
            relation_delta_max = max(
                relation_delta_max,
                float(reversed_out.hira.relation_delta.abs().max()),
            )
            mass_error = max(
                mass_error,
                abs(float(reversed_out.probabilities.sum()) - 1.0),
            )
            full_k = full_k and (
                int(reversed_out.hira.candidate_budget.item())
                == len(reversed_schema.options)
                and bool(reversed_out.hira.selected_mask.all())
            )

        vector = tuple(predictions[factor]["choice"] for factor in FACTOR_IDS)
        output.append({
            "case_id": case.case_id,
            "domain_id": case.domain_id,
            "state_encode_delta": model.state_encode_calls - before,
            "predictions": predictions,
            "factor_vector": vector,
            "severity": compose_severity(vector),
            "primitive_agreement": primitive_agreement,
            "option_order_invariant": order_invariant,
            "cache_consistent": cache_consistent,
            "full_k": full_k,
            "relation_delta_max_abs": relation_delta_max,
            "probability_mass_max_error": mass_error,
        })
    return output


def _runtime_domain_pass(rows) -> bool:
    return bool(rows) and all(
        int(row["state_encode_delta"]) == 1
        and bool(row["primitive_agreement"])
        and bool(row["option_order_invariant"])
        and bool(row["cache_consistent"])
        and bool(row["full_k"])
        and float(row["relation_delta_max_abs"]) == 0.0
        and float(row["probability_mass_max_error"]) <= 1e-6
        for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0_path = _validate_t0(args.t0_dir)
    candidate_path, training_receipt = _validate_training(args.candidate_dir)
    projection = load_rescued_projection_checkpoint(t0_path)

    encoder = _load_a13()
    runtime = build_hira_v0_w34_core(
        encoder,
        t0_path,
        candidate_path,
        expected_candidate_sha256=training_receipt["checkpoint_sha256"],
        hira=HIRACore(d_model=256, dropout=0.0),
        include_unbridged_baseline=True,
    )
    runtime.eval()

    rows = generate_w34_partition("confirm")
    before = runtime.state_encode_calls
    confirm_cache = compile_w34_cache(runtime, rows, partition="confirm")
    cache_encode_count = runtime.state_encode_calls - before
    if cache_encode_count != len(rows):
        raise RuntimeError("W34 confirm cache state encoding count changed")

    baseline = build_baseline_scorer(projection)
    candidate_state, _ = load_w34_candidate_checkpoint(
        candidate_path,
        expected_sha256=training_receipt["checkpoint_sha256"],
    )
    candidate = build_coevidence_scorer(
        projection,
        candidate_state,
        freeze=True,
    )

    baseline_eval = evaluate_w34_cache(confirm_cache, baseline)
    candidate_eval = evaluate_w34_cache(confirm_cache, candidate)

    schemas, reversed_choice = _compile_runtime_schemas(runtime)
    runtime_before = runtime.state_encode_calls
    runtime_rows = _runtime_rows(
        runtime,
        rows,
        schemas,
        reversed_choice,
        candidate_eval["predictions"],
    )
    runtime_encode_count = runtime.state_encode_calls - runtime_before

    quality_per_domain = {
        domain: quality_gate_pass(candidate_eval["per_domain"][domain])
        for domain in PARTITION_DOMAINS["confirm"]
    }
    runtime_per_domain = {
        domain: _runtime_domain_pass(
            [row for row in runtime_rows if row["domain_id"] == domain]
        )
        for domain in PARTITION_DOMAINS["confirm"]
    }
    transfer = transfer_gate_summary(
        baseline_eval["pooled"],
        candidate_eval["pooled"],
    )
    transfer_per_domain = {
        domain: transfer_gate_summary(
            baseline_eval["per_domain"][domain],
            candidate_eval["per_domain"][domain],
        )
        for domain in PARTITION_DOMAINS["confirm"]
    }

    ready = (
        all(quality_per_domain.values())
        and all(runtime_per_domain.values())
        and bool(transfer["pass"])
    )
    outcome = (
        "HIRA_V0_TRANSFER_CORE_READY"
        if ready
        else "W34_COEVIDENCE_COMPOSITION_FAIL"
    )

    audit = {
        "schema_version": "r8-w34-coevidence-audit-v1",
        "status": "PASS",
        "outcome": outcome,
        "domains": list(PARTITION_DOMAINS["confirm"]),
        "case_count": len(rows),
        "qualification_outcome": training_receipt["qualification_outcome"],
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "candidate_checkpoint_sha256": training_receipt["checkpoint_sha256"],
        "candidate_parameter_count": W34_CANDIDATE_PARAMETER_COUNT,
        "baseline": baseline_eval,
        "candidate": candidate_eval,
        "quality_per_domain": quality_per_domain,
        "runtime_per_domain": runtime_per_domain,
        "transfer": transfer,
        "transfer_per_domain": transfer_per_domain,
        "runtime_rows": runtime_rows,
        "runtime_state_encode_count": runtime_encode_count,
        "confirm_cache_state_encode_count": cache_encode_count,
        "candidate_truncation_used": False,
        "relation_refinement_used": False,
        "confirm_used_for_training_or_selection": False,
        "qualification_rows_used_for_training": False,
        "w33_rows_used": False,
        "w32_rows_used": False,
        "w31_rows_used": False,
        "w30_rows_used": False,
        "w29_rows_used": False,
        "older_authority_rows_used": False,
        "reference_outputs_used_as_model_inputs": False,
        "exact_text_overlap": [],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("W34_FINAL=" + json.dumps({
        "outcome": outcome,
        "quality": quality_per_domain,
        "runtime": runtime_per_domain,
        "transfer": transfer,
        "baseline": baseline_eval["pooled"],
        "candidate": candidate_eval["pooled"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
