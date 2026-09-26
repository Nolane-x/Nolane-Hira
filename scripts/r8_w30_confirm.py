from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import torch

from nmd.compositional_projection_reference import evaluate_reference_panel
from nmd.hira import HIRACore
from nmd.hira_v0_authority import all_w29_text_atoms
from nmd.hira_v0_eval import reference_domain_pass_w29
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.semantic_transfer_authority import (
    FACTOR_IDS,
    PARTITION_DOMAINS,
    PRIMITIVES,
    QUESTION_TEXT,
    REFERENCE_HYPOTHESES,
    all_w30_text_atoms,
    compose_severity,
    factor_options,
    generate_w30_partition,
)
from nmd.semantic_transfer_cache import compile_w30_cache
from nmd.semantic_transfer_core import (
    W30_BRIDGE_PARAMETER_COUNT,
    build_hira_v0_transfer_core,
    load_transfer_bridge_checkpoint,
)
from nmd.semantic_transfer_eval import evaluate_w30_cache
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

REFERENCE_PANEL = {
    "deberta_nli": {
        "repo": "cross-encoder/nli-deberta-v3-base",
        "revision": "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
        "weight_sha256": "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa",
    },
    "roberta_nli": {
        "repo": "cross-encoder/nli-roberta-base",
        "revision": "1be0567456f0543475805e758725f151f283705a",
        "weight_sha256": "efc90996d2ed80123c26c9091c91385ffddc6d2fd0b2bacf3187fbd6c5b87953",
    },
}
REFERENCE_NAMES = tuple(REFERENCE_PANEL)
REFERENCE_TASKS = ("F0", "F1", "U", "C", "F2")


def prior_text_atoms() -> set[str]:
    spec = importlib.util.spec_from_file_location(
        "w29eval",
        Path(__file__).with_name("r8_w29_evaluate.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load W29 prior-text firewall")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.prior_text_atoms()) | set(all_w29_text_atoms())


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"W30 A13 weight SHA mismatch: {actual}")
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
        raise RuntimeError("W30 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("W30 T0 identity mismatch")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W30 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W30 T0 checkpoint bytes changed")
    return checkpoint


def _validate_bridge(directory: Path) -> tuple[Path, dict[str, object]]:
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    checkpoint = directory / "bridge.pt"
    if receipt.get("schema_version") != "r8-w30-transfer-training-receipt-v1":
        raise RuntimeError("W30 unexpected training receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("W30 training receipt did not pass")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("W30 confirm leakage detected")
    if receipt.get("w29_rows_used") is not False:
        raise RuntimeError("W30 W29-row leakage detected")
    if receipt.get("older_authority_rows_used") is not False:
        raise RuntimeError("W30 older-authority leakage detected")
    if receipt.get("projection_training_performed") is not False:
        raise RuntimeError("W30 T0 projection was trained")
    if int(receipt.get("bridge_parameter_count", -1)) != W30_BRIDGE_PARAMETER_COUNT:
        raise RuntimeError("W30 bridge parameter count changed")
    actual = file_sha256(checkpoint)
    if actual != receipt.get("checkpoint_sha256"):
        raise RuntimeError("W30 bridge checkpoint/receipt SHA mismatch")
    load_transfer_bridge_checkpoint(checkpoint, expected_sha256=actual)
    return checkpoint, receipt


def _entailment_index(config) -> int:
    for key, value in (getattr(config, "id2label", {}) or {}).items():
        if "entail" in str(value).lower():
            return int(key)
    for key, value in (getattr(config, "label2id", {}) or {}).items():
        if "entail" in str(key).lower():
            return int(value)
    raise RuntimeError("W30 reference config has no entailment label")


def _load_reference(name: str, spec_data: dict[str, str]):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = Path(
        snapshot_download(repo_id=spec_data["repo"], revision=spec_data["revision"])
    )
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != spec_data["weight_sha256"]:
        raise RuntimeError(f"W30 {name} reference SHA mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, actual, _entailment_index(model.config)


@torch.inference_mode()
def _pair_scores(model, tokenizer, pairs, entailment_index: int):
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(pairs), 64):
        batch = pairs[start : start + 64]
        encoded = tokenizer(
            [a for a, _ in batch],
            [b for _, b in batch],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded, return_dict=True).logits
        probs = torch.softmax(logits.float(), dim=-1)[:, entailment_index]
        values.extend(float(value) for value in probs.cpu())
    return values


def _reference_scores(rows, model, tokenizer, entailment_index: int):
    lookup = {
        row.case_id: {task: [0.0, 0.0] for task in REFERENCE_TASKS}
        for row in rows
    }
    requests = []
    for row in rows:
        for task in REFERENCE_TASKS:
            for value, hypothesis in enumerate(REFERENCE_HYPOTHESES[task]):
                requests.append((row.case_id, task, value, row.state_text, hypothesis))
    scores = _pair_scores(
        model,
        tokenizer,
        [(text, hypothesis) for _, _, _, text, hypothesis in requests],
        entailment_index,
    )
    for request, score in zip(requests, scores):
        case_id, task, value, _, _ = request
        lookup[case_id][task][value] = score
    return lookup


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


def _runtime_rows(model, rows, schemas, reversed_choice, cache_predictions):
    result = []
    for case in rows:
        before = model.state_encode_calls
        memory = model.compile_state(case.state_text)
        predictions = {}
        relation_delta_max = 0.0
        mass_error = 0.0
        full_k = True
        order_invariant = True
        cache_consistent = True

        for index, factor in enumerate(FACTOR_IDS):
            predictions[factor] = {}
            for primitive in PRIMITIVES:
                schema = schemas[(case.domain_id, factor, primitive)]
                out = model.forward_compiled(
                    memory,
                    schema,
                    coarse_mode="bridged_symmetric_semantic",
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
                coarse_mode="bridged_symmetric_semantic",
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
        severity = compose_severity(vector)
        result.append({
            "case_id": case.case_id,
            "domain_id": case.domain_id,
            "state_encode_delta": model.state_encode_calls - before,
            "predictions": predictions,
            "gold_vector": tuple(case.factor_vector),
            "factor_vector_correct": vector == tuple(case.factor_vector),
            "severity_correct": severity == int(case.severity),
            "invalid_factor_vector": severity is None,
            "option_order_invariant": bool(order_invariant),
            "cache_consistent": bool(cache_consistent),
            "relation_delta_max_abs": float(relation_delta_max),
            "probability_mass_max_error": float(mass_error),
            "full_k": bool(full_k),
        })
    return result


def _runtime_summary(rows):
    out = {}
    for domain in PARTITION_DOMAINS["confirm"]:
        subset = [row for row in rows if row["domain_id"] == domain]
        agreements = {}
        for factor in FACTOR_IDS:
            agreements[factor] = {
                "choice_vs_score": sum(
                    row["predictions"][factor]["choice"]
                    == row["predictions"][factor]["score"]
                    for row in subset
                ) / len(subset),
                "choice_vs_noul": sum(
                    row["predictions"][factor]["choice"]
                    == row["predictions"][factor]["noul"]
                    for row in subset
                ) / len(subset),
                "score_vs_noul": sum(
                    row["predictions"][factor]["score"]
                    == row["predictions"][factor]["noul"]
                    for row in subset
                ) / len(subset),
            }
        out[domain] = {
            "state_once_rate": sum(row["state_encode_delta"] == 1 for row in subset) / len(subset),
            "option_order_invariance": sum(row["option_order_invariant"] for row in subset) / len(subset),
            "cache_consistency": sum(row["cache_consistent"] for row in subset) / len(subset),
            "full_k_rate": sum(row["full_k"] for row in subset) / len(subset),
            "relation_delta_max_abs": max(row["relation_delta_max_abs"] for row in subset),
            "probability_mass_max_error": max(row["probability_mass_max_error"] for row in subset),
            "cross_primitive_agreement": agreements,
        }
    return out


def _quality_domain_pass(row) -> bool:
    return bool(
        all(float(row["factors"][factor]["top1"]) >= 0.90 for factor in FACTOR_IDS)
        and all(float(row["factors"][factor]["balanced_accuracy"]) >= 0.88 for factor in FACTOR_IDS)
        and float(row["factor_vector_top1"]) >= 0.82
        and float(row["composed_severity_top1"]) >= 0.82
        and float(row["invalid_factor_vector_rate"]) <= 0.05
        and float(row["factor_probability_mass_max_error"]) <= 1e-6
    )


def _runtime_domain_pass(row) -> bool:
    return bool(
        float(row["state_once_rate"]) == 1.0
        and float(row["option_order_invariance"]) == 1.0
        and float(row["cache_consistency"]) == 1.0
        and float(row["full_k_rate"]) == 1.0
        and float(row["relation_delta_max_abs"]) == 0.0
        and float(row["probability_mass_max_error"]) <= 1e-6
        and all(
            float(value) == 1.0
            for factor in row["cross_primitive_agreement"].values()
            for value in factor.values()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    overlap = sorted(all_w30_text_atoms() & prior_text_atoms())
    if overlap:
        raise RuntimeError(f"W30 exact text overlaps exposed authorities: {overlap}")

    t0_path = _validate_t0(args.t0_dir)
    bridge_path, train_receipt = _validate_bridge(args.bridge_dir)
    bridge_sha = file_sha256(bridge_path)

    encoder = _load_a13()
    model = build_hira_v0_transfer_core(
        encoder,
        t0_path,
        bridge_path,
        expected_bridge_sha256=bridge_sha,
        hira=HIRACore(d_model=256, dropout=0.0),
        include_unbridged_baseline=True,
    )
    model.eval()
    if model.symmetric_semantic_scorer is None:
        raise RuntimeError("W30 baseline scorer missing")
    if model.bridged_symmetric_semantic_scorer is None:
        raise RuntimeError("W30 bridged scorer missing")

    confirm_rows = generate_w30_partition("confirm")
    before_cache = model.state_encode_calls
    confirm_cache = compile_w30_cache(model, confirm_rows, partition="confirm")
    cache_encodes = model.state_encode_calls - before_cache
    if cache_encodes != len(confirm_rows):
        raise RuntimeError("W30 confirm cache state-encode count changed")

    baseline_eval = evaluate_w30_cache(
        confirm_cache,
        model.symmetric_semantic_scorer,
    )
    bridged_eval = evaluate_w30_cache(
        confirm_cache,
        model.bridged_symmetric_semantic_scorer,
    )

    schemas, reversed_choice = _compile_runtime_schemas(model)
    runtime_rows = _runtime_rows(
        model,
        confirm_rows,
        schemas,
        reversed_choice,
        bridged_eval["predictions"],
    )
    runtime = _runtime_summary(runtime_rows)

    model_scores = {}
    reference_hashes = {}
    entailment_indices = {}
    for name in REFERENCE_NAMES:
        reference, tokenizer, actual, entailment_index = _load_reference(
            name,
            REFERENCE_PANEL[name],
        )
        model_scores[name] = _reference_scores(
            confirm_rows,
            reference,
            tokenizer,
            entailment_index,
        )
        reference_hashes[name] = actual
        entailment_indices[name] = entailment_index
        del reference
        gc.collect()

    panel = evaluate_reference_panel(
        [row.__dict__ for row in confirm_rows],
        model_scores,
    )

    reference_per_domain = {
        domain: reference_domain_pass_w29(panel, domain)
        for domain in PARTITION_DOMAINS["confirm"]
    }
    quality_per_domain = {
        domain: _quality_domain_pass(bridged_eval["per_domain"][domain])
        for domain in PARTITION_DOMAINS["confirm"]
    }
    runtime_per_domain = {
        domain: _runtime_domain_pass(runtime[domain])
        for domain in PARTITION_DOMAINS["confirm"]
    }

    baseline_pooled = baseline_eval["pooled"]
    bridged_pooled = bridged_eval["pooled"]
    severity_delta = (
        float(bridged_pooled["composed_severity_top1"])
        - float(baseline_pooled["composed_severity_top1"])
    )
    factor_regressions = {
        factor: (
            float(baseline_pooled["factors"][factor]["top1"])
            - float(bridged_pooled["factors"][factor]["top1"])
        )
        for factor in FACTOR_IDS
    }
    transfer_gate = bool(
        severity_delta >= 0.20
        and all(value <= 0.03 for value in factor_regressions.values())
    )

    global_runtime = bool(
        file_sha256(t0_path) == W28_T0_CHECKPOINT_SHA256
        and model.symmetric_semantic_scorer.trainable_parameter_count == 0
        and model.bridged_symmetric_semantic_scorer.trainable_parameter_count == 0
        and model.bridged_symmetric_semantic_scorer.bridge_parameter_count
        == W30_BRIDGE_PARAMETER_COUNT
        and train_receipt["confirm_case_count_used"] == 0
        and train_receipt["w29_rows_used"] is False
        and train_receipt["older_authority_rows_used"] is False
    )

    if not all(reference_per_domain.values()):
        outcome = "W30_REFERENCE_INADEQUATE"
    elif not (
        all(quality_per_domain.values())
        and all(runtime_per_domain.values())
        and transfer_gate
        and global_runtime
    ):
        outcome = "W30_TRANSFER_BRIDGE_FAIL"
    else:
        outcome = "HIRA_V0_TRANSFER_CORE_READY"

    result = {
        "schema_version": "r8-w30-transfer-core-audit-v1",
        "status": "PASS",
        "outcome": outcome,
        "domains": list(PARTITION_DOMAINS["confirm"]),
        "case_count": len(confirm_rows),
        "reference_per_domain": reference_per_domain,
        "quality_per_domain": quality_per_domain,
        "runtime_per_domain": runtime_per_domain,
        "transfer_gate": transfer_gate,
        "global_runtime_integrity": global_runtime,
        "severity_delta_vs_unbridged": severity_delta,
        "factor_regressions_vs_unbridged": factor_regressions,
        "baseline": baseline_eval,
        "bridged": bridged_eval,
        "runtime": runtime,
        "reference": panel,
        "bridge_checkpoint_sha256": bridge_sha,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "reference_weight_sha256": reference_hashes,
        "reference_entailment_label_index": entailment_indices,
        "training_receipt": train_receipt,
        "w29_rows_used": False,
        "older_authority_rows_used": False,
        "reference_outputs_used_as_model_inputs": False,
        "candidate_truncation_used": False,
        "relation_refinement_used": False,
        "confirm_used_for_training_or_selection": False,
        "exact_text_overlap": [],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("W30_FINAL=" + json.dumps({
        "outcome": outcome,
        "reference_per_domain": reference_per_domain,
        "quality_per_domain": quality_per_domain,
        "runtime_per_domain": runtime_per_domain,
        "transfer_gate": transfer_gate,
        "severity_delta_vs_unbridged": severity_delta,
        "factor_regressions_vs_unbridged": factor_regressions,
        "baseline_pooled": baseline_pooled,
        "bridged_pooled": bridged_pooled,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
