from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.compositional_projection_reference import (
    REFERENCE_NAMES,
    REFERENCE_TASKS,
    evaluate_reference_panel,
)
from nmd.hira import HIRACore
from nmd.hira_v0_authority import (
    DOMAINS,
    DOMAIN_SEEDS,
    FACTOR_IDS,
    PRIMITIVES,
    QUESTION_TEXT,
    REFERENCE_HYPOTHESES,
    all_w29_query_texts,
    all_w29_schema_texts,
    all_w29_text_atoms,
    compose_severity,
    factor_options,
    generate_w29_domains,
)
from nmd.hira_v0_eval import classify_w29, summarize_hira_rows
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import (
    W28_T0_CHECKPOINT_SHA256,
    build_hira_v0_semantic_core,
)
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


def prior_text_atoms() -> set[str]:
    from nmd.anchor_preserving_residual_authority import all_w15_text_atoms
    from nmd.anchor_residual_authority import all_w12_text_atoms
    from nmd.atomic_geometry_authority import all_w25_text_atoms
    from nmd.atomic_severity_authority import all_w24_text_atoms
    from nmd.compositional_projection_authority import all_w28_text_atoms
    from nmd.conjunctive_authority import all_w7_values
    from nmd.continuous_reliability_authority import all_w14_text_atoms
    from nmd.cross_encoder_authority import all_w23_text_atoms
    from nmd.f2_compositional_authority import all_w27_text_atoms
    from nmd.field_isolated_authority import all_w17_text_atoms
    from nmd.field_semantic_rescue_authority import all_w6h_values
    from nmd.freeform_attribution_authority import all_w7b_values
    from nmd.high_cardinality_decomposition_authority import all_w6j_values
    from nmd.high_k_localization_authority import all_w6f_values
    from nmd.interface_decomposition_authority import all_w11_text_atoms
    from nmd.latent_composition_authority import all_w18_text_atoms
    from nmd.latent_ordinal_authority import all_w19_text_atoms
    from nmd.pairwise_latent_authority import all_w20_text_atoms
    from nmd.projection_replication_authority import all_w26_text_atoms
    from nmd.prototype_latent_authority import all_w21_text_atoms
    from nmd.reference_panel_authority import all_w22_text_atoms
    from nmd.regime_transfer_authority import all_w16_text_atoms
    from nmd.representation_bridge_authority import all_w6i_values
    from nmd.representation_ceiling_authority import all_w10_text_atoms
    from nmd.second_order_localization_authority import all_w6g_values
    from nmd.semantic_alignment_authority import all_w9_text_atoms
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_consistency_authority import all_w13_text_atoms
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_cross_candidate_binding import all_w5i_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab
    from nmd.semantic_transfer_authority import all_w8_text_atoms
    from nmd.typed_competitive_authority import all_w6b_values
    from nmd.typed_domain_generalization_authority import all_w6d_values
    from nmd.typed_joint_replication_authority import all_w6e_values
    from nmd.typed_reliability_authority import all_w6c_values

    return (
        set(all_w5f_vocab()) | set(all_w5g_vocab()) | set(all_w5h_vocab())
        | set(all_w5i_vocab()) | set(all_w6b_values()) | set(all_w6c_values())
        | set(all_w6d_values()) | set(all_w6e_values()) | set(all_w6f_values())
        | set(all_w6g_values()) | set(all_w6h_values()) | set(all_w6i_values())
        | set(all_w6j_values()) | set(all_w7_values()) | set(all_w7b_values())
        | set(all_w8_text_atoms()) | set(all_w9_text_atoms(include_confirm=True))
        | set(all_w10_text_atoms()) | set(all_w11_text_atoms())
        | set(all_w12_text_atoms()) | set(all_w13_text_atoms())
        | set(all_w14_text_atoms()) | set(all_w15_text_atoms())
        | set(all_w16_text_atoms()) | set(all_w17_text_atoms())
        | set(all_w18_text_atoms()) | set(all_w19_text_atoms())
        | set(all_w20_text_atoms()) | set(all_w21_text_atoms())
        | set(all_w22_text_atoms()) | set(all_w23_text_atoms())
        | set(all_w24_text_atoms()) | set(all_w25_text_atoms())
        | set(all_w26_text_atoms()) | set(all_w27_text_atoms())
        | set(all_w28_text_atoms())
    )


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("W29 A13 weight SHA mismatch")
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


def _validate_t0_receipt(directory: Path) -> dict:
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    checkpoint_path = directory / "candidate.pt"
    if receipt.get("schema_version") != "r8-w28-candidate-receipt-v1":
        raise RuntimeError("W29 unexpected W28 T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("W29 W28 T0 receipt identity mismatch")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W29 W28 T0 receipt checkpoint SHA changed")
    if file_sha256(checkpoint_path) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W29 W28 T0 checkpoint bytes changed")
    if receipt.get("trainable_parameter_count") != 32768:
        raise RuntimeError("W29 W28 T0 parameter count changed")
    return receipt


def _compile_schemas(model):
    schemas = {}
    reversed_choice = {}
    cache_receipts = {}
    for factor in FACTOR_IDS:
        options = factor_options(factor)
        for primitive in PRIMITIVES:
            schema, first = model.compile_schema(
                primitive=primitive,
                question_text=QUESTION_TEXT[primitive][factor],
                options=options,
                use_cache=True,
                include_token_artifacts=True,
            )
            schema2, second = model.compile_schema(
                primitive=primitive,
                question_text=QUESTION_TEXT[primitive][factor],
                options=options,
                use_cache=True,
                include_token_artifacts=True,
            )
            if schema.schema_hash != schema2.schema_hash:
                raise RuntimeError("W29 identical schema hash changed across cache reuse")
            schemas[(factor, primitive)] = schema
            cache_receipts[f"{factor}:{primitive}"] = {
                "first_cache_hit": first.cache_hit,
                "second_cache_hit": second.cache_hit,
                "schema_hash": schema.schema_hash,
            }

        reversed_schema, _ = model.compile_schema(
            primitive="choice",
            question_text=QUESTION_TEXT["choice"][factor],
            options=tuple(reversed(options)),
            use_cache=True,
            include_token_artifacts=True,
        )
        reversed_choice[factor] = reversed_schema
    return schemas, reversed_choice, cache_receipts


def _evaluate_hira(model, rows, schemas, reversed_choice):
    result_rows = []
    for case in rows:
        before = model.state_encode_calls
        memory = model.compile_state(case.state_text)
        predictions = {}
        relation_delta_max = 0.0
        mass_error = 0.0
        full_k = True

        for factor_index, factor in enumerate(FACTOR_IDS):
            predictions[factor] = {}
            for primitive in PRIMITIVES:
                out = model.forward_compiled(
                    memory,
                    schemas[(factor, primitive)],
                    coarse_mode="symmetric_semantic",
                    relation_refinement=False,
                )
                option = schemas[(factor, primitive)].options[
                    int(out.probabilities.argmax())
                ]
                predictions[factor][primitive] = int(float(option.value))
                relation_delta_max = max(
                    relation_delta_max,
                    float(out.hira.relation_delta.abs().max()),
                )
                mass_error = max(
                    mass_error,
                    abs(float(out.probabilities.sum()) - 1.0),
                )
                full_k = full_k and (
                    int(out.hira.candidate_budget.item())
                    == len(schemas[(factor, primitive)].options)
                    and bool(out.hira.selected_mask.all())
                )

        order_invariant = True
        for factor in FACTOR_IDS:
            out = model.forward_compiled(
                memory,
                reversed_choice[factor],
                coarse_mode="symmetric_semantic",
                relation_refinement=False,
            )
            selected = reversed_choice[factor].options[
                int(out.probabilities.argmax())
            ]
            order_invariant = order_invariant and (
                int(float(selected.value))
                == int(predictions[factor]["choice"])
            )
            relation_delta_max = max(
                relation_delta_max,
                float(out.hira.relation_delta.abs().max()),
            )
            mass_error = max(
                mass_error,
                abs(float(out.probabilities.sum()) - 1.0),
            )
            full_k = full_k and (
                int(out.hira.candidate_budget.item())
                == len(reversed_choice[factor].options)
                and bool(out.hira.selected_mask.all())
            )

        state_delta = model.state_encode_calls - before
        choice_vector = tuple(
            int(predictions[factor]["choice"])
            for factor in FACTOR_IDS
        )
        severity = compose_severity(choice_vector)
        golds = {
            factor: int(case.factor_vector[index])
            for index, factor in enumerate(FACTOR_IDS)
        }
        result_rows.append(
            {
                "case_id": case.case_id,
                "domain_id": case.domain_id,
                "golds": golds,
                "predictions": predictions,
                "factor_vector_pred": choice_vector,
                "factor_vector_correct": choice_vector == tuple(case.factor_vector),
                "severity_pred": severity,
                "severity_correct": severity == int(case.severity),
                "invalid_factor_vector": severity is None,
                "option_order_invariant": bool(order_invariant),
                "state_encode_delta": int(state_delta),
                "relation_delta_max_abs": float(relation_delta_max),
                "probability_mass_max_error": float(mass_error),
                "full_k": bool(full_k),
            }
        )
    return result_rows


def _entailment_index(config) -> int:
    for key, value in (getattr(config, "id2label", {}) or {}).items():
        if "entail" in str(value).lower():
            return int(key)
    for key, value in (getattr(config, "label2id", {}) or {}).items():
        if "entail" in str(key).lower():
            return int(value)
    raise RuntimeError("W29 reference config has no entailment label")


def _load_reference(name: str, spec: dict[str, str]):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=spec["repo"], revision=spec["revision"]))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != spec["weight_sha256"]:
        raise RuntimeError(f"W29 {name} reference SHA mismatch")
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
        values.extend(float(x) for x in probs.cpu())
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
                requests.append(
                    (row.case_id, task, value, row.state_text, hypothesis)
                )
    scores = _pair_scores(
        model,
        tokenizer,
        [(query, hypothesis) for _, _, _, query, hypothesis in requests],
        entailment_index,
    )
    for request, score in zip(requests, scores):
        case_id, task, value, _, _ = request
        lookup[case_id][task][value] = score
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    values = all_w29_text_atoms()
    overlap = sorted(values & prior_text_atoms())
    if overlap:
        raise RuntimeError(f"W29 exact text atoms overlap prior authorities: {overlap}")
    if all_w29_query_texts() & all_w29_schema_texts():
        raise RuntimeError("W29 query/schema exact sentence overlap")

    rows = generate_w29_domains()
    if len(rows) != 384 or {row.domain_id for row in rows} != set(DOMAINS):
        raise RuntimeError("W29 authority count/domain set changed")

    t0_receipt = _validate_t0_receipt(args.t0_dir)
    encoder = _load_a13()
    model = build_hira_v0_semantic_core(
        encoder,
        args.t0_dir / "candidate.pt",
        expected_sha256=W28_T0_CHECKPOINT_SHA256,
        hira=HIRACore(d_model=256, dropout=0.0),
    )
    model.eval()

    schemas, reversed_choice, cache_receipts = _compile_schemas(model)
    hira_rows = _evaluate_hira(
        model,
        rows,
        schemas,
        reversed_choice,
    )
    hira_summary = summarize_hira_rows(hira_rows)

    model_scores = {}
    actual_hashes = {}
    entailment_indices = {}
    for name in REFERENCE_NAMES:
        reference, tokenizer, actual, index = _load_reference(
            name,
            REFERENCE_PANEL[name],
        )
        model_scores[name] = _reference_scores(
            rows,
            reference,
            tokenizer,
            index,
        )
        actual_hashes[name] = actual
        entailment_indices[name] = index
        del reference
        gc.collect()

    panel = evaluate_reference_panel(
        [row.__dict__ for row in rows],
        model_scores,
    )

    runtime_integrity = {
        "t0_checkpoint_exact": file_sha256(
            args.t0_dir / "candidate.pt"
        ) == W28_T0_CHECKPOINT_SHA256,
        "projection_trainable_parameter_count": int(
            model.symmetric_semantic_scorer.trainable_parameter_count
        ),
        "projection_parameter_count": sum(
            p.numel() for p in model.symmetric_semantic_scorer.parameters()
        ),
        "schema_cache_hit_after_first_compile": all(
            (not row["first_cache_hit"]) and row["second_cache_hit"]
            for row in cache_receipts.values()
        ),
        "relation_refinement_disabled": True,
        "candidate_truncation_used": False,
        "reference_outputs_used_as_model_inputs": False,
        "primary_query_count_per_case": 9,
        "invariance_query_count_per_case": 3,
        "state_encode_calls_total": int(model.state_encode_calls),
        "case_count": len(rows),
    }

    classification = classify_w29(
        panel=panel,
        hira_summary=hira_summary,
        runtime_integrity=runtime_integrity,
    )

    result = {
        "schema_version": "r8-w29-hira-v0-semantic-core-audit-v1",
        "status": "PASS",
        **classification,
        "hira": hira_summary,
        "reference": panel,
        "runtime_integrity": runtime_integrity,
        "schema_cache_receipts": cache_receipts,
        "t0_receipt": t0_receipt,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "reference_panel": REFERENCE_PANEL,
        "actual_reference_weight_sha256": actual_hashes,
        "reference_entailment_label_index": entailment_indices,
        "case_count": 384,
        "domains": list(DOMAINS),
        "domain_seeds": dict(DOMAIN_SEEDS),
        "projection_training_performed": False,
        "projection_selection_performed": False,
        "relation_refinement_used": False,
        "reliability_calibration_used": False,
        "reference_outputs_used_as_model_inputs": False,
        "prior_exact_text_overlap": [],
        "query_schema_exact_sentence_overlap": [],
        "w28_rows_used": False,
        "w27_rows_used": False,
        "w26_rows_used": False,
        "w25_rows_used": False,
        "w24_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
        "text_atom_sha256": sha256(
            "\n".join(sorted(values)).encode()
        ).hexdigest(),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "outcome": result["outcome"],
        "reference_per_domain": result["reference_per_domain"],
        "hira_per_domain": result["hira_per_domain"],
        "runtime_per_domain": result["runtime_per_domain"],
        "global_runtime_integrity": result["global_runtime_integrity"],
        "hira_pooled": result["hira"]["pooled"],
        "reference_pooled": result["reference"]["consensus"]["pooled"],
        "runtime_integrity": result["runtime_integrity"],
    }
    print("W29_AUDIT_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
