from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.atomic_geometry_authority import all_w25_text_atoms
from nmd.projection_replication_authority import (
    DOMAIN_SEEDS,
    FACTOR_IDS,
    FACTOR_REFERENCE_HYPOTHESES,
    all_w26_query_texts,
    all_w26_schema_texts,
    all_w26_text_atoms,
    generate_w26_partition,
)
from nmd.projection_replication_reference import (
    REFERENCE_NAMES,
    evaluate_reference_panel,
    qualification_result,
)
from nmd.typed_competitive_cache import file_sha256

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
    from nmd.atomic_severity_authority import all_w24_text_atoms
    from nmd.conjunctive_authority import all_w7_values
    from nmd.continuous_reliability_authority import all_w14_text_atoms
    from nmd.cross_encoder_authority import all_w23_text_atoms
    from nmd.field_isolated_authority import all_w17_text_atoms
    from nmd.field_semantic_rescue_authority import all_w6h_values
    from nmd.freeform_attribution_authority import all_w7b_values
    from nmd.high_cardinality_decomposition_authority import all_w6j_values
    from nmd.high_k_localization_authority import all_w6f_values
    from nmd.interface_decomposition_authority import all_w11_text_atoms
    from nmd.latent_composition_authority import all_w18_text_atoms
    from nmd.latent_ordinal_authority import all_w19_text_atoms
    from nmd.pairwise_latent_authority import all_w20_text_atoms
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
    )


def _entailment_index(config) -> int:
    id2label = getattr(config, "id2label", {}) or {}
    for key, value in id2label.items():
        if "entail" in str(value).lower():
            return int(key)
    label2id = getattr(config, "label2id", {}) or {}
    for key, value in label2id.items():
        if "entail" in str(key).lower():
            return int(value)
    raise RuntimeError("W26 reference config has no identifiable entailment label")


def _load_reference(name: str, spec: dict[str, str]):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=spec["repo"], revision=spec["revision"]))
    weight = snapshot / "model.safetensors"
    if not weight.exists():
        raise RuntimeError(f"W26 {name} model.safetensors missing")
    actual_sha = file_sha256(weight)
    if actual_sha != spec["weight_sha256"]:
        raise RuntimeError(f"W26 {name} pinned weight SHA mismatch: {actual_sha}")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot), local_files_only=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, actual_sha, _entailment_index(model.config)


@torch.inference_mode()
def _pair_entailment_scores(model, tokenizer, pairs, entailment_index: int):
    device = next(model.parameters()).device
    values: list[float] = []
    for start in range(0, len(pairs), 64):
        batch = pairs[start : start + 64]
        encoded = tokenizer(
            [left for left, _ in batch],
            [right for _, right in batch],
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


def _score_rows(rows, model, tokenizer, entailment_index: int):
    lookup = {
        row.case_id: {factor_id: [0.0, 0.0] for factor_id in FACTOR_IDS}
        for row in rows
    }
    requests = []
    for row in rows:
        for factor_id in FACTOR_IDS:
            for value, hypothesis in enumerate(FACTOR_REFERENCE_HYPOTHESES[factor_id]):
                requests.append(
                    (
                        row.case_id,
                        factor_id,
                        value,
                        row.severity_field,
                        hypothesis,
                    )
                )
    values = _pair_entailment_scores(
        model,
        tokenizer,
        [(query, hypothesis) for _, _, _, query, hypothesis in requests],
        entailment_index,
    )
    for request, score in zip(requests, values):
        case_id, factor_id, value, _, _ = request
        lookup[case_id][factor_id][value] = score
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    values = all_w26_text_atoms()
    overlap = sorted(values & prior_text_atoms())
    if overlap:
        raise RuntimeError(f"W26 exact text atoms overlap prior authorities: {overlap}")
    if all_w26_query_texts() & all_w26_schema_texts():
        raise RuntimeError("W26 query/schema exact sentence overlap")

    rows = generate_w26_partition("qualification")
    if len(rows) != 192 or {row.domain_id for row in rows} != {"EA", "EB"}:
        raise RuntimeError("W26 qualification authority count changed")

    model_scores = {}
    actual_hashes = {}
    entailment_indices = {}

    for name in REFERENCE_NAMES:
        spec = REFERENCE_PANEL[name]
        model, tokenizer, actual_sha, entailment_index = _load_reference(name, spec)
        model_scores[name] = _score_rows(rows, model, tokenizer, entailment_index)
        actual_hashes[name] = actual_sha
        entailment_indices[name] = entailment_index
        del model
        gc.collect()

    row_dicts = [row.__dict__ for row in rows]
    panel = evaluate_reference_panel(row_dicts, model_scores)
    qualification = qualification_result(panel)

    result = {
        "schema_version": "r8-w26-reference-qualification-v1",
        "status": "PASS",
        "qualification_status": qualification["status"],
        "outcome": qualification["outcome"],
        "qualification_per_domain": qualification["per_domain"],
        "panel": panel,
        "reference_panel": REFERENCE_PANEL,
        "actual_reference_weight_sha256": actual_hashes,
        "reference_entailment_label_index": entailment_indices,
        "factor_scoring_direction": "premise=query; hypothesis=factor-option; entailment-probability",
        "panel_option_score": "arithmetic-mean-ce0-ce1-entailment-probability",
        "case_count": 192,
        "domains": ["EA", "EB"],
        "domain_seeds": {"EA": DOMAIN_SEEDS["EA"], "EB": DOMAIN_SEEDS["EB"]},
        "text_atom_sha256": sha256("\n".join(sorted(values)).encode()).hexdigest(),
        "prior_exact_text_overlap": [],
        "query_schema_exact_sentence_overlap": [],
        "a13_materialized": False,
        "hira_materialized": False,
        "hira_train_dev_confirm_materialized": False,
        "reference_outputs_used_as_hira_training_targets": False,
        "qualification_rows_used_for_hira_selection": False,
        "w25_rows_used": False,
        "w24_rows_used": False,
        "w23_rows_used": False,
        "w22_rows_used": False,
        "w21_rows_used": False,
        "w20_rows_used": False,
        "w19_rows_used": False,
        "w18_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "qualification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
