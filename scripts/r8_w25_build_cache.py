from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.atomic_geometry_authority import (
    DOMAIN_SEEDS,
    PARTITION_DOMAINS,
    all_w25_query_texts,
    all_w25_schema_texts,
    all_w25_text_atoms,
    generate_w25_partition,
)
from nmd.atomic_geometry_cache import compile_w25_cache, save_w25_cache
from nmd.atomic_severity_authority import all_w24_text_atoms
from nmd.cross_encoder_authority import all_w23_text_atoms
from nmd.hira import HIRACore
from nmd.latent_composition_authority import all_w18_text_atoms
from nmd.latent_ordinal_authority import all_w19_text_atoms
from nmd.pairwise_latent_authority import all_w20_text_atoms
from nmd.prototype_latent_authority import all_w21_text_atoms
from nmd.reference_panel_authority import all_w22_text_atoms
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def prior_text_atoms() -> set[str]:
    from nmd.anchor_preserving_residual_authority import all_w15_text_atoms
    from nmd.anchor_residual_authority import all_w12_text_atoms
    from nmd.conjunctive_authority import all_w7_values
    from nmd.continuous_reliability_authority import all_w14_text_atoms
    from nmd.field_isolated_authority import all_w17_text_atoms
    from nmd.field_semantic_rescue_authority import all_w6h_values
    from nmd.freeform_attribution_authority import all_w7b_values
    from nmd.high_cardinality_decomposition_authority import all_w6j_values
    from nmd.high_k_localization_authority import all_w6f_values
    from nmd.interface_decomposition_authority import all_w11_text_atoms
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
        | set(all_w24_text_atoms())
    )


def _load_model() -> NolaneHira:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("W25 A13 weight SHA mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(
        base, tokenizer, revision=A13_REVISION, max_length=MAX_LENGTH
    )
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    return model


def _validate_freeze_receipts(paths: list[Path]) -> dict[str, dict]:
    receipts = {}
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != "r8-w25-candidate-receipt-v1":
            raise RuntimeError("W25 unexpected candidate receipt schema")
        if row.get("status") != "PASS":
            raise RuntimeError("W25 candidate freeze receipt not PASS")
        candidate = str(row.get("candidate", ""))
        if candidate in receipts:
            raise RuntimeError("W25 duplicate candidate freeze receipt")
        if row.get("confirm_exposed") is not False:
            raise RuntimeError("W25 candidate receipt already marks CONFIRM exposed")
        receipts[candidate] = row
    if set(receipts) != {"Q0", "Q1", "T0", "T1"}:
        raise RuntimeError("W25 CONFIRM requires frozen Q0/Q1/T0/T1 receipts")
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("train", "dev", "confirm"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, action="append", default=[])
    args = parser.parse_args()

    values = all_w25_text_atoms()
    overlap = sorted(values & prior_text_atoms())
    if overlap:
        raise RuntimeError(f"W25 exact text atoms overlap prior authorities: {overlap}")
    query_schema_overlap = sorted(all_w25_query_texts() & all_w25_schema_texts())
    if query_schema_overlap:
        raise RuntimeError(f"W25 query/schema exact sentence overlap: {query_schema_overlap}")

    freeze_receipts = {}
    if args.partition == "confirm":
        freeze_receipts = _validate_freeze_receipts(args.freeze_receipt)
    elif args.freeze_receipt:
        raise RuntimeError("W25 freeze receipts are only valid for sealed CONFIRM materialization")

    rows = generate_w25_partition(args.partition)
    expected_count = {"train": 384, "dev": 96, "confirm": 192}[args.partition]
    if len(rows) != expected_count:
        raise RuntimeError("W25 authority partition count changed")

    model = _load_model()
    cache = compile_w25_cache(model, rows, partition=args.partition)

    args.out.mkdir(parents=True, exist_ok=True)
    path = save_w25_cache(cache, args.out / f"w25-{args.partition}.pt")
    receipt = {
        "schema_version": "r8-w25-cache-receipt-v1",
        "status": "PASS",
        "partition": args.partition,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "case_count": expected_count,
        "domains": list(PARTITION_DOMAINS[args.partition]),
        "domain_seeds": {d: DOMAIN_SEEDS[d] for d in PARTITION_DOMAINS[args.partition]},
        "factor_ids": ["F0", "F1", "F2"],
        "factor_decoder": {"000": 0, "100": 1, "110": 2, "111": 3},
        "cache_sha256": file_sha256(path),
        "text_atom_sha256": sha256("\n".join(sorted(values)).encode()).hexdigest(),
        "prior_exact_text_overlap": [],
        "query_schema_exact_sentence_overlap": [],
        "training_performed": False,
        "selection_performed": False,
        "trainable_parameter_count": 0,
        "logical_state_compiles_per_case": 1,
        "a13_query_invocations_per_case": 1,
        "encoded_query_sequences_per_case": 1,
        "confirm_materialized_after_all_candidate_freezes": args.partition == "confirm",
        "candidate_freeze_receipts": (
            {name: row["checkpoint_sha256"] for name, row in freeze_receipts.items()}
            if args.partition == "confirm" else {}
        ),
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
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
