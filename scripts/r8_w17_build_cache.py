from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.field_isolated_authority import (
    all_w17_text_atoms,
    generate_w17_dev,
    generate_w17_train,
)
from nmd.field_isolated_cache import compile_w17_cache, save_w17_cache
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = (
    "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
)
MAX_LENGTH = 256


def _prior_text_atoms() -> set[str]:
    from nmd.anchor_preserving_residual_authority import all_w15_text_atoms
    from nmd.anchor_residual_authority import all_w12_text_atoms
    from nmd.conjunctive_authority import all_w7_values
    from nmd.continuous_reliability_authority import all_w14_text_atoms
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
        set(all_w5f_vocab())
        | set(all_w5g_vocab())
        | set(all_w5h_vocab())
        | set(all_w5i_vocab())
        | set(all_w6b_values())
        | set(all_w6c_values())
        | set(all_w6d_values())
        | set(all_w6e_values())
        | set(all_w6f_values())
        | set(all_w6g_values())
        | set(all_w6h_values())
        | set(all_w6i_values())
        | set(all_w6j_values())
        | set(all_w7_values())
        | set(all_w7b_values())
        | set(all_w8_text_atoms())
        | set(all_w9_text_atoms(include_confirm=True))
        | set(all_w10_text_atoms())
        | set(all_w11_text_atoms())
        | set(all_w12_text_atoms())
        | set(all_w13_text_atoms())
        | set(all_w14_text_atoms())
        | set(all_w15_text_atoms())
        | set(all_w16_text_atoms())
    )


def _hash_strings(values: set[str]) -> str:
    return sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _load_model() -> NolaneHira:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("W17 A13 weight SHA mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=MAX_LENGTH,
    )
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    values = all_w17_text_atoms()
    overlap = sorted(values & _prior_text_atoms())
    if overlap:
        raise RuntimeError(f"W17 exact text atoms overlap prior authorities: {overlap}")

    train = generate_w17_train()
    dev = generate_w17_dev()
    if len(train) != 384 or len(dev) != 96:
        raise RuntimeError("W17 train/dev count changed")

    model = _load_model()
    train_cache = compile_w17_cache(model, train, expected_split_prefix="train")
    dev_cache = compile_w17_cache(model, dev, expected_split_prefix="dev-co")

    args.out.mkdir(parents=True, exist_ok=True)
    train_path = save_w17_cache(train_cache, args.out / "train.pt")
    dev_path = save_w17_cache(dev_cache, args.out / "dev-co.pt")

    receipt = {
        "schema_version": "r8-w17-train-dev-cache-receipt-v1",
        "status": "PASS",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "train_case_count": len(train),
        "dev_case_count": len(dev),
        "train_decision_count": 5 * len(train),
        "dev_decision_count": 5 * len(dev),
        "train_domains": ["CK", "CL", "CM", "CN"],
        "dev_domain": "CO",
        "train_cache_sha256": file_sha256(train_path),
        "dev_cache_sha256": file_sha256(dev_path),
        "all_text_atom_sha256": _hash_strings(values),
        "prior_exact_text_overlap": [],
        "training_performed": False,
        "selection_performed": False,
        "confirm_cp_exposed": False,
        "confirm_cq_exposed": False,
        "logical_state_compiles_per_candidate_case": train_cache["metadata"][
            "logical_state_compiles_per_candidate_case"
        ],
        "a13_invocations_per_candidate_case": train_cache["metadata"][
            "a13_invocations_per_candidate_case"
        ],
        "encoded_sequences_per_candidate_case": train_cache["metadata"][
            "encoded_sequences_per_candidate_case"
        ],
        "w15_rows_used": False,
        "w16_rows_used": False,
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
