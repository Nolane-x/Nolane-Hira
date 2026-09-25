from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.freeform_attribution_authority import (
    CONFIRM_AS_SEED,
    CONFIRM_AT_SEED,
    DEV_AR_SEED,
    DOMAINS,
    SOURCE_AN_SEED,
    SOURCE_AO_SEED,
    SOURCE_AP_SEED,
    SOURCE_AQ_SEED,
    all_w7b_values,
    domain_role_sets,
    domain_template_sets,
    domain_value_sets,
    generate_w7b_dev,
    generate_w7b_train,
)
from nmd.conjunctive_cache import (
    compile_w7_cache,
    save_w7_cache,
)
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import file_sha256


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def list_hash(values) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return sha256(payload).hexdigest()


def prior_values() -> set[str]:
    from nmd.field_semantic_rescue_authority import all_w6h_values
    from nmd.high_cardinality_decomposition_authority import all_w6j_values
    from nmd.high_k_localization_authority import all_w6f_values
    from nmd.representation_bridge_authority import all_w6i_values
    from nmd.second_order_localization_authority import all_w6g_values
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_cross_candidate_binding import all_w5i_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab
    from nmd.typed_competitive_authority import all_w6b_values
    from nmd.typed_domain_generalization_authority import all_w6d_values
    from nmd.typed_joint_replication_authority import all_w6e_values
    from nmd.typed_reliability_authority import all_w6c_values
    from nmd.conjunctive_authority import all_w7_values

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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=A13_MODEL,
            revision=A13_REVISION,
        )
    )
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=MAX_LENGTH,
    )
    model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    model.eval()

    values = all_w7b_values()
    overlap = sorted(values & prior_values())
    if overlap:
        raise RuntimeError(
            f"W7b values overlap prior authority/diagnostic values: {overlap}"
        )

    train_cases = generate_w7b_train()
    dev_cases = generate_w7b_dev()

    train_cache = compile_w7_cache(
        model,
        train_cases,
        expected_split="train",
    )
    dev_cache = compile_w7_cache(
        model,
        dev_cases,
        expected_split="dev-ar",
    )

    train_path = save_w7_cache(
        train_cache,
        args.out / "train.pt",
    )
    dev_path = save_w7_cache(
        dev_cache,
        args.out / "dev-ar.pt",
    )

    value_sets = domain_value_sets()
    template_sets = domain_template_sets()
    role_sets = domain_role_sets()

    receipt = {
        "schema_version": "r8-w7b-cache-receipt-v1",
        "status": "PASS",
        "scope": "W7b AG-AJ TRAIN + DEV-AK only; CONFIRM-AL/AM sealed",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "source_seeds": {
            "AN": SOURCE_AN_SEED,
            "AO": SOURCE_AO_SEED,
            "AP": SOURCE_AP_SEED,
            "AQ": SOURCE_AQ_SEED,
        },
        "dev_ar_seed": DEV_AR_SEED,
        "reserved_confirm_as_seed": CONFIRM_AS_SEED,
        "reserved_confirm_at_seed": CONFIRM_AT_SEED,
        "train_case_count": len(train_cases),
        "dev_case_count": len(dev_cases),
        "train_decision_count": 5 * len(train_cases),
        "dev_decision_count": 5 * len(dev_cases),
        "train_domain_counts": {
            domain: sum(case.domain_id == domain for case in train_cases)
            for domain in ("AN", "AO", "AP", "AQ")
        },
        "dev_domain_counts": {
            "AR": sum(case.domain_id == "AR" for case in dev_cases),
        },
        "train_factor_identity_sha256": train_cache["metadata"][
            "factor_identity_sha256"
        ],
        "dev_factor_identity_sha256": dev_cache["metadata"][
            "factor_identity_sha256"
        ],
        "train_semantic_signature_sha256": train_cache["metadata"][
            "semantic_signature_sha256"
        ],
        "dev_semantic_signature_sha256": dev_cache["metadata"][
            "semantic_signature_sha256"
        ],
        "train_factor_encoder_batches": train_cache["metadata"][
            "factor_encoder_batches"
        ],
        "dev_factor_encoder_batches": dev_cache["metadata"][
            "factor_encoder_batches"
        ],
        "train_factor_text_count": train_cache["metadata"][
            "factor_text_count"
        ],
        "dev_factor_text_count": dev_cache["metadata"][
            "factor_text_count"
        ],
        "state_encodes_per_case_train": train_cache["metadata"][
            "state_encode_calls_per_case"
        ],
        "state_encodes_per_case_dev": dev_cache["metadata"][
            "state_encode_calls_per_case"
        ],
        "domain_value_sha256": {
            domain: list_hash(rows)
            for domain, rows in value_sets.items()
        },
        "domain_template_sha256": {
            domain: list_hash(rows)
            for domain, rows in template_sets.items()
        },
        "domain_role_sha256": {
            domain: list_hash(rows)
            for domain, rows in role_sets.items()
        },
        "all_value_count": len(values),
        "all_value_sha256": list_hash(values),
        "prior_authority_value_overlap": [],
        "train_cache_sha256": file_sha256(train_path),
        "dev_cache_sha256": file_sha256(dev_path),
        "confirm_as_exposed": False,
        "confirm_at_exposed": False,
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
        "w6h_confirm_rows_used": False,
        "w6i_diagnostic_rows_used": False,
        "w6j_diagnostic_rows_used": False,
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
