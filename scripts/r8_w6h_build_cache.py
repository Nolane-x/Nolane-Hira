from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.field_semantic_rescue import attach_one_field_pair_labels
from nmd.field_semantic_rescue_authority import (
    CONFIRM_Y_SEED,
    CONFIRM_Z_SEED,
    DEV_X_SEED,
    DOMAINS,
    SOURCE_T_SEED,
    SOURCE_U_SEED,
    SOURCE_V_SEED,
    SOURCE_W_SEED,
    all_w6h_values,
    domain_value_sets,
    generate_w6h_dev,
    generate_w6h_train,
)
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import (
    compile_w6b_cache,
    file_sha256,
    save_w6b_cache,
)

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def list_hash(values) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return sha256(payload).hexdigest()


def prior_values() -> set[str]:
    from nmd.high_k_localization_authority import all_w6f_values
    from nmd.second_order_localization_authority import all_w6g_values
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_cross_candidate_binding import all_w5i_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

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
    model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    model.eval()

    values = all_w6h_values()
    overlap = sorted(values & prior_values())
    if overlap:
        raise RuntimeError(f"W6h value vocabulary overlaps prior values: {overlap}")

    train_cases = generate_w6h_train()
    dev_cases = generate_w6h_dev()

    train_cache = compile_w6b_cache(model, train_cases, expected_split="train")
    dev_cache = compile_w6b_cache(model, dev_cases, expected_split="dev-x")
    attach_one_field_pair_labels(train_cache, train_cases)
    attach_one_field_pair_labels(dev_cache, dev_cases)

    train_path = save_w6b_cache(train_cache, args.out / "train.pt")
    dev_path = save_w6b_cache(dev_cache, args.out / "dev-x.pt")

    value_sets = domain_value_sets()
    receipt = {
        "schema_version": "r8-w6h-cache-receipt-v1",
        "status": "PASS",
        "scope": "W6h T-W TRAIN + DEV-X only; CONFIRM-Y/Z sealed",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "source_seeds": {
            "T": SOURCE_T_SEED,
            "U": SOURCE_U_SEED,
            "V": SOURCE_V_SEED,
            "W": SOURCE_W_SEED,
        },
        "dev_x_seed": DEV_X_SEED,
        "reserved_confirm_y_seed": CONFIRM_Y_SEED,
        "reserved_confirm_z_seed": CONFIRM_Z_SEED,
        "train_case_count": len(train_cases),
        "dev_case_count": len(dev_cases),
        "train_decision_count": 5 * len(train_cases),
        "dev_decision_count": 5 * len(dev_cases),
        "train_domain_counts": {
            domain: sum(case.domain_id == domain for case in train_cases)
            for domain in ("T", "U", "V", "W")
        },
        "domain_value_sha256": {
            domain: list_hash(rows)
            for domain, rows in value_sets.items()
        },
        "all_value_sha256": list_hash(values),
        "domain_template_ids": {
            domain: list(spec.templates)
            for domain, spec in DOMAINS.items()
        },
        "domain_roles": {
            domain: list(spec.roles)
            for domain, spec in DOMAINS.items()
        },
        "prior_authority_value_overlap": [],
        "state_encodes_per_case_train": train_cache["metadata"]["state_encode_calls_per_case"],
        "state_encodes_per_case_dev": dev_cache["metadata"]["state_encode_calls_per_case"],
        "train_cache_sha256": file_sha256(train_path),
        "dev_cache_sha256": file_sha256(dev_path),
        "confirm_y_exposed": False,
        "confirm_z_exposed": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
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
