from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import (
    compile_w6b_cache,
    file_sha256,
    save_w6b_cache,
)
from nmd.typed_joint_replication_authority import (
    CONFIRM_L_SEED,
    CONFIRM_M_SEED,
    DEV_K_SEED,
    DOMAINS,
    SOURCE_G_SEED,
    SOURCE_H_SEED,
    SOURCE_I_SEED,
    SOURCE_J_SEED,
    all_w6e_values,
    domain_value_sets,
    generate_w6e_dev,
    generate_w6e_multi_train,
)


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def list_hash(values) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return sha256(payload).hexdigest()


def prior_values() -> set[str]:
    from nmd.typed_competitive_authority import all_w6b_values
    from nmd.typed_reliability_authority import all_w6c_values
    from nmd.typed_domain_generalization_authority import all_w6d_values
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_cross_candidate_binding import all_w5i_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab

    return (
        set(all_w6b_values())
        | set(all_w6c_values())
        | set(all_w6d_values())
        | set(all_w5f_vocab())
        | set(all_w5g_vocab())
        | set(all_w5h_vocab())
        | set(all_w5i_vocab())
    )


def gold_semantic_hash(cases) -> str:
    values = []
    for case in cases:
        diagnosis = case.typed.decisions[0]
        values.append(
            diagnosis.options[diagnosis.gold_index].criterion_text
        )
    return list_hash(values)


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

    values = all_w6e_values()
    overlap = sorted(values & prior_values())
    if overlap:
        raise RuntimeError(
            f"W6e value vocabulary overlaps prior authority values: {overlap}"
        )

    train_cases = generate_w6e_multi_train()
    dev_cases = generate_w6e_dev()

    train_cache = compile_w6b_cache(
        model,
        train_cases,
        expected_split="train-multi",
    )
    dev_cache = compile_w6b_cache(
        model,
        dev_cases,
        expected_split="dev-k",
    )

    train_path = save_w6b_cache(
        train_cache,
        args.out / "train-multi.pt",
    )
    dev_path = save_w6b_cache(
        dev_cache,
        args.out / "dev-k.pt",
    )

    value_sets = domain_value_sets()
    template_ids = {
        domain_id: list(spec.templates)
        for domain_id, spec in DOMAINS.items()
    }
    role_values = {
        domain_id: list(spec.roles)
        for domain_id, spec in DOMAINS.items()
    }
    train_domains = {
        domain: sum(case.domain_id == domain for case in train_cases)
        for domain in ("G", "H", "I", "J")
    }

    receipt = {
        "schema_version": "r8-w6e-cache-receipt-v1",
        "status": "PASS",
        "scope": (
            "W6e source G-J + held-out DEV-K only; "
            "CONFIRM-L/M sealed"
        ),
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "source_seeds": {
            "G": SOURCE_G_SEED,
            "H": SOURCE_H_SEED,
            "I": SOURCE_I_SEED,
            "J": SOURCE_J_SEED,
        },
        "dev_k_seed": DEV_K_SEED,
        "reserved_confirm_l_seed": CONFIRM_L_SEED,
        "reserved_confirm_m_seed": CONFIRM_M_SEED,
        "train_case_count": len(train_cases),
        "dev_case_count": len(dev_cases),
        "train_decision_count": 5 * len(train_cases),
        "dev_decision_count": 5 * len(dev_cases),
        "train_domain_counts": train_domains,
        "train_gold_semantic_sha256": gold_semantic_hash(train_cases),
        "dev_gold_semantic_sha256": gold_semantic_hash(dev_cases),
        "domain_value_sha256": {
            domain: list_hash(rows)
            for domain, rows in value_sets.items()
        },
        "all_value_sha256": list_hash(values),
        "all_value_count": len(values),
        "domain_template_ids": template_ids,
        "template_sha256": list_hash(
            template
            for rows in template_ids.values()
            for template in rows
        ),
        "domain_roles": role_values,
        "role_sha256": list_hash(
            role
            for rows in role_values.values()
            for role in rows
        ),
        "prior_authority_value_overlap": [],
        "state_encodes_per_case_train": train_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "state_encodes_per_case_dev": dev_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "train_cache_sha256": file_sha256(train_path),
        "dev_cache_sha256": file_sha256(dev_path),
        "confirm_l_exposed": False,
        "confirm_m_exposed": False,
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
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
