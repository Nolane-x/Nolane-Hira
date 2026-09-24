from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.second_order_localization import (
    compile_w6g_cache,
    save_w6g_cache,
)
from nmd.second_order_localization_authority import (
    BASES_PER_DOMAIN,
    DOMAIN_Q_SEED,
    DOMAIN_R_SEED,
    DOMAIN_S_SEED,
    ROLE_KEYS,
    VIEW_IDS,
    all_w6g_values,
    domain_role_sets,
    domain_template_sets,
    domain_value_sets,
    generate_w6g_diagnostics,
)
from nmd.typed_competitive_cache import file_sha256


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def list_hash(values) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return sha256(payload).hexdigest()


def prior_values() -> set[str]:
    from nmd.high_k_localization_authority import all_w6f_values
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

    values = all_w6g_values()
    overlap = sorted(values & prior_values())
    if overlap:
        raise RuntimeError(
            f"W6g diagnostic values overlap prior authorities: {overlap}"
        )

    views = generate_w6g_diagnostics()
    cache = compile_w6g_cache(model, views)
    cache_path = save_w6g_cache(
        cache,
        args.out / "diagnostics.pt",
    )

    value_sets = domain_value_sets()
    template_sets = domain_template_sets()
    role_sets = domain_role_sets()

    receipt = {
        "schema_version": "r8-w6g-cache-receipt-v1",
        "status": "PASS",
        "scope": "fresh second-order diagnostic only; no training/no confirm",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "domain_seeds": {
            "Q": DOMAIN_Q_SEED,
            "R": DOMAIN_R_SEED,
            "S": DOMAIN_S_SEED,
        },
        "base_count_per_domain": BASES_PER_DOMAIN,
        "base_count": cache["metadata"]["base_count"],
        "view_count": cache["metadata"]["view_count"],
        "domain_base_counts": cache["metadata"]["domain_base_counts"],
        "view_counts": cache["metadata"]["view_counts"],
        "case_id_sha256": cache["metadata"]["case_id_sha256"],
        "semantic_view_sha256": cache["metadata"]["semantic_view_sha256"],
        "domain_semantic_view_sha256": cache[
            "metadata"
        ]["domain_semantic_view_sha256"],
        "view_ids": list(VIEW_IDS),
        "role_keys": list(ROLE_KEYS),
        "state_encode_calls": cache["metadata"]["state_encode_calls"],
        "state_encodes_per_base": cache[
            "metadata"
        ]["state_encode_calls_per_base"],
        "state_encodes_per_view": cache[
            "metadata"
        ]["state_encode_calls_per_view"],
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
        "cache_sha256": file_sha256(cache_path),
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
        "training_performed": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
