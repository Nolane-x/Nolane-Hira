from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_transfer_authority import (
    all_w8_text_atoms,
    generate_w8_diagnostics,
)
from nmd.semantic_transfer_cache import compile_w8_cache, save_w8_cache
from nmd.typed_competitive_cache import file_sha256


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def _prior_values() -> set[str]:
    from nmd.conjunctive_authority import all_w7_values
    from nmd.field_semantic_rescue_authority import all_w6h_values
    from nmd.freeform_attribution_authority import all_w7b_values
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
    )


def _hash_strings(values: set[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    values = all_w8_text_atoms()
    overlap = sorted(values & _prior_values())
    if overlap:
        raise RuntimeError(
            f"W8 exact text atoms overlap prior authorities: {overlap}"
        )

    rows = generate_w8_diagnostics()
    if len(rows) != 3072:
        raise RuntimeError("W8 requires exactly 3072 paired views")
    if len({row.base_id for row in rows}) != 256:
        raise RuntimeError("W8 requires exactly 256 base states")

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION)
    )
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("W8 A13 weight SHA mismatch")

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

    cache = compile_w8_cache(model, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = save_w8_cache(cache, args.out / "diagnostic.pt")

    domain_base_counts = {
        domain: len(
            {
                row.base_id
                for row in rows
                if row.domain_id == domain
            }
        )
        for domain in ("AU", "AV", "AW", "AX")
    }
    receipt = {
        "schema_version": "r8-w8-cache-receipt-v1",
        "status": "PASS",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "max_length": MAX_LENGTH,
        "base_count": cache["metadata"]["base_count"],
        "view_count": cache["metadata"]["view_count"],
        "state_encode_calls": cache["metadata"]["state_encode_calls"],
        "state_encodes_per_base": cache["metadata"][
            "state_encode_calls_per_base"
        ],
        "domain_base_counts": domain_base_counts,
        "base_id_sha256": cache["metadata"]["base_id_sha256"],
        "case_id_sha256": cache["metadata"]["case_id_sha256"],
        "all_text_atom_sha256": _hash_strings(values),
        "prior_exact_text_overlap": [],
        "cache_sha256": file_sha256(cache_path),
        "training_performed": False,
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
        "w6h_confirm_rows_used": False,
        "w6i_diagnostic_rows_used": False,
        "w6j_diagnostic_rows_used": False,
        "w7_confirm_rows_used": False,
        "w7b_confirm_rows_used": False,
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
