from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_alignment_authority import (
    BASES_PER_DOMAIN,
    CONFIRM_DOMAINS,
    DEV_DOMAINS,
    DOMAIN_SEEDS,
    TRAIN_DOMAINS,
    all_w9_text_atoms,
    generate_w9_split,
)
from nmd.semantic_alignment_cache import (
    compile_w9_cache,
    save_w9_cache,
)
from nmd.typed_competitive_cache import file_sha256


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def _hash(values) -> str:
    return sha256(
        "\n".join(sorted(map(str, values))).encode("utf-8")
    ).hexdigest()


def prior_text_atoms() -> set[str]:
    from nmd.conjunctive_authority import all_w7_values
    from nmd.freeform_attribution_authority import all_w7b_values
    from nmd.high_cardinality_decomposition_authority import all_w6j_values
    from nmd.semantic_transfer_authority import all_w8_text_atoms

    return (
        set(all_w7_values())
        | set(all_w7b_values())
        | set(all_w6j_values())
        | set(all_w8_text_atoms())
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
        raise RuntimeError("W9 A13 weight SHA-256 mismatch")

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

    current_atoms = all_w9_text_atoms(include_confirm=False)
    overlap = sorted(current_atoms & prior_text_atoms())
    if overlap:
        raise RuntimeError(
            f"W9 train/dev exact text atoms overlap prior evidence: {overlap}"
        )

    train_rows = generate_w9_split("train")
    dev_rows = generate_w9_split("dev")

    train_cache = compile_w9_cache(model, train_rows)
    dev_cache = compile_w9_cache(model, dev_rows)

    train_path = save_w9_cache(
        train_cache,
        args.out / "train.pt",
    )
    dev_path = save_w9_cache(
        dev_cache,
        args.out / "dev-bc.pt",
    )

    receipt = {
        "schema_version": "r8-w9-cache-receipt-v1",
        "status": "PASS",
        "scope": "W9 AY-BB TRAIN + DEV-BC only; CONFIRM-BD/BE sealed",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "train_domains": list(TRAIN_DOMAINS),
        "dev_domains": list(DEV_DOMAINS),
        "reserved_confirm_domains": list(CONFIRM_DOMAINS),
        "domain_seeds": dict(DOMAIN_SEEDS),
        "train_base_count": train_cache["metadata"]["base_count"],
        "dev_base_count": dev_cache["metadata"]["base_count"],
        "train_view_count": train_cache["metadata"]["view_count"],
        "dev_view_count": dev_cache["metadata"]["view_count"],
        "expected_train_base_count": len(TRAIN_DOMAINS) * BASES_PER_DOMAIN,
        "expected_dev_base_count": len(DEV_DOMAINS) * BASES_PER_DOMAIN,
        "state_encodes_per_base_train": train_cache["metadata"][
            "state_encode_calls_per_base"
        ],
        "state_encodes_per_base_dev": dev_cache["metadata"][
            "state_encode_calls_per_base"
        ],
        "train_base_id_sha256": train_cache["metadata"]["base_id_sha256"],
        "dev_base_id_sha256": dev_cache["metadata"]["base_id_sha256"],
        "train_case_id_sha256": train_cache["metadata"]["case_id_sha256"],
        "dev_case_id_sha256": dev_cache["metadata"]["case_id_sha256"],
        "train_cache_sha256": file_sha256(train_path),
        "dev_cache_sha256": file_sha256(dev_path),
        "w9_train_dev_text_atom_count": len(current_atoms),
        "w9_train_dev_text_atom_sha256": _hash(current_atoms),
        "prior_exact_text_overlap": [],
        "confirm_bd_exposed": False,
        "confirm_be_exposed": False,
        "w7_confirm_rows_used": False,
        "w7b_confirm_rows_used": False,
        "w8_diagnostic_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }

    if receipt["train_base_count"] != receipt["expected_train_base_count"]:
        raise RuntimeError("W9 TRAIN base count changed")
    if receipt["dev_base_count"] != receipt["expected_dev_base_count"]:
        raise RuntimeError("W9 DEV base count changed")
    if receipt["state_encodes_per_base_train"] != 1.0:
        raise RuntimeError("W9 TRAIN state-once contract failed")
    if receipt["state_encodes_per_base_dev"] != 1.0:
        raise RuntimeError("W9 DEV state-once contract failed")

    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
