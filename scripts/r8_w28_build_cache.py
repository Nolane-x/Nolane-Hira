from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.compositional_projection_authority import (
    DOMAIN_SEEDS,
    PARTITION_DOMAINS,
    all_w28_query_texts,
    all_w28_schema_texts,
    all_w28_text_atoms,
    generate_w28_partition,
)
from nmd.compositional_projection_cache import compile_w28_cache, save_w28_cache
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def _validate_qualification(path: Path) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("schema_version") != "r8-w28-reference-qualification-v1":
        raise RuntimeError("W28 qualification schema mismatch")
    if row.get("status") != "PASS":
        raise RuntimeError("W28 qualification execution not PASS")
    if row.get("qualification_status") != "PASS":
        raise RuntimeError("W28 reference qualification failed")
    if row.get("outcome") != "W28_REFERENCE_QUALIFIED":
        raise RuntimeError("W28 qualification outcome not authorized")
    if row.get("a13_materialized") is not False:
        raise RuntimeError("W28 qualification unexpectedly materialized A13")
    if row.get("hira_train_dev_confirm_materialized") is not False:
        raise RuntimeError("W28 qualification unexpectedly materialized HIRA data")
    if row.get("prior_exact_text_overlap") != []:
        raise RuntimeError("W28 qualification freshness failed")
    if row.get("query_schema_exact_sentence_overlap") != []:
        raise RuntimeError("W28 qualification query/schema overlap")
    expected_hash = sha256(
        "\n".join(sorted(all_w28_text_atoms())).encode()
    ).hexdigest()
    if row.get("text_atom_sha256") != expected_hash:
        raise RuntimeError("W28 qualification text universe hash mismatch")
    return row


def _validate_freeze_receipts(paths: list[Path]) -> dict[str, dict]:
    receipts = {}
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != "r8-w28-candidate-receipt-v1":
            raise RuntimeError("W28 candidate receipt schema mismatch")
        if row.get("status") != "PASS":
            raise RuntimeError("W28 candidate freeze receipt not PASS")
        candidate = str(row.get("candidate", ""))
        if candidate in receipts:
            raise RuntimeError("W28 duplicate candidate receipt")
        if row.get("confirm_exposed") is not False:
            raise RuntimeError("W28 candidate was not frozen pre-CONFIRM")
        receipts[candidate] = row
    if set(receipts) != {"T0", "T1"}:
        raise RuntimeError("W28 CONFIRM requires frozen T0/T1 receipts")
    return receipts


def _load_model() -> NolaneHira:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("W28 A13 weight SHA mismatch")
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
    parser.add_argument("--partition", choices=("train","dev","confirm"), required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, action="append", default=[])
    args = parser.parse_args()

    qualification = _validate_qualification(args.qualification)
    if all_w28_query_texts() & all_w28_schema_texts():
        raise RuntimeError("W28 query/schema exact sentence overlap")

    freeze_receipts = {}
    if args.partition == "confirm":
        freeze_receipts = _validate_freeze_receipts(args.freeze_receipt)
    elif args.freeze_receipt:
        raise RuntimeError("W28 freeze receipts only allowed for CONFIRM")

    rows = generate_w28_partition(args.partition)
    expected_count = {"train":384,"dev":96,"confirm":192}[args.partition]
    if len(rows) != expected_count:
        raise RuntimeError("W28 partition count changed")

    model = _load_model()
    cache = compile_w28_cache(model, rows, partition=args.partition)

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = save_w28_cache(cache, args.out / f"w28-{args.partition}.pt")
    receipt = {
        "schema_version": "r8-w28-cache-receipt-v1",
        "status": "PASS",
        "partition": args.partition,
        "qualification_outcome": qualification["outcome"],
        "qualification_text_atom_sha256": qualification["text_atom_sha256"],
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "case_count": expected_count,
        "domains": list(PARTITION_DOMAINS[args.partition]),
        "domain_seeds": {d:DOMAIN_SEEDS[d] for d in PARTITION_DOMAINS[args.partition]},
        "cache_sha256": file_sha256(cache_path),
        "factor_ids": ["F0","F1","F2"],
        "factor_decoder": {"000":0,"100":1,"110":2,"111":3},
        "primary_reference_f2": "U_AND_C",
        "training_performed": False,
        "selection_performed": False,
        "trainable_parameter_count": 0,
        "logical_state_compiles_per_case": 1,
        "a13_query_invocations_per_case": 1,
        "encoded_query_sequences_per_case": 1,
        "confirm_materialized_after_both_candidate_freezes": args.partition == "confirm",
        "candidate_freeze_receipts": (
            {name:row["checkpoint_sha256"] for name,row in freeze_receipts.items()}
            if args.partition == "confirm" else {}
        ),
        "reference_outputs_used_as_training_targets": False,
        "qualification_rows_used_for_hira_selection": False,
        "w27_rows_used": False,
        "w26_rows_used": False,
        "w25_rows_used": False,
        "w24_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out/"receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt,sort_keys=True))


if __name__ == "__main__":
    main()
