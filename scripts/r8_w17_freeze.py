from __future__ import annotations

import argparse
import json
from pathlib import Path

from nmd.typed_competitive_cache import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-receipt", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.cache_receipt.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w17-train-dev-cache-receipt-v1":
        raise RuntimeError("unexpected W17 cache receipt")
    if receipt.get("status") != "PASS":
        raise RuntimeError("W17 cache receipt is not PASS")
    if receipt.get("training_performed") is not False:
        raise RuntimeError("W17 must not train")
    if receipt.get("selection_performed") is not False:
        raise RuntimeError("W17 must not select on DEV")
    if receipt.get("confirm_cp_exposed") is not False:
        raise RuntimeError("W17 CP exposed before freeze")
    if receipt.get("confirm_cq_exposed") is not False:
        raise RuntimeError("W17 CQ exposed before freeze")
    if file_sha256(args.train_cache) != receipt.get("train_cache_sha256"):
        raise RuntimeError("W17 train cache SHA mismatch")
    if file_sha256(args.dev_cache) != receipt.get("dev_cache_sha256"):
        raise RuntimeError("W17 dev cache SHA mismatch")

    expected_logical = {
        "FULL_SINGLE": 1,
        "FULL_TRIPLICATE": 1,
        "FIELD_ISOLATED": 1,
    }
    expected_invocations = dict(expected_logical)
    expected_sequences = {
        "FULL_SINGLE": 1,
        "FULL_TRIPLICATE": 3,
        "FIELD_ISOLATED": 3,
    }
    if receipt.get("logical_state_compiles_per_candidate_case") != expected_logical:
        raise RuntimeError("W17 logical compile contract changed")
    if receipt.get("a13_invocations_per_candidate_case") != expected_invocations:
        raise RuntimeError("W17 A13 invocation contract changed")
    if receipt.get("encoded_sequences_per_candidate_case") != expected_sequences:
        raise RuntimeError("W17 sequence accounting changed")

    freeze = {
        "schema_version": "r8-w17-preconfirm-freeze-v1",
        "status": "PASS",
        "mechanism": "FIELD_ISOLATED_ZERO_PARAMETER",
        "training_performed": False,
        "selection_performed": False,
        "all_scientific_gates_frozen_before_confirm": True,
        "confirm_cp_exposed": False,
        "confirm_cq_exposed": False,
        "train_cache_sha256": receipt["train_cache_sha256"],
        "dev_cache_sha256": receipt["dev_cache_sha256"],
        "a13_revision": receipt["a13_revision"],
        "a13_weight_sha256": receipt["a13_weight_sha256"],
        "logical_state_compiles_per_candidate_case": expected_logical,
        "a13_invocations_per_candidate_case": expected_invocations,
        "encoded_sequences_per_candidate_case": expected_sequences,
        "trainable_parameter_count": 0,
        "w15_rows_used": False,
        "w16_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(freeze, sort_keys=True))


if __name__ == "__main__":
    main()
