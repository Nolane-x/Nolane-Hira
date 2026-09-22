from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.hira import HIRACore, count_parameters
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_decisions import load_pinned_typed_decisions_final
from nmd.typed_eval import evaluate_typed_cases
from nmd.typed_final import (
    A13_MAX_LENGTH,
    A13_MODEL,
    A13_REVISION,
    A13_WEIGHT_SHA256,
    FINAL_CASES,
    FINAL_DECISIONS,
    FINAL_SPLIT,
    STATE_SEGMENT_TOKENS,
    W3A_SELECTED_HEAD_SHA256,
    compare_laya_typed,
    laya_typed_metric_map,
    load_and_validate_marker,
)

EXPECTED_HEAD_PARAMS = 422_159


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--selected-head", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    marker = load_and_validate_marker(args.marker)
    selected_sha = file_sha256(args.selected_head)
    if selected_sha != W3A_SELECTED_HEAD_SHA256:
        raise RuntimeError(
            f"selected head SHA mismatch: {selected_sha}"
        )

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(
            repo_id=A13_MODEL,
            revision=A13_REVISION,
        )
    )
    weight_path = snapshot / "model.safetensors"
    if file_sha256(weight_path) != A13_WEIGHT_SHA256:
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
        max_length=A13_MAX_LENGTH,
    )
    hira = HIRACore(d_model=256, dropout=0.05)
    state = torch.load(
        args.selected_head,
        map_location="cpu",
        weights_only=True,
    )
    hira.load_state_dict(state, strict=True)
    if count_parameters(hira) != EXPECTED_HEAD_PARAMS:
        raise RuntimeError("HIRA head parameter count changed")

    model = NolaneHira(encoder, hira)
    model.eval()

    # This log line is the authoritative boundary: before it, final rows have
    # not been requested; after it, the frozen one-shot test exposure begins.
    print("R8_W3B_FINAL_TEST_EXPOSURE_BEGIN", flush=True)
    cases = load_pinned_typed_decisions_final()
    if len(cases) != FINAL_CASES:
        raise RuntimeError("final case count mismatch after load")

    metrics = evaluate_typed_cases(
        model,
        cases,
        adaptive_budget=False,
    )
    if metrics["case_count"] != FINAL_CASES:
        raise RuntimeError("final evaluator case count mismatch")
    if metrics["decision_count"] != FINAL_DECISIONS:
        raise RuntimeError("final evaluator decision count mismatch")
    if metrics["state_encode_calls"] != FINAL_CASES:
        raise RuntimeError("final state-once contract failed")
    if metrics["state_encode_calls_per_case"] != 1.0:
        raise RuntimeError("final state-once ratio failed")
    if metrics["decisions_per_state_encode"] != 5.0:
        raise RuntimeError("final decisions/state ratio failed")

    mapped = laya_typed_metric_map(metrics)
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    comparison = compare_laya_typed(
        targets,
        mapped,
        tolerance=float(
            targets.get("rules", {}).get("win_tolerance", 1e-9)
        ),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    results = {
        "schema_version": "r8-w3b-final-results-v1",
        "candidate": "hira-r8-w3-specialist",
        "metrics": mapped,
        "raw_metrics": metrics,
    }
    receipt = {
        "schema_version": "r8-w3b-final-receipt-v1",
        "status": "PASS",
        "scientific_authority": "ONE_SHOT_FINAL",
        "marker": marker,
        "selected_head_sha256": selected_sha,
        "head_parameter_count": count_parameters(hira),
        "final_test_exposed": True,
        "dataset_split": FINAL_SPLIT,
        "case_count": metrics["case_count"],
        "decision_count": metrics["decision_count"],
        "state_encode_calls": metrics["state_encode_calls"],
        "state_encode_calls_per_case": metrics[
            "state_encode_calls_per_case"
        ],
        "decisions_per_state_encode": metrics[
            "decisions_per_state_encode"
        ],
        "laya_typed_counts": comparison["counts"],
        "jev_cells_populated": False,
    }
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "typed-comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(receipt, sort_keys=True))
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
