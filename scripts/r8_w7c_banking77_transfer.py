from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.banking77_transfer import (
    CANDIDATES,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    END_INDEX,
    EXAMPLE_COUNT,
    START_INDEX,
    classify_transfer,
    evaluate_banking77_transfer,
    load_banking77_transfer_authority,
)
from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import file_sha256


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
A13_MAX_LENGTH = 256
FREEZE_SCHEMA = "r8-w7b-freeze-v1"


def _validate_freeze(freeze: dict, checkpoints: Path) -> dict[str, dict]:
    if freeze.get("schema_version") != FREEZE_SCHEMA:
        raise RuntimeError("unexpected W7b freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W7b freeze is not PASS")
    if freeze.get("all_candidates_frozen_before_confirm") is not True:
        raise RuntimeError("W7b freeze boundary invalid")
    rows = {row["candidate"]: row for row in freeze.get("candidates", [])}
    if set(rows) != set(CANDIDATES):
        raise RuntimeError("W7c requires all five frozen W7b candidates")
    for name, row in rows.items():
        hira_path = checkpoints / f"{name}-hira.pt"
        scorer_path = checkpoints / f"{name}-scorer.pt"
        if file_sha256(hira_path) != row["hira_sha256"]:
            raise RuntimeError(f"W7c {name} HIRA SHA mismatch")
        if file_sha256(scorer_path) != row["scorer_sha256"]:
            raise RuntimeError(f"W7c {name} scorer SHA mismatch")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    rows = _validate_freeze(freeze, args.checkpoints)

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION)
    )
    weight_path = snapshot / "model.safetensors"
    if file_sha256(weight_path) != A13_WEIGHT_SHA256:
        raise RuntimeError("W7c A13 weight SHA mismatch")

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

    print("R8_W7C_BANKING77_ROWS_400_799_EXPOSURE_BEGIN", flush=True)
    authority = load_banking77_transfer_authority()

    metrics: dict[str, dict[str, object]] = {}
    provenance: dict[str, dict[str, object]] = {}

    for name in CANDIDATES:
        row = rows[name]
        hira_path = args.checkpoints / f"{name}-hira.pt"
        scorer_path = args.checkpoints / f"{name}-scorer.pt"

        hira = HIRACore(d_model=256, dropout=0.0)
        hira.load_state_dict(
            torch.load(hira_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
        scorer.load_state_dict(
            torch.load(scorer_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        model = NolaneHira(
            encoder,
            hira,
            coarse_scorer=scorer,
        )
        model.eval()
        metrics[name] = evaluate_banking77_transfer(model, authority)
        provenance[name] = {
            "selected_epoch": row["selected_epoch"],
            "hira_sha256": row["hira_sha256"],
            "scorer_sha256": row["scorer_sha256"],
            "trainable_parameter_count": row["trainable_parameter_count"],
            "optimization_seed": row["optimization_seed"],
        }

    verdict, verdict_details = classify_transfer(metrics)

    result = {
        "schema_version": "r8-w7c-banking77-transfer-v1",
        "status": "PASS",
        "scientific_scope": "PUBLIC_TRANSFER_DIAGNOSTIC_ONLY",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "slice_start": START_INDEX,
            "slice_end_exclusive": END_INDEX,
            "case_count": EXAMPLE_COUNT,
        },
        "w4a_first400_reused": False,
        "banking77_task_training_used": False,
        "retrieval_used": False,
        "calibration_fit_used": False,
        "checkpoint_selection_on_w7c": False,
        "laya_scorecard_populated": False,
        "jev_scorecard_populated": False,
        "paths": metrics,
        "candidate_provenance": provenance,
        "verdict": verdict,
        "verdict_details": verdict_details,
        "w7b_frozen_verdict_overridden": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
