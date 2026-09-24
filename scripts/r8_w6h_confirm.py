from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.field_semantic_rescue import (
    CANDIDATES,
    SemanticAdaptedCompetitiveScorer,
    SemanticResidualAdapter,
    attach_one_field_pair_labels,
    evaluate_w6h,
    rescue_verdict,
)
from nmd.field_semantic_rescue_authority import (
    CONFIRM_Y_SEED,
    CONFIRM_Z_SEED,
    generate_w6h_confirm,
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


def _load_candidate(name: str, row: dict, checkpoints: Path):
    hira_path = checkpoints / f"{name}-hira.pt"
    scorer_path = checkpoints / f"{name}-scorer.pt"
    if file_sha256(hira_path) != row["hira_sha256"]:
        raise RuntimeError(f"W6h {name} HIRA SHA mismatch")
    if file_sha256(scorer_path) != row["scorer_sha256"]:
        raise RuntimeError(f"W6h {name} scorer SHA mismatch")

    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(hira_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    if name == "semantic-residual-adapter":
        scorer = SemanticAdaptedCompetitiveScorer(
            CompetitiveCoarseScorer(d_model=256, d_rel=128),
            SemanticResidualAdapter(),
        )
    else:
        scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.load_state_dict(
        torch.load(scorer_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    return hira, scorer


def _evaluate_paths(freeze: dict, checkpoints: Path, cache: dict):
    by_name = {row["candidate"]: row for row in freeze["candidates"]}
    metrics = {}
    for name in CANDIDATES:
        hira, scorer = _load_candidate(name, by_name[name], checkpoints)
        metrics[name] = evaluate_w6h(hira, scorer, cache["cases"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w6h-freeze-v1":
        raise RuntimeError("unexpected W6h freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W6h freeze is not PASS")
    if freeze.get("all_candidates_frozen_before_confirm") is not True:
        raise RuntimeError("W6h candidates were not frozen before CONFIRM")
    if freeze.get("confirm_y_exposed") is not False or freeze.get("confirm_z_exposed") is not False:
        raise RuntimeError("W6h freeze already exposed CONFIRM")
    if {row["candidate"] for row in freeze["candidates"]} != set(CANDIDATES):
        raise RuntimeError("W6h frozen candidate set changed")

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA mismatch")

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
    cache_model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    cache_model.eval()

    print("R8_W6H_CONFIRM_Y_GENERATION_BEGIN", flush=True)
    y_cases = generate_w6h_confirm("Y", allow_confirm=True)
    y_cache = compile_w6b_cache(
        cache_model,
        y_cases,
        expected_split="confirm-y",
    )
    attach_one_field_pair_labels(y_cache, y_cases)
    metrics_y = _evaluate_paths(freeze, args.checkpoints, y_cache)

    print("R8_W6H_CONFIRM_Z_GENERATION_BEGIN", flush=True)
    z_cases = generate_w6h_confirm("Z", allow_confirm=True)
    z_cache = compile_w6b_cache(
        cache_model,
        z_cases,
        expected_split="confirm-z",
    )
    attach_one_field_pair_labels(z_cache, z_cases)
    metrics_z = _evaluate_paths(freeze, args.checkpoints, z_cache)

    verdict, details = rescue_verdict(
        confirm_y=metrics_y,
        confirm_z=metrics_z,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    y_path = save_w6b_cache(y_cache, args.out / "confirm-y-cache.pt")
    z_path = save_w6b_cache(z_cache, args.out / "confirm-z-cache.pt")

    receipt = {
        "schema_version": "r8-w6h-confirm-v1",
        "status": "PASS",
        "confirm_generated_after_all_dev_freezes": True,
        "confirm_y_seed": CONFIRM_Y_SEED,
        "confirm_z_seed": CONFIRM_Z_SEED,
        "confirm_y_case_count": len(y_cases),
        "confirm_z_case_count": len(z_cases),
        "confirm_y_decision_count": 5 * len(y_cases),
        "confirm_z_decision_count": 5 * len(z_cases),
        "confirm_y_case_id_sha256": y_cache["metadata"]["case_id_sha256"],
        "confirm_z_case_id_sha256": z_cache["metadata"]["case_id_sha256"],
        "confirm_y_cache_sha256": file_sha256(y_path),
        "confirm_z_cache_sha256": file_sha256(z_path),
        "state_encodes_per_case_y": y_cache["metadata"]["state_encode_calls_per_case"],
        "state_encodes_per_case_z": z_cache["metadata"]["state_encode_calls_per_case"],
        "paths": {
            "Y": metrics_y,
            "Z": metrics_z,
        },
        "verdict_details": details,
        "verdict": verdict,
        "candidate_dev_freezes": freeze["candidates"],
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "confirm.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
