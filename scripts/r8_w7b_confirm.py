from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.freeform_attribution_authority import (
    CONFIRM_AS_SEED,
    CONFIRM_AT_SEED,
    generate_w7b_confirm,
)
from nmd.freeform_attribution_cache import (
    compile_w7b_cache,
    save_w7b_cache,
)
from nmd.freeform_attribution_training import (
    CANDIDATES,
    attribution_verdict,
    evaluate_w7,
)
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import file_sha256


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def _load_paths(freeze: dict, checkpoints: Path):
    frozen = {}
    for row in freeze["candidates"]:
        name = row["candidate"]
        hira_path = checkpoints / f"{name}-hira.pt"
        scorer_path = checkpoints / f"{name}-scorer.pt"
        if file_sha256(hira_path) != row["hira_sha256"]:
            raise RuntimeError(f"W7b {name} HIRA SHA mismatch")
        if file_sha256(scorer_path) != row["scorer_sha256"]:
            raise RuntimeError(f"W7b {name} scorer SHA mismatch")
        frozen[name] = (row, hira_path, scorer_path)
    return frozen


def _load_scorer(name: str, path: Path):
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True),
        strict=True,
    )
    return scorer


def _evaluate_paths(frozen, cache):
    metrics = {}
    for name in CANDIDATES:
        _, hira_path, scorer_path = frozen[name]
        hira = HIRACore(d_model=256, dropout=0.0)
        hira.load_state_dict(
            torch.load(hira_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        scorer = _load_scorer(name, scorer_path)
        metrics[name] = evaluate_w7(
            hira,
            scorer,
            cache["cases"],
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w7b-freeze-v1":
        raise RuntimeError("unexpected W7b freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W7b freeze is not PASS")
    if freeze.get("all_candidates_frozen_before_confirm") is not True:
        raise RuntimeError("W7b candidates were not frozen before CONFIRM")
    if freeze.get("confirm_as_exposed") is not False:
        raise RuntimeError("W7b freeze already exposed CONFIRM-AS")
    if freeze.get("confirm_at_exposed") is not False:
        raise RuntimeError("W7b freeze already exposed CONFIRM-AT")
    if {row["candidate"] for row in freeze["candidates"]} != set(CANDIDATES):
        raise RuntimeError("W7b frozen candidate set changed")

    frozen = _load_paths(freeze, args.checkpoints)

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(
            repo_id=A13_MODEL,
            revision=A13_REVISION,
        )
    )
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA mismatch")
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
    cache_model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    cache_model.eval()

    print("R8_W7B_CONFIRM_AS_GENERATION_BEGIN", flush=True)
    confirm_as_cases = generate_w7b_confirm("AS", allow_confirm=True)
    confirm_as_cache = compile_w7b_cache(
        cache_model,
        confirm_as_cases,
        expected_split="confirm-as",
    )
    metrics_as = _evaluate_paths(frozen, confirm_as_cache)

    print("R8_W7B_CONFIRM_AT_GENERATION_BEGIN", flush=True)
    confirm_at_cases = generate_w7b_confirm("AT", allow_confirm=True)
    confirm_at_cache = compile_w7b_cache(
        cache_model,
        confirm_at_cases,
        expected_split="confirm-at",
    )
    metrics_at = _evaluate_paths(frozen, confirm_at_cache)

    verdict, verdict_details = attribution_verdict(
        metrics_as,
        metrics_at,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    cache_as_path = save_w7b_cache(
        confirm_as_cache,
        args.out / "confirm-as-cache.pt",
    )
    cache_at_path = save_w7b_cache(
        confirm_at_cache,
        args.out / "confirm-at-cache.pt",
    )

    receipt = {
        "schema_version": "r8-w7b-confirm-v1",
        "status": "PASS",
        "confirm_generated_after_all_dev_freezes": True,
        "confirm_domains": ["AS", "AT"],
        "confirm_as_seed": CONFIRM_AS_SEED,
        "confirm_at_seed": CONFIRM_AT_SEED,
        "confirm_as_case_count": len(confirm_as_cases),
        "confirm_at_case_count": len(confirm_at_cases),
        "confirm_as_decision_count": 5 * len(confirm_as_cases),
        "confirm_at_decision_count": 5 * len(confirm_at_cases),
        "confirm_as_case_id_sha256": confirm_as_cache[
            "metadata"
        ]["case_id_sha256"],
        "confirm_at_case_id_sha256": confirm_at_cache[
            "metadata"
        ]["case_id_sha256"],
        "confirm_as_cache_sha256": file_sha256(cache_as_path),
        "confirm_at_cache_sha256": file_sha256(cache_at_path),
        "state_encodes_per_case_as": confirm_as_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "state_encodes_per_case_at": confirm_at_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "factor_encoder_batches_as": 0,
        "factor_encoder_batches_at": 0,
        "paths": {
            "AS": metrics_as,
            "AT": metrics_at,
        },
        "verdict_details": verdict_details,
        "verdict": verdict,
        "candidate_dev_freezes": freeze["candidates"],
        "equal_train_case_budget": freeze["equal_train_case_budget"],
        "equal_optimizer_steps": freeze["equal_optimizer_steps"],
        "primary_replica_seed_independence": freeze[
            "primary_replica_seed_independence"
        ],
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
        "w6h_confirm_rows_used": False,
        "w6i_diagnostic_rows_used": False,
        "w6j_diagnostic_rows_used": False,
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
