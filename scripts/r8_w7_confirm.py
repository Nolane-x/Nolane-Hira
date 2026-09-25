from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.conjunctive_authority import (
    CONFIRM_AL_SEED,
    CONFIRM_AM_SEED,
    generate_w7_confirm,
)
from nmd.conjunctive_cache import (
    compile_w7_cache,
    save_w7_cache,
)
from nmd.conjunctive_coarse import ConjunctiveEvidenceScorer
from nmd.conjunctive_training import (
    CANDIDATES,
    conjunctive_verdict,
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
            raise RuntimeError(f"W7 {name} HIRA SHA mismatch")
        if file_sha256(scorer_path) != row["scorer_sha256"]:
            raise RuntimeError(f"W7 {name} scorer SHA mismatch")
        frozen[name] = (row, hira_path, scorer_path)
    return frozen


def _load_scorer(name: str, path: Path):
    if name in {"conjunctive-primary", "conjunctive-replica"}:
        scorer = ConjunctiveEvidenceScorer(
            base=CompetitiveCoarseScorer(d_model=256, d_rel=128)
        )
    else:
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
    if freeze.get("schema_version") != "r8-w7-freeze-v1":
        raise RuntimeError("unexpected W7 freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W7 freeze is not PASS")
    if freeze.get("all_candidates_frozen_before_confirm") is not True:
        raise RuntimeError("W7 candidates were not frozen before CONFIRM")
    if freeze.get("confirm_al_exposed") is not False:
        raise RuntimeError("W7 freeze already exposed CONFIRM-AL")
    if freeze.get("confirm_am_exposed") is not False:
        raise RuntimeError("W7 freeze already exposed CONFIRM-AM")
    if {row["candidate"] for row in freeze["candidates"]} != set(CANDIDATES):
        raise RuntimeError("W7 frozen candidate set changed")

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

    print("R8_W7_CONFIRM_AL_GENERATION_BEGIN", flush=True)
    confirm_al_cases = generate_w7_confirm("AL", allow_confirm=True)
    confirm_al_cache = compile_w7_cache(
        cache_model,
        confirm_al_cases,
        expected_split="confirm-al",
    )
    metrics_al = _evaluate_paths(frozen, confirm_al_cache)

    print("R8_W7_CONFIRM_AM_GENERATION_BEGIN", flush=True)
    confirm_am_cases = generate_w7_confirm("AM", allow_confirm=True)
    confirm_am_cache = compile_w7_cache(
        cache_model,
        confirm_am_cases,
        expected_split="confirm-am",
    )
    metrics_am = _evaluate_paths(frozen, confirm_am_cache)

    verdict, verdict_details = conjunctive_verdict(
        metrics_al,
        metrics_am,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    cache_al_path = save_w7_cache(
        confirm_al_cache,
        args.out / "confirm-al-cache.pt",
    )
    cache_am_path = save_w7_cache(
        confirm_am_cache,
        args.out / "confirm-am-cache.pt",
    )

    receipt = {
        "schema_version": "r8-w7-confirm-v1",
        "status": "PASS",
        "confirm_generated_after_all_dev_freezes": True,
        "confirm_domains": ["AL", "AM"],
        "confirm_al_seed": CONFIRM_AL_SEED,
        "confirm_am_seed": CONFIRM_AM_SEED,
        "confirm_al_case_count": len(confirm_al_cases),
        "confirm_am_case_count": len(confirm_am_cases),
        "confirm_al_decision_count": 5 * len(confirm_al_cases),
        "confirm_am_decision_count": 5 * len(confirm_am_cases),
        "confirm_al_case_id_sha256": confirm_al_cache[
            "metadata"
        ]["case_id_sha256"],
        "confirm_am_case_id_sha256": confirm_am_cache[
            "metadata"
        ]["case_id_sha256"],
        "confirm_al_factor_identity_sha256": confirm_al_cache[
            "metadata"
        ]["factor_identity_sha256"],
        "confirm_am_factor_identity_sha256": confirm_am_cache[
            "metadata"
        ]["factor_identity_sha256"],
        "confirm_al_cache_sha256": file_sha256(cache_al_path),
        "confirm_am_cache_sha256": file_sha256(cache_am_path),
        "state_encodes_per_case_al": confirm_al_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "state_encodes_per_case_am": confirm_am_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "factor_encoder_batches_al": confirm_al_cache[
            "metadata"
        ]["factor_encoder_batches"],
        "factor_encoder_batches_am": confirm_am_cache[
            "metadata"
        ]["factor_encoder_batches"],
        "paths": {
            "AL": metrics_al,
            "AM": metrics_am,
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
