from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import (
    compile_w6b_cache,
    evaluate_w6b_cases,
    file_sha256,
    save_w6b_cache,
)
from nmd.typed_joint_replication import (
    CANDIDATES,
    replication_verdict,
)
from nmd.typed_joint_replication_authority import (
    CONFIRM_L_SEED,
    CONFIRM_M_SEED,
    generate_w6e_confirm,
)


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
            raise RuntimeError(f"W6e {name} HIRA SHA mismatch")
        if file_sha256(scorer_path) != row["scorer_sha256"]:
            raise RuntimeError(f"W6e {name} scorer SHA mismatch")
        frozen[name] = (row, hira_path, scorer_path)
    return frozen


def _evaluate_paths(frozen, cache):
    metrics = {}
    for name in CANDIDATES:
        _, hira_path, scorer_path = frozen[name]
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
        metrics[name] = evaluate_w6b_cases(
            hira,
            scorer,
            cache["cases"],
            competitive=True,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w6e-freeze-v1":
        raise RuntimeError("unexpected W6e freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W6e freeze is not PASS")
    if freeze.get("all_candidates_frozen_before_confirm") is not True:
        raise RuntimeError("W6e candidates were not frozen before CONFIRM")
    if freeze.get("confirm_l_exposed") is not False:
        raise RuntimeError("W6e freeze already exposed CONFIRM-L")
    if freeze.get("confirm_m_exposed") is not False:
        raise RuntimeError("W6e freeze already exposed CONFIRM-M")
    if {row["candidate"] for row in freeze["candidates"]} != set(CANDIDATES):
        raise RuntimeError("W6e frozen candidate set changed")

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

    print("R8_W6E_CONFIRM_L_GENERATION_BEGIN", flush=True)
    confirm_l_cases = generate_w6e_confirm("L", allow_confirm=True)
    confirm_l_cache = compile_w6b_cache(
        cache_model,
        confirm_l_cases,
        expected_split="confirm-l",
    )
    metrics_l = _evaluate_paths(frozen, confirm_l_cache)

    print("R8_W6E_CONFIRM_M_GENERATION_BEGIN", flush=True)
    confirm_m_cases = generate_w6e_confirm("M", allow_confirm=True)
    confirm_m_cache = compile_w6b_cache(
        cache_model,
        confirm_m_cases,
        expected_split="confirm-m",
    )
    metrics_m = _evaluate_paths(frozen, confirm_m_cache)

    verdict, verdict_details = replication_verdict(
        confirm_l=metrics_l,
        confirm_m=metrics_m,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    cache_l_path = save_w6b_cache(
        confirm_l_cache,
        args.out / "confirm-l-cache.pt",
    )
    cache_m_path = save_w6b_cache(
        confirm_m_cache,
        args.out / "confirm-m-cache.pt",
    )

    receipt = {
        "schema_version": "r8-w6e-confirm-v1",
        "status": "PASS",
        "confirm_generated_after_all_dev_freezes": True,
        "confirm_domains": ["L", "M"],
        "confirm_l_seed": CONFIRM_L_SEED,
        "confirm_m_seed": CONFIRM_M_SEED,
        "confirm_l_case_count": len(confirm_l_cases),
        "confirm_m_case_count": len(confirm_m_cases),
        "confirm_l_decision_count": 5 * len(confirm_l_cases),
        "confirm_m_decision_count": 5 * len(confirm_m_cases),
        "confirm_l_case_id_sha256": confirm_l_cache[
            "metadata"
        ]["case_id_sha256"],
        "confirm_m_case_id_sha256": confirm_m_cache[
            "metadata"
        ]["case_id_sha256"],
        "confirm_l_cache_sha256": file_sha256(cache_l_path),
        "confirm_m_cache_sha256": file_sha256(cache_m_path),
        "state_encodes_per_case_l": confirm_l_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "state_encodes_per_case_m": confirm_m_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "paths": {
            "L": metrics_l,
            "M": metrics_m,
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
