from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.anchor_preserving_residual import AnchorPreservingResidualMixer
from nmd.anchor_preserving_residual_authority import generate_w15_confirm
from nmd.anchor_preserving_residual_cache import (
    compile_w15_cache,
    save_w15_cache,
)
from nmd.anchor_preserving_residual_training import (
    CANDIDATES,
    TRAINABLE_CANDIDATES,
    evaluate_w15,
    w15_verdict,
)
from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import file_sha256


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = (
    "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
)
W9_HIRA_SHA256 = (
    "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
)
W9_SCORER_SHA256 = (
    "8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102"
)
MAX_LENGTH = 256


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION)
    )
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("W15 A13 weight SHA mismatch")
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
    return HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=MAX_LENGTH,
    )


def _load_base(hira_path: Path, scorer_path: Path):
    if file_sha256(hira_path) != W9_HIRA_SHA256:
        raise RuntimeError("W15 frozen HIRA SHA mismatch")
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W15 frozen scorer SHA mismatch")
    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(hira_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    hira.eval()
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.load_state_dict(
        torch.load(scorer_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return hira, scorer


def _load_mixer(row: dict, checkpoints: Path):
    name = row["candidate"]
    if name not in TRAINABLE_CANDIDATES:
        return None
    path = checkpoints / f"{name}-mixer.pt"
    if not path.exists():
        raise RuntimeError(f"W15 frozen mixer missing: {name}")
    if file_sha256(path) != row.get("mixer_sha256"):
        raise RuntimeError(f"W15 frozen mixer SHA mismatch: {name}")
    bounded = bool(row["mixer_bounded"])
    mixer = AnchorPreservingResidualMixer(bounded=bounded)
    mixer.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True),
        strict=True,
    )
    mixer.eval()
    return mixer


def _evaluate_all(freeze: dict, checkpoints: Path, hira, scorer, cache):
    by_name = {row["candidate"]: row for row in freeze["candidates"]}
    metrics = {}
    for name in CANDIDATES:
        mixer = _load_mixer(by_name[name], checkpoints)
        metrics[name] = evaluate_w15(
            name,
            hira,
            scorer,
            cache["cases"],
            mixer=mixer,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--w9-hira", type=Path, required=True)
    parser.add_argument("--w9-scorer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w15-freeze-v1":
        raise RuntimeError("unexpected W15 freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W15 freeze is not PASS")
    if freeze.get("all_candidates_frozen_before_confirm") is not True:
        raise RuntimeError("W15 candidates not all frozen before CONFIRM")
    if freeze.get("confirm_ce_exposed") is not False:
        raise RuntimeError("W15 freeze already exposed CE")
    if freeze.get("confirm_cf_exposed") is not False:
        raise RuntimeError("W15 freeze already exposed CF")
    if {row["candidate"] for row in freeze["candidates"]} != set(CANDIDATES):
        raise RuntimeError("W15 frozen candidate set changed")
    if freeze.get("primary_replica_seed_independence") is not True:
        raise RuntimeError("W15 primary/replica seed independence failed")

    hira, scorer = _load_base(args.w9_hira, args.w9_scorer)
    encoder = _load_a13()
    cache_model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    cache_model.eval()

    print("R8_W15_CONFIRM_CE_GENERATION_BEGIN", flush=True)
    ce_cases = generate_w15_confirm("CE", allow_confirm=True)
    ce_cache = compile_w15_cache(
        cache_model,
        ce_cases,
        expected_split="confirm-ce",
    )
    ce_metrics = _evaluate_all(
        freeze,
        args.checkpoints,
        hira,
        scorer,
        ce_cache,
    )

    print("R8_W15_CONFIRM_CF_GENERATION_BEGIN", flush=True)
    cf_cases = generate_w15_confirm("CF", allow_confirm=True)
    cf_cache = compile_w15_cache(
        cache_model,
        cf_cases,
        expected_split="confirm-cf",
    )
    cf_metrics = _evaluate_all(
        freeze,
        args.checkpoints,
        hira,
        scorer,
        cf_cache,
    )

    verdict, verdict_details = w15_verdict(ce_metrics, cf_metrics)

    args.out.mkdir(parents=True, exist_ok=True)
    ce_path = save_w15_cache(ce_cache, args.out / "confirm-ce.pt")
    cf_path = save_w15_cache(cf_cache, args.out / "confirm-cf.pt")

    receipt = {
        "schema_version": "r8-w15-confirm-v1",
        "status": "PASS",
        "verdict": verdict,
        "verdict_details": verdict_details,
        "confirm_generated_after_all_dev_freezes": True,
        "confirm_domains": ["CE", "CF"],
        "confirm_ce_case_count": len(ce_cases),
        "confirm_cf_case_count": len(cf_cases),
        "confirm_ce_decision_count": 5 * len(ce_cases),
        "confirm_cf_decision_count": 5 * len(cf_cases),
        "state_encodes_per_case_ce": ce_cache["metadata"][
            "state_encode_calls_per_case"
        ],
        "state_encodes_per_case_cf": cf_cache["metadata"][
            "state_encode_calls_per_case"
        ],
        "confirm_ce_case_id_sha256": ce_cache["metadata"][
            "case_id_sha256"
        ],
        "confirm_cf_case_id_sha256": cf_cache["metadata"][
            "case_id_sha256"
        ],
        "confirm_ce_cache_sha256": file_sha256(ce_path),
        "confirm_cf_cache_sha256": file_sha256(cf_path),
        "paths": {
            "CE": ce_metrics,
            "CF": cf_metrics,
        },
        "candidate_dev_freezes": freeze["candidates"],
        "equal_train_case_budget": freeze["equal_train_case_budget"],
        "equal_optimizer_steps": freeze["equal_optimizer_steps"],
        "primary_replica_seed_independence": freeze[
            "primary_replica_seed_independence"
        ],
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "hira_sha256": W9_HIRA_SHA256,
        "scorer_sha256": W9_SCORER_SHA256,
        "w12_rows_used": False,
        "w13_rows_used": False,
        "w14_rows_used": False,
        "banking77_rows_used": False,
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
