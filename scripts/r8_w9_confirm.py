from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_alignment_authority import (
    DOMAIN_SEEDS,
    all_w9_text_atoms,
    generate_w9_domain,
)
from nmd.semantic_alignment_bridge import SemanticAlignmentBridgeScorer
from nmd.semantic_alignment_cache import (
    compile_w9_cache,
    save_w9_cache,
)
from nmd.semantic_alignment_eval import evaluate_w9_checkpoint
from nmd.semantic_alignment_training import CANDIDATES
from nmd.semantic_alignment_verdict import semantic_alignment_verdict
from nmd.typed_competitive_cache import file_sha256


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def _load_scorer(candidate: str, path: Path):
    base = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    if candidate in {
        "frozen-w6e-control",
        "projection-semantic-control",
    }:
        scorer = base
    elif candidate == "shared-bridge-semantic-control":
        scorer = SemanticAlignmentBridgeScorer(
            base,
            mode="shared",
        )
    elif candidate in {
        "asymmetric-bridge-primary",
        "asymmetric-bridge-replica",
    }:
        scorer = SemanticAlignmentBridgeScorer(
            base,
            mode="asymmetric",
        )
    else:
        raise ValueError(f"unknown W9 candidate: {candidate}")

    scorer.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True),
        strict=True,
    )
    scorer.eval()
    return scorer


def _load_frozen(
    freeze: dict,
    checkpoints: Path,
) -> dict[str, tuple[Path, Path]]:
    rows = {}
    for row in freeze["candidates"]:
        candidate = row["candidate"]
        root = checkpoints / row["checkpoint_dir"]
        hira = root / "hira.pt"
        scorer = root / "scorer.pt"
        if file_sha256(hira) != row["hira_sha256"]:
            raise RuntimeError("W9 frozen HIRA SHA mismatch")
        if file_sha256(scorer) != row["scorer_sha256"]:
            raise RuntimeError("W9 frozen scorer SHA mismatch")
        rows[candidate] = (hira, scorer)
    if set(rows) != set(CANDIDATES):
        raise RuntimeError("W9 frozen candidate set changed")
    return rows


def _evaluate_paths(
    frozen: dict[str, tuple[Path, Path]],
    cache: dict,
) -> dict[str, dict[str, object]]:
    metrics = {}
    for candidate in CANDIDATES:
        hira_path, scorer_path = frozen[candidate]
        hira = HIRACore(d_model=256, dropout=0.0)
        hira.load_state_dict(
            torch.load(hira_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        hira.eval()
        scorer = _load_scorer(candidate, scorer_path)
        metrics[candidate] = evaluate_w9_checkpoint(
            hira,
            scorer,
            cache,
        )
    return metrics


def _combined_for_verdict(
    bd: dict[str, dict[str, object]],
    be: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    combined = {}
    for candidate in CANDIDATES:
        left = bd[candidate]
        right = be[candidate]
        combined[candidate] = {
            "per_domain": {
                "BD": left["per_domain"]["BD"],
                "BE": right["per_domain"]["BE"],
            },
            "alignment": {
                "per_domain": {
                    "BD": left["alignment"]["per_domain"]["BD"],
                    "BE": right["alignment"]["per_domain"]["BE"],
                },
            },
            "probability_mass_max_error": max(
                float(left["probability_mass_max_error"]),
                float(right["probability_mass_max_error"]),
            ),
            "state_encodes_per_base": 1.0,
        }
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w9-freeze-v1":
        raise RuntimeError("unexpected W9 freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W9 freeze is not PASS")
    if freeze.get("all_candidates_frozen_before_confirm") is not True:
        raise RuntimeError("W9 candidates were not frozen before CONFIRM")
    if freeze.get("confirm_bd_exposed") is not False:
        raise RuntimeError("W9 freeze already exposed CONFIRM-BD")
    if freeze.get("confirm_be_exposed") is not False:
        raise RuntimeError("W9 freeze already exposed CONFIRM-BE")
    if {row["candidate"] for row in freeze["candidates"]} != set(CANDIDATES):
        raise RuntimeError("W9 frozen candidate set changed")

    frozen = _load_frozen(freeze, args.checkpoints)

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
        raise RuntimeError("W9 A13 weight SHA mismatch")

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

    train_dev_atoms = all_w9_text_atoms(include_confirm=False)
    all_atoms = all_w9_text_atoms(include_confirm=True)
    confirm_atoms = all_atoms - train_dev_atoms
    if train_dev_atoms & confirm_atoms:
        raise RuntimeError("W9 exact train/dev-confirm atom overlap")

    print("R8_W9_CONFIRM_BD_GENERATION_BEGIN", flush=True)
    bd_rows = generate_w9_domain("BD", allow_confirm=True)
    bd_cache = compile_w9_cache(cache_model, bd_rows)
    metrics_bd = _evaluate_paths(frozen, bd_cache)

    print("R8_W9_CONFIRM_BE_GENERATION_BEGIN", flush=True)
    be_rows = generate_w9_domain("BE", allow_confirm=True)
    be_cache = compile_w9_cache(cache_model, be_rows)
    metrics_be = _evaluate_paths(frozen, be_cache)

    combined = _combined_for_verdict(metrics_bd, metrics_be)
    verdict, verdict_details = semantic_alignment_verdict(combined)

    args.out.mkdir(parents=True, exist_ok=True)
    bd_cache_path = save_w9_cache(
        bd_cache,
        args.out / "confirm-bd-cache.pt",
    )
    be_cache_path = save_w9_cache(
        be_cache,
        args.out / "confirm-be-cache.pt",
    )

    external_authorized = verdict in {
        "ASYMMETRIC_SEMANTIC_BRIDGE_RESCUE",
        "SEMANTIC_ALIGNMENT_RESCUE_NO_ASYMMETRY",
    }

    receipt = {
        "schema_version": "r8-w9-confirm-v1",
        "status": "PASS",
        "confirm_generated_after_all_dev_freezes": True,
        "confirm_domains": ["BD", "BE"],
        "confirm_bd_seed": DOMAIN_SEEDS["BD"],
        "confirm_be_seed": DOMAIN_SEEDS["BE"],
        "confirm_bd_base_count": bd_cache["metadata"]["base_count"],
        "confirm_be_base_count": be_cache["metadata"]["base_count"],
        "confirm_bd_view_count": bd_cache["metadata"]["view_count"],
        "confirm_be_view_count": be_cache["metadata"]["view_count"],
        "confirm_bd_cache_sha256": file_sha256(bd_cache_path),
        "confirm_be_cache_sha256": file_sha256(be_cache_path),
        "state_encodes_per_base_bd": bd_cache["metadata"][
            "state_encode_calls_per_base"
        ],
        "state_encodes_per_base_be": be_cache["metadata"][
            "state_encode_calls_per_base"
        ],
        "paths": {
            "BD": metrics_bd,
            "BE": metrics_be,
        },
        "verdict": verdict,
        "verdict_details": verdict_details,
        "external_public_phase_authorized": external_authorized,
        "candidate_dev_freezes": freeze["candidates"],
        "equal_train_base_budget": freeze["equal_train_base_budget"],
        "equal_optimizer_steps": freeze["equal_optimizer_steps"],
        "primary_replica_seed_independence": freeze[
            "primary_replica_seed_independence"
        ],
        "w7_confirm_rows_used": False,
        "w7b_confirm_rows_used": False,
        "w8_diagnostic_rows_used": False,
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
