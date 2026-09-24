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
from nmd.typed_domain_generalization import (
    CANDIDATES,
    confirm_verdict,
)
from nmd.typed_domain_generalization_authority import (
    CONFIRM_F_SEED,
    generate_w6d_confirm,
)


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w6d-freeze-v1":
        raise RuntimeError("unexpected W6d freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W6d freeze is not PASS")
    if freeze.get("all_candidates_frozen_before_confirm") is not True:
        raise RuntimeError("W6d candidates were not frozen before CONFIRM")
    if freeze.get("confirm_exposed") is not False:
        raise RuntimeError("W6d freeze already exposed CONFIRM")
    if {row["candidate"] for row in freeze["candidates"]} != set(CANDIDATES):
        raise RuntimeError("W6d frozen candidate set changed")

    frozen = {}
    for row in freeze["candidates"]:
        name = row["candidate"]
        hira_path = args.checkpoints / f"{name}-hira.pt"
        scorer_path = args.checkpoints / f"{name}-scorer.pt"
        if file_sha256(hira_path) != row["hira_sha256"]:
            raise RuntimeError(f"W6d {name} HIRA SHA mismatch")
        if file_sha256(scorer_path) != row["scorer_sha256"]:
            raise RuntimeError(f"W6d {name} scorer SHA mismatch")
        frozen[name] = (row, hira_path, scorer_path)

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

    print("R8_W6D_CONFIRM_F_GENERATION_BEGIN", flush=True)
    confirm_cases = generate_w6d_confirm(allow_confirm=True)
    confirm_cache = compile_w6b_cache(
        cache_model,
        confirm_cases,
        expected_split="confirm",
    )

    metrics = {}
    for name in CANDIDATES:
        row, hira_path, scorer_path = frozen[name]
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
            confirm_cache["cases"],
            competitive=True,
        )

    verdict, competence, diversity, joint_gates = confirm_verdict(
        single=metrics["single-source-scorer-only"],
        multi=metrics["multi-source-scorer-only"],
        joint=metrics["multi-source-joint"],
    )

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = save_w6b_cache(
        confirm_cache,
        args.out / "confirm-f-cache.pt",
    )
    receipt = {
        "schema_version": "r8-w6d-confirm-v1",
        "status": "PASS",
        "confirm_generated_after_all_dev_freezes": True,
        "confirm_domain": "F",
        "confirm_seed": CONFIRM_F_SEED,
        "confirm_case_count": len(confirm_cases),
        "confirm_decision_count": 5 * len(confirm_cases),
        "confirm_case_id_sha256": confirm_cache[
            "metadata"
        ]["case_id_sha256"],
        "confirm_cache_sha256": file_sha256(cache_path),
        "state_encodes_per_case": confirm_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "paths": metrics,
        "multi_source_semantic_competence": competence,
        "multi_vs_single_diversity_gates": diversity,
        "joint_localization_gates": joint_gates,
        "verdict": verdict,
        "candidate_dev_freezes": freeze["candidates"],
        "scorer_only_equal_case_budget": freeze[
            "scorer_only_equal_case_budget"
        ],
        "scorer_only_equal_optimizer_steps": freeze[
            "scorer_only_equal_optimizer_steps"
        ],
        "scorer_only_equal_trainable_parameters": freeze[
            "scorer_only_equal_trainable_parameters"
        ],
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
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
