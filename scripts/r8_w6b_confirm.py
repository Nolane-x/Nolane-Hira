from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_authority import (
    CONFIRM_SEED,
    generate_w6b_authority,
)
from nmd.typed_competitive_cache import (
    confirm_verdict,
    compile_w6b_cache,
    evaluate_w6b_cases,
    file_sha256,
    save_w6b_cache,
)


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
W3_HEAD_SHA256 = "2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c"
W5I_SCORER_SHA256 = "5ce9cdceb5a87c8b18b8d50e68396f23184e1dfb9acad338f25f24277d438a0d"
MAX_LENGTH = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--competitive-hira", type=Path, required=True)
    parser.add_argument("--competitive-scorer", type=Path, required=True)
    parser.add_argument("--legacy-hira", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(
        args.selection.read_text(encoding="utf-8")
    )
    if selection.get("schema_version") != "r8-w6b-selection-receipt-v1":
        raise RuntimeError("unexpected W6b selection schema")
    if selection.get("status") != "PASS":
        raise RuntimeError("W6b selection is not PASS")
    if selection.get("selection_authority") != "DEV_ONLY":
        raise RuntimeError("W6b selection authority is not DEV-only")
    if selection.get("confirm_exposed") is not False:
        raise RuntimeError("W6b selection already exposed CONFIRM")
    if selection.get("typed_decisions_final_or_test_used") is not False:
        raise RuntimeError("forbidden typed final/test exposure recorded")
    if selection.get("base_w3_head_sha256") != W3_HEAD_SHA256:
        raise RuntimeError("unexpected W3 base identity")
    if selection.get("base_w5i_scorer_sha256") != W5I_SCORER_SHA256:
        raise RuntimeError("unexpected W5i scorer identity")

    if (
        file_sha256(args.competitive_hira)
        != selection["selected_competitive_hira_sha256"]
    ):
        raise RuntimeError("selected competitive HIRA SHA mismatch")
    if (
        file_sha256(args.competitive_scorer)
        != selection["selected_competitive_scorer_sha256"]
    ):
        raise RuntimeError("selected competitive scorer SHA mismatch")
    if file_sha256(args.legacy_hira) != selection["legacy_hira_sha256"]:
        raise RuntimeError("legacy HIRA SHA mismatch")

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
        max_length=MAX_LENGTH,
    )
    cache_model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    cache_model.eval()

    print("R8_W6B_CONFIRM_GENERATION_BEGIN", flush=True)
    confirm_cases = generate_w6b_authority(
        "confirm",
        allow_confirm=True,
    )
    confirm_cache = compile_w6b_cache(
        cache_model,
        confirm_cases,
        expected_split="confirm",
    )

    competitive_hira = HIRACore(d_model=256, dropout=0.0)
    competitive_hira.load_state_dict(
        torch.load(
            args.competitive_hira,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    competitive_scorer = CompetitiveCoarseScorer(
        d_model=256,
        d_rel=128,
    )
    competitive_scorer.load_state_dict(
        torch.load(
            args.competitive_scorer,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    legacy_hira = HIRACore(d_model=256, dropout=0.0)
    legacy_hira.load_state_dict(
        torch.load(
            args.legacy_hira,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )

    selected_metrics = evaluate_w6b_cases(
        competitive_hira,
        competitive_scorer,
        confirm_cache["cases"],
        competitive=True,
    )
    legacy_metrics = evaluate_w6b_cases(
        legacy_hira,
        None,
        confirm_cache["cases"],
        competitive=False,
    )
    verdict, absolute, mechanism = confirm_verdict(
        selected_metrics,
        legacy_metrics,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = save_w6b_cache(
        confirm_cache,
        args.out / "confirm-cache.pt",
    )
    receipt = {
        "schema_version": "r8-w6b-confirm-v1",
        "status": "PASS",
        "selected_competitive_candidate": selection[
            "selected_competitive_candidate"
        ],
        "selected_competitive_epoch": selection[
            "selected_competitive_epoch"
        ],
        "selected_competitive_hira_sha256": selection[
            "selected_competitive_hira_sha256"
        ],
        "selected_competitive_scorer_sha256": selection[
            "selected_competitive_scorer_sha256"
        ],
        "legacy_candidate": selection["legacy_candidate"],
        "legacy_epoch": selection["legacy_epoch"],
        "legacy_hira_sha256": selection["legacy_hira_sha256"],
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "base_w3_head_sha256": W3_HEAD_SHA256,
        "base_w5i_scorer_sha256": W5I_SCORER_SHA256,
        "train_cache_sha256": selection["train_cache_sha256"],
        "dev_cache_sha256": selection["dev_cache_sha256"],
        "selected_competitive_trainable_parameter_count": selection[
            "selected_competitive_trainable_parameter_count"
        ],
        "selected_competitive_total_parameter_count": selection[
            "selected_competitive_total_parameter_count"
        ],
        "legacy_trainable_parameter_count": selection[
            "legacy_trainable_parameter_count"
        ],
        "legacy_total_parameter_count": selection[
            "legacy_total_parameter_count"
        ],
        "confirm_seed": CONFIRM_SEED,
        "confirm_generated_after_selection_freeze": True,
        "confirm_case_count": len(confirm_cases),
        "confirm_decision_count": 5 * len(confirm_cases),
        "confirm_case_id_sha256": confirm_cache[
            "metadata"
        ]["case_id_sha256"],
        "confirm_cache_sha256": file_sha256(cache_path),
        "state_encodes_per_case": confirm_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "selected": selected_metrics,
        "legacy_control": legacy_metrics,
        "absolute_gates": absolute,
        "mechanism_gates": mechanism,
        "verdict": verdict,
        "candidate_dev_histories": selection["candidates"],
        "typed_decisions_final_or_test_used": False,
        "w5_semantic_confirm_rows_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "confirm.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
