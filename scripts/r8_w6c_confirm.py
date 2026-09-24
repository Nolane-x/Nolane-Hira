from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.calibration import TypedReliabilityCalibrator
from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_reliability_authority import (
    CONFIRM_SEED,
    generate_w6c_authority,
)
from nmd.typed_reliability_cache import (
    W6B_HIRA_SHA256,
    W6B_SCORER_SHA256,
    confirm_verdict,
    evaluate_w6c_cases,
    file_sha256,
    compile_w6c_logit_cache,
    save_w6c_logit_cache,
)


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--w6b-hira", type=Path, required=True)
    parser.add_argument("--w6b-scorer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(
        args.selection.read_text(encoding="utf-8")
    )
    if (
        selection.get("schema_version")
        != "r8-w6c-selection-receipt-v1"
    ):
        raise RuntimeError("unexpected W6c selection schema")
    if selection.get("status") != "PASS":
        raise RuntimeError("W6c selection is not PASS")
    if selection.get("selection_authority") != "DEV_ONLY":
        raise RuntimeError("W6c selection is not DEV-only")
    if selection.get("confirm_exposed") is not False:
        raise RuntimeError("W6c selection already exposed CONFIRM")
    if selection.get("w6b_confirm_rows_used") is not False:
        raise RuntimeError("W6b CONFIRM rows were used")
    if (
        selection.get("typed_decisions_final_or_test_used")
        is not False
    ):
        raise RuntimeError("forbidden typed final/test exposure")
    if (
        selection.get("base_w6b_hira_sha256")
        != W6B_HIRA_SHA256
    ):
        raise RuntimeError("unexpected W6b HIRA identity")
    if (
        selection.get("base_w6b_scorer_sha256")
        != W6B_SCORER_SHA256
    ):
        raise RuntimeError("unexpected W6b scorer identity")
    if (
        file_sha256(args.calibrator)
        != selection["selected_calibrator_sha256"]
    ):
        raise RuntimeError(
            "selected W6c calibrator SHA mismatch"
        )
    if file_sha256(args.w6b_hira) != W6B_HIRA_SHA256:
        raise RuntimeError("W6b HIRA file SHA mismatch")
    if file_sha256(args.w6b_scorer) != W6B_SCORER_SHA256:
        raise RuntimeError("W6b scorer file SHA mismatch")

    mode = selection["selected_mode"]
    calibrator = TypedReliabilityCalibrator(mode)
    calibrator.load_state_dict(
        torch.load(
            args.calibrator,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    if (
        calibrator.trainable_parameter_count
        != selection["selected_trainable_parameter_count"]
    ):
        raise RuntimeError(
            "W6c calibrator parameter count mismatch"
        )
    calibrator.eval()

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
    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(
            args.w6b_hira,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    scorer = CompetitiveCoarseScorer(
        d_model=256,
        d_rel=128,
    )
    scorer.load_state_dict(
        torch.load(
            args.w6b_scorer,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)

    model = NolaneHira(
        encoder,
        hira,
        coarse_scorer=scorer,
    )
    model.eval()

    print("R8_W6C_CONFIRM_GENERATION_BEGIN", flush=True)
    confirm_cases = generate_w6c_authority(
        "confirm",
        allow_confirm=True,
    )
    confirm_cache = compile_w6c_logit_cache(
        model,
        confirm_cases,
        expected_split="confirm",
    )

    selected_metrics = evaluate_w6c_cases(
        calibrator,
        confirm_cache,
    )
    control_metrics = evaluate_w6c_cases(
        None,
        confirm_cache,
    )
    verdict, absolute, mechanism = confirm_verdict(
        selected_metrics,
        control_metrics,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = save_w6c_logit_cache(
        confirm_cache,
        args.out / "confirm-cache.pt",
    )
    receipt = {
        "schema_version": "r8-w6c-confirm-v1",
        "status": "PASS",
        "selected_candidate": selection[
            "selected_candidate"
        ],
        "selected_mode": mode,
        "selected_epoch": selection["selected_epoch"],
        "selected_parameters": selection[
            "selected_parameters"
        ],
        "selected_calibrator_sha256": selection[
            "selected_calibrator_sha256"
        ],
        "selected_trainable_parameter_count": selection[
            "selected_trainable_parameter_count"
        ],
        "selected_total_parameter_count": selection[
            "selected_total_parameter_count"
        ],
        "control_candidate": selection["control_candidate"],
        "control_trainable_parameter_count": 0,
        "control_total_parameter_count": selection[
            "control_total_parameter_count"
        ],
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "base_w6b_hira_sha256": W6B_HIRA_SHA256,
        "base_w6b_scorer_sha256": W6B_SCORER_SHA256,
        "train_cache_sha256": selection[
            "train_cache_sha256"
        ],
        "dev_cache_sha256": selection["dev_cache_sha256"],
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
        "frozen_production_control": control_metrics,
        "absolute_gates": absolute,
        "mechanism_gates": mechanism,
        "verdict": verdict,
        "candidate_dev_histories": selection["candidates"],
        "cached_production_logits": True,
        "hira_frozen": True,
        "competitive_scorer_frozen": True,
        "w6b_confirm_rows_used": False,
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
