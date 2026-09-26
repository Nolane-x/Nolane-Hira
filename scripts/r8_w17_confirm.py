from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.competitive import CompetitiveCoarseScorer
from nmd.field_isolated_authority import generate_w17_confirm
from nmd.field_isolated_cache import compile_w17_cache, save_w17_cache
from nmd.field_isolated_eval import (
    evaluate_w17,
    reference_summary_from_scores,
    w17_verdict,
)
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
REFERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REFERENCE_WEIGHT_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
MAX_LENGTH = 256


def _load_a13_model() -> NolaneHira:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("W17 A13 weight SHA mismatch")
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
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    return model


def _load_w9_base(freeze_path: Path, checkpoints: Path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w9-freeze-v1":
        raise RuntimeError("unexpected W9 freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W9 freeze is not PASS")
    rows = {row["candidate"]: row for row in freeze["candidates"]}
    row = rows.get("projection-semantic-control")
    if row is None:
        raise RuntimeError("W17 projection-semantic-control missing")
    root = checkpoints / row["checkpoint_dir"]
    hira_path = root / "hira.pt"
    scorer_path = root / "scorer.pt"
    if file_sha256(hira_path) != W9_HIRA_SHA256:
        raise RuntimeError("W17 frozen HIRA SHA mismatch")
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W17 frozen scorer SHA mismatch")

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
    hira.eval()
    scorer.eval()
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return freeze, row, hira, scorer


@torch.inference_mode()
def _reference_embeddings(model, tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    out: dict[str, torch.Tensor] = {}
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        result = model(**encoded, return_dict=True)
        tokens = result.last_hidden_state
        mask = encoded["attention_mask"].to(tokens.dtype)[..., None]
        pooled = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = F.normalize(pooled, dim=-1).cpu()
        for text, vector in zip(batch, pooled):
            out[text] = vector
    return out


@torch.inference_mode()
def _reference_for_cache(cache: dict) -> tuple[dict[str, object], str]:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(repo_id=REFERENCE_MODEL, revision=REFERENCE_REVISION)
    )
    weight = snapshot / "model.safetensors"
    if not weight.exists():
        raise RuntimeError("W17 reference model.safetensors missing")
    weight_sha = file_sha256(weight)
    if weight_sha != REFERENCE_WEIGHT_SHA256:
        raise RuntimeError("W17 pinned reference SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    texts: set[str] = set()
    for case in cache["cases"]:
        texts.add(str(case["fields"]["intent"]))
        diagnosis = next(
            row for row in case["decisions"] if row["question_id"] == "diagnosis"
        )
        texts.update(str(x) for x in diagnosis["views"]["D0"]["option_texts"])
    embeddings = _reference_embeddings(model, tokenizer, sorted(texts))

    lookup: dict[str, list[float]] = {}
    for case in cache["cases"]:
        state = embeddings[str(case["fields"]["intent"])]
        diagnosis = next(
            row for row in case["decisions"] if row["question_id"] == "diagnosis"
        )
        options = torch.stack(
            [
                embeddings[str(text)]
                for text in diagnosis["views"]["D0"]["option_texts"]
            ],
            dim=0,
        )
        scores = torch.einsum("d,kd->k", state, options)
        lookup[str(case["case_id"])] = [float(value) for value in scores]
    return reference_summary_from_scores(cache["cases"], lookup), weight_sha


def _validate_preconfirm(path: Path) -> dict:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w17-preconfirm-freeze-v1":
        raise RuntimeError("unexpected W17 preconfirm freeze")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W17 preconfirm freeze is not PASS")
    if freeze.get("all_scientific_gates_frozen_before_confirm") is not True:
        raise RuntimeError("W17 gates were not frozen before confirm")
    if freeze.get("training_performed") is not False:
        raise RuntimeError("W17 training is forbidden")
    if freeze.get("selection_performed") is not False:
        raise RuntimeError("W17 DEV selection is forbidden")
    if freeze.get("confirm_cp_exposed") is not False:
        raise RuntimeError("W17 CP already exposed")
    if freeze.get("confirm_cq_exposed") is not False:
        raise RuntimeError("W17 CQ already exposed")
    if int(freeze.get("trainable_parameter_count", -1)) != 0:
        raise RuntimeError("W17 must remain zero-parameter")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preconfirm-freeze", type=Path, required=True)
    parser.add_argument("--w9-freeze", type=Path, required=True)
    parser.add_argument("--w9-checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preconfirm = _validate_preconfirm(args.preconfirm_freeze)
    w9_freeze, w9_row, hira, scorer = _load_w9_base(
        args.w9_freeze, args.w9_checkpoints
    )
    cache_model = _load_a13_model()

    print("R8_W17_CONFIRM_CP_GENERATION_BEGIN", flush=True)
    cp_rows = generate_w17_confirm("CP", allow_confirm=True)
    cp_cache = compile_w17_cache(
        cache_model, cp_rows, expected_split_prefix="confirm-cp"
    )
    cp_metrics = evaluate_w17(hira, scorer, cp_cache)
    cp_reference, reference_sha = _reference_for_cache(cp_cache)

    print("R8_W17_CONFIRM_CQ_GENERATION_BEGIN", flush=True)
    cq_rows = generate_w17_confirm("CQ", allow_confirm=True)
    cq_cache = compile_w17_cache(
        cache_model, cq_rows, expected_split_prefix="confirm-cq"
    )
    cq_metrics = evaluate_w17(hira, scorer, cq_cache)
    cq_reference, reference_sha_cq = _reference_for_cache(cq_cache)
    if reference_sha_cq != reference_sha:
        raise RuntimeError("W17 reference SHA changed across confirm domains")

    verdict, verdict_details = w17_verdict(
        cp_metrics, cq_metrics, cp_reference, cq_reference
    )

    args.out.mkdir(parents=True, exist_ok=True)
    cp_path = save_w17_cache(cp_cache, args.out / "confirm-cp.pt")
    cq_path = save_w17_cache(cq_cache, args.out / "confirm-cq.pt")

    result = {
        "schema_version": "r8-w17-confirm-v1",
        "status": "PASS",
        "verdict": verdict,
        "verdict_details": verdict_details,
        "training_performed": False,
        "selection_performed": False,
        "trainable_parameter_count": 0,
        "confirm_generated_after_preconfirm_freeze": True,
        "confirm_domains": ["CP", "CQ"],
        "confirm_cp_case_count": len(cp_rows),
        "confirm_cq_case_count": len(cq_rows),
        "confirm_cp_decision_count": 5 * len(cp_rows),
        "confirm_cq_decision_count": 5 * len(cq_rows),
        "confirm_cp_cache_sha256": file_sha256(cp_path),
        "confirm_cq_cache_sha256": file_sha256(cq_path),
        "paths": {
            "CP": cp_metrics,
            "CQ": cq_metrics,
        },
        "reference": {
            "CP": cp_reference,
            "CQ": cq_reference,
        },
        "reference_model": REFERENCE_MODEL,
        "reference_revision": REFERENCE_REVISION,
        "reference_weight_sha256": reference_sha,
        "reference_is_hira_candidate": False,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "w9_hira_sha256": W9_HIRA_SHA256,
        "w9_scorer_sha256": W9_SCORER_SHA256,
        "w9_freeze_status": w9_freeze["status"],
        "w9_projection_scorer_sha256": w9_row["scorer_sha256"],
        "preconfirm_freeze_status": preconfirm["status"],
        "w15_rows_used": False,
        "w16_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "confirm.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
