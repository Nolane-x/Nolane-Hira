from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from nmd.atomic_geometry_cache import load_w25_cache
from nmd.atomic_geometry_eval import (
    AtomicFactorProbe,
    classify_w25,
    evaluate_probe,
    evaluate_raw_a13,
    evaluate_semantic_projection,
    reference_evaluation_from_scores,
)
from nmd.typed_competitive_cache import file_sha256

W9_HIRA_SHA256 = "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
W9_SCORER_SHA256 = "8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102"

REFERENCE = {
    "repo": "cross-encoder/nli-deberta-v3-base",
    "revision": "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
    "weight_sha256": "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa",
}


def _load_w9_projection(freeze_path: Path, checkpoints: Path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w9-freeze-v1" or freeze.get("status") != "PASS":
        raise RuntimeError("W25 unexpected W9 freeze")
    rows = {row["candidate"]: row for row in freeze["candidates"]}
    row = rows.get("projection-semantic-control")
    if row is None:
        raise RuntimeError("W25 W9 projection-semantic-control missing")
    root = checkpoints / row["checkpoint_dir"]
    hira_path = root / "hira.pt"
    scorer_path = root / "scorer.pt"
    if file_sha256(hira_path) != W9_HIRA_SHA256:
        raise RuntimeError("W25 frozen HIRA SHA mismatch")
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W25 frozen scorer SHA mismatch")
    state = torch.load(scorer_path, map_location="cpu", weights_only=True)
    projection = state["projection.weight"].detach().float().clone()
    if tuple(projection.shape) != (128, 256):
        raise RuntimeError("W25 W9 projection shape changed")
    return freeze, row, projection


def _load_candidate(directory: Path, candidate: str):
    receipt_path = directory / "receipt.json"
    checkpoint_path = directory / "candidate.pt"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w25-candidate-receipt-v1":
        raise RuntimeError(f"W25 {candidate} receipt schema mismatch")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != candidate:
        raise RuntimeError(f"W25 {candidate} receipt identity mismatch")
    if receipt.get("confirm_exposed") is not False:
        raise RuntimeError(f"W25 {candidate} was not frozen before CONFIRM")
    if file_sha256(checkpoint_path) != receipt.get("checkpoint_sha256"):
        raise RuntimeError(f"W25 {candidate} checkpoint SHA mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "r8-w25-candidate-checkpoint-v1":
        raise RuntimeError(f"W25 {candidate} checkpoint schema mismatch")
    if checkpoint.get("candidate") != candidate:
        raise RuntimeError(f"W25 {candidate} checkpoint identity mismatch")
    return receipt, checkpoint


def _entailment_index(config) -> int:
    id2label = getattr(config, "id2label", {}) or {}
    for key, value in id2label.items():
        if "entail" in str(value).lower():
            return int(key)
    label2id = getattr(config, "label2id", {}) or {}
    for key, value in label2id.items():
        if "entail" in str(key).lower():
            return int(value)
    raise RuntimeError("W25 reference config has no identifiable entailment label")


def _load_reference():
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=REFERENCE["repo"], revision=REFERENCE["revision"]))
    weight = snapshot / "model.safetensors"
    if not weight.exists():
        raise RuntimeError("W25 reference model.safetensors missing")
    actual_sha = file_sha256(weight)
    if actual_sha != REFERENCE["weight_sha256"]:
        raise RuntimeError(f"W25 reference weight SHA mismatch: {actual_sha}")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot), local_files_only=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, actual_sha, _entailment_index(model.config)


@torch.inference_mode()
def _pair_entailment_scores(model, tokenizer, pairs, entailment_index: int):
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(pairs), 64):
        batch = pairs[start : start + 64]
        encoded = tokenizer(
            [a for a, _ in batch],
            [b for _, b in batch],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded, return_dict=True).logits
        probs = torch.softmax(logits.float(), dim=-1)[:, entailment_index]
        values.extend(float(x) for x in probs.cpu())
    return values


@torch.inference_mode()
def _reference_scores(cache: dict, model, tokenizer, entailment_index: int):
    lookup = {
        str(case["case_id"]): {"F0": [0.0, 0.0], "F1": [0.0, 0.0], "F2": [0.0, 0.0]}
        for case in cache["cases"]
    }
    requests = []
    for case in cache["cases"]:
        case_id = str(case["case_id"])
        domain = str(case["domain_id"])
        query = str(case["severity_field"])
        hypotheses = cache["schemas"][domain]["factor_reference_hypotheses"]
        for factor_id in ("F0", "F1", "F2"):
            for value, hypothesis in enumerate(hypotheses[factor_id]):
                requests.append((case_id, factor_id, value, query, str(hypothesis)))
    scores = _pair_entailment_scores(
        model,
        tokenizer,
        [(query, hypothesis) for _, _, _, query, hypothesis in requests],
        entailment_index,
    )
    for request, score in zip(requests, scores):
        case_id, factor_id, value, _, _ = request
        lookup[case_id][factor_id][value] = score
    return lookup


def _evaluate_probe_checkpoint(checkpoint: dict, cache: dict, w9_projection: torch.Tensor | None):
    d_in = int(checkpoint["input_dim"])
    probe = AtomicFactorProbe(d_in)
    probe.load_state_dict(checkpoint["state_dict"], strict=True)
    probe.eval()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    return evaluate_probe(probe, cache, w9_projection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-cache", type=Path, required=True)
    parser.add_argument("--w9-freeze", type=Path, required=True)
    parser.add_argument("--w9-checkpoints", type=Path, required=True)
    parser.add_argument("--q0-dir", type=Path, required=True)
    parser.add_argument("--q1-dir", type=Path, required=True)
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--t1-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w25_cache(args.confirm_cache, expected_partition="confirm")
    if set(cache["metadata"]["domains"]) != {"DY", "DZ"}:
        raise RuntimeError("W25 CONFIRM domains changed")

    freeze, w9_row, w9_projection = _load_w9_projection(
        args.w9_freeze, args.w9_checkpoints
    )

    receipts = {}
    checkpoints = {}
    for name, directory in {
        "Q0": args.q0_dir,
        "Q1": args.q1_dir,
        "T0": args.t0_dir,
        "T1": args.t1_dir,
    }.items():
        receipt, checkpoint = _load_candidate(directory, name)
        receipts[name] = receipt
        checkpoints[name] = checkpoint

    evaluations = {
        "A0": evaluate_raw_a13(cache),
        "P0": evaluate_semantic_projection(cache, w9_projection),
        "Q0": _evaluate_probe_checkpoint(checkpoints["Q0"], cache, None),
        "Q1": _evaluate_probe_checkpoint(checkpoints["Q1"], cache, w9_projection),
        "T0": evaluate_semantic_projection(
            cache, checkpoints["T0"]["projection_weight"].float()
        ),
        "T1": evaluate_semantic_projection(
            cache, checkpoints["T1"]["projection_weight"].float()
        ),
    }

    model, tokenizer, actual_sha, entailment_index = _load_reference()
    score_lookup = _reference_scores(cache, model, tokenizer, entailment_index)
    reference = reference_evaluation_from_scores(cache["cases"], score_lookup)
    del model
    gc.collect()

    classification = classify_w25(evaluations, reference)

    result = {
        "schema_version": "r8-w25-atomic-geometry-audit-v1",
        "status": "PASS",
        **classification,
        "evaluations": evaluations,
        "reference": reference,
        "candidate_receipts": receipts,
        "reference_model": REFERENCE,
        "actual_reference_weight_sha256": actual_sha,
        "reference_entailment_label_index": entailment_index,
        "reference_scoring_direction": "premise=query; hypothesis=factor-option; entailment-probability",
        "w9_freeze_status": freeze["status"],
        "w9_projection_scorer_sha256": w9_row["scorer_sha256"],
        "w9_hira_sha256": W9_HIRA_SHA256,
        "w9_scorer_sha256": W9_SCORER_SHA256,
        "confirm_domains": ["DY", "DZ"],
        "confirm_materialized_after_all_candidate_freezes": True,
        "a13_frozen_all_paths": True,
        "reference_supplied_training_targets": False,
        "w24_rows_used": False,
        "w23_rows_used": False,
        "w22_rows_used": False,
        "w21_rows_used": False,
        "w20_rows_used": False,
        "w19_rows_used": False,
        "w18_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
