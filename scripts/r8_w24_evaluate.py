from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from nmd.atomic_severity_eval import (
    atomic_reference_evaluation_from_scores,
    classify_w24,
    evaluate_w24,
)
from nmd.atomic_severity_cache import load_w24_cache
from nmd.competitive import CompetitiveCoarseScorer
from nmd.cross_encoder_eval import prototype_reference_evaluation_from_scores
from nmd.typed_competitive_cache import file_sha256

W9_HIRA_SHA256 = "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
W9_SCORER_SHA256 = "8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102"

CROSS_ENCODER_PANEL = {
    "deberta_nli": {
        "repo": "cross-encoder/nli-deberta-v3-base",
        "revision": "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
        "weight_sha256": "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa",
    },
    "roberta_nli": {
        "repo": "cross-encoder/nli-roberta-base",
        "revision": "1be0567456f0543475805e758725f151f283705a",
        "weight_sha256": "efc90996d2ed80123c26c9091c91385ffddc6d2fd0b2bacf3187fbd6c5b87953",
    },
}


def _load_scorer(freeze_path: Path, checkpoints: Path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w9-freeze-v1" or freeze.get("status") != "PASS":
        raise RuntimeError("unexpected W9 freeze")
    rows = {row["candidate"]: row for row in freeze["candidates"]}
    row = rows.get("projection-semantic-control")
    if row is None:
        raise RuntimeError("W24 projection-semantic-control missing")
    root = checkpoints / row["checkpoint_dir"]
    hira_path = root / "hira.pt"
    scorer_path = root / "scorer.pt"
    if file_sha256(hira_path) != W9_HIRA_SHA256:
        raise RuntimeError("W24 frozen HIRA SHA mismatch")
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W24 frozen scorer SHA mismatch")

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.load_state_dict(
        torch.load(scorer_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return freeze, row, scorer


def _entailment_index(config) -> int:
    id2label = getattr(config, "id2label", {}) or {}
    for key, value in id2label.items():
        if "entail" in str(value).lower():
            return int(key)
    label2id = getattr(config, "label2id", {}) or {}
    for key, value in label2id.items():
        if "entail" in str(key).lower():
            return int(value)
    raise RuntimeError("W24 cross-encoder config has no identifiable entailment label")


def _load_cross_encoder(name: str, spec: dict[str, str]):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=spec["repo"], revision=spec["revision"]))
    weight = snapshot / "model.safetensors"
    if not weight.exists():
        raise RuntimeError(f"W24 {name} model.safetensors missing")
    actual_sha = file_sha256(weight)
    if actual_sha != spec["weight_sha256"]:
        raise RuntimeError(f"W24 {name} pinned weight SHA mismatch: {actual_sha}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot), local_files_only=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, actual_sha, _entailment_index(model.config)


@torch.inference_mode()
def _pair_entailment_scores(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    entailment_index: int,
) -> list[float]:
    device = next(model.parameters()).device
    values: list[float] = []
    for start in range(0, len(pairs), 64):
        batch = pairs[start : start + 64]
        left = [a for a, _ in batch]
        right = [b for _, b in batch]
        encoded = tokenizer(
            left,
            right,
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
def _atomic_factor_scores(cache: dict, model, tokenizer, entailment_index: int):
    lookup = {
        str(case["case_id"]): {"F0": [0.0, 0.0], "F1": [0.0, 0.0], "F2": [0.0, 0.0]}
        for case in cache["cases"]
    }
    requests: list[tuple[str, str, int, str, str]] = []
    for case in cache["cases"]:
        case_id = str(case["case_id"])
        domain = str(case["domain_id"])
        severity_text = str(case["fields"]["severity"])
        hypotheses = cache["schemas"][domain]["factor_reference_hypotheses"]
        for factor_id in ("F0", "F1", "F2"):
            for value, hypothesis in enumerate(hypotheses[factor_id]):
                requests.append(
                    (case_id, factor_id, value, severity_text, str(hypothesis))
                )

    pairs = [(premise, hypothesis) for _, _, _, premise, hypothesis in requests]
    scores = _pair_entailment_scores(model, tokenizer, pairs, entailment_index)
    for request, score in zip(requests, scores):
        case_id, factor_id, value, _, _ = request
        lookup[case_id][factor_id][value] = score
    return lookup


@torch.inference_mode()
def _prototype_control_scores(cache: dict, model, tokenizer, entailment_index: int):
    lookup = {
        str(case["case_id"]): {
            "severity_prototypes": [0.0] * 12,
            "confidence_prototypes": [0.0] * 9,
        }
        for case in cache["cases"]
    }
    requests: list[tuple[str, str, int, str, str]] = []
    for case in cache["cases"]:
        case_id = str(case["case_id"])
        domain = str(case["domain_id"])
        pack = cache["schemas"][domain]
        severity_text = str(case["fields"]["severity"])
        confidence_text = str(case["fields"]["confidence"])
        for index, option in enumerate(pack["severity_prototypes"]["option_texts"]):
            requests.append((case_id, "severity_prototypes", index, severity_text, str(option)))
        for index, option in enumerate(pack["confidence_prototypes"]["option_texts"]):
            requests.append((case_id, "confidence_prototypes", index, confidence_text, str(option)))

    forward_pairs = [(query, option) for _, _, _, query, option in requests]
    reverse_pairs = [(option, query) for _, _, _, query, option in requests]
    forward_scores = _pair_entailment_scores(
        model, tokenizer, forward_pairs, entailment_index
    )
    reverse_scores = _pair_entailment_scores(
        model, tokenizer, reverse_pairs, entailment_index
    )
    for request, forward, reverse in zip(requests, forward_scores, reverse_scores):
        case_id, field, index, _, _ = request
        lookup[case_id][field][index] = (forward + reverse) * 0.5
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--w9-freeze", type=Path, required=True)
    parser.add_argument("--w9-checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w24_cache(args.cache)
    freeze, row, scorer = _load_scorer(args.w9_freeze, args.w9_checkpoints)
    metrics = evaluate_w24(scorer, cache)

    controls = {}
    control_predictions = {}
    actual_hashes = {}
    entailment_indices = {}
    atomic_reference = None

    for name, spec in CROSS_ENCODER_PANEL.items():
        model, tokenizer, actual_sha, entailment_index = _load_cross_encoder(name, spec)
        actual_hashes[name] = actual_sha
        entailment_indices[name] = entailment_index

        if name == "deberta_nli":
            factor_lookup = _atomic_factor_scores(
                cache, model, tokenizer, entailment_index
            )
            atomic_reference = atomic_reference_evaluation_from_scores(
                cache["cases"], factor_lookup
            )

        control_lookup = _prototype_control_scores(
            cache, model, tokenizer, entailment_index
        )
        control_eval = prototype_reference_evaluation_from_scores(
            cache["cases"], control_lookup
        )
        controls[name] = control_eval["summary"]
        control_predictions[name] = control_eval["predictions"]

        del model
        gc.collect()

    if atomic_reference is None:
        raise RuntimeError("W24 primary atomic reference missing")

    classification = classify_w24(
        metrics,
        atomic_reference["summary"],
        controls,
        control_predictions,
    )

    result = {
        "schema_version": "r8-w24-atomic-severity-audit-v1",
        "status": "PASS",
        **classification,
        "metrics": metrics,
        "atomic_reference": atomic_reference["summary"],
        "atomic_reference_predictions": atomic_reference["predictions"],
        "cross_encoder_controls": controls,
        "cross_encoder_control_predictions": control_predictions,
        "cross_encoder_panel": CROSS_ENCODER_PANEL,
        "actual_cross_encoder_weight_sha256": actual_hashes,
        "cross_encoder_entailment_label_index": entailment_indices,
        "factor_scoring_direction": "premise=query; hypothesis=factor-option; entailment-probability",
        "direct_control_scoring": "w23-bidirectional-prototype-entailment",
        "reference_models_are_hira_candidates": False,
        "reference_protocol_frozen_before_exposure": True,
        "training_performed": False,
        "selection_performed": False,
        "trainable_parameter_count": 0,
        "w9_freeze_status": freeze["status"],
        "w9_projection_scorer_sha256": row["scorer_sha256"],
        "w9_hira_sha256": W9_HIRA_SHA256,
        "w9_scorer_sha256": W9_SCORER_SHA256,
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
