from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.competitive import CompetitiveCoarseScorer
from nmd.cross_encoder_cache import load_w23_cache
from nmd.cross_encoder_eval import (
    classify_w23,
    evaluate_w23,
    prototype_reference_evaluation_from_scores,
    reference_evaluation_from_scores,
)
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

BIENCODER_PANEL = {
    "minilm": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "weight_sha256": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        "pooling": "mean",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "mpnet": {
        "repo": "sentence-transformers/all-mpnet-base-v2",
        "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "weight_sha256": "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658",
        "pooling": "mean",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5": {
        "repo": "intfloat/e5-small-v2",
        "revision": "93da57dece4e396a19773b2658fe3ffdd358e5eb",
        "weight_sha256": "45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1",
        "pooling": "mean",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "bge": {
        "repo": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "weight_sha256": "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
        "pooling": "cls",
        "query_prefix": "",
        "passage_prefix": "",
    },
}


def _load_scorer(freeze_path: Path, checkpoints: Path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w9-freeze-v1" or freeze.get("status") != "PASS":
        raise RuntimeError("unexpected W9 freeze")
    rows = {row["candidate"]: row for row in freeze["candidates"]}
    row = rows.get("projection-semantic-control")
    if row is None:
        raise RuntimeError("W23 projection-semantic-control missing")
    root = checkpoints / row["checkpoint_dir"]
    hira_path = root / "hira.pt"
    scorer_path = root / "scorer.pt"
    if file_sha256(hira_path) != W9_HIRA_SHA256:
        raise RuntimeError("W23 frozen HIRA SHA mismatch")
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W23 frozen scorer SHA mismatch")

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
    raise RuntimeError("W23 cross-encoder config has no identifiable entailment label")


@torch.inference_mode()
def _pair_entailment_scores(model, tokenizer, pairs: list[tuple[str, str]], entailment_index: int) -> list[float]:
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
def _bidirectional_entailment_scores(model, tokenizer, query: str, options: list[str], entailment_index: int) -> list[float]:
    forward = [(query, option) for option in options]
    reverse = [(option, query) for option in options]
    forward_scores = _pair_entailment_scores(model, tokenizer, forward, entailment_index)
    reverse_scores = _pair_entailment_scores(model, tokenizer, reverse, entailment_index)
    return [(a + b) * 0.5 for a, b in zip(forward_scores, reverse_scores)]


def _cross_reference_scores(cache: dict, model, tokenizer, entailment_index: int):
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


@torch.inference_mode()
def _evaluate_cross_encoder(cache: dict, name: str, spec: dict[str, str]):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=spec["repo"], revision=spec["revision"]))
    weight = snapshot / "model.safetensors"
    if not weight.exists():
        raise RuntimeError(f"W23 {name} model.safetensors missing")
    actual_sha = file_sha256(weight)
    if actual_sha != spec["weight_sha256"]:
        raise RuntimeError(f"W23 {name} pinned weight SHA mismatch: {actual_sha}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot), local_files_only=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    entailment_index = _entailment_index(model.config)

    lookup = _cross_reference_scores(cache, model, tokenizer, entailment_index)
    evaluation = prototype_reference_evaluation_from_scores(cache["cases"], lookup)

    del model
    gc.collect()
    return evaluation, actual_sha, entailment_index


@torch.inference_mode()
def _encode_texts(model, tokenizer, texts: list[str], *, prefix: str, pooling: str):
    device = next(model.parameters()).device
    out: dict[str, torch.Tensor] = {}
    for start in range(0, len(texts), 64):
        originals = texts[start : start + 64]
        batch = [prefix + text for text in originals]
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
        if pooling == "cls":
            pooled = tokens[:, 0]
        elif pooling == "mean":
            mask = encoded["attention_mask"].to(tokens.dtype)[..., None]
            pooled = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
        else:
            raise ValueError(f"unsupported W23 pooling: {pooling}")
        pooled = F.normalize(pooled, dim=-1).cpu()
        for text, vector in zip(originals, pooled):
            out[text] = vector
    return out


@torch.inference_mode()
def _evaluate_biencoder(cache: dict, name: str, spec: dict[str, str]):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=spec["repo"], revision=spec["revision"]))
    weight = snapshot / "model.safetensors"
    if not weight.exists():
        raise RuntimeError(f"W23 {name} model.safetensors missing")
    actual_sha = file_sha256(weight)
    if actual_sha != spec["weight_sha256"]:
        raise RuntimeError(f"W23 {name} pinned weight SHA mismatch: {actual_sha}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    query_texts = set()
    passage_texts = set()
    for case in cache["cases"]:
        query_texts.update(str(case["fields"][field]) for field in ("severity", "confidence"))
    for pack in cache["schemas"].values():
        for view_id in ("D0", "D1", "D2"):
            passage_texts.update(str(x) for x in pack["abstract_severity"][view_id]["option_texts"])
            passage_texts.update(str(x) for x in pack["abstract_confidence"][view_id]["option_texts"])
        passage_texts.update(str(x) for x in pack["severity_prototypes"]["option_texts"])
        passage_texts.update(str(x) for x in pack["confidence_prototypes"]["option_texts"])

    query_emb = _encode_texts(
        model, tokenizer, sorted(query_texts),
        prefix=spec["query_prefix"], pooling=spec["pooling"]
    )
    passage_emb = _encode_texts(
        model, tokenizer, sorted(passage_texts),
        prefix=spec["passage_prefix"], pooling=spec["pooling"]
    )

    def scores(state_text: str, option_texts) -> list[float]:
        state = query_emb[str(state_text)]
        options = torch.stack([passage_emb[str(text)] for text in option_texts], dim=0)
        logits = torch.einsum("d,kd->k", state, options)
        return [float(x) for x in logits]

    def multiview_scores(state_text: str, views) -> list[float]:
        logits = torch.stack(
            [
                torch.tensor(scores(state_text, views[view_id]["option_texts"]), dtype=torch.float32)
                for view_id in ("D0", "D1", "D2")
            ],
            dim=0,
        ).mean(dim=0)
        return [float(x) for x in logits]

    lookup = {}
    for case in cache["cases"]:
        domain = str(case["domain_id"])
        pack = cache["schemas"][domain]
        lookup[str(case["case_id"])] = {
            "abstract_severity": multiview_scores(
                str(case["fields"]["severity"]), pack["abstract_severity"]
            ),
            "abstract_confidence": multiview_scores(
                str(case["fields"]["confidence"]), pack["abstract_confidence"]
            ),
            "severity_prototypes": scores(
                str(case["fields"]["severity"]), pack["severity_prototypes"]["option_texts"]
            ),
            "confidence_prototypes": scores(
                str(case["fields"]["confidence"]), pack["confidence_prototypes"]["option_texts"]
            ),
        }

    evaluation = reference_evaluation_from_scores(cache["cases"], lookup)
    del model
    gc.collect()
    return evaluation, actual_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--w9-freeze", type=Path, required=True)
    parser.add_argument("--w9-checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w23_cache(args.cache)
    freeze, row, scorer = _load_scorer(args.w9_freeze, args.w9_checkpoints)
    metrics = evaluate_w23(scorer, cache)

    cross_references = {}
    cross_predictions = {}
    cross_hashes = {}
    cross_entailment_indices = {}
    for name, spec in CROSS_ENCODER_PANEL.items():
        evaluation, actual_sha, entailment_index = _evaluate_cross_encoder(cache, name, spec)
        cross_references[name] = evaluation["summary"]
        cross_predictions[name] = evaluation["predictions"]
        cross_hashes[name] = actual_sha
        cross_entailment_indices[name] = entailment_index

    biencoder_references = {}
    biencoder_predictions = {}
    biencoder_hashes = {}
    for name, spec in BIENCODER_PANEL.items():
        evaluation, actual_sha = _evaluate_biencoder(cache, name, spec)
        biencoder_references[name] = evaluation["summary"]
        biencoder_predictions[name] = evaluation["predictions"]
        biencoder_hashes[name] = actual_sha

    classification = classify_w23(
        metrics,
        cross_references,
        cross_predictions,
        biencoder_references,
    )

    result = {
        "schema_version": "r8-w23-cross-encoder-audit-v1",
        "status": "PASS",
        **classification,
        "metrics": metrics,
        "cross_encoder_references": cross_references,
        "cross_encoder_predictions": cross_predictions,
        "cross_encoder_panel": CROSS_ENCODER_PANEL,
        "actual_cross_encoder_weight_sha256": cross_hashes,
        "cross_encoder_entailment_label_index": cross_entailment_indices,
        "biencoder_references": biencoder_references,
        "biencoder_predictions": biencoder_predictions,
        "biencoder_panel": BIENCODER_PANEL,
        "actual_biencoder_weight_sha256": biencoder_hashes,
        "reference_panels_frozen_before_exposure": True,
        "reference_models_are_hira_candidates": False,
        "training_performed": False,
        "selection_performed": False,
        "trainable_parameter_count": 0,
        "w9_freeze_status": freeze["status"],
        "w9_projection_scorer_sha256": row["scorer_sha256"],
        "w9_hira_sha256": W9_HIRA_SHA256,
        "w9_scorer_sha256": W9_SCORER_SHA256,
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
