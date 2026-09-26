from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.competitive import CompetitiveCoarseScorer
from nmd.latent_composition_cache import load_w18_cache
from nmd.latent_composition_eval import (
    classify_w18,
    evaluate_w18,
    reference_summary_from_scores,
)
from nmd.typed_competitive_cache import file_sha256

W9_HIRA_SHA256 = "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
W9_SCORER_SHA256 = "8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102"
REFERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REFERENCE_WEIGHT_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"


def _load_scorer(freeze_path: Path, checkpoints: Path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w9-freeze-v1" or freeze.get("status") != "PASS":
        raise RuntimeError("unexpected W9 freeze")
    rows = {row["candidate"]: row for row in freeze["candidates"]}
    row = rows.get("projection-semantic-control")
    if row is None:
        raise RuntimeError("W18 projection-semantic-control missing")
    root = checkpoints / row["checkpoint_dir"]
    hira_path = root / "hira.pt"
    scorer_path = root / "scorer.pt"
    if file_sha256(hira_path) != W9_HIRA_SHA256:
        raise RuntimeError("W18 frozen HIRA SHA mismatch")
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W18 frozen scorer SHA mismatch")

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.load_state_dict(
        torch.load(scorer_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return freeze, row, scorer


@torch.inference_mode()
def _embeddings(model, tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    out = {}
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
def _reference(cache: dict):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=REFERENCE_MODEL, revision=REFERENCE_REVISION))
    weight = snapshot / "model.safetensors"
    if not weight.exists() or file_sha256(weight) != REFERENCE_WEIGHT_SHA256:
        raise RuntimeError("W18 pinned MiniLM SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    texts = set()
    for case in cache["cases"]:
        texts.update(str(case["fields"][name]) for name in ("intent", "severity", "confidence"))
        diagnosis = next(row for row in case["decisions"] if row["question_id"] == "diagnosis")
        texts.update(str(x) for x in diagnosis["views"]["D0"]["option_texts"])
        texts.update(str(x) for x in case["latent_views"]["severity"]["D0"]["option_texts"])
        texts.update(str(x) for x in case["latent_views"]["confidence"]["D0"]["option_texts"])
    emb = _embeddings(model, tokenizer, sorted(texts))

    lookup = {}
    for case in cache["cases"]:
        diagnosis = next(row for row in case["decisions"] if row["question_id"] == "diagnosis")
        def scores(field, option_texts):
            state = emb[str(case["fields"][field])]
            options = torch.stack([emb[str(text)] for text in option_texts], dim=0)
            return [float(x) for x in torch.einsum("d,kd->k", state, options)]
        lookup[str(case["case_id"])] = {
            "intent": scores("intent", diagnosis["views"]["D0"]["option_texts"]),
            "severity": scores(
                "severity", case["latent_views"]["severity"]["D0"]["option_texts"]
            ),
            "confidence": scores(
                "confidence", case["latent_views"]["confidence"]["D0"]["option_texts"]
            ),
        }
    return reference_summary_from_scores(cache["cases"], lookup), file_sha256(weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--w9-freeze", type=Path, required=True)
    parser.add_argument("--w9-checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w18_cache(args.cache)
    freeze, row, scorer = _load_scorer(args.w9_freeze, args.w9_checkpoints)
    metrics = evaluate_w18(scorer, cache)
    reference, reference_sha = _reference(cache)
    classification = classify_w18(metrics, reference)

    oracle_acc = {
        domain: metrics["per_domain"][domain]["paths"]["ORACLE_LATENT_COMPOSED"]["accuracy"]
        for domain in sorted(metrics["per_domain"])
    }
    if any(float(value) != 1.0 for value in oracle_acc.values()):
        raise RuntimeError(f"W18 oracle sanity failed: {oracle_acc}")

    result = {
        "schema_version": "r8-w18-audit-v1",
        "status": "PASS",
        **classification,
        "metrics": metrics,
        "reference": reference,
        "oracle_accuracy_per_domain": oracle_acc,
        "training_performed": False,
        "selection_performed": False,
        "trainable_parameter_count": 0,
        "reference_model": REFERENCE_MODEL,
        "reference_revision": REFERENCE_REVISION,
        "reference_weight_sha256": reference_sha,
        "reference_is_hira_candidate": False,
        "w9_freeze_status": freeze["status"],
        "w9_projection_scorer_sha256": row["scorer_sha256"],
        "w9_hira_sha256": W9_HIRA_SHA256,
        "w9_scorer_sha256": W9_SCORER_SHA256,
        "w17_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
