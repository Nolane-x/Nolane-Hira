from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.competitive import CompetitiveCoarseScorer
from nmd.continuous_reliability import (
    DOMAINS,
    classify_domain,
    cross_domain_outcome,
)
from nmd.continuous_reliability_cache import load_w14_cache
from nmd.continuous_reliability_eval import (
    evaluate_continuous_reliability,
    reference_summary,
)
from nmd.hira import HIRACore
from nmd.typed_competitive_cache import file_sha256

REFERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REFERENCE_WEIGHT_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)


def _load_w9_freeze(freeze_path: Path, checkpoints: Path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w9-freeze-v1":
        raise RuntimeError("unexpected W9 freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W9 freeze is not PASS")
    if freeze.get("all_candidates_frozen_before_confirm") is not True:
        raise RuntimeError("W9 candidate freeze boundary invalid")

    rows = {row["candidate"]: row for row in freeze["candidates"]}
    required = {"frozen-w6e-control", "projection-semantic-control"}
    if not required <= set(rows):
        raise RuntimeError("W14 required W9 checkpoint set missing")

    loaded = {}
    for name in required:
        row = rows[name]
        root = checkpoints / row["checkpoint_dir"]
        hira_path = root / "hira.pt"
        scorer_path = root / "scorer.pt"
        if file_sha256(hira_path) != row["hira_sha256"]:
            raise RuntimeError(f"W14 {name} HIRA SHA mismatch")
        if file_sha256(scorer_path) != row["scorer_sha256"]:
            raise RuntimeError(f"W14 {name} scorer SHA mismatch")
        loaded[name] = (row, hira_path, scorer_path)
    return freeze, loaded


@torch.inference_mode()
def _reference_embeddings(model, tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    out: dict[str, torch.Tensor] = {}
    for start in range(0, len(texts), 64):
        batch_text = texts[start : start + 64]
        encoded = tokenizer(
            batch_text,
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
        for text, embedding in zip(batch_text, pooled):
            out[text] = embedding
    return out


@torch.inference_mode()
def _reference_scores(cache: dict) -> tuple[dict[str, list[float]], str]:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(repo_id=REFERENCE_MODEL, revision=REFERENCE_REVISION)
    )
    weight = snapshot / "model.safetensors"
    if not weight.exists():
        raise RuntimeError("W14 reference model.safetensors missing")
    weight_sha = file_sha256(weight)
    if weight_sha != REFERENCE_WEIGHT_SHA256:
        raise RuntimeError("W14 pinned reference weight SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    texts: set[str] = set()
    for base in cache["bases"]:
        texts.add(str(base["state_text"]))
        for view in base["views"]:
            if view["view_id"] == "D0":
                texts.update(str(text) for text in view["option_texts"])

    embeddings = _reference_embeddings(model, tokenizer, sorted(texts))
    lookup: dict[str, list[float]] = {}
    for base in cache["bases"]:
        state = embeddings[str(base["state_text"])]
        for view in base["views"]:
            if view["view_id"] != "D0":
                continue
            options = torch.stack(
                [embeddings[str(text)] for text in view["option_texts"]],
                dim=0,
            )
            scores = torch.einsum("d,kd->k", state, options)
            lookup[str(view["case_id"])] = [float(value) for value in scores]
    return lookup, weight_sha


def _domain_metrics(staged: dict, reference: dict, domain: str) -> dict[str, object]:
    return {
        "reference": reference["per_domain"][domain],
        "ensemble": staged["stages"]["E"]["per_domain"][domain],
        "final": staged["stages"]["F"]["per_domain"][domain],
        "tertiles": staged["per_domain"][domain]["tertiles"],
        "guards": staged["per_domain"][domain]["guards"],
        "correlations": staged["per_domain"][domain]["correlations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w14_cache(args.cache)
    freeze, paths = _load_w9_freeze(args.freeze, args.checkpoints)

    frozen_row, frozen_hira_path, _ = paths["frozen-w6e-control"]
    w9_row, _, w9_scorer_path = paths["projection-semantic-control"]

    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(frozen_hira_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    hira.eval()

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.load_state_dict(
        torch.load(w9_scorer_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    scorer.eval()

    staged = evaluate_continuous_reliability(hira, scorer, cache)
    reference_lookup, reference_weight_sha = _reference_scores(cache)
    reference = reference_summary(cache, reference_lookup)

    per_domain = {}
    for domain in DOMAINS:
        metrics = _domain_metrics(staged, reference, domain)
        per_domain[domain] = {
            **classify_domain(metrics),
            "metrics": metrics,
            "ensemble_to_final": staged["per_domain"][domain][
                "ensemble_to_final"
            ],
        }

    outcome = cross_domain_outcome(per_domain)

    result = {
        "schema_version": "r8-w14-continuous-reliability-v1",
        "status": "PASS",
        "scope": "diagnostic only; no training",
        "training_performed": False,
        "stages": staged["stages"],
        "guards": staged["guards"],
        "per_domain": staged["per_domain"],
        "reliability_meta": staged["reliability_meta"],
        "reference": reference,
        "per_domain_classification": per_domain,
        "outcome": outcome["outcome"],
        "stable_classification": outcome["stable_classification"],
        "classification_counts": outcome["counts"],
        "reference_model": REFERENCE_MODEL,
        "reference_revision": REFERENCE_REVISION,
        "reference_weight_sha256": reference_weight_sha,
        "reference_is_hira_candidate": False,
        "w9_projection_scorer_sha256": w9_row["scorer_sha256"],
        "hira_sha256": frozen_row["hira_sha256"],
        "probability_mass_max_error": staged["probability_mass_max_error"],
        "state_encodes_per_base": staged["state_encodes_per_base"],
        "cached_base_count": staged["cached_base_count"],
        "cached_view_count": staged["cached_view_count"],
        "w8_diagnostic_rows_used": False,
        "w9_confirm_rows_used": False,
        "w10_diagnostic_rows_used": False,
        "w11_diagnostic_rows_used": False,
        "w12_diagnostic_rows_used": False,
        "w13_diagnostic_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
        "w9_freeze_status": freeze["status"],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
