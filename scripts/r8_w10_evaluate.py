from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.representation_ceiling import classify_domain, cross_domain_outcome
from nmd.representation_ceiling_cache import load_w10_cache
from nmd.representation_ceiling_eval import (
    evaluate_hira_representation_operators,
    make_reference_records,
    summarize_operator,
)
from nmd.typed_competitive_cache import file_sha256

REFERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


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
        raise RuntimeError("W10 required W9 checkpoint set missing")

    loaded = {}
    for name in required:
        row = rows[name]
        root = checkpoints / row["checkpoint_dir"]
        hira_path = root / "hira.pt"
        scorer_path = root / "scorer.pt"
        if file_sha256(hira_path) != row["hira_sha256"]:
            raise RuntimeError(f"W10 {name} HIRA SHA mismatch")
        if file_sha256(scorer_path) != row["scorer_sha256"]:
            raise RuntimeError(f"W10 {name} scorer SHA mismatch")
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
        encoded = {k: v.to(device) for k, v in encoded.items()}
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
        snapshot_download(
            repo_id=REFERENCE_MODEL,
            revision=REFERENCE_REVISION,
        )
    )
    weight = snapshot / "model.safetensors"
    if not weight.exists():
        raise RuntimeError("W10 reference model.safetensors missing")
    weight_sha = file_sha256(weight)

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    texts: set[str] = set()
    for base in cache["bases"]:
        texts.add(str(base["state_text"]))
        for view in base["views"]:
            texts.update(str(x) for x in view["option_texts"])
    embeddings = _reference_embeddings(model, tokenizer, sorted(texts))

    lookup: dict[str, list[float]] = {}
    for base in cache["bases"]:
        state = embeddings[str(base["state_text"])]
        for view in base["views"]:
            options = torch.stack(
                [embeddings[str(text)] for text in view["option_texts"]],
                dim=0,
            )
            scores = torch.einsum("d,kd->k", state, options)
            lookup[str(view["case_id"])] = [float(x) for x in scores]
    return lookup, weight_sha


def _metric(summary: dict, domain: str, k: int) -> float:
    return float(
        summary["per_domain"][domain]["views"]["definition"][str(k)]["top1"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w10_cache(args.cache)
    freeze, paths = _load_w9_freeze(args.freeze, args.checkpoints)

    frozen_row, frozen_hira_path, frozen_scorer_path = paths["frozen-w6e-control"]
    w9_row, _, w9_scorer_path = paths["projection-semantic-control"]

    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(frozen_hira_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    hira.eval()

    w6e_scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    w6e_scorer.load_state_dict(
        torch.load(frozen_scorer_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    w6e_scorer.eval()

    w9_scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    w9_scorer.load_state_dict(
        torch.load(w9_scorer_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    w9_scorer.eval()

    hira_results = evaluate_hira_representation_operators(
        hira,
        w6e_scorer,
        w9_scorer,
        cache,
    )

    reference_lookup, reference_weight_sha = _reference_scores(cache)
    r0_records = make_reference_records(cache, reference_lookup)
    r0_summary = summarize_operator(r0_records)

    operators = dict(hira_results["operators"])
    operators["R0"] = r0_summary

    domain_classification = {}
    for domain in ("BF", "BG", "BH", "BI"):
        metrics = {
            "a0_k4_top1": _metric(operators["A0"], domain, 4),
            "a1_k4_top1": _metric(operators["A1"], domain, 4),
            "a1_k16_top1": _metric(operators["A1"], domain, 16),
            "p0_k4_top1": _metric(operators["P0"], domain, 4),
            "p1_k4_top1": _metric(operators["P1"], domain, 4),
            "p1_k16_top1": _metric(operators["P1"], domain, 16),
            "s1_k4_top1": _metric(operators["S1"], domain, 4),
            "s1_k16_top1": _metric(operators["S1"], domain, 16),
            "r0_k4_top1": _metric(operators["R0"], domain, 4),
        }
        domain_classification[domain] = {
            **classify_domain(metrics),
            "metrics": metrics,
        }

    outcome = cross_domain_outcome(domain_classification)

    result = {
        "schema_version": "r8-w10-representation-ceiling-v1",
        "status": "PASS",
        "scope": "diagnostic only; no training",
        "training_performed": False,
        "operators": operators,
        "per_domain_classification": domain_classification,
        "outcome": outcome["outcome"],
        "stable_classification": outcome["stable_classification"],
        "classification_counts": outcome["counts"],
        "reference_ambiguous_domains": outcome["reference_ambiguous_domains"],
        "reference_model": REFERENCE_MODEL,
        "reference_revision": REFERENCE_REVISION,
        "reference_weight_sha256": reference_weight_sha,
        "reference_is_hira_candidate": False,
        "w6e_scorer_sha256": frozen_row["scorer_sha256"],
        "w9_projection_scorer_sha256": w9_row["scorer_sha256"],
        "hira_sha256": frozen_row["hira_sha256"],
        "probability_mass_max_error": hira_results["probability_mass_max_error"],
        "state_encodes_per_base": hira_results["state_encodes_per_base"],
        "cached_base_count": hira_results["cached_base_count"],
        "cached_view_count": hira_results["cached_view_count"],
        "w8_diagnostic_rows_used": False,
        "w9_confirm_rows_used": False,
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
