from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.interface_decomposition import classify_domain, cross_domain_outcome
from nmd.interface_decomposition_cache import load_w11_cache
from nmd.interface_decomposition_eval import (
    evaluate_interface_stages,
    make_reference_records,
    summarize_reference,
)
from nmd.typed_competitive_cache import file_sha256

REFERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REFERENCE_WEIGHT_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"


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
        raise RuntimeError("W11 required W9 checkpoint set missing")

    loaded = {}
    for name in required:
        row = rows[name]
        root = checkpoints / row["checkpoint_dir"]
        hira_path = root / "hira.pt"
        scorer_path = root / "scorer.pt"
        if file_sha256(hira_path) != row["hira_sha256"]:
            raise RuntimeError(f"W11 {name} HIRA SHA mismatch")
        if file_sha256(scorer_path) != row["scorer_sha256"]:
            raise RuntimeError(f"W11 {name} scorer SHA mismatch")
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
        raise RuntimeError("W11 reference model.safetensors missing")
    weight_sha = file_sha256(weight)
    if weight_sha != REFERENCE_WEIGHT_SHA256:
        raise RuntimeError("W11 pinned reference weight SHA mismatch")

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


def _definition_cell(summary: dict, domain: str, k: int) -> dict:
    return summary["per_domain"][domain]["views"]["definition"][str(k)]


def _domain_metrics(
    stages: dict,
    transitions: dict,
    reference: dict,
    domain: str,
) -> dict[str, object]:
    stage_metrics = {}
    for stage, summary in stages.items():
        stage_metrics[stage] = {
            "4": _definition_cell(summary, domain, 4),
            "8": _definition_cell(summary, domain, 8),
            "16": _definition_cell(summary, domain, 16),
        }
    ref = {
        "4": _definition_cell(reference, domain, 4),
        "8": _definition_cell(reference, domain, 8),
        "16": _definition_cell(reference, domain, 16),
    }
    return {
        "stages": stage_metrics,
        "transitions": transitions[domain],
        "reference": ref,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w11_cache(args.cache)
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

    staged = evaluate_interface_stages(hira, scorer, cache)

    for k, row in staged["q5_q6_equivalence"].items():
        if float(row["top1_identity_rate"]) != 1.0:
            raise RuntimeError(f"W11 Q5/Q6 top1 mismatch at K{k}")
        if float(row["full_rank_order_identity_rate"]) != 1.0:
            raise RuntimeError(f"W11 Q5/Q6 rank-order mismatch at K{k}")

    reference_lookup, reference_weight_sha = _reference_scores(cache)
    reference_records = make_reference_records(cache, reference_lookup)
    reference = summarize_reference(reference_records)

    per_domain = {}
    for domain in ("BJ", "BK", "BL", "BM"):
        metrics = _domain_metrics(
            staged["stages"],
            staged["per_domain_transitions"],
            reference,
            domain,
        )
        per_domain[domain] = {
            **classify_domain(metrics),
            "metrics": metrics,
            "first_events": staged["first_events"][domain],
        }

    outcome = cross_domain_outcome(per_domain)

    result = {
        "schema_version": "r8-w11-interface-decomposition-v1",
        "status": "PASS",
        "scope": "diagnostic only; no training",
        "training_performed": False,
        "stages": staged["stages"],
        "transitions": staged["transitions"],
        "per_domain_transitions": staged["per_domain_transitions"],
        "first_events": staged["first_events"],
        "q5_q6_equivalence": staged["q5_q6_equivalence"],
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
