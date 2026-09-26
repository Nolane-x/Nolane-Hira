from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.competitive import CompetitiveCoarseScorer
from nmd.regime_transfer_authority import DOMAINS
from nmd.regime_transfer_cache import load_w16_cache
from nmd.regime_transfer_eval import (
    classify_domain,
    cross_domain_outcome,
    evaluate_regime_transfer,
    reference_summary,
)
from nmd.typed_competitive_cache import file_sha256

REFERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REFERENCE_WEIGHT_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
W9_SCORER_SHA256 = (
    "8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102"
)


def _load_w9_projection(freeze_path: Path, checkpoints: Path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w9-freeze-v1":
        raise RuntimeError("unexpected W9 freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W9 freeze is not PASS")
    rows = {row["candidate"]: row for row in freeze["candidates"]}
    row = rows.get("projection-semantic-control")
    if row is None:
        raise RuntimeError("W16 W9 projection semantic checkpoint missing")
    root = checkpoints / row["checkpoint_dir"]
    scorer_path = root / "scorer.pt"
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W16 W9 scorer SHA mismatch")
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.load_state_dict(
        torch.load(scorer_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    scorer.eval()
    return freeze, row, scorer


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
        raise RuntimeError("W16 reference model.safetensors missing")
    weight_sha = file_sha256(weight)
    if weight_sha != REFERENCE_WEIGHT_SHA256:
        raise RuntimeError("W16 pinned reference weight SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    texts: set[str] = set()
    for base in cache["bases"]:
        texts.add(str(base["bare_state_text"]))
        texts.add(str(base["decorated_state_text"]))
        for view in base["views"]:
            if view["view_id"] == "D0":
                texts.update(str(text) for text in view["option_texts"])

    embeddings = _reference_embeddings(model, tokenizer, sorted(texts))
    lookup: dict[str, list[float]] = {}
    for base in cache["bases"]:
        states = {
            "R0": embeddings[str(base["bare_state_text"])],
            "R1": embeddings[str(base["decorated_state_text"])],
        }
        for view in base["views"]:
            if view["view_id"] != "D0":
                continue
            options = torch.stack(
                [embeddings[str(text)] for text in view["option_texts"]],
                dim=0,
            )
            for rendering, state in states.items():
                scores = torch.einsum("d,kd->k", state, options)
                lookup[f"{rendering}|{view['case_id']}"] = [
                    float(value) for value in scores
                ]
    return lookup, weight_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w16_cache(args.cache)
    freeze, row, scorer = _load_w9_projection(args.freeze, args.checkpoints)
    staged = evaluate_regime_transfer(scorer, cache)

    reference_lookup, reference_weight_sha = _reference_scores(cache)
    reference = reference_summary(cache, reference_lookup)

    per_domain_classification = {}
    for domain in DOMAINS:
        per_domain_classification[domain] = classify_domain(
            staged["metrics"]["per_domain"][domain],
            reference["per_domain"][domain],
        )
    outcome = cross_domain_outcome(per_domain_classification)

    result = {
        "schema_version": "r8-w16-regime-transfer-v1",
        "status": "PASS",
        "scope": "diagnostic only; no training",
        "training_performed": False,
        "metrics": staged["metrics"],
        "transitions": staged["transitions"],
        "directional_attribution": staged["directional_attribution"],
        "token_accounting": staged["token_accounting"],
        "reference": reference,
        "per_domain_classification": per_domain_classification,
        "outcome": outcome["outcome"],
        "stable_classification": outcome["stable_classification"],
        "classification_counts": outcome["counts"],
        "reference_model": REFERENCE_MODEL,
        "reference_revision": REFERENCE_REVISION,
        "reference_weight_sha256": reference_weight_sha,
        "reference_is_hira_candidate": False,
        "w9_projection_scorer_sha256": row["scorer_sha256"],
        "state_encoder_batches_per_base": staged["state_encoder_batches_per_base"],
        "encoded_state_texts_per_base": staged["encoded_state_texts_per_base"],
        "r2_additional_encoder_calls": staged["r2_additional_encoder_calls"],
        "prefix_token_identity_rate": staged["prefix_token_identity_rate"],
        "cached_base_count": staged["cached_base_count"],
        "cached_view_count": staged["cached_view_count"],
        "w14_rows_used": False,
        "w15_rows_used": False,
        "w15_ce_cf_rows_used": False,
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
