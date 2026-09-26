from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.competitive import CompetitiveCoarseScorer
from nmd.prototype_latent_cache import load_w21_cache
from nmd.prototype_latent_eval import (
    classify_w21,
    evaluate_w21,
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
        raise RuntimeError("W21 projection-semantic-control missing")
    root = checkpoints / row["checkpoint_dir"]
    hira_path = root / "hira.pt"
    scorer_path = root / "scorer.pt"
    if file_sha256(hira_path) != W9_HIRA_SHA256:
        raise RuntimeError("W21 frozen HIRA SHA mismatch")
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W21 frozen scorer SHA mismatch")

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
        raise RuntimeError("W21 pinned MiniLM SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    texts = set()
    for case in cache["cases"]:
        texts.update(str(case["fields"][name]) for name in ("severity", "confidence"))
    for pack in cache["schemas"].values():
        for view_id in ("D0", "D1", "D2"):
            texts.update(str(x) for x in pack["abstract_severity"][view_id]["option_texts"])
            texts.update(str(x) for x in pack["abstract_confidence"][view_id]["option_texts"])
        texts.update(str(x) for x in pack["severity_prototypes"]["option_texts"])
        texts.update(str(x) for x in pack["confidence_prototypes"]["option_texts"])
    emb = _embeddings(model, tokenizer, sorted(texts))

    def scores(state_text: str, option_texts) -> list[float]:
        state = emb[str(state_text)]
        options = torch.stack([emb[str(text)] for text in option_texts], dim=0)
        logits = torch.einsum("d,kd->k", state, options)
        return [float(x) for x in logits]

    def multiview_scores(state_text: str, views) -> list[float]:
        logits = torch.stack(
            [
                torch.tensor(
                    scores(state_text, views[view_id]["option_texts"]),
                    dtype=torch.float32,
                )
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
                str(case["fields"]["severity"]),
                pack["abstract_severity"],
            ),
            "abstract_confidence": multiview_scores(
                str(case["fields"]["confidence"]),
                pack["abstract_confidence"],
            ),
            "severity_prototypes": scores(
                str(case["fields"]["severity"]),
                pack["severity_prototypes"]["option_texts"],
            ),
            "confidence_prototypes": scores(
                str(case["fields"]["confidence"]),
                pack["confidence_prototypes"]["option_texts"],
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

    cache = load_w21_cache(args.cache)
    freeze, row, scorer = _load_scorer(args.w9_freeze, args.w9_checkpoints)
    metrics = evaluate_w21(scorer, cache)
    reference, reference_sha = _reference(cache)
    classification = classify_w21(metrics, reference)

    result = {
        "schema_version": "r8-w21-audit-v1",
        "status": "PASS",
        **classification,
        "metrics": metrics,
        "reference": reference,
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
        "w20_rows_used": False,
        "w19_rows_used": False,
        "w18_rows_used": False,
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
