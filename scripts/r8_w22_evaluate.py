from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.competitive import CompetitiveCoarseScorer
from nmd.reference_panel_cache import load_w22_cache
from nmd.reference_panel_eval import (
    classify_w22,
    evaluate_w22,
    reference_evaluation_from_scores,
)
from nmd.typed_competitive_cache import file_sha256

W9_HIRA_SHA256 = "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
W9_SCORER_SHA256 = "8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102"

REFERENCE_PANEL = {
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
        raise RuntimeError("W22 projection-semantic-control missing")
    root = checkpoints / row["checkpoint_dir"]
    hira_path = root / "hira.pt"
    scorer_path = root / "scorer.pt"
    if file_sha256(hira_path) != W9_HIRA_SHA256:
        raise RuntimeError("W22 frozen HIRA SHA mismatch")
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W22 frozen scorer SHA mismatch")

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
def _encode_texts(
    model,
    tokenizer,
    texts: list[str],
    *,
    prefix: str,
    pooling: str,
) -> dict[str, torch.Tensor]:
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
            raise ValueError(f"unsupported W22 pooling: {pooling}")
        pooled = F.normalize(pooled, dim=-1).cpu()
        for text, vector in zip(originals, pooled):
            out[text] = vector
    return out


@torch.inference_mode()
def _evaluate_reference(cache: dict, name: str, spec: dict[str, str]):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(repo_id=spec["repo"], revision=spec["revision"])
    )
    weight = snapshot / "model.safetensors"
    if not weight.exists():
        raise RuntimeError(f"W22 {name} model.safetensors missing")
    actual_sha = file_sha256(weight)
    if actual_sha != spec["weight_sha256"]:
        raise RuntimeError(
            f"W22 {name} pinned weight SHA mismatch: {actual_sha}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    query_texts = set()
    passage_texts = set()
    for case in cache["cases"]:
        query_texts.update(
            str(case["fields"][field]) for field in ("severity", "confidence")
        )
    for pack in cache["schemas"].values():
        for view_id in ("D0", "D1", "D2"):
            passage_texts.update(
                str(x) for x in pack["abstract_severity"][view_id]["option_texts"]
            )
            passage_texts.update(
                str(x) for x in pack["abstract_confidence"][view_id]["option_texts"]
            )
        passage_texts.update(
            str(x) for x in pack["severity_prototypes"]["option_texts"]
        )
        passage_texts.update(
            str(x) for x in pack["confidence_prototypes"]["option_texts"]
        )

    query_emb = _encode_texts(
        model,
        tokenizer,
        sorted(query_texts),
        prefix=spec["query_prefix"],
        pooling=spec["pooling"],
    )
    passage_emb = _encode_texts(
        model,
        tokenizer,
        sorted(passage_texts),
        prefix=spec["passage_prefix"],
        pooling=spec["pooling"],
    )

    def scores(state_text: str, option_texts) -> list[float]:
        state = query_emb[str(state_text)]
        options = torch.stack(
            [passage_emb[str(text)] for text in option_texts],
            dim=0,
        )
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

    evaluation = reference_evaluation_from_scores(cache["cases"], lookup)
    return evaluation, actual_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--w9-freeze", type=Path, required=True)
    parser.add_argument("--w9-checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w22_cache(args.cache)
    freeze, row, scorer = _load_scorer(args.w9_freeze, args.w9_checkpoints)
    metrics = evaluate_w22(scorer, cache)

    references = {}
    predictions = {}
    actual_reference_hashes = {}
    for name, spec in REFERENCE_PANEL.items():
        evaluation, actual_sha = _evaluate_reference(cache, name, spec)
        references[name] = evaluation["summary"]
        predictions[name] = evaluation["predictions"]
        actual_reference_hashes[name] = actual_sha

    classification = classify_w22(metrics, references, predictions)

    result = {
        "schema_version": "r8-w22-reference-panel-audit-v1",
        "status": "PASS",
        **classification,
        "metrics": metrics,
        "references": references,
        "reference_panel": REFERENCE_PANEL,
        "actual_reference_weight_sha256": actual_reference_hashes,
        "reference_panel_frozen_before_exposure": True,
        "reference_models_are_hira_candidates": False,
        "training_performed": False,
        "selection_performed": False,
        "trainable_parameter_count": 0,
        "w9_freeze_status": freeze["status"],
        "w9_projection_scorer_sha256": row["scorer_sha256"],
        "w9_hira_sha256": W9_HIRA_SHA256,
        "w9_scorer_sha256": W9_SCORER_SHA256,
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
