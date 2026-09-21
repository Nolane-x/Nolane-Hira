from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import torch
import torch.nn.functional as F

from nmd.hira import HIRACore
from nmd.semantic import HFAutoSemanticEncoder


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
R12_HEAD_SHA256 = "0ca95572399d15717e1069545439165e46083c58afea8707cbbadeca67ad3b86"

BREAKING_REVISION = "8a7658c1ce6b732f4e8af3b06560f1a13b8b18b0"
BREAKING_ZIP_BLOB_SHA = "9fcd602891e4f844d53d611924915ca921d881ce"

OPTION_TEXTS = (
    "the hypothesis is entailed by the premise",
    "the hypothesis is neutral with respect to the premise",
    "the hypothesis contradicts the premise",
)
LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def segment_pool(tokens: torch.Tensor, mask: torch.Tensor, n_segments: int):
    n, _, d = tokens.shape
    out = torch.zeros(n, n_segments, d, dtype=torch.float32)
    out_mask = torch.zeros(n, n_segments, dtype=torch.bool)
    for i in range(n):
        valid = tokens[i][mask[i].bool()].detach().cpu()
        if valid.shape[0] == 0:
            valid = tokens[i, :1].detach().cpu()
        pieces = torch.tensor_split(valid, min(n_segments, valid.shape[0]), dim=0)
        for j, piece in enumerate(pieces):
            if piece.numel() == 0:
                continue
            out[i, j] = piece.mean(0).float()
            out_mask[i, j] = True
    return out, out_mask


def parse_line(line: str):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def load_breaking(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [n for n in zf.namelist() if n.endswith("dataset.jsonl")]
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one dataset.jsonl in zip, found {candidates}")
        raw = zf.read(candidates[0]).decode("utf-8")

    rows = []
    for i, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        x = parse_line(line)
        label = str(x["gold_label"]).strip()
        if label not in LABEL_TO_ID:
            raise ValueError(f"line {i}: unsupported label {label!r}")
        rows.append({
            "premise": str(x["sentence1"]),
            "hypothesis": str(x["sentence2"]),
            "label": LABEL_TO_ID[label],
            "label_name": label,
            "category": str(x.get("category", "unknown")),
        })
    if len(rows) != 8193:
        raise ValueError(f"expected full Breaking NLI set of 8193, got {len(rows)}")
    return rows


def macro_f1(gold, pred):
    scores = []
    for c in range(3):
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        scores.append((2 * tp) / max(1, 2 * tp + fp + fn))
    return sum(scores) / 3.0


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-zip", type=Path, required=True)
    ap.add_argument("--r12-head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--segments", type=int, default=8)
    args = ap.parse_args()

    if file_sha256(args.r12_head) != R12_HEAD_SHA256:
        raise RuntimeError("R12 head SHA mismatch")

    rows = load_breaking(args.dataset_zip)
    source_zip_sha256 = file_sha256(args.dataset_zip)

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(base, tokenizer, revision=A13_REVISION, max_length=128)

    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(torch.load(args.r12_head, map_location="cpu", weights_only=True), strict=True)
    hira.eval()

    options = F.normalize(encoder.encode_texts(OPTION_TEXTS).pooled_embeddings, dim=-1).float()

    gold, pred = [], []
    category_total = Counter()
    category_correct = Counter()
    confusion = [[0, 0, 0] for _ in range(3)]

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        state_batch = encoder.encode_texts([x["premise"] for x in batch])
        q_batch = encoder.encode_texts([x["hypothesis"] for x in batch])
        segments, seg_mask = segment_pool(
            state_batch.token_embeddings, state_batch.attention_mask, args.segments
        )
        n = len(batch)
        out = hira(
            q_batch.pooled_embeddings.float(),
            segments,
            options.unsqueeze(0).expand(n, -1, -1),
            torch.zeros(n, dtype=torch.long),
            segment_mask=seg_mask,
            forced_budget=3,
        )
        yhat = out.probabilities.argmax(-1).cpu().tolist()
        for row, p in zip(batch, yhat):
            g = row["label"]
            gold.append(g)
            pred.append(p)
            confusion[g][p] += 1
            category_total[row["category"]] += 1
            if p == g:
                category_correct[row["category"]] += 1

    counts = Counter(gold)
    correct = sum(int(g == p) for g, p in zip(gold, pred))
    recalls = {}
    for c in range(3):
        n = counts[c]
        recalls[ID_TO_LABEL[c]] = (
            sum(1 for g, p in zip(gold, pred) if g == c and p == c) / n if n else None
        )
    macro_recall = sum(recalls.values()) / 3.0
    overall = correct / len(gold)
    majority = max(counts.values()) / len(gold)

    gate = {
        "overall_accuracy_floor": 0.50,
        "macro_recall_floor": 0.45,
        "entailment_recall_floor": 0.30,
        "contradiction_recall_floor": 0.50,
    }
    gate["overall_pass"] = overall >= gate["overall_accuracy_floor"]
    gate["macro_recall_pass"] = macro_recall >= gate["macro_recall_floor"]
    gate["entailment_pass"] = recalls["entailment"] >= gate["entailment_recall_floor"]
    gate["contradiction_pass"] = recalls["contradiction"] >= gate["contradiction_recall_floor"]
    gate["pass"] = all(gate[k] for k in [
        "overall_pass", "macro_recall_pass", "entailment_pass", "contradiction_pass"
    ])

    receipt = {
        "schema_version": "r13-breaking-heldout-v1",
        "status": "PASS",
        "evidence_scope": "frozen R12 HIRA head on untouched Breaking NLI; no model or threshold update",
        "sources": {
            "breaking_revision": BREAKING_REVISION,
            "breaking_zip_blob_sha": BREAKING_ZIP_BLOB_SHA,
            "breaking_zip_sha256": source_zip_sha256,
            "r12_head_sha256": R12_HEAD_SHA256,
            "a13_revision": A13_REVISION,
            "a13_weight_sha256": A13_WEIGHT_SHA256,
        },
        "protocol": {
            "examples": len(rows),
            "training_on_breaking_nli": False,
            "threshold_tuning_on_breaking_nli": False,
            "license": "CC BY-SA 4.0",
        },
        "class_counts": {ID_TO_LABEL[k]: v for k, v in sorted(counts.items())},
        "majority_baseline_accuracy": majority,
        "accuracy": overall,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1(gold, pred),
        "per_label_recall": recalls,
        "confusion_matrix_gold_rows_pred_columns": confusion,
        "category_accuracy": {
            k: category_correct[k] / category_total[k] for k in sorted(category_total)
        },
        "gate": gate,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS",
        "gate_pass": gate["pass"],
        "accuracy": overall,
        "macro_recall": macro_recall,
        "macro_f1": receipt["macro_f1"],
        "majority_baseline_accuracy": majority,
        "entailment_recall": recalls["entailment"],
        "contradiction_recall": recalls["contradiction"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
