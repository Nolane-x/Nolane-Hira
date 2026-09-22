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

R13_HANS_NON_ENTAILMENT = 0.08373333333333334
R13_BREAKING_CONTRADICTION = 0.12618648799553323

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


def load_hans(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        rows.append({
            "premise": str(x["sentence1"]),
            "hypothesis": str(x["sentence2"]),
            "gold": str(x["gold_label"]),
            "heuristic": str(x.get("heuristic", "unknown")),
        })
    if len(rows) != 30000:
        raise ValueError(f"expected 30000 HANS rows, got {len(rows)}")
    return rows


def parse_breaking_line(line: str):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def load_breaking(path: Path):
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith("dataset.jsonl")]
        if len(names) != 1:
            raise ValueError("expected one Breaking NLI dataset.jsonl")
        raw = zf.read(names[0]).decode("utf-8")
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        x = parse_breaking_line(line)
        label = str(x["gold_label"])
        rows.append({
            "premise": str(x["sentence1"]),
            "hypothesis": str(x["sentence2"]),
            "label": LABEL_TO_ID[label],
            "label_name": label,
            "category": str(x.get("category", "unknown")),
        })
    if len(rows) != 8193:
        raise ValueError(f"expected 8193 Breaking NLI rows, got {len(rows)}")
    return rows


@torch.inference_mode()
def predict_batches(rows, encoder, hira, option_embeddings, *, batch_size: int, segments: int):
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        s = encoder.encode_texts([x["premise"] for x in batch])
        q = encoder.encode_texts([x["hypothesis"] for x in batch])
        seg, mask = segment_pool(s.token_embeddings, s.attention_mask, segments)
        n = len(batch)
        out = hira(
            q.pooled_embeddings.float(),
            seg,
            option_embeddings.unsqueeze(0).expand(n, -1, -1),
            torch.zeros(n, dtype=torch.long),
            segment_mask=mask,
            forced_budget=3,
        )
        yield batch, out.probabilities.cpu()


def eval_hans(rows, encoder, hira, option_embeddings, *, batch_size: int, segments: int):
    overall = Counter()
    heur = defaultdict(Counter)
    for batch, probs in predict_batches(
        rows, encoder, hira, option_embeddings, batch_size=batch_size, segments=segments
    ):
        pred = probs.argmax(-1).tolist()
        for row, p in zip(batch, pred):
            binary = "entailment" if p == 0 else "non-entailment"
            gold = row["gold"]
            ok = binary == gold
            overall["n"] += 1
            overall["correct"] += int(ok)
            overall[f"{gold}_n"] += 1
            overall[f"{gold}_correct"] += int(ok)
            h = heur[row["heuristic"]]
            h["n"] += 1
            h["correct"] += int(ok)
            h[f"{gold}_n"] += 1
            h[f"{gold}_correct"] += int(ok)

    def finish(c):
        return {
            "n": c["n"],
            "accuracy": c["correct"] / c["n"],
            "entailment_accuracy": c["entailment_correct"] / c["entailment_n"],
            "non_entailment_accuracy": c["non-entailment_correct"] / c["non-entailment_n"],
        }

    final = finish(overall)
    return {
        "overall": final,
        "heuristics": {k: finish(v) for k, v in sorted(heur.items())},
        "r13_non_entailment_baseline": R13_HANS_NON_ENTAILMENT,
        "non_entailment_gain": final["non_entailment_accuracy"] - R13_HANS_NON_ENTAILMENT,
    }


def macro_f1(gold, pred):
    scores = []
    for c in range(3):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return sum(scores) / 3


def eval_breaking(rows, encoder, hira, option_embeddings, *, batch_size: int, segments: int):
    gold, pred = [], []
    cats = defaultdict(Counter)
    for batch, probs in predict_batches(
        rows, encoder, hira, option_embeddings, batch_size=batch_size, segments=segments
    ):
        pp = probs.argmax(-1).tolist()
        for row, yhat in zip(batch, pp):
            y = row["label"]
            gold.append(y)
            pred.append(yhat)
            cats[row["category"]]["n"] += 1
            cats[row["category"]]["correct"] += int(y == yhat)

    counts = Counter(gold)
    recalls = {
        ID_TO_LABEL[c]: sum(g == c and p == c for g, p in zip(gold, pred)) / counts[c]
        for c in range(3)
    }
    accuracy = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    contradiction_gain = recalls["contradiction"] - R13_BREAKING_CONTRADICTION
    return {
        "accuracy": accuracy,
        "macro_recall": sum(recalls.values()) / 3,
        "macro_f1": macro_f1(gold, pred),
        "per_label_recall": recalls,
        "category_accuracy": {k: v["correct"] / v["n"] for k, v in sorted(cats.items())},
        "r13_contradiction_baseline": R13_BREAKING_CONTRADICTION,
        "contradiction_gain": contradiction_gain,
    }


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair-receipt", type=Path, required=True)
    ap.add_argument("--repaired-head", type=Path, required=True)
    ap.add_argument("--hans-jsonl", type=Path, required=True)
    ap.add_argument("--breaking-zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--segments", type=int, default=8)
    args = ap.parse_args()

    repair = json.loads(args.repair_receipt.read_text())
    if repair.get("schema_version") != "r15-retention-repair-v1":
        raise RuntimeError("unexpected repair receipt schema")
    if not repair["gates"]["primary_pass"]:
        raise RuntimeError("R15 primary gate did not pass")
    expected_sha = repair["selected_head_sha256"]
    actual_sha = file_sha256(args.repaired_head)
    if actual_sha != expected_sha:
        raise RuntimeError("R15 selected head SHA mismatch")

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    if file_sha256(snapshot / "model.safetensors") != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA mismatch")
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(base, tok, revision=A13_REVISION, max_length=128)

    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(args.repaired_head, map_location="cpu", weights_only=True),
        strict=True,
    )
    hira.eval()
    options = F.normalize(encoder.encode_texts(OPTION_TEXTS).pooled_embeddings, dim=-1).float()

    hans = eval_hans(
        load_hans(args.hans_jsonl), encoder, hira, options,
        batch_size=args.batch_size, segments=args.segments
    )
    breaking = eval_breaking(
        load_breaking(args.breaking_zip), encoder, hira, options,
        batch_size=args.batch_size, segments=args.segments
    )

    gates = {
        "hans_non_entailment_gain_floor": 0.15,
        "hans_non_entailment_gain": hans["non_entailment_gain"],
        "hans_repair_pass": hans["non_entailment_gain"] >= 0.15,
        "breaking_contradiction_gain_floor": 0.15,
        "breaking_contradiction_gain": breaking["contradiction_gain"],
        "breaking_repair_pass": breaking["contradiction_gain"] >= 0.15,
    }
    gates["diagnostic_pass"] = gates["hans_repair_pass"] and gates["breaking_repair_pass"]

    payload = {
        "schema_version": "r15-adapted-diagnostics-v1",
        "status": "PASS",
        "evidence_scope": "adapted diagnostics after R13 failure discovery; no weights or thresholds updated",
        "repair_head_sha256": actual_sha,
        "repair_primary_gates": repair["gates"],
        "protocol": {
            "hans_used_for_model_selection": False,
            "breaking_used_for_model_selection": False,
            "weights_updated_during_diagnostics": False,
        },
        "hans": hans,
        "breaking_nli": breaking,
        "gates": gates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS",
        "diagnostic_pass": gates["diagnostic_pass"],
        "hans_non_entailment_accuracy": hans["overall"]["non_entailment_accuracy"],
        "hans_gain": hans["non_entailment_gain"],
        "breaking_contradiction_recall": breaking["per_label_recall"]["contradiction"],
        "breaking_gain": breaking["contradiction_gain"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
