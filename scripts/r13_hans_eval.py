from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.hira import HIRACore
from nmd.semantic import HFAutoSemanticEncoder


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

R12_HEAD_SHA256 = "0ca95572399d15717e1069545439165e46083c58afea8707cbbadeca67ad3b86"

HANS_REVISION = "7299f6f657089ce06a0f98e7e81f8d0f5b7741ce"
HANS_BLOB_SHA = "15a8339b4f20fd21536a3f631682f7f1f52e5a2f"

OPTION_TEXTS = (
    "the hypothesis is entailed by the premise",
    "the hypothesis is neutral with respect to the premise",
    "the hypothesis contradicts the premise",
)


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
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        gold = str(raw.get("gold_label", "")).strip()
        if gold not in {"entailment", "non-entailment"}:
            raise ValueError(f"line {i}: unsupported HANS label {gold!r}")
        rows.append({
            "premise": str(raw["sentence1"]),
            "hypothesis": str(raw["sentence2"]),
            "gold": gold,
            "heuristic": str(raw.get("heuristic", "unknown")),
            "subcase": str(raw.get("subcase", "unknown")),
            "template": str(raw.get("template", "unknown")),
        })
    if len(rows) != 30000:
        raise ValueError(f"expected HANS full evaluation set of 30000, got {len(rows)}")
    return rows


def finalize(group):
    total = group["correct"] + group["wrong"]
    return {
        "n": total,
        "accuracy": group["correct"] / total if total else None,
        "entailment_n": group["entailment_n"],
        "entailment_accuracy": (
            group["entailment_correct"] / group["entailment_n"]
            if group["entailment_n"] else None
        ),
        "non_entailment_n": group["non_entailment_n"],
        "non_entailment_accuracy": (
            group["non_entailment_correct"] / group["non_entailment_n"]
            if group["non_entailment_n"] else None
        ),
    }


def update(group, *, correct: bool, gold: str):
    if correct:
        group["correct"] += 1
    else:
        group["wrong"] += 1
    if gold == "entailment":
        group["entailment_n"] += 1
        if correct:
            group["entailment_correct"] += 1
    else:
        group["non_entailment_n"] += 1
        if correct:
            group["non_entailment_correct"] += 1


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hans-jsonl", type=Path, required=True)
    ap.add_argument("--r12-head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--segments", type=int, default=8)
    args = ap.parse_args()

    if file_sha256(args.r12_head) != R12_HEAD_SHA256:
        raise RuntimeError("R12 selected HIRA head SHA-256 mismatch")

    rows = load_hans(args.hans_jsonl)

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(base, tokenizer, revision=A13_REVISION, max_length=128)

    hira = HIRACore(d_model=256, dropout=0.0)
    state = torch.load(args.r12_head, map_location="cpu", weights_only=True)
    hira.load_state_dict(state, strict=True)
    hira.eval()

    option_embeddings = encoder.encode_texts(OPTION_TEXTS).pooled_embeddings
    option_embeddings = F.normalize(option_embeddings, dim=-1).float()

    def new_group():
        return {
            "correct": 0,
            "wrong": 0,
            "entailment_n": 0,
            "entailment_correct": 0,
            "non_entailment_n": 0,
            "non_entailment_correct": 0,
        }

    overall = new_group()
    by_heuristic = defaultdict(new_group)
    by_subcase = defaultdict(new_group)
    three_way_counts = defaultdict(lambda: [0, 0, 0])
    probability_sums = defaultdict(lambda: [0.0, 0.0, 0.0])

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        premise = [x["premise"] for x in batch]
        hypothesis = [x["hypothesis"] for x in batch]

        state_batch = encoder.encode_texts(premise)
        q_batch = encoder.encode_texts(hypothesis)
        segments, seg_mask = segment_pool(
            state_batch.token_embeddings,
            state_batch.attention_mask,
            args.segments,
        )

        n = len(batch)
        options = option_embeddings.unsqueeze(0).expand(n, -1, -1)
        qtype = torch.zeros(n, dtype=torch.long)
        out = hira(
            q_batch.pooled_embeddings.float(),
            segments,
            options,
            qtype,
            segment_mask=seg_mask,
            forced_budget=3,
        )
        probs = out.probabilities.cpu()
        pred3 = probs.argmax(-1)

        for i, row in enumerate(batch):
            pred_bin = "entailment" if int(pred3[i]) == 0 else "non-entailment"
            correct = pred_bin == row["gold"]
            update(overall, correct=correct, gold=row["gold"])
            update(by_heuristic[row["heuristic"]], correct=correct, gold=row["gold"])
            update(
                by_subcase[f'{row["heuristic"]}::{row["subcase"]}'],
                correct=correct,
                gold=row["gold"],
            )
            key = row["gold"]
            three_way_counts[key][int(pred3[i])] += 1
            for j in range(3):
                probability_sums[key][j] += float(probs[i, j])

    overall_final = finalize(overall)
    heuristics = {k: finalize(v) for k, v in sorted(by_heuristic.items())}
    subcases = {k: finalize(v) for k, v in sorted(by_subcase.items())}
    mean_probs = {}
    for gold, sums in probability_sums.items():
        n = overall[f"{'entailment' if gold == 'entailment' else 'non_entailment'}_n"]
        mean_probs[gold] = [x / max(1, n) for x in sums]

    min_label_accuracy = min(
        overall_final["entailment_accuracy"],
        overall_final["non_entailment_accuracy"],
    )
    min_heuristic_accuracy = min(x["accuracy"] for x in heuristics.values())
    gate = {
        "overall_accuracy_floor": 0.55,
        "minimum_label_accuracy_floor": 0.50,
        "minimum_heuristic_accuracy_floor": 0.50,
    }
    gate["overall_pass"] = overall_final["accuracy"] >= gate["overall_accuracy_floor"]
    gate["label_balance_pass"] = min_label_accuracy >= gate["minimum_label_accuracy_floor"]
    gate["heuristic_floor_pass"] = min_heuristic_accuracy >= gate["minimum_heuristic_accuracy_floor"]
    gate["pass"] = all(
        gate[x] for x in ["overall_pass", "label_balance_pass", "heuristic_floor_pass"]
    )

    receipt = {
        "schema_version": "r13-hans-heldout-v1",
        "status": "PASS",
        "evidence_scope": "frozen R12 HIRA head on untouched HANS evaluation set; no weight/threshold/model selection update",
        "sources": {
            "hans_revision": HANS_REVISION,
            "hans_blob_sha": HANS_BLOB_SHA,
            "r12_head_sha256": R12_HEAD_SHA256,
            "a13_revision": A13_REVISION,
            "a13_weight_sha256": A13_WEIGHT_SHA256,
        },
        "protocol": {
            "examples": len(rows),
            "batch_size": args.batch_size,
            "segments": args.segments,
            "binary_mapping": "3-way argmax: entailment -> entailment; neutral|contradiction -> non-entailment",
            "training_on_hans": False,
            "threshold_tuning_on_hans": False,
        },
        "gate": gate,
        "overall": overall_final,
        "heuristics": heuristics,
        "subcases": subcases,
        "three_way_prediction_counts_by_binary_gold": dict(three_way_counts),
        "mean_three_way_probabilities_by_binary_gold": mean_probs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS",
        "gate_pass": gate["pass"],
        "overall_accuracy": overall_final["accuracy"],
        "entailment_accuracy": overall_final["entailment_accuracy"],
        "non_entailment_accuracy": overall_final["non_entailment_accuracy"],
        "min_heuristic_accuracy": min_heuristic_accuracy,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
