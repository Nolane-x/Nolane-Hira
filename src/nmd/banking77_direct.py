from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch

from .contracts import LogicalOption
from .runtime import NolaneHira

DATASET_ID = "mteb/banking77"
DATASET_REVISION = "18072d2685ea682290f7b8924d94c62acc19c0b2"
DATASET_SPLIT = "test"
EXPECTED_EXAMPLES = 400
EXPECTED_LABELS = 77
QUESTION = "Which banking intent does `message` express?"
LAYA_TARGET = 0.492
FORCED_BUDGET = 255

W3A_RUN_ID = 35748778854
SELECTED_ARTIFACT_ID = 10706135649
SELECTED_ARTIFACT_DIGEST = "sha256:8a2efd31a6ccea3b987957d328c4b73267ab00ff9024c469bcb7f3cd578d5d92"
SELECTED_HEAD_SHA256 = "2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c"

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
A13_MAX_LENGTH = 256
STATE_SEGMENT_TOKENS = 32


@dataclass(frozen=True)
class Banking77Example:
    state_text: str
    gold_index: int
    gold_label: str


@dataclass(frozen=True)
class Banking77Authority:
    options: tuple[LogicalOption, ...]
    labels: tuple[str, ...]
    examples: tuple[Banking77Example, ...]


EXPECTED_MARKER = {
    "schema_version": "r8-w4a-banking77-direct-authority-v1",
    "authorization": "AUTHORIZED_ONE_SHOT",
    "dataset_id": DATASET_ID,
    "dataset_revision": DATASET_REVISION,
    "split": DATASET_SPLIT,
    "selection": "first_400",
    "expected_examples": EXPECTED_EXAMPLES,
    "expected_labels": EXPECTED_LABELS,
    "selected_head_sha256": SELECTED_HEAD_SHA256,
    "selected_artifact_id": SELECTED_ARTIFACT_ID,
    "selected_artifact_digest": SELECTED_ARTIFACT_DIGEST,
    "w3a_run_id": W3A_RUN_ID,
    "a13_model": A13_MODEL,
    "a13_revision": A13_REVISION,
    "a13_weight_sha256": A13_WEIGHT_SHA256,
    "forced_budget": FORCED_BUDGET,
    "adaptive_budget": False,
}


def load_and_validate_marker(path: str | Path) -> dict:
    marker = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError("W4a marker must be a JSON object")
    missing = sorted(set(EXPECTED_MARKER) - set(marker))
    if missing:
        raise ValueError(f"W4a marker missing fields: {missing}")
    mismatches = {
        k: {"expected": v, "actual": marker.get(k)}
        for k, v in EXPECTED_MARKER.items()
        if marker.get(k) != v
    }
    if mismatches:
        raise ValueError(f"W4a marker mismatch: {mismatches}")
    extra = sorted(set(marker) - set(EXPECTED_MARKER))
    if extra:
        raise ValueError(f"W4a marker has unauthorized fields: {extra}")
    return marker


def _row_value(row: Mapping[str, object], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"Banking77 row requires non-empty {key}")
    return value


def load_banking77_authority(
    *,
    dataset_loader: Callable[..., object] | None = None,
) -> Banking77Authority:
    if dataset_loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Install the research stack to load Banking77"
            ) from exc
        dataset_loader = load_dataset

    rows = dataset_loader(
        DATASET_ID,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    if len(rows) < EXPECTED_EXAMPLES:
        raise ValueError(
            f"Banking77 test must contain at least {EXPECTED_EXAMPLES} rows"
        )

    label_texts = [str(x).strip() for x in rows["label_text"]]
    if any(not x for x in label_texts):
        raise ValueError("Banking77 label_text contains empty values")
    labels = tuple(sorted(set(label_texts)))
    if len(labels) != EXPECTED_LABELS:
        raise ValueError(
            f"expected exactly {EXPECTED_LABELS} Banking77 labels, "
            f"got {len(labels)}"
        )

    semantic_labels = tuple(label.replace("_", " ") for label in labels)
    if len(set(semantic_labels)) != EXPECTED_LABELS:
        raise ValueError("semantic Banking77 labels are not unique")

    options = tuple(
        LogicalOption(
            option_id=f"intent-{index:03d}",
            criterion_text=semantic,
        )
        for index, semantic in enumerate(semantic_labels)
    )
    index_by_semantic = {
        semantic: index
        for index, semantic in enumerate(semantic_labels)
    }

    examples: list[Banking77Example] = []
    for row in list(rows)[:EXPECTED_EXAMPLES]:
        message = _row_value(row, "text")
        label_text = _row_value(row, "label_text").replace("_", " ")
        if label_text not in index_by_semantic:
            raise ValueError("Banking77 gold label is outside frozen label space")
        state_text = json.dumps(
            {"message": message},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        examples.append(
            Banking77Example(
                state_text=state_text,
                gold_index=index_by_semantic[label_text],
                gold_label=label_text,
            )
        )

    if len(examples) != EXPECTED_EXAMPLES:
        raise ValueError("Banking77 direct selection count mismatch")
    return Banking77Authority(
        options=options,
        labels=semantic_labels,
        examples=tuple(examples),
    )


def _ece15(conf: list[float], correct: list[float]) -> float:
    n = len(conf)
    total = 0.0
    for b in range(15):
        lo, hi = b / 15, (b + 1) / 15
        idx = [i for i, p in enumerate(conf) if p > lo and p <= hi]
        if not idx:
            continue
        c = sum(conf[i] for i in idx) / len(idx)
        a = sum(correct[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(c - a)
    return total


def _macro_f1(y_true: list[int], y_pred: list[int], k: int) -> float:
    scores = []
    for label in range(k):
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(scores) / k


def _aurc(conf: list[float], correct: list[float]) -> float:
    order = sorted(range(len(conf)), key=lambda i: (-conf[i], i))
    errors = 0.0
    total = 0.0
    for rank, i in enumerate(order, start=1):
        errors += 1.0 - correct[i]
        total += errors / rank
    return total / len(order)


@torch.inference_mode()
def evaluate_banking77_direct(
    model: NolaneHira,
    authority: Banking77Authority,
) -> dict[str, object]:
    model.eval()
    before = model.state_encode_calls
    schema, receipt = model.compile_schema(
        primitive="choice",
        question_text=QUESTION,
        options=authority.options,
        use_cache=True,
    )
    if receipt.option_count != EXPECTED_LABELS:
        raise RuntimeError("Banking77 schema option count mismatch")

    y_true: list[int] = []
    y_pred: list[int] = []
    confidence: list[float] = []
    correctness: list[float] = []
    hard_brier = 0.0
    nll = 0.0
    max_mass_error = 0.0
    tail_mass_max = 0.0
    candidate_budgets: list[int] = []
    latencies_ms: list[float] = []

    for example in authority.examples:
        start = time.perf_counter()
        memory = model.compile_state(
            example.state_text,
            segment_tokens=STATE_SEGMENT_TOKENS,
        )
        out = model.forward_compiled(
            memory,
            schema,
            forced_budget=FORCED_BUDGET,
            adaptive_budget=False,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        p = out.probabilities.detach().cpu().to(torch.float64)
        if p.shape != (EXPECTED_LABELS,):
            raise RuntimeError("Banking77 probability shape mismatch")
        pred = int(p.argmax().item())
        gold = example.gold_index
        ok = float(pred == gold)

        y_true.append(gold)
        y_pred.append(pred)
        confidence.append(float(p.max()))
        correctness.append(ok)
        onehot = torch.zeros_like(p)
        onehot[gold] = 1.0
        hard_brier += float(((p - onehot) ** 2).sum())
        nll += -math.log(max(float(p[gold]), 1e-12))
        max_mass_error = max(max_mass_error, abs(float(p.sum()) - 1.0))
        candidate_budgets.append(int(out.hira.candidate_budget[0].item()))
        tail_mass_max = max(
            tail_mass_max,
            float(out.hira.tail_mass[0].detach().cpu()),
        )

    n = len(authority.examples)
    state_calls = model.state_encode_calls - before
    if state_calls != EXPECTED_EXAMPLES:
        raise RuntimeError("Banking77 state-once contract failed")
    if set(candidate_budgets) != {EXPECTED_LABELS}:
        raise RuntimeError("Banking77 full-K candidate budget contract failed")

    ordered_latency = sorted(latencies_ms)
    def percentile(q: float) -> float:
        pos = min(len(ordered_latency) - 1, int(round(q * (len(ordered_latency) - 1))))
        return ordered_latency[pos]

    accuracy = sum(correctness) / n
    return {
        "case_count": n,
        "label_count": EXPECTED_LABELS,
        "accuracy": accuracy,
        "macro_f1": _macro_f1(y_true, y_pred, EXPECTED_LABELS),
        "hard_brier": hard_brier / n,
        "nll": nll / n,
        "ece": _ece15(confidence, correctness),
        "mean_confidence": sum(confidence) / n,
        "aurc": _aurc(confidence, correctness),
        "p50_ms_cpu_ci": percentile(0.50),
        "p95_ms_cpu_ci": percentile(0.95),
        "probability_mass_max_error": max_mass_error,
        "state_encode_calls": state_calls,
        "state_encode_calls_per_case": state_calls / n,
        "candidate_budget_min": min(candidate_budgets),
        "candidate_budget_max": max(candidate_budgets),
        "tail_mass_max": tail_mass_max,
        "laya_target": LAYA_TARGET,
        "laya_status": (
            "WIN" if accuracy > LAYA_TARGET
            else "TIE" if accuracy == LAYA_TARGET
            else "LOSS"
        ),
    }
