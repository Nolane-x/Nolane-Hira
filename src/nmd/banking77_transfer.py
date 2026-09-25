from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Callable, Mapping

import torch

from .contracts import LogicalOption
from .runtime import NolaneHira

DATASET_ID = "mteb/banking77"
DATASET_REVISION = "18072d2685ea682290f7b8924d94c62acc19c0b2"
DATASET_SPLIT = "test"
START_INDEX = 400
EXAMPLE_COUNT = 400
END_INDEX = START_INDEX + EXAMPLE_COUNT
EXPECTED_LABELS = 77
QUESTION = "Which banking intent does `message` express?"
FORCED_BUDGET = 77

CANDIDATES = (
    "frozen-w6e-control",
    "typed-only-retune",
    "pair-only-retune",
    "typed-plus-pair-primary",
    "typed-plus-pair-replica",
)


@dataclass(frozen=True)
class Banking77TransferExample:
    state_text: str
    gold_index: int
    gold_label: str


@dataclass(frozen=True)
class Banking77TransferAuthority:
    options: tuple[LogicalOption, ...]
    labels: tuple[str, ...]
    examples: tuple[Banking77TransferExample, ...]


def _row_value(row: Mapping[str, object], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"Banking77 row requires non-empty {key}")
    return value


def load_banking77_transfer_authority(
    *,
    dataset_loader: Callable[..., object] | None = None,
) -> Banking77TransferAuthority:
    if dataset_loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the research stack to load Banking77") from exc
        dataset_loader = load_dataset

    rows = dataset_loader(
        DATASET_ID,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    if len(rows) < END_INDEX:
        raise ValueError(
            f"Banking77 test must contain at least {END_INDEX} rows"
        )

    label_texts = [str(x).strip() for x in rows["label_text"]]
    if any(not x for x in label_texts):
        raise ValueError("Banking77 label_text contains empty values")
    labels = tuple(sorted(set(label_texts)))
    if len(labels) != EXPECTED_LABELS:
        raise ValueError(
            f"expected exactly {EXPECTED_LABELS} Banking77 labels, got {len(labels)}"
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

    selected = list(rows)[START_INDEX:END_INDEX]
    if len(selected) != EXAMPLE_COUNT:
        raise ValueError("Banking77 transfer selection count mismatch")

    examples: list[Banking77TransferExample] = []
    for row in selected:
        message = _row_value(row, "text")
        label_text = _row_value(row, "label_text").replace("_", " ")
        if label_text not in index_by_semantic:
            raise ValueError("Banking77 gold label outside frozen label space")
        examples.append(
            Banking77TransferExample(
                state_text=json.dumps(
                    {"message": message},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                gold_index=index_by_semantic[label_text],
                gold_label=label_text,
            )
        )

    return Banking77TransferAuthority(
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
def evaluate_banking77_transfer(
    model: NolaneHira,
    authority: Banking77TransferAuthority,
) -> dict[str, object]:
    model.eval()
    before = model.state_encode_calls
    schema, receipt = model.compile_schema(
        primitive="choice",
        question_text=QUESTION,
        options=authority.options,
        use_cache=True,
        include_token_artifacts=True,
    )
    if receipt.option_count != EXPECTED_LABELS:
        raise RuntimeError("Banking77 transfer schema option count mismatch")

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
        memory = model.compile_state(example.state_text, segment_tokens=32)
        out = model.forward_compiled(
            memory,
            schema,
            forced_budget=FORCED_BUDGET,
            adaptive_budget=False,
            relation_mode="pooled",
            coarse_mode="competitive",
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        p = out.probabilities.detach().cpu().to(torch.float64)
        if p.shape != (EXPECTED_LABELS,):
            raise RuntimeError("Banking77 transfer probability shape mismatch")
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
    if n != EXAMPLE_COUNT:
        raise RuntimeError("Banking77 transfer example count changed")
    state_calls = model.state_encode_calls - before
    if state_calls != EXAMPLE_COUNT:
        raise RuntimeError("Banking77 transfer state-once contract failed")
    if set(candidate_budgets) != {EXPECTED_LABELS}:
        raise RuntimeError("Banking77 transfer full-K contract failed")

    ordered = sorted(latencies_ms)

    def percentile(q: float) -> float:
        pos = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
        return ordered[pos]

    return {
        "case_count": n,
        "label_count": EXPECTED_LABELS,
        "slice_start": START_INDEX,
        "slice_end_exclusive": END_INDEX,
        "accuracy": sum(correctness) / n,
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
    }


def classify_transfer(
    metrics: Mapping[str, Mapping[str, object]],
) -> tuple[str, dict[str, object]]:
    if set(metrics) != set(CANDIDATES):
        raise ValueError("W7c candidate metric set changed")
    frozen = metrics["frozen-w6e-control"]
    primary = metrics["typed-plus-pair-primary"]
    replica = metrics["typed-plus-pair-replica"]

    frozen_acc = float(frozen["accuracy"])
    primary_acc = float(primary["accuracy"])
    replica_acc = float(replica["accuracy"])
    primary_gain = primary_acc - frozen_acc
    replica_gain = replica_acc - frozen_acc

    integrity = {}
    for name in CANDIDATES:
        row = metrics[name]
        integrity[name] = {
            "state_once": float(row["state_encode_calls_per_case"]) == 1.0,
            "probability_mass": float(row["probability_mass_max_error"]) <= 1e-6,
            "full_k": (
                int(row["candidate_budget_min"]) == EXPECTED_LABELS
                and int(row["candidate_budget_max"]) == EXPECTED_LABELS
            ),
        }

    signal = (
        primary_gain >= 0.10
        and replica_gain >= 0.10
        and primary_acc >= 0.25
        and replica_acc >= 0.25
        and all(all(g.values()) for g in integrity.values())
    )
    weak = (
        not signal
        and max(primary_gain, replica_gain) >= 0.05
        and primary_gain >= -0.05
        and replica_gain >= -0.05
        and all(all(g.values()) for g in integrity.values())
    )

    if signal:
        verdict = "PUBLIC_HIGH_K_TRANSFER_SIGNAL"
    elif weak:
        verdict = "PUBLIC_HIGH_K_TRANSFER_WEAK"
    else:
        verdict = "PUBLIC_HIGH_K_TRANSFER_ABSENT"

    details = {
        "frozen_accuracy": frozen_acc,
        "primary_accuracy": primary_acc,
        "replica_accuracy": replica_acc,
        "primary_gain_vs_frozen": primary_gain,
        "replica_gain_vs_frozen": replica_gain,
        "typed_only_gain_vs_frozen": (
            float(metrics["typed-only-retune"]["accuracy"]) - frozen_acc
        ),
        "pair_only_gain_vs_frozen": (
            float(metrics["pair-only-retune"]["accuracy"]) - frozen_acc
        ),
        "primary_gain_vs_typed_only": (
            primary_acc - float(metrics["typed-only-retune"]["accuracy"])
        ),
        "replica_minus_primary": replica_acc - primary_acc,
        "integrity": integrity,
    }
    return verdict, details
