from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import torch

from nmd.banking77_direct import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    EXPECTED_EXAMPLES,
    EXPECTED_LABELS,
    LAYA_TARGET,
    QUESTION,
    load_banking77_authority,
)
from nmd.local_runtime import load_hira_v0_m4_bundle


SCHEMA_VERSION = "hira-v0-mainline-m5-banking77-direct-v1"
OUTCOME_READY = "HIRA_V0_M5_BANKING77_DIRECT_READY"
OUTCOME_FAIL = "HIRA_V0_M5_BANKING77_DIRECT_FAIL"
M5_CONTRACT_SCHEMA = "hira-v0-mainline-m5-contract-v1"
EXPECTED_BASE = "97840f88b592ddfa10ae66e156fab781a4f18af7"
EXPECTED_CONTRACT_ARTIFACT_ID = 10943744649
EXPECTED_CONTRACT_DIGEST = (
    "sha256:856031dec357cb0a0d0f1b0fff82a5d7661832a5c67aebc7c2b45d309ec45230"
)


def state_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def ece15(confidence: list[float], correct: list[float]) -> float:
    n = len(confidence)
    if n == 0:
        raise ValueError("ECE requires examples")
    total = 0.0
    for b in range(15):
        lo, hi = b / 15.0, (b + 1) / 15.0
        idx = [
            i
            for i, p in enumerate(confidence)
            if p > lo and p <= hi
        ]
        if not idx:
            continue
        c = sum(confidence[i] for i in idx) / len(idx)
        a = sum(correct[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(c - a)
    return total


def macro_f1(y_true: list[int], y_pred: list[int], k: int) -> float:
    scores: list[float] = []
    for label in range(k):
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return sum(scores) / k


def aurc(confidence: list[float], correct: list[float]) -> float:
    order = sorted(range(len(confidence)), key=lambda i: (-confidence[i], i))
    errors = 0.0
    total = 0.0
    for rank, i in enumerate(order, start=1):
        errors += 1.0 - correct[i]
        total += errors / rank
    return total / len(order)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[pos]


def summarize_rows(
    rows: list[dict[str, object]],
    *,
    label_count: int,
) -> dict[str, object]:
    if not rows:
        raise ValueError("Banking77 summary requires rows")

    y_true = [int(row["gold_index"]) for row in rows]
    y_pred = [int(row["predicted_index"]) for row in rows]
    confidence = [float(row["confidence"]) for row in rows]
    correct = [float(bool(row["correct"])) for row in rows]
    hard_brier = [float(row["hard_brier"]) for row in rows]
    nll = [float(row["nll"]) for row in rows]
    latency = [float(row["latency_ms"]) for row in rows]
    mass_error = [float(row["probability_mass_error"]) for row in rows]
    budgets = [int(row["candidate_budget"]) for row in rows]
    relation_delta = [float(row["relation_delta_max_abs"]) for row in rows]

    accuracy = sum(correct) / len(rows)
    return {
        "case_count": len(rows),
        "label_count": label_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1(y_true, y_pred, label_count),
        "hard_brier": sum(hard_brier) / len(rows),
        "nll": sum(nll) / len(rows),
        "ece_15bin": ece15(confidence, correct),
        "mean_confidence": sum(confidence) / len(rows),
        "aurc": aurc(confidence, correct),
        "p50_ms_cpu_ci": percentile(latency, 0.50),
        "p95_ms_cpu_ci": percentile(latency, 0.95),
        "mean_ms_cpu_ci": sum(latency) / len(rows),
        "probability_mass_max_error": max(mass_error),
        "candidate_budget_min": min(budgets),
        "candidate_budget_max": max(budgets),
        "relation_delta_max_abs": max(relation_delta),
    }


def load_contract(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != M5_CONTRACT_SCHEMA:
        raise RuntimeError("unexpected M5 contract schema")
    if payload.get("base_main") != EXPECTED_BASE:
        raise RuntimeError("M5 base changed")
    frozen = payload.get("frozen_hira") or {}
    if frozen.get("m4_package_artifact_id") != 10944174953:
        raise RuntimeError("M5 M4 package authority changed")
    if frozen.get("production_ready") is not False:
        raise RuntimeError("M5 contract production-ready overclaim")
    claims = payload.get("claim_policy") or {}
    if claims.get("final_test_selection_forbidden") is not True:
        raise RuntimeError("M5 final-selection rule changed")
    return payload


def load_contract_receipt(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("outcome") != "HIRA_V0_M5_CONTRACT_READY":
        raise RuntimeError("M5-B requires qualified M5-A contract")
    if payload.get("status") != "PASS":
        raise RuntimeError("M5-A contract receipt is not PASS")
    if payload.get("final_scores_exposed") is not False:
        raise RuntimeError("M5-A unexpectedly exposed final scores")
    return payload


@torch.inference_mode()
def run_banking77(
    *,
    bundle_dir: Path,
    semantic_snapshot_dir: Path | None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    model = load_hira_v0_m4_bundle(
        bundle_dir,
        semantic_snapshot_dir=semantic_snapshot_dir,
    )
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M5-B runtime has trainable parameters")
    if model.manifest.production_ready:
        raise RuntimeError("M5-B runtime overclaims production readiness")

    authority = load_banking77_authority()
    if len(authority.examples) != EXPECTED_EXAMPLES:
        raise RuntimeError("Banking77 first-400 authority count changed")
    if len(authority.options) != EXPECTED_LABELS:
        raise RuntimeError("Banking77 option count changed")

    schema, schema_receipt = model.runtime.compile_schema(
        primitive="choice",
        question_text=QUESTION,
        options=authority.options,
        use_cache=True,
        include_token_artifacts=True,
    )
    if schema_receipt.option_count != EXPECTED_LABELS:
        raise RuntimeError("Banking77 compiled schema K changed")

    before_calls = model.runtime.state_encode_calls
    rows: list[dict[str, object]] = []

    for index, example in enumerate(authority.examples):
        started = time.perf_counter()
        session = model.open_session(example.state_text)
        output = session.decide_compiled(schema)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        p = output.probabilities.detach().cpu().to(torch.float64)
        if p.shape != (EXPECTED_LABELS,):
            raise RuntimeError("Banking77 probability shape changed")
        if not bool(torch.isfinite(p).all()):
            raise RuntimeError("Banking77 produced non-finite probabilities")

        predicted = int(p.argmax().item())
        gold = int(example.gold_index)
        onehot = torch.zeros_like(p)
        onehot[gold] = 1.0
        candidate_budget = int(output.hira.candidate_budget.item())
        relation_delta = float(
            output.hira.relation_delta.detach().abs().max().cpu()
        )

        rows.append({
            "index": index,
            "state_sha256": state_sha256(example.state_text),
            "gold_index": gold,
            "gold_label": example.gold_label,
            "predicted_index": predicted,
            "predicted_label": authority.labels[predicted],
            "correct": predicted == gold,
            "confidence": float(p.max()),
            "gold_probability": float(p[gold]),
            "hard_brier": float(((p - onehot) ** 2).sum()),
            "nll": -math.log(max(float(p[gold]), 1e-12)),
            "probability_mass_error": abs(float(p.sum()) - 1.0),
            "candidate_budget": candidate_budget,
            "relation_delta_max_abs": relation_delta,
            "latency_ms": elapsed_ms,
            "probabilities": [float(value) for value in p.tolist()],
        })

    state_calls = model.runtime.state_encode_calls - before_calls
    if state_calls != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Banking77 state-once violation: {state_calls} != {EXPECTED_EXAMPLES}"
        )

    metrics = summarize_rows(rows, label_count=EXPECTED_LABELS)
    if metrics["candidate_budget_min"] != EXPECTED_LABELS:
        raise RuntimeError("Banking77 candidate budget dropped below full-K")
    if metrics["candidate_budget_max"] != EXPECTED_LABELS:
        raise RuntimeError("Banking77 candidate budget exceeded full-K")
    if float(metrics["relation_delta_max_abs"]) != 0.0:
        raise RuntimeError("Banking77 relation refinement changed")
    if float(metrics["probability_mass_max_error"]) > 1e-6:
        raise RuntimeError("Banking77 probability mass contract failed")

    metrics["state_encode_calls"] = state_calls
    metrics["state_encode_calls_per_case"] = state_calls / EXPECTED_EXAMPLES
    metrics["schema_hash"] = schema.schema_hash
    metrics["schema_cache_info"] = model.schema_cache_info()

    runtime = {
        "manifest": model.manifest.to_dict(),
        "parameter_report": model.parameter_report().to_dict(),
    }
    return metrics, rows, runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--semantic-snapshot", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-receipt", type=Path, required=True)
    parser.add_argument("--contract-artifact-id", type=int, required=True)
    parser.add_argument("--contract-artifact-digest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    load_contract(args.contract)
    load_contract_receipt(args.contract_receipt)
    if args.contract_artifact_id != EXPECTED_CONTRACT_ARTIFACT_ID:
        raise RuntimeError("M5-A contract artifact ID changed")
    if args.contract_artifact_digest != EXPECTED_CONTRACT_DIGEST:
        raise RuntimeError("M5-A contract artifact digest changed")

    print("HIRA_V0_M5_BANKING77_FINAL_EXPOSURE_BEGIN", flush=True)
    metrics, rows, runtime = run_banking77(
        bundle_dir=args.bundle,
        semantic_snapshot_dir=args.semantic_snapshot,
    )

    accuracy = float(metrics["accuracy"])
    if accuracy > LAYA_TARGET:
        status = "WIN"
    elif accuracy < LAYA_TARGET:
        status = "LOSS"
    else:
        status = "TIE"

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME_READY,
        "scientific_authority": "M5_MATCHED_HELD_OUT_DIRECT",
        "m5_contract_artifact_id": args.contract_artifact_id,
        "m5_contract_artifact_digest": args.contract_artifact_digest,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "selection": "first_400",
            "examples": EXPECTED_EXAMPLES,
            "labels": EXPECTED_LABELS,
        },
        "protocol": {
            "question": QUESTION,
            "option_semantics": "label_text.replace('_', ' ')",
            "label_order": "sorted(set(label_text))",
            "banking77_task_training_used": False,
            "retrieved_labeled_examples_per_prediction": 0,
            "full_k": True,
            "adaptive_budget": False,
            "relation_refinement": False,
            "state_once": True,
        },
        "runtime": runtime,
        "metrics": metrics,
        "headline": {
            "id": "laya.app.banking77_full",
            "direction": "higher",
            "target": LAYA_TARGET,
            "value": accuracy,
            "status": status,
            "matched": True,
        },
        "jev_banking77_24shot": {
            "status": "NOT_COMPARABLE",
            "reason": (
                "Jev 0.924 authority uses 24 retrieved labeled examples per "
                "prediction; this Hira lane uses zero Banking77 demonstrations."
            ),
            "score_populated": False,
        },
        "used_for_model_selection": False,
        "model_weights_changed_after_exposure": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out / "per-example.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    scorecard = {
        "schema_version": "hira-v0-mainline-m5-scorecard-input-v1",
        "candidate": "hira-v0-m4-frozen",
        "metrics": {
            "laya.app.banking77_full": accuracy,
        },
        "explicit_status": {
            "jev.banking77.24shot.fulltest": "NOT_COMPARABLE",
        },
    }
    (args.out / "scorecard-input.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("HIRA_V0_M5_BANKING77_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
