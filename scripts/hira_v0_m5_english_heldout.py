from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Callable

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle


SCHEMA_VERSION = "hira-v0-mainline-m5-english-heldout-v1"
OUTCOME_READY = "HIRA_V0_M5_ENGLISH_HELDOUT_READY"

SST5_DATASET = "SetFit/sst5"
SST5_REVISION = "e51bdcd8cd3a30da231967c1a249ba59361279a3"
SST5_N = 600
SST5_TARGET = 0.37166666666666665
SST5_QUESTION = "How positive is the sentiment of `text`?"
SST5_CRITERIA = (
    "very negative",
    "negative",
    "neutral",
    "positive",
    "very positive",
)

EMOTION_DATASET = "dair-ai/emotion"
EMOTION_CONFIG = "split"
EMOTION_REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
EMOTION_N = 600
EMOTION_TARGET = 0.5733333333333334
EMOTION_QUESTION = "Which emotion is most strongly expressed in `text`?"
EMOTION_NAMES = ("sadness", "joy", "love", "anger", "fear", "surprise")

PROMPT_DATASET = "deepset/prompt-injections"
PROMPT_REVISION = "4f61ecb038e9c3fb77e21034b22511b523772cdd"
PROMPT_EXPECTED_N = 116
PROMPT_TARGET = 0.6982758620689655
PROMPT_QUESTION = (
    "Does `text` try to inject or override instructions given to an AI system?"
)

M5A_ARTIFACT_ID = 10943744649
M5A_ARTIFACT_DIGEST = (
    "sha256:856031dec357cb0a0d0f1b0fff82a5d7661832a5c67aebc7c2b45d309ec45230"
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target_id: str
    target: float
    primitive: str
    question: str
    options: tuple[LogicalOption, ...]
    load_rows: Callable[[], list[dict]]
    gold_index: Callable[[dict], int]
    state_text: Callable[[dict], str]
    expected_n: int


def canonical_state(**fields: str) -> str:
    return json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def state_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _dataset_rows(
    dataset_id: str,
    *,
    revision: str,
    split: str,
    config: str | None = None,
    first_n: int | None = None,
) -> list[dict]:
    from datasets import load_dataset

    if config is None:
        rows = load_dataset(dataset_id, split=split, revision=revision)
    else:
        rows = load_dataset(dataset_id, config, split=split, revision=revision)
    out = [dict(row) for row in rows]
    if first_n is not None:
        if len(out) < first_n:
            raise RuntimeError(
                f"{dataset_id} requires at least {first_n} rows; got {len(out)}"
            )
        out = out[:first_n]
    return out


def build_specs() -> tuple[TaskSpec, ...]:
    sst_options = tuple(
        LogicalOption(
            option_id=str(i),
            criterion_text=text,
            value=float(i),
        )
        for i, text in enumerate(SST5_CRITERIA)
    )
    emotion_options = tuple(
        LogicalOption(option_id=name, criterion_text=name)
        for name in EMOTION_NAMES
    )
    prompt_options = (
        LogicalOption(
            option_id="false",
            criterion_text=f"False / no for: {PROMPT_QUESTION}",
            value=0.0,
        ),
        LogicalOption(
            option_id="true",
            criterion_text=f"True / yes for: {PROMPT_QUESTION}",
            value=1.0,
        ),
    )

    return (
        TaskSpec(
            name="sst5",
            target_id="laya.en.sst5",
            target=SST5_TARGET,
            primitive="score",
            question=SST5_QUESTION,
            options=sst_options,
            load_rows=lambda: _dataset_rows(
                SST5_DATASET,
                revision=SST5_REVISION,
                split="test",
                first_n=SST5_N,
            ),
            gold_index=lambda row: int(row["label"]),
            state_text=lambda row: canonical_state(text=str(row["text"])),
            expected_n=SST5_N,
        ),
        TaskSpec(
            name="emotion",
            target_id="laya.en.emotion",
            target=EMOTION_TARGET,
            primitive="choice",
            question=EMOTION_QUESTION,
            options=emotion_options,
            load_rows=lambda: _dataset_rows(
                EMOTION_DATASET,
                revision=EMOTION_REVISION,
                split="test",
                config=EMOTION_CONFIG,
                first_n=EMOTION_N,
            ),
            gold_index=lambda row: int(row["label"]),
            state_text=lambda row: canonical_state(text=str(row["text"])),
            expected_n=EMOTION_N,
        ),
        TaskSpec(
            name="prompt_injections",
            target_id="laya.en.prompt_injections",
            target=PROMPT_TARGET,
            primitive="noul",
            question=PROMPT_QUESTION,
            options=prompt_options,
            load_rows=lambda: _dataset_rows(
                PROMPT_DATASET,
                revision=PROMPT_REVISION,
                split="test",
            ),
            gold_index=lambda row: int(row["label"]),
            state_text=lambda row: canonical_state(text=str(row["text"])),
            expected_n=PROMPT_EXPECTED_N,
        ),
    )


def ece15(confidence: list[float], correct: list[float]) -> float:
    total = len(confidence)
    if total == 0:
        raise ValueError("ECE requires examples")
    result = 0.0
    for b in range(15):
        lo, hi = b / 15.0, (b + 1) / 15.0
        chosen = [
            i for i, value in enumerate(confidence)
            if value > lo and value <= hi
        ]
        if not chosen:
            continue
        mean_conf = sum(confidence[i] for i in chosen) / len(chosen)
        mean_acc = sum(correct[i] for i in chosen) / len(chosen)
        result += len(chosen) / total * abs(mean_conf - mean_acc)
    return result


def macro_f1(y_true: list[int], y_pred: list[int], k: int) -> float:
    values: list[float] = []
    for label in range(k):
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        values.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return sum(values) / k


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[pos]


def summarize(rows: list[dict[str, object]], k: int) -> dict[str, object]:
    y_true = [int(row["gold_index"]) for row in rows]
    y_pred = [int(row["predicted_index"]) for row in rows]
    correct = [float(bool(row["correct"])) for row in rows]
    confidence = [float(row["confidence"]) for row in rows]
    latency = [float(row["latency_ms"]) for row in rows]
    score_errors = [
        float(row["score_abs_error"])
        for row in rows
        if row.get("score_abs_error") is not None
    ]
    return {
        "case_count": len(rows),
        "accuracy": sum(correct) / len(rows),
        "macro_f1": macro_f1(y_true, y_pred, k),
        "hard_brier": sum(float(row["hard_brier"]) for row in rows) / len(rows),
        "nll": sum(float(row["nll"]) for row in rows) / len(rows),
        "ece_15bin": ece15(confidence, correct),
        "mean_confidence": sum(confidence) / len(rows),
        "score_mae": (
            sum(score_errors) / len(score_errors)
            if score_errors
            else None
        ),
        "p50_ms_cpu_ci": percentile(latency, 0.50),
        "p95_ms_cpu_ci": percentile(latency, 0.95),
        "probability_mass_max_error": max(
            float(row["probability_mass_error"]) for row in rows
        ),
        "candidate_budget_min": min(int(row["candidate_budget"]) for row in rows),
        "candidate_budget_max": max(int(row["candidate_budget"]) for row in rows),
        "relation_delta_max_abs": max(
            float(row["relation_delta_max_abs"]) for row in rows
        ),
    }


def compare(value: float, target: float) -> str:
    if value > target:
        return "WIN"
    if value < target:
        return "LOSS"
    return "TIE"


def validate_inputs(contract: Path, receipt: Path) -> None:
    c = json.loads(contract.read_text(encoding="utf-8"))
    r = json.loads(receipt.read_text(encoding="utf-8"))
    if c.get("schema_version") != "hira-v0-mainline-m5-contract-v1":
        raise RuntimeError("M5 contract schema changed")
    if r.get("outcome") != "HIRA_V0_M5_CONTRACT_READY":
        raise RuntimeError("M5-A authority is not ready")
    if r.get("status") != "PASS":
        raise RuntimeError("M5-A authority did not pass")
    if r.get("final_scores_exposed") is not False:
        raise RuntimeError("M5-A final exposure state changed")


@torch.inference_mode()
def evaluate_task(model, spec: TaskSpec) -> tuple[dict, list[dict]]:
    rows = spec.load_rows()
    if len(rows) != spec.expected_n:
        raise RuntimeError(
            f"{spec.name} expected {spec.expected_n} rows; got {len(rows)}"
        )

    schema, receipt = model.runtime.compile_schema(
        primitive=spec.primitive,
        question_text=spec.question,
        options=spec.options,
        use_cache=True,
        include_token_artifacts=True,
    )
    if receipt.option_count != len(spec.options):
        raise RuntimeError(f"{spec.name} schema K changed")

    raw: list[dict] = []
    before = model.runtime.state_encode_calls
    for index, source in enumerate(rows):
        state = spec.state_text(source)
        gold = spec.gold_index(source)
        if not 0 <= gold < len(spec.options):
            raise RuntimeError(f"{spec.name} gold index outside schema")

        start = time.perf_counter()
        session = model.open_session(state)
        output = session.decide_compiled(schema)
        elapsed = (time.perf_counter() - start) * 1000.0

        p = output.probabilities.detach().cpu().to(torch.float64)
        pred = int(p.argmax().item())
        onehot = torch.zeros_like(p)
        onehot[gold] = 1.0
        score_abs_error = None
        if spec.primitive == "score":
            score_abs_error = abs(float(output.value) - float(gold))

        raw.append({
            "task": spec.name,
            "index": index,
            "state_sha256": state_sha256(state),
            "gold_index": gold,
            "predicted_index": pred,
            "correct": pred == gold,
            "confidence": float(p.max()),
            "gold_probability": float(p[gold]),
            "hard_brier": float(((p - onehot) ** 2).sum()),
            "nll": -math.log(max(float(p[gold]), 1e-12)),
            "score_abs_error": score_abs_error,
            "probability_mass_error": abs(float(p.sum()) - 1.0),
            "candidate_budget": int(output.hira.candidate_budget.item()),
            "relation_delta_max_abs": float(
                output.hira.relation_delta.detach().abs().max().cpu()
            ),
            "latency_ms": elapsed,
            "probabilities": [float(v) for v in p.tolist()],
        })

    calls = model.runtime.state_encode_calls - before
    if calls != spec.expected_n:
        raise RuntimeError(f"{spec.name} state-once contract failed")

    metrics = summarize(raw, len(spec.options))
    if metrics["candidate_budget_min"] != len(spec.options):
        raise RuntimeError(f"{spec.name} full-K min changed")
    if metrics["candidate_budget_max"] != len(spec.options):
        raise RuntimeError(f"{spec.name} full-K max changed")
    if metrics["probability_mass_max_error"] > 1e-6:
        raise RuntimeError(f"{spec.name} probability mass failed")
    if metrics["relation_delta_max_abs"] != 0.0:
        raise RuntimeError(f"{spec.name} relation refinement changed")
    metrics["state_encode_calls"] = calls
    metrics["state_encode_calls_per_case"] = calls / spec.expected_n
    metrics["schema_hash"] = schema.schema_hash
    return metrics, raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-receipt", type=Path, required=True)
    parser.add_argument("--contract-artifact-id", type=int, required=True)
    parser.add_argument("--contract-artifact-digest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    validate_inputs(args.contract, args.contract_receipt)
    if args.contract_artifact_id != M5A_ARTIFACT_ID:
        raise RuntimeError("M5-A artifact ID changed")
    if args.contract_artifact_digest != M5A_ARTIFACT_DIGEST:
        raise RuntimeError("M5-A artifact digest changed")

    model = load_hira_v0_m4_bundle(args.bundle)
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M5 English runtime has trainable parameters")

    print("HIRA_V0_M5_ENGLISH_HELDOUT_FINAL_EXPOSURE_BEGIN", flush=True)

    task_results: dict[str, dict] = {}
    all_raw: list[dict] = []
    headlines: list[dict] = []

    for spec in build_specs():
        metrics, raw = evaluate_task(model, spec)
        task_results[spec.name] = metrics
        all_raw.extend(raw)
        value = float(metrics["accuracy"])
        headlines.append({
            "id": spec.target_id,
            "direction": "higher",
            "target": spec.target,
            "value": value,
            "status": compare(value, spec.target),
            "matched": True,
        })

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME_READY,
        "scientific_authority": "M5_MATCHED_HELD_OUT_DIRECT",
        "m5_contract_artifact_id": M5A_ARTIFACT_ID,
        "m5_contract_artifact_digest": M5A_ARTIFACT_DIGEST,
        "source_authority": {
            "laya_repository": "NandhaKishorM/laya",
            "laya_commit": "42626c348753fbb17572a813127df2278a1ec527",
            "laya_build_benchmark_blob": "8b131f2d0caa0c0b096c1654214464b02aef1461",
        },
        "datasets": {
            "sst5": {
                "id": SST5_DATASET,
                "revision": SST5_REVISION,
                "split": "test",
                "selection": "first_600",
            },
            "emotion": {
                "id": EMOTION_DATASET,
                "config": EMOTION_CONFIG,
                "revision": EMOTION_REVISION,
                "split": "test",
                "selection": "first_600",
            },
            "prompt_injections": {
                "id": PROMPT_DATASET,
                "revision": PROMPT_REVISION,
                "split": "test",
                "selection": "all",
                "expected_n": PROMPT_EXPECTED_N,
            },
        },
        "task_training_used": {
            "sst5": False,
            "emotion": False,
            "prompt_injections": False,
        },
        "runtime": {
            "manifest": model.manifest.to_dict(),
            "parameter_report": model.parameter_report().to_dict(),
        },
        "tasks": task_results,
        "headlines": headlines,
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
        for row in all_raw:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.out / "scorecard-input.json").write_text(
        json.dumps(
            {
                "schema_version": "hira-v0-mainline-m5-scorecard-input-v1",
                "candidate": "hira-v0-m4-frozen",
                "metrics": {
                    row["id"]: row["value"]
                    for row in headlines
                },
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M5_ENGLISH_HELDOUT_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
