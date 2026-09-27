from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle


SCHEMA_VERSION = "hira-v0-mainline-m5-jev-btzsc-v1"
OUTCOME = "HIRA_V0_M5_JEV_BTZSC_READY"

JEV_REPOSITORY = "AbdelStark/jev-benchmarks"
JEV_COMMIT = "0d610cc53e79bcbec691312b0c4adb4a0e371642"
JEV_DATA_BLOB = "efdf58ab89dddc7dff7687f6625747c06ec9551f"
JEV_ADAPTER_BLOB = "328a5536bde2b4e0809b4b1727ccbbb6bd6da3fe"
JEV_METRICS_BLOB = "eab6060c3aa54a1ab771dbf2a05dab526b5a9be8"

DATASET_ID = "btzsc/btzsc"
DATASET_REVISION = "fef2a2ac62b69c58670047dddf045c53d7c3cb5e"
SEED = 20260917
SAMPLES_PER_DATASET = 100
QUESTION = "Which single label best describes the input text?"
EXPECTED_MANIFEST_SHA256 = (
    "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
)
CONDITIONS = (
    ("agnews", "topic", 4),
    ("emotiondair", "emotion", 6),
    ("banking77", "intent", 72),
)

JEV_TARGETS = {
    "jev.zero.ag_news.accuracy": ("agnews", "accuracy", 0.91, "higher"),
    "jev.zero.ag_news.brier": ("agnews", "brier", 0.145914, "lower"),
    "jev.zero.ag_news.coverage5": (
        "agnews", "coverage_at_error_budget", 0.83, "higher"
    ),
    "jev.zero.banking77.accuracy": ("banking77", "accuracy", 0.87, "higher"),
    "jev.zero.banking77.brier": (
        "banking77", "brier", 0.17912525885113767, "lower"
    ),
    "jev.zero.banking77.coverage5": (
        "banking77", "coverage_at_error_budget", 0.86, "higher"
    ),
    "jev.zero.emotion.accuracy": ("emotiondair", "accuracy", 0.48, "higher"),
    "jev.zero.emotion.brier": (
        "emotiondair", "brier", 0.8462892588511376, "lower"
    ),
}


@dataclass(frozen=True)
class Example:
    dataset: str
    task: str
    example_id: str
    text: str
    text_sha256: str
    labels: tuple[str, ...]
    target_index: int

    def to_dict(self) -> dict:
        row = asdict(self)
        row["labels"] = list(self.labels)
        return row


def _class_count(texts: list[str]) -> int:
    first = texts[0]
    for index in range(1, len(texts)):
        if texts[index] != first:
            return index
    raise ValueError("could not infer class count from repeated BTZSC texts")


def _balanced_indices(targets: list[int], limit: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)
    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)
    chosen: list[int] = []
    classes = sorted(by_class)
    while len(chosen) < min(limit, len(targets)):
        made_progress = False
        for class_id in classes:
            if by_class[class_id] and len(chosen) < limit:
                chosen.append(by_class[class_id].pop())
                made_progress = True
        if not made_progress:
            break
    return sorted(chosen)


def load_examples() -> list[Example]:
    from datasets import load_dataset

    output: list[Example] = []
    for dataset_offset, (name, task, expected_classes) in enumerate(CONDITIONS):
        rows = load_dataset(
            DATASET_ID,
            name=name,
            split="test",
            revision=DATASET_REVISION,
        )
        binary = [int(value) for value in rows["labels"]]
        texts = [str(value) for value in rows["text"]]
        n_classes = _class_count(texts)
        if n_classes != expected_classes:
            raise RuntimeError(
                f"{name} class count changed: {n_classes} != {expected_classes}"
            )
        total = len(rows) // n_classes
        labels = tuple(str(rows[index]["hypothesis"]) for index in range(n_classes))
        valid_sample_indices: list[int] = []
        targets: list[int] = []
        for sample_index in range(total):
            offset = sample_index * n_classes
            values = binary[offset : offset + n_classes]
            if sum(values) == 1:
                valid_sample_indices.append(sample_index)
                targets.append(values.index(1))
        selected_positions = _balanced_indices(
            targets,
            SAMPLES_PER_DATASET,
            SEED + dataset_offset,
        )
        for position in selected_positions:
            sample_index = valid_sample_indices[position]
            offset = sample_index * n_classes
            text = texts[offset]
            output.append(
                Example(
                    dataset=name,
                    task=task,
                    example_id=f"{name}:{sample_index}",
                    text=text,
                    text_sha256=sha256(text.encode()).hexdigest(),
                    labels=labels,
                    target_index=targets[position],
                )
            )
    return output


def write_manifest(path: Path, examples: list[Example]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    example.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Jev BTZSC manifest identity mismatch: "
            f"{digest} != {EXPECTED_MANIFEST_SHA256}"
        )
    return digest


def canonical_state(text: str) -> str:
    # Mirrors Jev's structured state={"text": example.text} at the Hira
    # string-state boundary using deterministic JSON serialization.
    return json.dumps(
        {"text": text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _macro_f1(targets: np.ndarray, predictions: np.ndarray, n_classes: int) -> float:
    scores = []
    for class_id in range(n_classes):
        tp = int(np.sum((targets == class_id) & (predictions == class_id)))
        fp = int(np.sum((targets != class_id) & (predictions == class_id)))
        fn = int(np.sum((targets == class_id) & (predictions != class_id)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def score_rows(rows: list[dict], *, ece_bins: int = 10, error_budget: float = 0.05) -> dict:
    targets = np.array([int(row["target_index"]) for row in rows], dtype=int)
    predicted = np.array([int(row["predicted_index"]) for row in rows], dtype=int)
    probabilities = np.array([row["probabilities"] for row in rows], dtype=float)
    confidence = probabilities.max(axis=1)
    correct = predicted == targets
    one_hot = np.eye(probabilities.shape[1])[targets]

    ece = 0.0
    edges = np.linspace(0, 1, ece_bins + 1)
    for bin_index in range(ece_bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        mask = (confidence >= lower) & (
            (confidence <= upper)
            if bin_index == ece_bins - 1
            else (confidence < upper)
        )
        if np.any(mask):
            confidence_gap = abs(
                float(np.mean(confidence[mask])) - float(np.mean(correct[mask]))
            )
            ece += float(np.mean(mask)) * confidence_gap

    selected_count = 0
    selected_risk: float | None = None
    selected_threshold: float | None = None
    for threshold in sorted(set(confidence), reverse=True):
        accepted = confidence >= threshold
        risk = float(np.mean(~correct[accepted]))
        count = int(np.sum(accepted))
        if risk <= error_budget and count > selected_count:
            selected_count = count
            selected_risk = risk
            selected_threshold = float(threshold)

    latencies = np.array([float(row["latency_seconds"]) for row in rows])
    return {
        "n": len(rows),
        "valid": len(rows),
        "failures": 0,
        "accuracy": float(np.mean(targets == predicted)),
        "macro_f1": _macro_f1(targets, predicted, probabilities.shape[1]),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "nll": float(
            -np.mean(
                np.log(
                    np.clip(
                        probabilities[np.arange(len(rows)), targets],
                        1e-12,
                        1,
                    )
                )
            )
        ),
        "ece": ece,
        "mean_confidence": float(np.mean(confidence)),
        "true_label_zero_rate": float(
            np.mean(probabilities[np.arange(len(rows)), targets] == 0)
        ),
        "coverage_at_error_budget": selected_count / len(rows),
        "risk_at_selected_coverage": selected_risk,
        "confidence_threshold_at_error_budget": selected_threshold,
        "latency_p50_seconds_cpu_ci": float(np.quantile(latencies, 0.50)),
        "latency_p95_seconds_cpu_ci": float(np.quantile(latencies, 0.95)),
    }


def status(value: float, target: float, direction: str) -> str:
    if math.isclose(value, target, rel_tol=0.0, abs_tol=1e-15):
        return "TIE"
    if direction == "higher":
        return "WIN" if value > target else "LOSS"
    if direction == "lower":
        return "WIN" if value < target else "LOSS"
    raise ValueError(direction)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--contract-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    authority = json.loads(args.contract_receipt.read_text(encoding="utf-8"))
    if authority.get("outcome") != "HIRA_V0_M5_CONTRACT_READY":
        raise RuntimeError("M5-A contract is not qualified")

    examples = load_examples()
    if len(examples) != 300:
        raise RuntimeError(f"BTZSC pilot size changed: {len(examples)}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_digest = write_manifest(args.out / "manifest.jsonl", examples)

    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        grouped[example.dataset].append(example)
    if {name: len(rows) for name, rows in grouped.items()} != {
        "agnews": 100,
        "emotiondair": 100,
        "banking77": 100,
    }:
        raise RuntimeError("BTZSC condition counts changed")

    model = load_hira_v0_m4_bundle(args.bundle)
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("Jev matched lane runtime has trainable parameters")

    print("HIRA_V0_M5_JEV_BTZSC_FINAL_EXPOSURE_BEGIN", flush=True)

    raw: list[dict] = []
    scores: dict[str, dict] = {}
    before = model.runtime.state_encode_calls

    for name, _task, expected_k in CONDITIONS:
        condition = grouped[name]
        labels = condition[0].labels
        if len(labels) != expected_k:
            raise RuntimeError(f"{name} K changed")
        if any(row.labels != labels for row in condition):
            raise RuntimeError(f"{name} label order changed inside manifest")

        options = tuple(
            LogicalOption(
                option_id=f"label_{index:03d}",
                criterion_text=label,
            )
            for index, label in enumerate(labels)
        )
        schema, schema_receipt = model.runtime.compile_schema(
            primitive="choice",
            question_text=QUESTION,
            options=options,
            use_cache=True,
            include_token_artifacts=True,
        )
        if schema_receipt.option_count != expected_k:
            raise RuntimeError(f"{name} compiled K changed")

        condition_rows: list[dict] = []
        for example in condition:
            state = canonical_state(example.text)
            start = time.perf_counter()
            session = model.open_session(state)
            output = session.decide_compiled(schema)
            elapsed = time.perf_counter() - start

            p = output.probabilities.detach().cpu().to(torch.float64)
            if p.shape != (expected_k,):
                raise RuntimeError(f"{name} probability shape changed")
            if not bool(torch.isfinite(p).all()):
                raise RuntimeError(f"{name} non-finite probability vector")
            mass_error = abs(float(p.sum()) - 1.0)
            if mass_error > 1e-6:
                raise RuntimeError(f"{name} probability mass failed")
            if int(output.hira.candidate_budget.item()) != expected_k:
                raise RuntimeError(f"{name} full-K failed")
            relation_delta = float(
                output.hira.relation_delta.detach().abs().max().cpu()
            )
            if relation_delta != 0.0:
                raise RuntimeError(f"{name} relation refinement changed")

            predicted = int(p.argmax().item())
            row = {
                "dataset": name,
                "example_id": example.example_id,
                "text_sha256": example.text_sha256,
                "state_sha256": sha256(state.encode()).hexdigest(),
                "target_index": example.target_index,
                "predicted_index": predicted,
                "labels": list(labels),
                "probabilities": [float(v) for v in p.tolist()],
                "latency_seconds": elapsed,
                "probability_mass_error": mass_error,
                "candidate_budget": expected_k,
                "relation_delta_max_abs": relation_delta,
            }
            condition_rows.append(row)
            raw.append(row)

        scores[name] = score_rows(condition_rows)

    state_calls = model.runtime.state_encode_calls - before
    if state_calls != 300:
        raise RuntimeError(f"BTZSC state-once count changed: {state_calls}")

    headlines = []
    scorecard = {}
    for target_id, (dataset, metric, target, direction) in JEV_TARGETS.items():
        value = float(scores[dataset][metric])
        headlines.append({
            "id": target_id,
            "dataset": dataset,
            "metric": metric,
            "direction": direction,
            "target": target,
            "value": value,
            "status": status(value, target, direction),
            "matched": True,
        })
        scorecard[target_id] = value

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "M5_JEV_EXACT_BTZSC_ZERO_SHOT",
        "jev_authority": {
            "repository": JEV_REPOSITORY,
            "commit": JEV_COMMIT,
            "data_blob": JEV_DATA_BLOB,
            "adapter_blob": JEV_ADAPTER_BLOB,
            "metrics_blob": JEV_METRICS_BLOB,
            "resolved_model": "jev-1.13.0",
            "protocol": "pilot-v1-preregistered",
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "seed": SEED,
            "samples_per_condition": SAMPLES_PER_DATASET,
            "manifest_sha256_observed": manifest_digest,
        },
        "input_contract": {
            "question": QUESTION,
            "option_id_format": "label_{index:03d}",
            "criterion_text": "exact BTZSC hypothesis string",
            "state": '{"text": <exact example text>}',
            "task_specific_demos": 0,
            "weight_updates": False,
        },
        "runtime": {
            "manifest": model.manifest.to_dict(),
            "parameter_report": model.parameter_report().to_dict(),
        },
        "scores": scores,
        "headlines": headlines,
        "state_encode_calls": state_calls,
        "systems_comparison": {
            "status": "NOT_COMPARABLE",
            "reason": (
                "Hira latency is local GitHub-hosted CPU; Jev authority latency "
                "is hosted service from France."
            ),
        },
        "used_for_model_selection": False,
        "model_weights_changed_after_exposure": False,
        "production_ready_claimed": False,
    }

    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out / "per-example.jsonl").open("w", encoding="utf-8") as handle:
        for row in raw:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.out / "scorecard-input.json").write_text(
        json.dumps(
            {
                "schema_version": "hira-v0-mainline-m5-scorecard-input-v1",
                "candidate": "hira-v0-m4-frozen",
                "metrics": scorecard,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M5_JEV_BTZSC_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
