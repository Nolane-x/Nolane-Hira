from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle


SCHEMA_VERSION = "hira-v0-mainline-m5-xnli-v1"
OUTCOME = "HIRA_V0_M5_XNLI_READY"

DATASET_ID = "facebook/xnli"
DATASET_REVISION = "b8dd5d7af51114dbda02c0e3f6133f332186418e"
SPLIT = "test"
PER_LANGUAGE = 300
LANGUAGES = (
    "en", "de", "fr", "es", "ru", "tr", "ar", "hi", "ur",
    "vi", "th", "el", "bg", "zh", "sw",
)
QUESTION = "What is the relationship between `premise` and `hypothesis`?"
OPTIONS = (
    LogicalOption("entailment", "the premise implies the hypothesis is true"),
    LogicalOption("neutral", "the premise neither implies nor contradicts the hypothesis"),
    LogicalOption("contradiction", "the premise implies the hypothesis is false"),
)
TARGETS = {
    "laya.xnli.en": 0.86,
    "laya.xnli.vi": 0.7233333333333334,
    "laya.xnli.other14_macro": 0.731,
}
M5A_ARTIFACT_ID = 10943744649
M5A_ARTIFACT_DIGEST = (
    "sha256:856031dec357cb0a0d0f1b0fff82a5d7661832a5c67aebc7c2b45d309ec45230"
)


def canonical_state(premise: str, hypothesis: str) -> str:
    return json.dumps(
        {"premise": premise, "hypothesis": hypothesis},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def text_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def compare(value: float, target: float) -> str:
    if value > target:
        return "WIN"
    if value < target:
        return "LOSS"
    return "TIE"


def load_language(language: str) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset(
        DATASET_ID,
        language,
        split=SPLIT,
        revision=DATASET_REVISION,
    )
    if len(rows) < PER_LANGUAGE:
        raise RuntimeError(
            f"XNLI {language} requires >= {PER_LANGUAGE} rows; got {len(rows)}"
        )
    return [dict(row) for row in list(rows)[:PER_LANGUAGE]]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--contract-receipt", type=Path, required=True)
    parser.add_argument("--contract-artifact-id", type=int, required=True)
    parser.add_argument("--contract-artifact-digest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.contract_receipt.read_text(encoding="utf-8"))
    if receipt.get("outcome") != "HIRA_V0_M5_CONTRACT_READY":
        raise RuntimeError("M5-A contract is not qualified")
    if args.contract_artifact_id != M5A_ARTIFACT_ID:
        raise RuntimeError("M5-A artifact ID changed")
    if args.contract_artifact_digest != M5A_ARTIFACT_DIGEST:
        raise RuntimeError("M5-A artifact digest changed")

    model = load_hira_v0_m4_bundle(args.bundle)
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("XNLI M5 runtime has trainable parameters")

    schema, schema_receipt = model.runtime.compile_schema(
        primitive="choice",
        question_text=QUESTION,
        options=OPTIONS,
        use_cache=True,
        include_token_artifacts=True,
    )
    if schema_receipt.option_count != 3:
        raise RuntimeError("XNLI schema K changed")

    print("HIRA_V0_M5_XNLI_FINAL_EXPOSURE_BEGIN", flush=True)

    raw: list[dict] = []
    per_language: dict[str, dict] = {}
    total_before = model.runtime.state_encode_calls

    for language in LANGUAGES:
        rows = load_language(language)
        correct = 0
        latencies: list[float] = []
        language_before = model.runtime.state_encode_calls

        for index, row in enumerate(rows):
            state = canonical_state(str(row["premise"]), str(row["hypothesis"]))
            gold = int(row["label"])
            if gold not in (0, 1, 2):
                raise RuntimeError(f"XNLI {language} invalid gold label {gold}")

            start = time.perf_counter()
            session = model.open_session(state)
            output = session.decide_compiled(schema)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            p = output.probabilities.detach().cpu().to(torch.float64)
            pred = int(p.argmax().item())
            ok = pred == gold
            correct += int(ok)

            raw.append({
                "language": language,
                "index": index,
                "state_sha256": text_sha256(state),
                "gold_index": gold,
                "predicted_index": pred,
                "correct": ok,
                "confidence": float(p.max()),
                "probability_mass_error": abs(float(p.sum()) - 1.0),
                "candidate_budget": int(output.hira.candidate_budget.item()),
                "relation_delta_max_abs": float(
                    output.hira.relation_delta.detach().abs().max().cpu()
                ),
                "latency_ms": elapsed_ms,
                "probabilities": [float(v) for v in p.tolist()],
            })

        calls = model.runtime.state_encode_calls - language_before
        if calls != PER_LANGUAGE:
            raise RuntimeError(f"XNLI {language} state-once violation")
        per_language[language] = {
            "cases": PER_LANGUAGE,
            "accuracy": correct / PER_LANGUAGE,
            "state_encode_calls": calls,
            "mean_ms_cpu_ci": sum(latencies) / len(latencies),
        }

    total_calls = model.runtime.state_encode_calls - total_before
    expected_total = PER_LANGUAGE * len(LANGUAGES)
    if total_calls != expected_total:
        raise RuntimeError("XNLI total state-once count changed")
    if max(float(r["probability_mass_error"]) for r in raw) > 1e-6:
        raise RuntimeError("XNLI probability mass contract failed")
    if set(int(r["candidate_budget"]) for r in raw) != {3}:
        raise RuntimeError("XNLI full-K contract failed")
    if max(float(r["relation_delta_max_abs"]) for r in raw) != 0.0:
        raise RuntimeError("XNLI relation refinement changed")

    en = float(per_language["en"]["accuracy"])
    vi = float(per_language["vi"]["accuracy"])
    other14_languages = [x for x in LANGUAGES if x != "en"]
    other14 = sum(float(per_language[x]["accuracy"]) for x in other14_languages) / 14.0

    values = {
        "laya.xnli.en": en,
        "laya.xnli.vi": vi,
        "laya.xnli.other14_macro": other14,
    }
    headlines = [
        {
            "id": key,
            "direction": "higher",
            "target": TARGETS[key],
            "value": values[key],
            "status": compare(values[key], TARGETS[key]),
            "matched": True,
        }
        for key in TARGETS
    ]

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "M5_MATCHED_MULTILINGUAL_HELD_OUT",
        "source_authority": {
            "laya_repository": "NandhaKishorM/laya",
            "laya_commit": "42626c348753fbb17572a813127df2278a1ec527",
            "laya_build_benchmark_blob": "8b131f2d0caa0c0b096c1654214464b02aef1461",
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": SPLIT,
            "languages": list(LANGUAGES),
            "selection": "first_300_each",
            "total_cases": expected_total,
        },
        "protocol": {
            "question": QUESTION,
            "option_ids": [o.option_id for o in OPTIONS],
            "task_training_used": False,
            "full_k": True,
            "adaptive_budget": False,
            "relation_refinement": False,
            "state_once": True,
        },
        "runtime": {
            "manifest": model.manifest.to_dict(),
            "parameter_report": model.parameter_report().to_dict(),
        },
        "per_language": per_language,
        "headlines": headlines,
        "total_state_encode_calls": total_calls,
        "probability_mass_max_error": max(
            float(r["probability_mass_error"]) for r in raw
        ),
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
        for row in raw:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.out / "scorecard-input.json").write_text(
        json.dumps(
            {
                "schema_version": "hira-v0-mainline-m5-scorecard-input-v1",
                "candidate": "hira-v0-m4-frozen",
                "metrics": values,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M5_XNLI_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
