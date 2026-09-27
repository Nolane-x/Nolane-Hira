from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle


SCHEMA_VERSION = "hira-v0-mainline-m5-massive-intent-v1"
OUTCOME = "HIRA_V0_M5_MASSIVE_INTENT_READY"

LAYA_REPOSITORY = "NandhaKishorM/laya"
LAYA_COMMIT = "42626c348753fbb17572a813127df2278a1ec527"
LAYA_BUILD_BENCHMARK_BLOB = "8b131f2d0caa0c0b096c1654214464b02aef1461"

DATASET_ID = "mteb/amazon_massive_intent"
DATASET_REVISION = "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
LANGUAGES = (
    "en", "de", "fr", "es", "pt", "ru", "tr",
    "ar", "hi", "ta", "zh-CN", "ja", "ko", "sw",
)
SEED = 13
N_OPTS = 20
PER_LANG = 300
QUESTION = "What is the user asking for in `utterance`?"
TARGETS = {
    "laya.massive.intent.en": 0.7833333333333333,
    "laya.massive.intent.other13_macro": 0.451,
}


def canonical_state(text: str) -> str:
    return json.dumps(
        {"utterance": text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def criterion_text(label: str) -> str:
    return label.replace("_", " ").replace(".", ": ")


def build_options(
    rng: random.Random,
    gold: str,
    all_labels: list[str],
) -> tuple[tuple[LogicalOption, ...], int, list[str]]:
    pool = [x for x in all_labels if x != gold]
    keys = [gold] + rng.sample(pool, min(N_OPTS - 1, len(pool)))
    rng.shuffle(keys)
    options = tuple(
        LogicalOption(option_id=key, criterion_text=criterion_text(key))
        for key in keys
    )
    return options, keys.index(gold), keys


def compare(value: float, target: float) -> str:
    if value > target:
        return "WIN"
    if value < target:
        return "LOSS"
    return "TIE"


def load_language(language: str) -> tuple[list[dict], list[str]]:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        language,
        split="test",
        revision=DATASET_REVISION,
    )
    if len(dataset) < PER_LANG:
        raise RuntimeError(
            f"MASSIVE {language} requires >= {PER_LANG} rows; got {len(dataset)}"
        )
    rows = [dict(row) for row in list(dataset)[:PER_LANG]]
    all_labels = sorted(set(str(x) for x in dataset["label_text"]))
    if len(all_labels) < N_OPTS:
        raise RuntimeError(
            f"MASSIVE {language} label space too small: {len(all_labels)}"
        )
    return rows, all_labels


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

    model = load_hira_v0_m4_bundle(args.bundle)
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("MASSIVE authority runtime has trainable parameters")

    print("HIRA_V0_M5_MASSIVE_INTENT_FINAL_EXPOSURE_BEGIN", flush=True)

    raw: list[dict] = []
    per_language: dict[str, dict] = {}
    total_before = model.runtime.state_encode_calls

    for language in LANGUAGES:
        rows, all_labels = load_language(language)
        rng = random.Random(SEED)
        correct = 0
        compile_ms: list[float] = []
        decision_ms: list[float] = []
        language_before = model.runtime.state_encode_calls

        for index, row in enumerate(rows):
            gold_label = str(row["label_text"])
            options, gold_index, option_keys = build_options(
                rng, gold_label, all_labels
            )
            state = canonical_state(str(row["text"]))

            t0 = time.perf_counter()
            schema, receipt = model.runtime.compile_schema(
                primitive="choice",
                question_text=QUESTION,
                options=options,
                use_cache=True,
                include_token_artifacts=True,
            )
            compile_elapsed = (time.perf_counter() - t0) * 1000.0
            if receipt.option_count != N_OPTS:
                raise RuntimeError("MASSIVE compiled K changed")

            t1 = time.perf_counter()
            session = model.open_session(state)
            output = session.decide_compiled(schema)
            decision_elapsed = (time.perf_counter() - t1) * 1000.0

            p = output.probabilities.detach().cpu().to(torch.float64)
            if p.shape != (N_OPTS,):
                raise RuntimeError("MASSIVE probability shape changed")
            if abs(float(p.sum()) - 1.0) > 1e-6:
                raise RuntimeError("MASSIVE probability mass failed")
            if int(output.hira.candidate_budget.item()) != N_OPTS:
                raise RuntimeError("MASSIVE full-K failed")
            relation_delta = float(
                output.hira.relation_delta.detach().abs().max().cpu()
            )
            if relation_delta != 0.0:
                raise RuntimeError("MASSIVE relation refinement changed")

            pred = int(p.argmax().item())
            ok = pred == gold_index
            correct += int(ok)
            compile_ms.append(compile_elapsed)
            decision_ms.append(decision_elapsed)

            raw.append({
                "language": language,
                "index": index,
                "state_sha256": sha256(state.encode()).hexdigest(),
                "gold_label": gold_label,
                "gold_index": gold_index,
                "option_keys": option_keys,
                "predicted_index": pred,
                "predicted_label": option_keys[pred],
                "correct": ok,
                "probabilities": [float(v) for v in p.tolist()],
                "schema_hash": schema.schema_hash,
                "schema_compile_ms": compile_elapsed,
                "decision_ms_cpu_ci": decision_elapsed,
                "candidate_budget": int(output.hira.candidate_budget.item()),
                "relation_delta_max_abs": relation_delta,
            })

        calls = model.runtime.state_encode_calls - language_before
        if calls != PER_LANG:
            raise RuntimeError(f"MASSIVE {language} state-once violation")
        per_language[language] = {
            "cases": PER_LANG,
            "label_space": len(all_labels),
            "accuracy": correct / PER_LANG,
            "state_encode_calls": calls,
            "mean_schema_compile_ms_cpu_ci": sum(compile_ms) / len(compile_ms),
            "mean_decision_ms_cpu_ci": sum(decision_ms) / len(decision_ms),
        }

    total_calls = model.runtime.state_encode_calls - total_before
    expected = PER_LANG * len(LANGUAGES)
    if total_calls != expected:
        raise RuntimeError("MASSIVE total state-once count changed")

    en = float(per_language["en"]["accuracy"])
    other = [lang for lang in LANGUAGES if lang != "en"]
    other13 = sum(float(per_language[x]["accuracy"]) for x in other) / 13.0
    values = {
        "laya.massive.intent.en": en,
        "laya.massive.intent.other13_macro": other13,
    }
    headlines = [
        {
            "id": key,
            "direction": "higher",
            "target": target,
            "value": values[key],
            "status": compare(values[key], target),
            "matched": True,
        }
        for key, target in TARGETS.items()
    ]

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "M5_MATCHED_DYNAMIC_SCHEMA_MULTILINGUAL",
        "laya_authority": {
            "repository": LAYA_REPOSITORY,
            "commit": LAYA_COMMIT,
            "build_benchmark_blob": LAYA_BUILD_BENCHMARK_BLOB,
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "languages": list(LANGUAGES),
            "selection": "first_300_each",
            "seed": SEED,
            "n_options": N_OPTS,
            "total_cases": expected,
        },
        "protocol": {
            "question": QUESTION,
            "option_sampling": "gold + 19 seeded distractors; shuffle",
            "criterion_transform": "replace '_' with ' '; replace '.' with ': '",
            "task_training_used": False,
            "full_k": True,
            "adaptive_budget": False,
            "relation_refinement": False,
            "state_once": True,
        },
        "runtime": {
            "manifest": model.manifest.to_dict(),
            "parameter_report": model.parameter_report().to_dict(),
            "cache_info_final": model.schema_cache_info(),
        },
        "per_language": per_language,
        "headlines": headlines,
        "total_state_encode_calls": total_calls,
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
    print("HIRA_V0_M5_MASSIVE_INTENT_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
