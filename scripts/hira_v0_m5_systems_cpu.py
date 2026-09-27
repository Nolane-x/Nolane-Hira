from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time

import numpy as np
import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import A13_MODEL, A13_REVISION, load_hira_v0_m4_bundle


SCHEMA_VERSION = "hira-v0-mainline-m5-systems-cpu-v1"
OUTCOME = "HIRA_V0_M5_SYSTEMS_CPU_READY"

LAYA_REPOSITORY = "NandhaKishorM/laya"
LAYA_COMMIT = "42626c348753fbb17572a813127df2278a1ec527"
LAYA_BENCH_LATENCY_BLOB = "e97ddbb499fcfd86ee626ceab238608cddbfb426"
Q_COUNTS = (1, 5, 10, 50)
WARMUP = 2
REPS = 10

LAYA_T4_TARGETS = {
    1: 32.8,
    5: 40.1,
    10: 72.3,
    50: 337.4,
}

STATE_EN = {
    "ticket": {
        "subject": "Payout failing",
        "messages": [
            {
                "from": "customer",
                "text": (
                    "Hi, my Stripe payouts have failed for 3 days and I am losing "
                    "sales. Please help ASAP. " * 6
                ),
            }
        ],
    }
}
Q_NOUL = {
    "type": "noul",
    "instructions": "Does `ticket.messages[0].text` express urgency?",
}
Q_CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
        "billing": "payments",
        "technical": "bugs and integrations",
        "sales": "pricing",
    },
}


def state_text() -> str:
    return json.dumps(
        STATE_EN,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def choice_options() -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(option_id=key, criterion_text=value)
        for key, value in Q_CHOICE["criteria"].items()
    )


def noul_options() -> tuple[LogicalOption, ...]:
    question = Q_NOUL["instructions"]
    return (
        LogicalOption(
            option_id="false",
            criterion_text=f"False / no for: {question}",
            value=0.0,
        ),
        LogicalOption(
            option_id="true",
            criterion_text=f"True / yes for: {question}",
            value=1.0,
        ),
    )


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "min_ms": float(np.min(values)),
    }


def directory_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


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

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION)

    rss_before_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    model = load_hira_v0_m4_bundle(
        args.bundle,
        semantic_snapshot_dir=snapshot,
    )
    local_load_ms = (time.perf_counter() - started) * 1000.0
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("systems authority runtime has trainable parameters")
    if model.manifest.production_ready:
        raise RuntimeError("systems authority overclaims production readiness")

    c_options = choice_options()
    n_options = noul_options()

    # Force the two dynamic schemas into the bounded cache before timing. Laya's
    # raw benchmark also performs warmups before measured repetitions.
    warm = model.open_session(state_text())
    warm.decide(
        primitive="choice",
        question_text=Q_CHOICE["instructions"],
        options=c_options,
        use_schema_cache=True,
    )
    warm.decide(
        primitive="noul",
        question_text=Q_NOUL["instructions"],
        options=n_options,
        use_schema_cache=True,
    )

    before_measured_calls = model.runtime.state_encode_calls
    per_q: dict[str, dict] = {}

    for n in Q_COUNTS:
        def run_call() -> None:
            session = model.open_session(state_text())
            for index in range(n):
                if index % 2 == 0:
                    session.decide(
                        primitive="choice",
                        question_text=Q_CHOICE["instructions"],
                        options=c_options,
                        use_schema_cache=True,
                    )
                else:
                    session.decide(
                        primitive="noul",
                        question_text=Q_NOUL["instructions"],
                        options=n_options,
                        use_schema_cache=True,
                    )
            if session.query_count != n:
                raise RuntimeError("systems query count changed")

        for _ in range(WARMUP):
            run_call()
        values: list[float] = []
        for _ in range(REPS):
            t = time.perf_counter()
            run_call()
            values.append((time.perf_counter() - t) * 1000.0)

        summary = summarize_ms(values)
        summary["ms_per_question_p50"] = summary["p50_ms"] / n
        summary["warmup_calls"] = WARMUP
        summary["measured_calls"] = REPS
        summary["questions_per_call"] = n
        summary["laya_t4_target_ms"] = LAYA_T4_TARGETS[n]
        summary["laya_t4_status"] = "NOT_COMPARABLE"
        summary["laya_t4_reason"] = (
            "Hira authority uses GitHub-hosted CPU; Laya target is T4 GPU."
        )
        per_q[str(n)] = summary

    measured_state_calls = model.runtime.state_encode_calls - before_measured_calls
    expected_state_calls = len(Q_COUNTS) * (WARMUP + REPS)
    if measured_state_calls != expected_state_calls:
        raise RuntimeError(
            f"systems state-once call count changed: "
            f"{measured_state_calls} != {expected_state_calls}"
        )

    rss_after_benchmark = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    wheels = list((args.bundle / "python").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("systems authority expected exactly one packaged wheel")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "M5_SYSTEMS_CPU_MATCHED_CALL_SHAPE",
        "laya_authority": {
            "repository": LAYA_REPOSITORY,
            "commit": LAYA_COMMIT,
            "bench_latency_blob": LAYA_BENCH_LATENCY_BLOB,
            "call_shape": "one state + Q typed questions",
            "warmup": WARMUP,
            "reps": REPS,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "runtime": {
            "local_load_ms_excluding_snapshot_download": local_load_ms,
            "parameter_report": model.parameter_report().to_dict(),
            "manifest": model.manifest.to_dict(),
            "schema_cache_info": model.schema_cache_info(),
            "bundle_bytes": directory_bytes(args.bundle),
            "wheel_bytes": wheels[0].stat().st_size,
        },
        "memory": {
            "process_peak_rss_before_load_kib": int(rss_before_load),
            "process_peak_rss_after_load_kib": int(rss_after_load),
            "process_peak_rss_after_benchmark_kib": int(rss_after_benchmark),
            "note": (
                "ru_maxrss is process-level peak RSS on the authority runner, "
                "not standalone model residency."
            ),
        },
        "latency": per_q,
        "measured_state_encode_calls": measured_state_calls,
        "expected_state_encode_calls": expected_state_calls,
        "state_once": measured_state_calls == expected_state_calls,
        "systems_target_cells": {
            f"laya.latency.t4.q{n}": "NOT_COMPARABLE"
            for n in Q_COUNTS
        },
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M5_SYSTEMS_CPU_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
