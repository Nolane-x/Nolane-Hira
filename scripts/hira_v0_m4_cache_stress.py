from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import resource
import time

import torch

from nmd.contracts import LogicalOption
from nmd.high_cardinality_stress import ACTIONS, COLORS, OBJECTS
from nmd.mainline import (
    HIRA_V0_MAX_K,
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    build_hira_v0_m4_runtime,
)
from nmd.schema import (
    DEFAULT_SCHEMA_CACHE_MAX_BYTES,
    DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
)
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

M4A_RUN_ID = 36327904288
M4A_ARTIFACT_ID = 10934506746
M4A_ARTIFACT_DIGEST = (
    "sha256:447c394a5a6b2246a354b21d49e9ee848c2271a636eedd019ba2ae23496324eb"
)

M4_CACHE_STRESS_SCHEMA = "hira-v0-mainline-m4-cache-stress-v1"
STRESS_UNIQUE_SCHEMAS = 21
TOUCH_AFTER = 8
PROBABILITY_TOL = 1e-6
EQUIVALENCE_TOL = 2e-5


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M4-B A13 weight SHA mismatch: {actual}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=256,
    )


def _validate_t0(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w28-candidate-receipt-v1":
        raise RuntimeError("M4-B unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M4-B T0 identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M4-B T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M4-B T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M4-B unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M4-B W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M4-B W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M4-B W34 checkpoint bytes changed")
    return checkpoint


def _validate_m4a(directory: Path) -> dict[str, object]:
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "hira-v0-mainline-m4-schema-benchmark-v1":
        raise RuntimeError("M4-B unexpected M4-A receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M4-B M4-A infrastructure did not pass")
    if receipt.get("outcome") != "HIRA_V0_M4_SCHEMA_BATCHING_READY":
        raise RuntimeError("M4-B requires qualified schema batching")
    if not all(receipt.get("gates", {}).values()):
        raise RuntimeError("M4-B M4-A gates changed")
    return receipt


def _phrase_space() -> tuple[str, ...]:
    return tuple(
        f"{color} {obj} {action} route"
        for color, obj, action in itertools.product(COLORS, OBJECTS, ACTIONS)
    )


PHRASES = _phrase_space()


def _options() -> tuple[LogicalOption, ...]:
    if len(PHRASES) < HIRA_V0_MAX_K:
        raise RuntimeError("M4-B phrase space is too small")
    return tuple(
        LogicalOption(
            option_id=f"m4-cache-route-{index:03d}",
            criterion_text=PHRASES[index],
        )
        for index in range(HIRA_V0_MAX_K)
    )


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--m4a-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    m4a = _validate_m4a(args.m4a_dir)

    encoder = _load_a13()
    model = build_hira_v0_m4_runtime(
        encoder,
        t0,
        w34,
        schema_cache_max_entries=DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
        schema_cache_max_bytes=DEFAULT_SCHEMA_CACHE_MAX_BYTES,
    )
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M4-B cache stress requires frozen runtime")
    if model.manifest.production_ready:
        raise RuntimeError("M4-B cannot promote production readiness")

    options = _options()
    state = (
        "The route record explicitly names amber anchor align route. "
        "All other route descriptions are alternatives only."
    )
    session = model.open_session(state)

    before_rss = _rss_kib()
    anchor_question = "Which route phrase matches cache authority anchor 000?"
    anchor_schema, anchor_receipt = model.runtime.compile_schema(
        primitive="choice",
        question_text=anchor_question,
        options=options,
        use_cache=True,
        include_token_artifacts=True,
    )
    anchor_output = session.decide_compiled(anchor_schema)
    anchor_probabilities = anchor_output.probabilities.detach().cpu().clone()
    anchor_option = anchor_output.selected_option_id

    early_question = "Which route phrase matches cache authority filler 001?"
    early_schema, early_receipt = model.runtime.compile_schema(
        primitive="choice",
        question_text=early_question,
        options=options,
        use_cache=True,
        include_token_artifacts=True,
    )
    early_output = session.decide_compiled(early_schema)
    early_probabilities = early_output.probabilities.detach().cpu().clone()
    early_option = early_output.selected_option_id

    rows = [
        {
            "tag": "anchor",
            "cache_hit": anchor_receipt.cache_hit,
            "cache_stored": anchor_receipt.cache_stored,
            "entry_bytes": anchor_receipt.cache_entry_bytes,
        },
        {
            "tag": "early",
            "cache_hit": early_receipt.cache_hit,
            "cache_stored": early_receipt.cache_stored,
            "entry_bytes": early_receipt.cache_entry_bytes,
        },
    ]

    compile_ms = []
    # Two unique schemas already exist. Add 6 more before touching anchor.
    for index in range(2, TOUCH_AFTER):
        question = f"Which route phrase matches cache authority filler {index:03d}?"
        start = time.perf_counter()
        _schema, receipt = model.runtime.compile_schema(
            primitive="choice",
            question_text=question,
            options=options,
            use_cache=True,
            include_token_artifacts=True,
        )
        compile_ms.append((time.perf_counter() - start) * 1000.0)
        rows.append({
            "tag": f"filler-{index:03d}",
            "cache_hit": receipt.cache_hit,
            "cache_stored": receipt.cache_stored,
            "entry_bytes": receipt.cache_entry_bytes,
        })

    # Refresh anchor to MRU before applying additional pressure.
    anchor_hit_schema, anchor_hit_receipt = model.runtime.compile_schema(
        primitive="choice",
        question_text=anchor_question,
        options=options,
        use_cache=True,
        include_token_artifacts=True,
    )
    if not anchor_hit_receipt.cache_hit or anchor_hit_schema is not anchor_schema:
        raise RuntimeError("M4-B anchor cache touch failed before pressure")

    for index in range(TOUCH_AFTER, STRESS_UNIQUE_SCHEMAS):
        question = f"Which route phrase matches cache authority filler {index:03d}?"
        start = time.perf_counter()
        _schema, receipt = model.runtime.compile_schema(
            primitive="choice",
            question_text=question,
            options=options,
            use_cache=True,
            include_token_artifacts=True,
        )
        compile_ms.append((time.perf_counter() - start) * 1000.0)
        rows.append({
            "tag": f"filler-{index:03d}",
            "cache_hit": receipt.cache_hit,
            "cache_stored": receipt.cache_stored,
            "entry_bytes": receipt.cache_entry_bytes,
        })

    after_pressure = model.schema_cache_info()
    anchor_final_schema, anchor_final_receipt = model.runtime.compile_schema(
        primitive="choice",
        question_text=anchor_question,
        options=options,
        use_cache=True,
        include_token_artifacts=True,
    )
    anchor_survived = bool(
        anchor_final_receipt.cache_hit
        and anchor_final_schema is anchor_schema
    )

    # The early schema is deliberately old enough to be evicted.
    early_recompiled, early_recompile_receipt = model.runtime.compile_schema(
        primitive="choice",
        question_text=early_question,
        options=options,
        use_cache=True,
        include_token_artifacts=True,
    )
    early_recompiled_output = session.decide_compiled(early_recompiled)
    probability_error = _max_error(
        early_probabilities,
        early_recompiled_output.probabilities.detach().cpu(),
    )
    embedding_error = _max_error(
        early_schema.option_embeddings.detach().cpu(),
        early_recompiled.option_embeddings.detach().cpu(),
    )
    selected_match = early_option == early_recompiled_output.selected_option_id
    after_recompile = model.schema_cache_info()

    after_rss = _rss_kib()
    gates = {
        "entry_bound": after_pressure["entries"] <= DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
        "byte_bound": after_pressure["bytes"] <= DEFAULT_SCHEMA_CACHE_MAX_BYTES,
        "eviction_observed": after_pressure["evictions"] > 0,
        "anchor_survived_lru": anchor_survived,
        "early_was_evicted": not early_recompile_receipt.cache_hit,
        "early_semantic_equivalent": embedding_error <= EQUIVALENCE_TOL,
        "early_probability_equivalent": probability_error <= PROBABILITY_TOL,
        "early_selected_match": selected_match,
        "full_k": (
            anchor_output.probabilities.numel() == HIRA_V0_MAX_K
            and early_recompiled_output.probabilities.numel() == HIRA_V0_MAX_K
        ),
        "state_once": model.runtime.state_encode_calls == 1,
        "relation_delta_zero": (
            float(anchor_output.hira.relation_delta.abs().max()) == 0.0
            and float(early_recompiled_output.hira.relation_delta.abs().max()) == 0.0
        ),
    }
    outcome = (
        "HIRA_V0_M4_BOUNDED_CACHE_READY"
        if all(gates.values())
        else "HIRA_V0_M4_BOUNDED_CACHE_FAIL"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": M4_CACHE_STRESS_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "m4a_authority": {
            "run_id": M4A_RUN_ID,
            "artifact_id": M4A_ARTIFACT_ID,
            "artifact_digest": M4A_ARTIFACT_DIGEST,
            "outcome": m4a["outcome"],
        },
        "cache_limits": {
            "max_entries": DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
            "max_bytes": DEFAULT_SCHEMA_CACHE_MAX_BYTES,
        },
        "unique_schema_count": STRESS_UNIQUE_SCHEMAS,
        "k": HIRA_V0_MAX_K,
        "after_pressure": after_pressure,
        "after_recompile": after_recompile,
        "anchor_survived": anchor_survived,
        "early_recompile_cache_hit": early_recompile_receipt.cache_hit,
        "early_embedding_max_error": embedding_error,
        "early_probability_max_error": probability_error,
        "early_selected_option_match": selected_match,
        "mean_pressure_compile_ms": (
            sum(compile_ms) / len(compile_ms) if compile_ms else 0.0
        ),
        "rss_peak_before_kib": before_rss,
        "rss_peak_after_kib": after_rss,
        "rss_peak_delta_kib": max(0, after_rss - before_rss),
        "state_encode_count": model.runtime.state_encode_calls,
        "gates": gates,
        "semantic_quality_promoted": False,
        "reliability_promoted": False,
        "multilingual_promoted": False,
        "production_ready_claimed": False,
        "rows": rows,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M4_CACHE_STRESS_FINAL=" + json.dumps({
        "outcome": outcome,
        "gates": gates,
        "after_pressure": after_pressure,
        "after_recompile": after_recompile,
        "anchor_survived": anchor_survived,
        "early_probability_max_error": probability_error,
        "rss_peak_delta_kib": receipt["rss_peak_delta_kib"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
