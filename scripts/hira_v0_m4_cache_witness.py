from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

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
M4B1_RUN_ID = 36328918665
M4B1_ARTIFACT_ID = 10934986918
M4B1_ARTIFACT_DIGEST = (
    "sha256:e2624ce369a17c068cf8bb78fffb70615d86581a630dc49d31c3ae9d3cfc5ed4"
)

M4_CACHE_WITNESS_SCHEMA = "hira-v0-mainline-m4-cache-witness-v2"
PROBABILITY_TOL = 1e-6
EQUIVALENCE_TOL = 2e-5


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("M4-B2 A13 weight identity changed")

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


def _receipt(directory: Path) -> dict[str, object]:
    return json.loads((directory / "receipt.json").read_text(encoding="utf-8"))


def _validate_t0(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    row = _receipt(directory)
    if row.get("schema_version") != "r8-w28-candidate-receipt-v1":
        raise RuntimeError("M4-B2 unexpected T0 receipt")
    if row.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M4-B2 T0 identity changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M4-B2 T0 bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    row = _receipt(directory)
    if row.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M4-B2 unexpected W34 receipt")
    if row.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M4-B2 W34 identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M4-B2 W34 bytes changed")
    return checkpoint


def _validate_m4a(directory: Path) -> dict[str, object]:
    row = _receipt(directory)
    if row.get("schema_version") != "hira-v0-mainline-m4-schema-benchmark-v1":
        raise RuntimeError("M4-B2 unexpected M4-A receipt")
    if row.get("outcome") != "HIRA_V0_M4_SCHEMA_BATCHING_READY":
        raise RuntimeError("M4-B2 requires qualified M4-A")
    return row


def _validate_m4b1(directory: Path) -> dict[str, object]:
    row = _receipt(directory)
    if row.get("schema_version") != "hira-v0-mainline-m4-cache-stress-v1":
        raise RuntimeError("M4-B2 unexpected M4-B1 receipt")
    if row.get("outcome") != "HIRA_V0_M4_BOUNDED_CACHE_FAIL":
        raise RuntimeError("M4-B2 requires frozen M4-B1 failure")
    gates = row.get("gates", {})
    if gates.get("anchor_survived_lru") is not False:
        raise RuntimeError("M4-B2 expected anchor-only witness failure")
    for key, value in gates.items():
        if key != "anchor_survived_lru" and value is not True:
            raise RuntimeError(f"M4-B2 unexpected additional M4-B1 failure: {key}")
    return row


def _options() -> tuple[LogicalOption, ...]:
    phrases = tuple(
        f"{color} {obj} {action} route"
        for color, obj, action in itertools.product(COLORS, OBJECTS, ACTIONS)
    )
    return tuple(
        LogicalOption(
            option_id=f"m4-cache-route-{index:03d}",
            criterion_text=phrases[index],
        )
        for index in range(HIRA_V0_MAX_K)
    )


def _question(index: int) -> str:
    return f"Which route phrase matches cache witness sequence {index:03d}?"


def _compile(model, options, index: int):
    return model.runtime.compile_schema(
        primitive="choice",
        question_text=_question(index),
        options=options,
        use_cache=True,
        include_token_artifacts=True,
    )


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--m4a-dir", type=Path, required=True)
    parser.add_argument("--m4b1-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    m4a = _validate_m4a(args.m4a_dir)
    m4b1 = _validate_m4b1(args.m4b1_dir)

    encoder = _load_a13()
    model = build_hira_v0_m4_runtime(
        encoder,
        t0,
        w34,
        schema_cache_max_entries=DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
        schema_cache_max_bytes=DEFAULT_SCHEMA_CACHE_MAX_BYTES,
    )
    options = _options()
    session = model.open_session(
        "The route record explicitly names amber anchor align route. "
        "All other route descriptions are alternatives only."
    )

    saved = {}
    for index in range(10):
        schema, receipt = _compile(model, options, index)
        if receipt.cache_hit or not receipt.cache_stored:
            raise RuntimeError("M4-B2 initial working set did not cache")
        output = session.decide_compiled(schema)
        saved[index] = {
            "schema": schema,
            "probabilities": output.probabilities.detach().cpu().clone(),
            "selected_option_id": output.selected_option_id,
        }

    before_touch = model.schema_cache_info()
    if before_touch["entries"] != 10 or before_touch["evictions"] != 0:
        raise RuntimeError("M4-B2 initial ten-schema working set changed")

    # Index 0 is the LRU entry. A cache hit must move it to MRU.
    touched_schema, touched_receipt = _compile(model, options, 0)
    if not touched_receipt.cache_hit or touched_schema is not saved[0]["schema"]:
        raise RuntimeError("M4-B2 failed to hit LRU candidate")

    # Four new entries force byte-budget eviction. Since index 0 was touched,
    # older indices 1+ must leave before index 0.
    for index in range(10, 14):
        _schema, receipt = _compile(model, options, index)
        if receipt.cache_hit or not receipt.cache_stored:
            raise RuntimeError("M4-B2 pressure schema did not enter cache")

    after_pressure = model.schema_cache_info()

    touched_again, touched_again_receipt = _compile(model, options, 0)
    touched_survived = bool(
        touched_again_receipt.cache_hit
        and touched_again is saved[0]["schema"]
    )

    old_recompiled, old_receipt = _compile(model, options, 1)
    old_output = session.decide_compiled(old_recompiled)
    old_evicted = not old_receipt.cache_hit
    probability_error = _max_error(
        saved[1]["probabilities"],
        old_output.probabilities.detach().cpu(),
    )
    embedding_error = _max_error(
        saved[1]["schema"].option_embeddings.detach().cpu(),
        old_recompiled.option_embeddings.detach().cpu(),
    )
    selected_match = (
        saved[1]["selected_option_id"] == old_output.selected_option_id
    )
    final_info = model.schema_cache_info()

    gates = {
        "initial_working_set_resident": (
            before_touch["entries"] == 10
            and before_touch["evictions"] == 0
        ),
        "touch_hit": touched_receipt.cache_hit,
        "entry_bound": after_pressure["entries"] <= DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
        "byte_bound": after_pressure["bytes"] <= DEFAULT_SCHEMA_CACHE_MAX_BYTES,
        "eviction_observed": after_pressure["evictions"] > 0,
        "touched_lru_survived": touched_survived,
        "older_resident_evicted": old_evicted,
        "recompiled_semantic_equivalent": embedding_error <= EQUIVALENCE_TOL,
        "recompiled_probability_equivalent": probability_error <= PROBABILITY_TOL,
        "recompiled_selected_match": selected_match,
        "full_k": old_output.probabilities.numel() == HIRA_V0_MAX_K,
        "state_once": model.runtime.state_encode_calls == 1,
        "relation_delta_zero": float(old_output.hira.relation_delta.abs().max()) == 0.0,
    }
    outcome = (
        "HIRA_V0_M4_BOUNDED_CACHE_READY"
        if all(gates.values())
        else "HIRA_V0_M4_BOUNDED_CACHE_FAIL"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": M4_CACHE_WITNESS_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
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
        "m4b1_authority": {
            "run_id": M4B1_RUN_ID,
            "artifact_id": M4B1_ARTIFACT_ID,
            "artifact_digest": M4B1_ARTIFACT_DIGEST,
            "outcome": m4b1["outcome"],
        },
        "cache_limits": {
            "max_entries": DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
            "max_bytes": DEFAULT_SCHEMA_CACHE_MAX_BYTES,
        },
        "before_touch": before_touch,
        "after_pressure": after_pressure,
        "final_cache_info": final_info,
        "touched_survived": touched_survived,
        "old_recompile_cache_hit": old_receipt.cache_hit,
        "recompiled_embedding_max_error": embedding_error,
        "recompiled_probability_max_error": probability_error,
        "recompiled_selected_option_match": selected_match,
        "state_encode_count": model.runtime.state_encode_calls,
        "gates": gates,
        "semantic_quality_promoted": False,
        "reliability_promoted": False,
        "multilingual_promoted": False,
        "production_ready_claimed": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M4_CACHE_WITNESS_FINAL=" + json.dumps({
        "outcome": outcome,
        "gates": gates,
        "before_touch": before_touch,
        "after_pressure": after_pressure,
        "final_cache_info": final_info,
        "touched_survived": touched_survived,
        "recompiled_probability_max_error": probability_error,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
