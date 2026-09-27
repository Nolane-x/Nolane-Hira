from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import resource
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from nmd.contracts import LogicalOption
from nmd.high_cardinality_stress import ACTIONS, COLORS, OBJECTS
from nmd.mainline import (
    HIRA_V0_MAX_K,
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    build_hira_v0_m4_runtime,
)
from nmd.mainline_high_k import (
    M2_K_LADDER,
    compiled_schema_tensor_bytes,
)
from nmd.semantic import HFAutoSemanticEncoder, TextBatch, TextSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

M2_RUN_ID = 36310118240
M2_ARTIFACT_ID = 10928771833
M2_ARTIFACT_DIGEST = (
    "sha256:99e8c02aeb48fa32910a591d71b92e3e6f96258c0fa8501f38682ea4a2b7f453"
)

M4_BENCHMARK_SCHEMA = "hira-v0-mainline-m4-schema-benchmark-v1"
EQUIVALENCE_TOL = 2e-5
PROBABILITY_TOL = 1e-6


class CountingSemanticEncoder(TextSemanticEncoder):
    def __init__(self, base: TextSemanticEncoder):
        super().__init__()
        self.base = base
        self.d_model = int(base.d_model)
        self.encode_text_calls = 0

    @property
    def encoder_hash(self) -> str:
        return self.base.encoder_hash

    @property
    def tokenizer_hash(self) -> str:
        return self.base.tokenizer_hash

    def encode_texts(self, texts: Sequence[str]) -> TextBatch:
        self.encode_text_calls += 1
        return self.base.encode_texts(texts)

    def reset_calls(self) -> None:
        self.encode_text_calls = 0


def _load_a13() -> CountingSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M4 A13 weight SHA mismatch: {actual}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=256,
    )
    return CountingSemanticEncoder(encoder)


def _validate_t0(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w28-candidate-receipt-v1":
        raise RuntimeError("M4 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M4 T0 identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M4 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M4 T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M4 unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M4 W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M4 W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M4 W34 checkpoint bytes changed")
    return checkpoint


def _validate_m2(directory: Path) -> dict[str, object]:
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "hira-v0-mainline-m2-mechanics-integration-v1":
        raise RuntimeError("M4 unexpected M2 mechanics receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M4 M2 mechanics receipt did not pass")
    if receipt.get("outcome") != "HIRA_V0_M2_MECHANICS_READY":
        raise RuntimeError("M4 requires promoted M2 mechanics")
    if receipt.get("k_ladder") != list(M2_K_LADDER):
        raise RuntimeError("M4 M2 K ladder changed")
    return receipt


def _phrase_space() -> tuple[str, ...]:
    return tuple(
        f"{color} {obj} {action} route"
        for color, obj, action in itertools.product(COLORS, OBJECTS, ACTIONS)
    )


PHRASES = _phrase_space()


def _options(k: int) -> tuple[LogicalOption, ...]:
    if not 4 <= k <= HIRA_V0_MAX_K:
        raise ValueError(f"M4 benchmark invalid K: {k}")
    return tuple(
        LogicalOption(
            option_id=f"m2-route-{index:03d}",
            criterion_text=PHRASES[index],
        )
        for index in range(k)
    )


@torch.inference_mode()
def _legacy_semantic_compile(
    encoder: CountingSemanticEncoder,
    question_text: str,
    options: tuple[LogicalOption, ...],
) -> dict[str, object]:
    encoder.reset_calls()
    start = time.perf_counter()

    q = encoder.encode_texts([question_text]).pooled_embeddings[0]
    logical = []
    view_texts = []
    for option in options:
        texts = [
            option.criterion_text,
            *option.aliases,
            *option.exemplars,
        ]
        texts = [text for text in texts if text and text.strip()]
        if not texts:
            raise RuntimeError("legacy benchmark option has no semantic text")
        proto = encoder.encode_texts(texts).pooled_embeddings
        emb = F.normalize(proto, dim=-1).mean(0)
        logical.append(F.normalize(emb, dim=-1))
        view_texts.extend(texts)

    token_batch = encoder.encode_texts(
        [question_text, *[option.criterion_text for option in options]]
    )
    view_batch = encoder.encode_texts(view_texts)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "encoder_calls": encoder.encode_text_calls,
        "question_embedding": q,
        "option_embeddings": torch.stack(logical),
        "question_tokens": token_batch.token_embeddings[0],
        "criterion_tokens": token_batch.token_embeddings[1:],
        "criterion_attention": token_batch.attention_mask[1:].bool(),
        "view_tokens": view_batch.token_embeddings,
        "view_attention": view_batch.attention_mask.bool(),
    }


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max())


def _historical_by_k(m2: dict[str, object]) -> dict[int, dict[str, object]]:
    reports = m2["suite"]["query_reports"]
    return {
        int(report["requested_k"]): report
        for report in reports
    }


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--m2-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    m2 = _validate_m2(args.m2_dir)
    historical = _historical_by_k(m2)

    encoder = _load_a13()
    model = build_hira_v0_m4_runtime(encoder, t0, w34)
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M4 benchmark requires frozen runtime")

    # Warm model/tokenizer allocations before timed comparisons.
    encoder.encode_texts(["M4 warmup"])
    encoder.reset_calls()

    session = model.open_session(
        "The route record explicitly names amber anchor align route. "
        "The remaining route phrases are alternatives only."
    )
    question = "Which route phrase matches the route record?"

    rows = []
    before_rss = _rss_kib()
    for k in M2_K_LADDER:
        options = _options(k)

        legacy = _legacy_semantic_compile(encoder, question, options)
        expected_legacy_calls = k + 3
        if int(legacy["encoder_calls"]) != expected_legacy_calls:
            raise RuntimeError("M4 inferred legacy encoder-call contract changed")

        model.runtime.schema_compiler.clear()
        encoder.reset_calls()
        cold_start = time.perf_counter()
        schema, cold_receipt = model.runtime.compile_schema(
            primitive="choice",
            question_text=question,
            options=options,
            use_cache=True,
            include_token_artifacts=True,
        )
        cold_ms = (time.perf_counter() - cold_start) * 1000.0
        cold_calls = encoder.encode_text_calls

        encoder.reset_calls()
        warm_start = time.perf_counter()
        warm_schema, warm_receipt = model.runtime.compile_schema(
            primitive="choice",
            question_text=question,
            options=options,
            use_cache=True,
            include_token_artifacts=True,
        )
        warm_ms = (time.perf_counter() - warm_start) * 1000.0
        warm_calls = encoder.encode_text_calls

        decision_start = time.perf_counter()
        out = session.decide_compiled(schema)
        decision_ms = (time.perf_counter() - decision_start) * 1000.0

        q_error = _max_error(
            legacy["question_embedding"],
            schema.question_embedding,
        )
        option_error = _max_error(
            legacy["option_embeddings"],
            schema.option_embeddings,
        )
        question_token_error = _max_error(
            legacy["question_tokens"],
            schema.question_token_embeddings,
        )
        criterion_token_error = _max_error(
            legacy["criterion_tokens"],
            schema.option_token_embeddings,
        )

        # M2 integration options have exactly one positive view per option.
        optimized_views = schema.option_view_token_embeddings[:, 0]
        view_error = _max_error(
            legacy["view_tokens"],
            optimized_views,
        )

        mass_error = abs(float(out.probabilities.sum()) - 1.0)
        relation_delta = float(out.hira.relation_delta.abs().max())
        finite = bool(
            torch.isfinite(out.logits).all()
            and torch.isfinite(out.probabilities).all()
        )
        full_k = bool(
            int(out.hira.candidate_budget.item()) == k
            and bool(out.hira.selected_mask.all())
            and out.probabilities.numel() == k
        )

        old = historical[k]
        selected_match = out.selected_option_id == old["selected_option_id"]
        mechanics_pass = bool(
            finite
            and full_k
            and mass_error <= PROBABILITY_TOL
            and relation_delta == 0.0
            and selected_match
        )
        equivalence_pass = bool(
            q_error <= EQUIVALENCE_TOL
            and option_error <= EQUIVALENCE_TOL
            and question_token_error <= EQUIVALENCE_TOL
            and criterion_token_error <= EQUIVALENCE_TOL
            and view_error <= EQUIVALENCE_TOL
        )

        rows.append({
            "k": k,
            "legacy_encoder_calls": int(legacy["encoder_calls"]),
            "optimized_cold_encoder_calls": cold_calls,
            "optimized_warm_encoder_calls": warm_calls,
            "legacy_same_runner_semantic_ms": float(legacy["elapsed_ms"]),
            "optimized_cold_schema_ms": cold_ms,
            "optimized_warm_schema_ms": warm_ms,
            "optimized_decision_ms": decision_ms,
            "same_runner_speedup": (
                float(legacy["elapsed_ms"]) / cold_ms
                if cold_ms > 0.0
                else float("inf")
            ),
            "historical_m2_schema_ms": float(old["schema_compile_ms"]),
            "historical_m2_decision_ms": float(old["decision_ms"]),
            "schema_tensor_bytes": compiled_schema_tensor_bytes(schema),
            "historical_schema_tensor_bytes": int(old["schema_tensor_bytes"]),
            "question_embedding_max_error": q_error,
            "option_embedding_max_error": option_error,
            "question_token_max_error": question_token_error,
            "criterion_token_max_error": criterion_token_error,
            "view_token_max_error": view_error,
            "selected_option_id": out.selected_option_id,
            "historical_selected_option_id": old["selected_option_id"],
            "selected_option_match": selected_match,
            "probability_mass_error": mass_error,
            "relation_delta_max": relation_delta,
            "finite": finite,
            "full_k": full_k,
            "cold_cache_hit": cold_receipt.cache_hit,
            "warm_cache_hit": warm_receipt.cache_hit,
            "warm_schema_identity": warm_schema is schema,
            "equivalence_pass": equivalence_pass,
            "mechanics_pass": mechanics_pass,
        })

    after_rss = _rss_kib()
    k255 = next(row for row in rows if row["k"] == 255)

    gates = {
        "all_equivalent": all(row["equivalence_pass"] for row in rows),
        "all_mechanics": all(row["mechanics_pass"] for row in rows),
        "cold_calls_le_3": all(
            int(row["optimized_cold_encoder_calls"]) <= 3 for row in rows
        ),
        "warm_calls_zero": all(
            int(row["optimized_warm_encoder_calls"]) == 0 for row in rows
        ),
        "cold_cache_miss": all(not row["cold_cache_hit"] for row in rows),
        "warm_cache_hit": all(row["warm_cache_hit"] for row in rows),
        "k255_same_runner_faster": (
            float(k255["optimized_cold_schema_ms"])
            < float(k255["legacy_same_runner_semantic_ms"])
        ),
    }
    outcome = (
        "HIRA_V0_M4_SCHEMA_BATCHING_READY"
        if all(gates.values())
        else "HIRA_V0_M4_SCHEMA_BATCHING_FAIL"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": M4_BENCHMARK_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "m2_authority": {
            "run_id": M2_RUN_ID,
            "artifact_id": M2_ARTIFACT_ID,
            "artifact_digest": M2_ARTIFACT_DIGEST,
        },
        "manifest": model.manifest.to_dict(),
        "parameter_report": model.parameter_report().to_dict(),
        "k_ladder": list(M2_K_LADDER),
        "rows": rows,
        "gates": gates,
        "rss_peak_before_kib": before_rss,
        "rss_peak_after_kib": after_rss,
        "rss_peak_delta_kib": max(0, after_rss - before_rss),
        "semantic_quality_promoted": False,
        "reliability_promoted": False,
        "multilingual_promoted": False,
        "production_ready_claimed": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("HIRA_V0_M4_SCHEMA_FINAL=" + json.dumps({
        "outcome": outcome,
        "gates": gates,
        "k255": k255,
        "rss_peak_delta_kib": receipt["rss_peak_delta_kib"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
