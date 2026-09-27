from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.mainline import (
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    build_hira_v0_m2_mechanics,
)
from nmd.mainline_m2_authority import generate_m2_high_k_semantic
from nmd.mainline_m2_eval import evaluate_m2_semantic_cases
from nmd.mainline_m2_r1_authority import generate_m2_r1_dev
from nmd.mainline_m2_r1_training import (
    M2_R1_CANDIDATES,
    M2_R1_EPOCHS,
    M2_R1_GRAD_CLIP,
    M2_R1_LR,
    M2_R1_PAIR_MARGIN,
    M2_R1_PAIR_MARGIN_WEIGHT,
    M2_R1_SEEDS,
    M2_R1_TEMPERATURE,
    M2_R1_WEIGHT_DECAY,
    compile_m2_r1_cache,
    install_m2_r1_candidate,
    m2_r1_runtime_gate,
    select_m2_r1_family,
    train_m2_r1_candidate,
)
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

M2_BASELINE_RUN_ID = 36311890201
M2_BASELINE_ARTIFACT_ID = 10928764745
M2_BASELINE_ARTIFACT_DIGEST = (
    "sha256:18903d770c1917db94e395c49feb72c0e2fea1c01ef70e976c54ad40ed223148"
)
M2_R1_RECEIPT_SCHEMA = "hira-v0-mainline-m2-r1-train-dev-v1"
M2_R1_CHECKPOINT_SCHEMA = "hira-v0-mainline-m2-r1-high-k-scorer-v1"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M2-R1 A13 weight SHA mismatch: {actual}")

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
        raise RuntimeError("M2-R1 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M2-R1 T0 identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M2-R1 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M2-R1 T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M2-R1 unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M2-R1 W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2-R1 W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2-R1 W34 checkpoint bytes changed")
    if int(receipt.get("candidate_parameter_count", -1)) != 8192:
        raise RuntimeError("M2-R1 W34 candidate capacity changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M2-R1 W34 training receipt contains confirm leakage")
    return checkpoint


def _candidate_summary(row: dict[str, object]) -> dict[str, object]:
    return {
        "candidate": row["candidate"],
        "family": row["family"],
        "seed": row["seed"],
        "parameter_count": row["parameter_count"],
        "selected_epoch": row["selected_epoch"],
        "selected_dev": row["selected_dev"],
        "selection_key": row["selection_key"],
        "history": row["history"],
    }


def _save_checkpoint(
    path: Path,
    row: dict[str, object],
    *,
    role: str,
) -> str:
    payload = {
        "schema_version": M2_R1_CHECKPOINT_SCHEMA,
        "kind": "coevidence-semantic-high-k-rescue",
        "role": role,
        "candidate": row["candidate"],
        "family": row["family"],
        "seed": row["seed"],
        "parameter_count": row["parameter_count"],
        "selected_epoch": row["selected_epoch"],
        "base_w34_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "candidate_state_dict": row["selected_state_dict"],
    }
    torch.save(payload, path)
    return file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    encoder = _load_a13()
    model = build_hira_v0_m2_mechanics(encoder, t0, w34)

    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M2-R1 base mainline must be frozen")
    frozen_w34 = model.runtime.coevidence_symmetric_semantic_scorer
    if frozen_w34 is None:
        raise RuntimeError("M2-R1 W34 scorer missing")
    if frozen_w34.trainable_parameter_count != 0:
        raise RuntimeError("M2-R1 base W34 scorer must be frozen")

    # Evidence boundary: HKA-HKD TRAIN + wholly fresh HKH/HKI DEV only.
    # HKE/HKF are exposed historical evidence and are never generated here.
    # HKG remains sealed and is never generated here.
    train_rows = generate_m2_high_k_semantic("train")
    dev_rows = generate_m2_r1_dev()

    if len(train_rows) != 64 or {row.k for row in train_rows} != {4, 8, 16, 32}:
        raise RuntimeError("M2-R1 TRAIN authority surface changed")
    if len(dev_rows) != 48 or {row.k for row in dev_rows} != {64, 128}:
        raise RuntimeError("M2-R1 fresh DEV authority surface changed")

    train_cache = compile_m2_r1_cache(model, train_rows, split="train")
    dev_cache = compile_m2_r1_cache(model, dev_rows, split="dev")

    results = []
    for candidate in M2_R1_CANDIDATES:
        results.append(
            train_m2_r1_candidate(
                candidate,
                frozen_w34,
                train_cache,
                dev_cache,
            )
        )

    selection = select_m2_r1_family(results)
    primary = selection["primary"]
    replica = selection["replica"]

    # Cached selection is necessary but insufficient. Re-run selected primary
    # and replica through the complete mainline canonical+rotation evaluator.
    before_primary = model.runtime.state_encode_calls
    install_m2_r1_candidate(model, primary["selected_state_dict"])
    primary_runtime = evaluate_m2_semantic_cases(model, dev_rows)
    primary_state_delta = model.runtime.state_encode_calls - before_primary
    primary_gate = m2_r1_runtime_gate(primary_runtime, replica=False)

    before_replica = model.runtime.state_encode_calls
    install_m2_r1_candidate(model, replica["selected_state_dict"])
    replica_runtime = evaluate_m2_semantic_cases(model, dev_rows)
    replica_state_delta = model.runtime.state_encode_calls - before_replica
    replica_gate = m2_r1_runtime_gate(replica_runtime, replica=True)

    dev_qualified = bool(primary_gate["pass"] and replica_gate["pass"])

    args.out.mkdir(parents=True, exist_ok=True)
    train_cache_path = args.out / "train-cache.pt"
    dev_cache_path = args.out / "dev-cache.pt"
    torch.save(train_cache, train_cache_path)
    torch.save(dev_cache, dev_cache_path)

    primary_path = args.out / "primary.pt"
    replica_path = args.out / "replica.pt"
    primary_sha = _save_checkpoint(primary_path, primary, role="primary")
    replica_sha = _save_checkpoint(replica_path, replica, role="replica")

    receipt = {
        "schema_version": M2_R1_RECEIPT_SCHEMA,
        "status": "PASS",
        "outcome": (
            "HIRA_V0_M2_R1_DEV_QUALIFIED"
            if dev_qualified
            else "HIRA_V0_M2_R1_DEV_FAIL"
        ),
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "base_w34_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "baseline_failure_provenance": {
            "run_id": M2_BASELINE_RUN_ID,
            "artifact_id": M2_BASELINE_ARTIFACT_ID,
            "artifact_digest": M2_BASELINE_ARTIFACT_DIGEST,
            "rows_loaded_for_training_or_selection": False,
        },
        "train_partitions": ["HKA", "HKB", "HKC", "HKD"],
        "train_k": [4, 8, 16, 32],
        "train_case_count": len(train_rows),
        "dev_partitions": ["HKH", "HKI"],
        "dev_k": [64, 128],
        "dev_case_count": len(dev_rows),
        "candidate_count": len(results),
        "candidate_parameter_count": 8192,
        "candidate_names": list(M2_R1_CANDIDATES),
        "seeds": dict(M2_R1_SEEDS),
        "training": {
            "epochs": M2_R1_EPOCHS,
            "lr": M2_R1_LR,
            "weight_decay": M2_R1_WEIGHT_DECAY,
            "gradient_clip": M2_R1_GRAD_CLIP,
            "temperature": M2_R1_TEMPERATURE,
            "pair_margin": M2_R1_PAIR_MARGIN,
            "pair_margin_weight": M2_R1_PAIR_MARGIN_WEIGHT,
        },
        "candidates": [_candidate_summary(row) for row in results],
        "selected_family": selection["family"],
        "primary_candidate": primary["candidate"],
        "replica_candidate": replica["candidate"],
        "primary_checkpoint_sha256": primary_sha,
        "replica_checkpoint_sha256": replica_sha,
        "primary_cached_gate": selection["primary_cached_gate"],
        "replica_cached_gate": selection["replica_cached_gate"],
        "primary_runtime": primary_runtime,
        "replica_runtime": replica_runtime,
        "primary_runtime_gate": primary_gate,
        "replica_runtime_gate": replica_gate,
        "dev_qualification": {
            "pass": dev_qualified,
            "primary": primary_gate,
            "replica": replica_gate,
        },
        "cache_sha256": {
            "train": file_sha256(train_cache_path),
            "dev": file_sha256(dev_cache_path),
        },
        "cache_state_encode_count": (
            int(train_cache["metadata"]["state_encode_count"])
            + int(dev_cache["metadata"]["state_encode_count"])
        ),
        "primary_runtime_state_encode_count": primary_state_delta,
        "replica_runtime_state_encode_count": replica_state_delta,
        "decision_core_trainable_parameter_count": 0,
        "encoder_gradient_updates": False,
        "w28_projection_gradient_updates": False,
        "hira_core_gradient_updates": False,
        "only_w34_candidate_parameters_trained": True,
        "hke_hkf_rows_used_for_training": False,
        "hke_hkf_rows_used_for_selection": False,
        "hkg_exposed": False,
        "hkg_rows_used": False,
        "sealed_exposure_authorized": dev_qualified,
        "candidate_pruning_used": False,
        "relation_refinement_used": False,
        "adaptive_budget_used": False,
        "quality_claim_made": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("HIRA_V0_M2_R1_TRAIN_DEV_FINAL=" + json.dumps({
        "outcome": receipt["outcome"],
        "selected_family": receipt["selected_family"],
        "primary_candidate": receipt["primary_candidate"],
        "replica_candidate": receipt["replica_candidate"],
        "primary_runtime": receipt["primary_runtime"],
        "replica_runtime": receipt["replica_runtime"],
        "dev_qualification": receipt["dev_qualification"],
        "sealed_exposure_authorized": receipt["sealed_exposure_authorized"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
