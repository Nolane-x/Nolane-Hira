from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import resource

import torch

from nmd.high_cardinality_stress import (
    K_VALUES,
    evaluate_mechanical_stress,
    evaluate_semantic_key_stress,
)
from nmd.hira import HIRACore, count_parameters
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder


SELECTED_HEAD_SHA256 = "2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c"
EXPECTED_HEAD_PARAMS = 422_159

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
A13_MAX_LENGTH = 256


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    selected_sha = file_sha256(args.selected_head)
    if selected_sha != SELECTED_HEAD_SHA256:
        raise RuntimeError("selected HIRA head SHA mismatch")

    hira = HIRACore(d_model=256, dropout=0.05)
    state = torch.load(
        args.selected_head,
        map_location="cpu",
        weights_only=True,
    )
    hira.load_state_dict(state, strict=True)
    hira.eval()
    if count_parameters(hira) != EXPECTED_HEAD_PARAMS:
        raise RuntimeError("HIRA head parameter count changed")

    mechanical_rows = []
    mechanics_pass = True
    rss_before = rss_kib()
    for k in K_VALUES:
        before_k = rss_kib()
        result = evaluate_mechanical_stress(hira, k=k)
        after_k = rss_kib()
        row = asdict(result)
        row["peak_rss_kib_before_k"] = before_k
        row["peak_rss_kib_after_k"] = after_k
        row["peak_rss_kib_growth_from_previous_peak"] = max(
            0,
            after_k - before_k,
        )
        mechanical_rows.append(row)
        mechanics_pass = mechanics_pass and bool(result.mechanics_pass)

    mechanical_receipt = {
        "schema_version": "r8-w4b-mechanical-v1",
        "selected_head_sha256": selected_sha,
        "head_parameter_count": count_parameters(hira),
        "mechanics_verdict": (
            "MECHANICS_PASS" if mechanics_pass else "MECHANICS_FAIL"
        ),
        "rss_kib_before_track_a": rss_before,
        "rss_kib_after_track_a": rss_kib(),
        "k_results": mechanical_rows,
        "banking77_data_used": False,
        "campaign_cells_populated": 0,
    }
    save_json(args.out / "mechanical.json", mechanical_receipt)
    print(json.dumps(mechanical_receipt, sort_keys=True), flush=True)

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(
            repo_id=A13_MODEL,
            revision=A13_REVISION,
        )
    )
    weight_path = snapshot / "model.safetensors"
    if file_sha256(weight_path) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=A13_MAX_LENGTH,
    )
    model = NolaneHira(encoder, hira)
    model.eval()

    semantic_rows = [
        evaluate_semantic_key_stress(model, k=k)
        for k in K_VALUES
    ]
    semantic_receipt = {
        "schema_version": "r8-w4b-semantic-key-v1",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "selected_head_sha256": selected_sha,
        "k_results": semantic_rows,
        "diagnostic_only": True,
        "banking77_data_used": False,
        "campaign_cells_populated": 0,
    }
    save_json(args.out / "semantic-key.json", semantic_receipt)

    final = {
        "schema_version": "r8-w4b-final-v1",
        "status": "PASS",
        "mechanics_verdict": mechanical_receipt["mechanics_verdict"],
        "mechanical": mechanical_receipt,
        "semantic_key": semantic_receipt,
        "scientific_scope": (
            "Fresh synthetic K128/K255 stress only; no Banking77 data and "
            "no Laya/Jev campaign cell populated."
        ),
    }
    save_json(args.out / "result.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
