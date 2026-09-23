from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_balanced_binding import (
    CONFIRM_SEED,
    DEV_SEED,
    MAX_LENGTH,
    TRAIN_SEED,
    all_w5h_vocab,
    compile_binding_cache,
    generate_binding_authority,
    save_binding_cache,
)

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_hash(values) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return sha256(payload).hexdigest()


def prior_semantic_vocab() -> set[str]:
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab
    from nmd import semantic_alignment_probes as w5c
    from nmd import semantic_capacity_control as w5e
    from nmd import semantic_encoder_adaptation as w5d
    from nmd import semantic_routing_curriculum as w5a
    from nmd import semantic_token_curriculum as w5b

    prior = set(all_w5g_vocab()) | set(all_w5f_vocab())
    for module in (w5a, w5b, w5c, w5d, w5e):
        for name, value in vars(module).items():
            if not (name.startswith("TRAIN_") or name.startswith("CONFIRM_")):
                continue
            if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
                prior.update(value)
    return prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=MAX_LENGTH,
    )

    w5h_vocab = all_w5h_vocab()
    overlap = sorted(w5h_vocab & prior_semantic_vocab())
    if overlap:
        raise RuntimeError(f"W5h vocabulary overlaps W5a-W5g: {overlap}")

    train = generate_binding_authority("train")
    dev = generate_binding_authority("dev")
    train_cache = compile_binding_cache(encoder, train)
    dev_cache = compile_binding_cache(encoder, dev)

    train_path = args.out / "train.pt"
    dev_path = args.out / "dev.pt"
    save_binding_cache(train_cache, train_path)
    save_binding_cache(dev_cache, dev_path)

    receipt = {
        "schema_version": "r8-w5h-cache-receipt-v1",
        "status": "PASS",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "max_length": MAX_LENGTH,
        "train_seed": TRAIN_SEED,
        "dev_seed": DEV_SEED,
        "reserved_confirm_seed": CONFIRM_SEED,
        "w5a_through_w5g_vocab_disjoint": True,
        "train_case_count": len(train),
        "dev_case_count": len(dev),
        "train_case_id_sha256": list_hash(case.case_id for case in train),
        "dev_case_id_sha256": list_hash(case.case_id for case in dev),
        "vocab_sha256": list_hash(w5h_vocab),
        "train_encoder_calls": train_cache["encoder_calls"],
        "dev_encoder_calls": dev_cache["encoder_calls"],
        "state_text_encodes_per_case": 1.0,
        "train_cache_sha256": file_sha256(train_path),
        "dev_cache_sha256": file_sha256(dev_path),
        "confirm_exposed": False,
        "forbidden_benchmark_data_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
