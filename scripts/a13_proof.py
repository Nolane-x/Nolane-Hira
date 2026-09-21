from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore, count_parameters
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder


MODEL_ID = "microsoft/xtremedistil-l6-h256-uncased"
REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_assets(snapshot: Path) -> dict[str, dict[str, object]]:
    wanted = {
        "model.safetensors",
        "pytorch_model.bin",
        "config.json",
        "vocab.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    out = {}
    for path in sorted(snapshot.iterdir()):
        if path.is_file() and path.name in wanted:
            out[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    root = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION))
    assets = hash_assets(root)
    weight_name = "model.safetensors" if "model.safetensors" in assets else "pytorch_model.bin"
    if weight_name not in assets:
        raise RuntimeError("no supported model weight file was materialized")

    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    base = AutoModel.from_pretrained(str(root), local_files_only=True)
    base.eval()

    hidden = int(base.config.hidden_size)
    layers = int(base.config.num_hidden_layers)
    total_params = sum(p.numel() for p in base.parameters())
    pooler_params = sum(p.numel() for n, p in base.named_parameters() if n.startswith("pooler."))
    encoder_params_no_pooler = total_params - pooler_params

    if hidden != 256:
        raise AssertionError(f"expected hidden_size 256, got {hidden}")
    if layers != 6:
        raise AssertionError(f"expected 6 layers, got {layers}")

    encoder = HFAutoSemanticEncoder(base, tokenizer, revision=REVISION, max_length=128)
    texts = [
        "my bank transfer is still pending",
        "the cash withdrawal did not arrive",
        "my card payment was declined",
    ]
    with torch.no_grad():
        batch = encoder.encode_texts(texts)
    if batch.pooled_embeddings.shape != (3, 256):
        raise AssertionError(f"unexpected pooled shape: {batch.pooled_embeddings.shape}")

    hira = HIRACore(d_model=256, dropout=0.0)
    model = NolaneHira(encoder, hira)
    model.eval()

    options = [
        LogicalOption("opaque_a", "cash withdrawal issue"),
        LogicalOption("opaque_b", "pending bank transfer"),
        LogicalOption("opaque_c", "card payment declined"),
    ]
    with torch.no_grad():
        memory = model.compile_state("my transfer has not completed yet")
        schema, schema_receipt = model.compile_schema(
            primitive="choice",
            question_text="what issue is described?",
            options=options,
            use_cache=True,
        )
        decision = model.forward_compiled(memory, schema, forced_budget=3)

    payload = {
        "schema_version": "r10-a13-proof-v1",
        "status": "PASS",
        "evidence_scope": "checkpoint materialization + semantic encoder + R9 runtime integration; HIRA head remains untrained",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "assets": assets,
        "primary_weight_file": weight_name,
        "primary_weight_sha256": assets[weight_name]["sha256"],
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "model_total_params": total_params,
        "pooler_params": pooler_params,
        "encoder_params_no_pooler": encoder_params_no_pooler,
        "hira_params": count_parameters(hira),
        "pooled_shape": list(batch.pooled_embeddings.shape),
        "token_shape": list(batch.token_embeddings.shape),
        "state_segments": int(memory.segment_embeddings.shape[0]),
        "schema_hash": schema.schema_hash,
        "schema_cache_hit": schema_receipt.cache_hit,
        "decision_probabilities": [float(x) for x in decision.probabilities],
        "decision_sum": float(decision.probabilities.sum()),
        "semantic_quality_claim": false,
        "laya_jev_benchmark_claim": false
    }

    # JSON has no lowercase Python booleans; set after construction for explicitness.
    payload["semantic_quality_claim"] = False
    payload["laya_jev_benchmark_claim"] = False

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "revision": REVISION,
        "weight_sha256": payload["primary_weight_sha256"],
        "encoder_params_no_pooler": encoder_params_no_pooler,
        "hira_params": payload["hira_params"],
        "receipt": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
