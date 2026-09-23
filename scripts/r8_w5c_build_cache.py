from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_alignment_probes import (
    compile_alignment_cache,
    evaluate_probe,
    generate_alignment_authority,
    save_alignment_cache,
)

A13_MODEL="microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION="4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256="5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
A13_MAX_LENGTH=256


def file_sha256(path):
    h=sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModel,AutoTokenizer

    snapshot=Path(snapshot_download(repo_id=A13_MODEL,revision=A13_REVISION))
    weight=snapshot/"model.safetensors"
    if file_sha256(weight)!=A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA mismatch")
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    base=AutoModel.from_pretrained(str(snapshot),local_files_only=True)
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)
    enc=HFAutoSemanticEncoder(base,tok,revision=A13_REVISION,max_length=A13_MAX_LENGTH)
    model=NolaneHira(enc,HIRACore(d_model=256,dropout=0.0))
    model.eval()

    train_cases=generate_alignment_authority("train")
    dev_cases=generate_alignment_authority("dev")
    if len(train_cases)!=768 or len(dev_cases)!=192:
        raise RuntimeError("W5c TRAIN/DEV counts changed")
    train_cache=compile_alignment_cache(model,train_cases)
    dev_cache=compile_alignment_cache(model,dev_cases)

    args.out.mkdir(parents=True,exist_ok=True)
    train_path=args.out/"train-cache.pt"
    dev_path=args.out/"dev-cache.pt"
    save_alignment_cache(train_cache,train_path)
    save_alignment_cache(dev_cache,dev_path)

    raw={
        "pooled_cosine":evaluate_probe("pooled_cosine",dev_cache),
        "token_max":evaluate_probe("token_max",dev_cache),
    }
    receipt={
        "schema_version":"r8-w5c-cache-receipt-v1",
        "status":"PASS",
        "scope":"fresh W5c TRAIN/DEV only",
        "a13_model":A13_MODEL,
        "a13_revision":A13_REVISION,
        "a13_weight_sha256":A13_WEIGHT_SHA256,
        "train_case_count":train_cache["case_count"],
        "dev_case_count":dev_cache["case_count"],
        "train_state_encode_calls":train_cache["state_encode_calls"],
        "dev_state_encode_calls":dev_cache["state_encode_calls"],
        "train_cache_sha256":file_sha256(train_path),
        "dev_cache_sha256":file_sha256(dev_path),
        "raw_dev_probes":raw,
        "confirm_exposed":False,
        "forbidden_benchmark_data_used":False,
    }
    (args.out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(receipt,sort_keys=True))


if __name__=="__main__":
    main()
