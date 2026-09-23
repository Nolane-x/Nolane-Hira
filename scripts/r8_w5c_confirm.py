from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_alignment_probes import (
    BilinearAlignmentProbe,
    PairMLPAlignmentProbe,
    classify_confirm,
    compile_alignment_cache,
    evaluate_probe,
    generate_alignment_authority,
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


def load_probe(kind,path,receipt_path):
    receipt=json.loads(Path(receipt_path).read_text())
    if receipt.get("schema_version")!="r8-w5c-learned-probe-v1":
        raise RuntimeError("unexpected W5c probe receipt")
    if receipt.get("status")!="PASS" or receipt.get("probe")!=kind:
        raise RuntimeError("W5c probe receipt mismatch")
    if receipt.get("confirm_exposed") is not False:
        raise RuntimeError("W5c confirm already exposed")
    if file_sha256(path)!=receipt.get("probe_sha256"):
        raise RuntimeError("W5c probe SHA mismatch")
    model=BilinearAlignmentProbe() if kind=="bilinear" else PairMLPAlignmentProbe(dropout=0.05)
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
    model.eval()
    return model,receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bilinear",type=Path,required=True)
    p.add_argument("--bilinear-receipt",type=Path,required=True)
    p.add_argument("--pair-mlp",type=Path,required=True)
    p.add_argument("--pair-receipt",type=Path,required=True)
    p.add_argument("--cache-receipt",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()

    cache_receipt=json.loads(args.cache_receipt.read_text())
    if cache_receipt.get("schema_version")!="r8-w5c-cache-receipt-v1":
        raise RuntimeError("unexpected W5c cache receipt")
    if cache_receipt.get("confirm_exposed") is not False:
        raise RuntimeError("W5c confirm already exposed")

    bilinear,br=load_probe("bilinear",args.bilinear,args.bilinear_receipt)
    pair,pr=load_probe("pair_mlp",args.pair_mlp,args.pair_receipt)

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
    runtime=NolaneHira(enc,HIRACore(d_model=256,dropout=0.0))
    runtime.eval()

    print("R8_W5C_CONFIRM_GENERATION_BEGIN",flush=True)
    cases=generate_alignment_authority("confirm")
    if len(cases)!=256:
        raise RuntimeError("W5c confirm count changed")
    cache=compile_alignment_cache(runtime,cases)

    metrics={
        "pooled_cosine":evaluate_probe("pooled_cosine",cache),
        "token_max":evaluate_probe("token_max",cache),
        "bilinear":evaluate_probe("bilinear",cache,model=bilinear),
        "pair_mlp":evaluate_probe("pair_mlp",cache,model=pair),
    }
    verdict,gates=classify_confirm(metrics)
    raw_dev=cache_receipt["raw_dev_probes"]
    gains={
        "bilinear_dev_vs_pooled":float(br["selected_dev_metrics"]["accuracy"])-float(raw_dev["pooled_cosine"]["accuracy"]),
        "pair_mlp_dev_vs_pooled":float(pr["selected_dev_metrics"]["accuracy"])-float(raw_dev["pooled_cosine"]["accuracy"]),
        "pair_mlp_dev_vs_bilinear":float(pr["selected_dev_metrics"]["accuracy"])-float(br["selected_dev_metrics"]["accuracy"]),
    }
    result={
        "schema_version":"r8-w5c-confirm-v1",
        "status":"PASS",
        "verdict":verdict,
        "confirm_generated_after_probe_freeze":True,
        "confirm_case_count":256,
        "confirm_state_encode_calls":cache["state_encode_calls"],
        "confirm_state_encode_calls_per_case":cache["state_encode_calls_per_case"],
        "metrics":metrics,
        "gates":gates,
        "dev_mechanism_gains":gains,
        "learned_probe_sha256":{
            "bilinear":br["probe_sha256"],
            "pair_mlp":pr["probe_sha256"],
        },
        "forbidden_benchmark_data_used":False,
        "campaign_cells_populated":0,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"confirm.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
