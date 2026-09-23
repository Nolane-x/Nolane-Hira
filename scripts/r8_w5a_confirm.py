from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.hira import HIRACore, count_parameters
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_routing_curriculum import (
    compile_routing_cache,
    evaluate_routing_cache,
    generate_authority,
)

A13_MODEL="microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION="4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256="5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
A13_MAX_LENGTH=256
EXPECTED_PARAMS=422_159


def file_sha256(path: str | Path) -> str:
    h=sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--selected-head",type=Path,required=True)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    selection=json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("schema_version")!="r8-w5a-selection-v1":
        raise RuntimeError("unexpected W5a selection schema")
    if selection.get("status")!="PASS":
        raise RuntimeError("W5a selection is not PASS")
    if selection.get("candidate_count")!=4:
        raise RuntimeError("W5a selection candidate count mismatch")
    if selection.get("confirm_exposed") is not False:
        raise RuntimeError("selection already exposed to confirm")
    selected_sha=file_sha256(args.selected_head)
    if selected_sha!=selection.get("selected_head_sha256"):
        raise RuntimeError("selected head SHA mismatch")

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot=Path(snapshot_download(repo_id=A13_MODEL,revision=A13_REVISION))
    weight=snapshot/"model.safetensors"
    if file_sha256(weight)!=A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA mismatch")
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    base=AutoModel.from_pretrained(str(snapshot),local_files_only=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    encoder=HFAutoSemanticEncoder(
        base,tokenizer,revision=A13_REVISION,max_length=A13_MAX_LENGTH
    )
    hira=HIRACore(d_model=256,dropout=0.05)
    state=torch.load(args.selected_head,map_location="cpu",weights_only=True)
    hira.load_state_dict(state,strict=True)
    hira.eval()
    if count_parameters(hira)!=EXPECTED_PARAMS:
        raise RuntimeError("HIRA head parameter count changed")
    model=NolaneHira(encoder,hira)
    model.eval()

    print("R8_W5A_CONFIRM_GENERATION_BEGIN",flush=True)
    confirm_cases=generate_authority("confirm")
    if len(confirm_cases)!=256:
        raise RuntimeError("W5a confirm count mismatch")
    confirm_cache=compile_routing_cache(model,confirm_cases)
    metrics=evaluate_routing_cache(hira,confirm_cache)

    per=metrics["per_k"]
    gates={
        "overall_accuracy":float(metrics["accuracy"])>=0.50,
        "k128_accuracy":float(per["128"]["accuracy"])>=0.30,
        "k255_accuracy":float(per["255"]["accuracy"])>=0.20,
        "k255_top5":float(per["255"]["top5_recall"])>=0.50,
        "probability_mass":float(metrics["probability_mass_max_error"])<=1e-6,
        "state_once":confirm_cache["state_encode_calls"]==256,
        "budget_k32":per["32"]["candidate_budget_min"]==32 and per["32"]["candidate_budget_max"]==32,
        "budget_k64":per["64"]["candidate_budget_min"]==64 and per["64"]["candidate_budget_max"]==64,
        "budget_k128":per["128"]["candidate_budget_min"]==128 and per["128"]["candidate_budget_max"]==128,
        "budget_k255":per["255"]["candidate_budget_min"]==255 and per["255"]["candidate_budget_max"]==255,
    }
    verdict=(
        "SEMANTIC_COMPETENCE_PASS"
        if all(gates.values())
        else "SEMANTIC_COMPETENCE_FAIL"
    )
    args.out.mkdir(parents=True,exist_ok=True)
    result={
        "schema_version":"r8-w5a-confirm-v1",
        "status":"PASS",
        "verdict":verdict,
        "selected_candidate":selection["selected_candidate"],
        "selected_head_sha256":selected_sha,
        "selection_authority":"DEV_ONLY",
        "confirm_generated_after_selection":True,
        "confirm_case_count":256,
        "confirm_state_encode_calls":confirm_cache["state_encode_calls"],
        "confirm_state_encode_calls_per_case":confirm_cache["state_encode_calls_per_case"],
        "metrics":metrics,
        "gates":gates,
        "forbidden_benchmark_data_used":False,
        "campaign_cells_populated":0,
    }
    (args.out/"confirm.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
