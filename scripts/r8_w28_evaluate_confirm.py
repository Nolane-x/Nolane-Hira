from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from nmd.compositional_projection_cache import load_w28_cache
from nmd.compositional_projection_eval import (
    evaluate_projection,
    primary_rescue,
    replica_rescue,
)
from nmd.compositional_projection_reference import (
    REFERENCE_NAMES,
    REFERENCE_TASKS,
    confirm_domain_pass,
    evaluate_reference_panel,
)
from nmd.typed_competitive_cache import file_sha256

W9_HIRA_SHA256 = "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
W9_SCORER_SHA256 = "8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102"

REFERENCE_PANEL = {
    "deberta_nli": {
        "repo": "cross-encoder/nli-deberta-v3-base",
        "revision": "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
        "weight_sha256": "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa",
    },
    "roberta_nli": {
        "repo": "cross-encoder/nli-roberta-base",
        "revision": "1be0567456f0543475805e758725f151f283705a",
        "weight_sha256": "efc90996d2ed80123c26c9091c91385ffddc6d2fd0b2bacf3187fbd6c5b87953",
    },
}


def _load_w9_projection(freeze_path: Path, checkpoints: Path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w9-freeze-v1" or freeze.get("status") != "PASS":
        raise RuntimeError("W28 unexpected W9 freeze")
    rows = {row["candidate"]:row for row in freeze["candidates"]}
    row = rows.get("projection-semantic-control")
    if row is None:
        raise RuntimeError("W28 W9 projection control missing")
    root = checkpoints / row["checkpoint_dir"]
    hira_path = root/"hira.pt"
    scorer_path = root/"scorer.pt"
    if file_sha256(hira_path) != W9_HIRA_SHA256:
        raise RuntimeError("W28 HIRA SHA mismatch")
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W28 scorer SHA mismatch")
    state = torch.load(scorer_path,map_location="cpu",weights_only=True)
    projection = state["projection.weight"].detach().float().clone()
    if tuple(projection.shape)!=(128,256):
        raise RuntimeError("W28 W9 projection shape changed")
    return freeze,row,projection


def _load_candidate(directory: Path, candidate: str):
    receipt = json.loads((directory/"receipt.json").read_text(encoding="utf-8"))
    checkpoint_path = directory/"candidate.pt"
    if receipt.get("schema_version")!="r8-w28-candidate-receipt-v1":
        raise RuntimeError(f"W28 {candidate} receipt schema mismatch")
    if receipt.get("status")!="PASS" or receipt.get("candidate")!=candidate:
        raise RuntimeError(f"W28 {candidate} receipt identity mismatch")
    if receipt.get("confirm_exposed") is not False:
        raise RuntimeError(f"W28 {candidate} was not frozen before CONFIRM")
    if receipt.get("selection_partition")!="ET":
        raise RuntimeError(f"W28 {candidate} DEV selection partition changed")
    if receipt.get("optimizer_steps")!=96 or receipt.get("trainable_parameter_count")!=32768:
        raise RuntimeError(f"W28 {candidate} frozen recipe changed")
    if file_sha256(checkpoint_path)!=receipt.get("checkpoint_sha256"):
        raise RuntimeError(f"W28 {candidate} checkpoint SHA mismatch")
    checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if checkpoint.get("schema_version")!="r8-w28-candidate-checkpoint-v1":
        raise RuntimeError(f"W28 {candidate} checkpoint schema mismatch")
    if checkpoint.get("candidate")!=candidate:
        raise RuntimeError(f"W28 {candidate} checkpoint identity mismatch")
    return receipt,checkpoint


def _entailment_index(config) -> int:
    for key,value in (getattr(config,"id2label",{}) or {}).items():
        if "entail" in str(value).lower():
            return int(key)
    for key,value in (getattr(config,"label2id",{}) or {}).items():
        if "entail" in str(key).lower():
            return int(value)
    raise RuntimeError("W28 reference has no entailment label")


def _load_reference(name: str, spec: dict[str,str]):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot=Path(snapshot_download(repo_id=spec["repo"],revision=spec["revision"]))
    weight=snapshot/"model.safetensors"
    actual=file_sha256(weight)
    if actual!=spec["weight_sha256"]:
        raise RuntimeError(f"W28 {name} reference SHA mismatch")
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(str(snapshot),local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model,tokenizer,actual,_entailment_index(model.config)


@torch.inference_mode()
def _pair_scores(model,tokenizer,pairs,entailment_index:int):
    device=next(model.parameters()).device
    values=[]
    for start in range(0,len(pairs),64):
        batch=pairs[start:start+64]
        encoded=tokenizer(
            [a for a,_ in batch],
            [b for _,b in batch],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded={k:v.to(device) for k,v in encoded.items()}
        logits=model(**encoded,return_dict=True).logits
        probs=torch.softmax(logits.float(),dim=-1)[:,entailment_index]
        values.extend(float(x) for x in probs.cpu())
    return values


def _reference_scores(cache,model,tokenizer,entailment_index:int):
    lookup={
        str(case["case_id"]):{task:[0.0,0.0] for task in REFERENCE_TASKS}
        for case in cache["cases"]
    }
    requests=[]
    for case in cache["cases"]:
        case_id=str(case["case_id"])
        domain=str(case["domain_id"])
        query=str(case["severity_field"])
        hypotheses=cache["schemas"][domain]["reference_hypotheses"]
        for task in REFERENCE_TASKS:
            for value,hypothesis in enumerate(hypotheses[task]):
                requests.append((case_id,task,value,query,str(hypothesis)))
    scores=_pair_scores(
        model,
        tokenizer,
        [(q,h) for _,_,_,q,h in requests],
        entailment_index,
    )
    for req,score in zip(requests,scores):
        case_id,task,value,_,_=req
        lookup[case_id][task][value]=score
    return lookup


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--confirm-cache",type=Path,required=True)
    parser.add_argument("--w9-freeze",type=Path,required=True)
    parser.add_argument("--w9-checkpoints",type=Path,required=True)
    parser.add_argument("--t0-dir",type=Path,required=True)
    parser.add_argument("--t1-dir",type=Path,required=True)
    parser.add_argument("--qualification",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    qualification=json.loads(args.qualification.read_text(encoding="utf-8"))
    if qualification.get("outcome")!="W28_REFERENCE_QUALIFIED":
        raise RuntimeError("W28 final evaluation requires qualified reference")

    cache=load_w28_cache(args.confirm_cache,expected_partition="confirm")
    if set(cache["metadata"]["domains"])!={"EU","EV"}:
        raise RuntimeError("W28 CONFIRM domain set changed")

    freeze,w9_row,w9_projection=_load_w9_projection(args.w9_freeze,args.w9_checkpoints)
    receipts={}
    checkpoints={}
    for name,directory in {"T0":args.t0_dir,"T1":args.t1_dir}.items():
        receipts[name],checkpoints[name]=_load_candidate(directory,name)

    evaluations={
        "P0":evaluate_projection(cache,w9_projection),
        "T0":evaluate_projection(cache,checkpoints["T0"]["projection_weight"].float()),
        "T1":evaluate_projection(cache,checkpoints["T1"]["projection_weight"].float()),
    }

    model_scores={}
    actual_hashes={}
    entailment_indices={}
    for name in REFERENCE_NAMES:
        model,tokenizer,actual,index=_load_reference(name,REFERENCE_PANEL[name])
        model_scores[name]=_reference_scores(cache,model,tokenizer,index)
        actual_hashes[name]=actual
        entailment_indices[name]=index
        del model
        gc.collect()

    panel=evaluate_reference_panel(cache["cases"],model_scores)
    per_domain={}
    for domain in ("EU","EV"):
        p0=evaluations["P0"]["per_domain"][domain]
        t0=evaluations["T0"]["per_domain"][domain]
        t1=evaluations["T1"]["per_domain"][domain]
        per_domain[domain]={
            "reference_adequate":confirm_domain_pass(panel,domain),
            "t0_rescue":primary_rescue(t0,p0),
            "t1_rescue":replica_rescue(t1,p0),
            "p0_composed":float(p0["composed_severity_top1"]),
            "t0_composed":float(t0["composed_severity_top1"]),
            "t1_composed":float(t1["composed_severity_top1"]),
        }

    if not all(row["reference_adequate"] for row in per_domain.values()):
        outcome="W28_CONFIRM_REFERENCE_INADEQUATE"
    elif all(row["t0_rescue"] and row["t1_rescue"] for row in per_domain.values()):
        outcome="REPLICATED_COMPOSITIONAL_PROJECTION_RESCUE"
    else:
        outcome="COMPOSITIONAL_PROJECTION_RESCUE_NONREPLICATING"

    result={
        "schema_version":"r8-w28-compositional-projection-audit-v1",
        "status":"PASS",
        "outcome":outcome,
        "per_domain":per_domain,
        "evaluations":evaluations,
        "reference_panel_result":panel,
        "candidate_receipts":receipts,
        "reference_panel":REFERENCE_PANEL,
        "actual_reference_weight_sha256":actual_hashes,
        "reference_entailment_label_index":entailment_indices,
        "qualification_outcome":qualification["outcome"],
        "primary_f2_authority":"U_AND_C",
        "direct_f2_role":"diagnostic_only",
        "w9_freeze_status":freeze["status"],
        "w9_projection_scorer_sha256":w9_row["scorer_sha256"],
        "w9_hira_sha256":W9_HIRA_SHA256,
        "w9_scorer_sha256":W9_SCORER_SHA256,
        "confirm_domains":["EU","EV"],
        "confirm_materialized_after_both_candidate_freezes":True,
        "a13_frozen_all_paths":True,
        "reference_outputs_used_as_training_targets":False,
        "qualification_rows_used_for_hira_selection":False,
        "w27_rows_used":False,
        "w26_rows_used":False,
        "w25_rows_used":False,
        "w24_rows_used":False,
        "banking77_rows_used":False,
        "typed_decisions_final_or_test_used":False,
        "campaign_cells_populated":0,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"audit.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
