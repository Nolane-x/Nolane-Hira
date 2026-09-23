from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from nmd.semantic_routing_curriculum import dev_selection_key

EXPECTED_CANDIDATES=4


def file_sha256(path: str | Path) -> str:
    h=sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_key(receipt: dict) -> tuple:
    metrics=receipt["selected_dev_metrics"]
    epoch=int(receipt["selected_epoch"])
    init_rank=0 if receipt["init"]=="w3" else 1
    return (
        *dev_selection_key(metrics,epoch),
        init_rank,
        float(receipt["lr"]),
        str(receipt["candidate_name"]),
    )


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--candidates",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    rows=[]
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version")!="r8-w5a-candidate-v1":
            continue
        if receipt.get("status")!="PASS":
            raise RuntimeError("candidate receipt not PASS")
        if receipt.get("confirm_exposed") is not False:
            raise RuntimeError("candidate was exposed to confirm authority")
        head=receipt_path.parent/"hira-head.pt"
        if not head.exists():
            raise RuntimeError("candidate head missing")
        if file_sha256(head)!=receipt["head_sha256"]:
            raise RuntimeError("candidate head SHA mismatch")
        rows.append((receipt,head))

    names=[receipt["candidate_name"] for receipt,_ in rows]
    if len(rows)!=EXPECTED_CANDIDATES or len(set(names))!=EXPECTED_CANDIDATES:
        raise RuntimeError("expected exactly four unique W5a candidates")

    selected_receipt,selected_head=min(rows,key=lambda row:candidate_key(row[0]))
    args.out.mkdir(parents=True,exist_ok=True)
    out_head=args.out/"hira-head.pt"
    shutil.copyfile(selected_head,out_head)

    result={
        "schema_version":"r8-w5a-selection-v1",
        "status":"PASS",
        "candidate_count":len(rows),
        "selected_candidate":selected_receipt["candidate_name"],
        "selected_init":selected_receipt["init"],
        "selected_lr":selected_receipt["lr"],
        "selected_epoch":selected_receipt["selected_epoch"],
        "selected_dev_metrics":selected_receipt["selected_dev_metrics"],
        "selected_head_sha256":file_sha256(out_head),
        "confirm_exposed":False,
        "selection_authority":"DEV_ONLY",
        "forbidden_benchmark_data_used":False,
        "candidates":[
            {
                "candidate_name":r["candidate_name"],
                "init":r["init"],
                "lr":r["lr"],
                "selected_epoch":r["selected_epoch"],
                "selected_dev_metrics":r["selected_dev_metrics"],
                "head_sha256":r["head_sha256"],
            }
            for r,_ in sorted(rows,key=lambda row:row[0]["candidate_name"])
        ],
    }
    (args.out/"selection.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
