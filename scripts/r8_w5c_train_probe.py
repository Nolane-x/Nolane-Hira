from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.semantic_alignment_probes import (
    BilinearAlignmentProbe,
    PairMLPAlignmentProbe,
    EPOCHS,
    GLOBAL_SEED,
    dev_selection_key,
    load_alignment_cache,
    probe_parameter_count,
    train_learned_probe,
)


def file_sha256(path):
    h=sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-cache",type=Path,required=True)
    p.add_argument("--dev-cache",type=Path,required=True)
    p.add_argument("--probe",choices=["bilinear","pair_mlp"],required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()

    train=load_alignment_cache(args.train_cache)
    dev=load_alignment_cache(args.dev_cache)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if args.probe=="bilinear":
        model=BilinearAlignmentProbe()
    else:
        model=PairMLPAlignmentProbe(dropout=0.05)

    history,state,best=train_learned_probe(
        model,train,dev,probe=args.probe,lr=3e-4,epochs=EPOCHS,seed=GLOBAL_SEED
    )
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/"probe.pt"
    torch.save(state,path)
    epoch=int(best["epoch"])
    receipt={
        "schema_version":"r8-w5c-learned-probe-v1",
        "status":"PASS",
        "probe":args.probe,
        "lr":3e-4,
        "weight_decay":0.01,
        "epochs":EPOCHS,
        "global_seed":GLOBAL_SEED,
        "selected_epoch":epoch,
        "selected_dev_metrics":best,
        "selection_key":list(dev_selection_key(best,epoch)),
        "probe_sha256":file_sha256(path),
        "parameter_count":probe_parameter_count(model),
        "confirm_exposed":False,
        "forbidden_benchmark_data_used":False,
        "history":history,
    }
    (args.out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "probe":args.probe,
        "selected_epoch":epoch,
        "selected_dev_metrics":best,
        "probe_sha256":receipt["probe_sha256"],
        "parameter_count":receipt["parameter_count"],
    },sort_keys=True))


if __name__=="__main__":
    main()
