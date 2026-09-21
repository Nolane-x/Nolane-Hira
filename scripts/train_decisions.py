from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.checkpoint import save_checkpoint
from nmd.dataset import load_jsonl
from nmd.hira import HIRACore
from nmd.losses import LossWeights
from nmd.receipts import RunReceipt, canonical_json_hash
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder, TrainableSemanticEncoder
from nmd.training import optimizer_step


MUTABLE_REVISIONS = {"main", "master", "latest"}


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_model(args) -> NolaneHira:
    if args.encoder == "toy":
        encoder = TrainableSemanticEncoder(
            vocab_size=args.toy_vocab,
            d_model=256,
            n_layers=args.toy_layers,
            n_heads=4,
        )
    else:
        if not args.model_id or not args.revision:
            raise SystemExit("--model-id and immutable --revision are required for --encoder hf")
        if args.revision.lower() in MUTABLE_REVISIONS:
            raise SystemExit("mutable HF revisions are forbidden")
        encoder = HFAutoSemanticEncoder.from_pretrained(
            args.model_id,
            revision=args.revision,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
        )
        if encoder.d_model != 256:
            raise SystemExit(f"R9 proof runtime expects hidden_size=256, got {encoder.d_model}")
    return NolaneHira(encoder, HIRACore(d_model=256))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--code-revision", required=True)
    ap.add_argument("--encoder", choices=["toy", "hf"], default="toy")
    ap.add_argument("--model-id")
    ap.add_argument("--revision")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--toy-vocab", type=int, default=8192)
    ap.add_argument("--toy-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--forced-budget", type=int)
    ap.add_argument("--hard-ce", type=float, default=1.0)
    ap.add_argument("--teacher-kl", type=float, default=0.0)
    ap.add_argument("--brier", type=float, default=0.1)
    ap.add_argument("--soft-brier", type=float, default=0.0)
    ap.add_argument("--ordinal-mae", type=float, default=0.0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = load_jsonl(args.data)
    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    weights = LossWeights(
        hard_ce=args.hard_ce,
        teacher_kl=args.teacher_kl,
        brier=args.brier,
        soft_brier=args.soft_brier,
        ordinal_mae=args.ordinal_mae,
    )

    losses = []
    for _ in range(args.epochs):
        order = list(range(len(examples)))
        random.shuffle(order)
        for i in order:
            losses.append(
                optimizer_step(
                    model,
                    examples[i],
                    optimizer,
                    weights=weights,
                    forced_budget=args.forced_budget,
                )
            )

    config = {
        "encoder": args.encoder,
        "model_id": args.model_id,
        "encoder_revision": args.revision or model.encoder.encoder_hash,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "forced_budget": args.forced_budget,
        "loss_weights": weights.__dict__,
        "dataset_sha256": file_sha256(args.data),
        "examples": len(examples),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = save_checkpoint(
        model,
        args.out / "checkpoint",
        config=config,
        code_revision=args.code_revision,
        encoder_revision=config["encoder_revision"],
    )
    mean_loss = float(sum(losses) / max(1, len(losses)))
    receipt = RunReceipt(
        run_id=args.run_id,
        config_hash=canonical_json_hash(config),
        code_revision=args.code_revision,
        dataset_revisions={"jsonl_sha256": config["dataset_sha256"]},
        checkpoint_revisions={
            "encoder": str(config["encoder_revision"]),
            "weights_sha256": manifest.weights_sha256,
        },
        seed=args.seed,
        status="PASS",
        metrics={"mean_train_loss": mean_loss, "optimizer_steps": float(len(losses))},
        notes=("training metric only; no benchmark/generalization claim",),
    )
    receipt.write_once(args.out / "receipt.json")
    print(json.dumps({
        "status": "PASS",
        "examples": len(examples),
        "steps": len(losses),
        "mean_train_loss": mean_loss,
        "weights_sha256": manifest.weights_sha256,
        "receipt": str(args.out / "receipt.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
