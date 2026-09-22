from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from nmd.typed_feature_cache import dev_selection_key


EXPECTED_CANDIDATES = 8


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tie_key(receipt: dict) -> tuple:
    metrics = receipt["selected_dev_metrics"]
    epoch = int(receipt["selected_epoch"])
    init_rank = 0 if receipt["init"] == "r15" else 1
    loss_rank = 0 if receipt["loss_family"] == "balanced" else 1
    return (
        *dev_selection_key(metrics, epoch=epoch),
        init_rank,
        float(receipt["lr"]),
        loss_rank,
        str(receipt["candidate_name"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[dict, Path]] = []
    for receipt_path in sorted(
        args.candidates.rglob("receipt.json")
    ):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "r8-w3-candidate-receipt-v1":
            continue
        if receipt.get("status") != "PASS":
            raise RuntimeError("W3 candidate receipt is not PASS")
        head_path = receipt_path.parent / "hira-head.pt"
        if not head_path.exists():
            raise RuntimeError("W3 candidate head missing")
        if file_sha256(head_path) != receipt["head_sha256"]:
            raise RuntimeError("W3 candidate head SHA mismatch")
        rows.append((receipt, head_path))

    names = [row[0]["candidate_name"] for row in rows]
    if len(rows) != EXPECTED_CANDIDATES or len(set(names)) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"expected exactly {EXPECTED_CANDIDATES} unique W3 candidates"
        )

    selected_receipt, selected_head = min(
        rows,
        key=lambda row: tie_key(row[0]),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out_head = args.out / "hira-head.pt"
    shutil.copyfile(selected_head, out_head)

    summary = {
        "schema_version": "r8-w3-selection-receipt-v1",
        "status": "PASS",
        "scope": (
            "TRAIN/DEV-only candidate freeze before any typed-decisions "
            "final-test exposure"
        ),
        "candidate_count": len(rows),
        "selection_rule": [
            "higher DEV accuracy",
            "lower DEV hard Brier",
            "higher DEV soft accuracy",
            "lower DEV score MAE",
            "lower DEV ECE",
            "earlier epoch",
            "r15 init before fresh13",
            "lower learning rate",
            "balanced before soft",
        ],
        "selected_candidate": selected_receipt["candidate_name"],
        "selected_config": {
            "init": selected_receipt["init"],
            "loss_family": selected_receipt["loss_family"],
            "loss_weights": selected_receipt["loss_weights"],
            "lr": selected_receipt["lr"],
            "selected_epoch": selected_receipt["selected_epoch"],
            "global_seed": selected_receipt["global_seed"],
        },
        "selected_dev_metrics": selected_receipt[
            "selected_dev_metrics"
        ],
        "selected_source_head_sha256": selected_receipt["head_sha256"],
        "selected_head_sha256": file_sha256(out_head),
        "candidates": [
            {
                "candidate_name": receipt["candidate_name"],
                "init": receipt["init"],
                "loss_family": receipt["loss_family"],
                "lr": receipt["lr"],
                "selected_epoch": receipt["selected_epoch"],
                "selected_dev_metrics": receipt[
                    "selected_dev_metrics"
                ],
                "head_sha256": receipt["head_sha256"],
            }
            for receipt, _ in sorted(
                rows,
                key=lambda row: row[0]["candidate_name"],
            )
        ],
        "final_test_exposed": False,
    }
    (args.out / "selection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
