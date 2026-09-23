from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from nmd.semantic_late_interaction import dev_selection_key

ORDER = {
    "raw-maxsim": 0,
    "proj64-maxsim": 1,
    "proj128-maxsim": 2,
}


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key(receipt: dict) -> tuple:
    metrics = receipt["selected_dev_metrics"]
    epoch = int(receipt["selected_epoch"])
    return (
        *dev_selection_key(metrics, epoch),
        ORDER[receipt["candidate"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "r8-w5f-candidate-receipt-v1":
            continue
        state_path = receipt_path.parent / "matcher.pt"
        if file_sha256(state_path) != receipt["matcher_sha256"]:
            raise RuntimeError("W5f matcher SHA mismatch")
        rows.append((receipt, state_path))

    names = {receipt["candidate"] for receipt, _ in rows}
    if names != set(ORDER) or len(rows) != 3:
        raise RuntimeError(f"expected exactly three W5f candidates, got {names}")

    selected, state_path = min(rows, key=lambda row: key(row[0]))
    args.out.mkdir(parents=True, exist_ok=True)
    out_state = args.out / "matcher.pt"
    shutil.copyfile(state_path, out_state)

    summary = {
        "schema_version": "r8-w5f-selection-receipt-v1",
        "status": "PASS",
        "selected_candidate": selected["candidate"],
        "projection_dim": selected["projection_dim"],
        "selected_epoch": selected["selected_epoch"],
        "selected_dev_metrics": selected["selected_dev_metrics"],
        "selected_matcher_sha256": file_sha256(out_state),
        "confirm_exposed": False,
        "candidates": [
            {
                "candidate": receipt["candidate"],
                "projection_dim": receipt["projection_dim"],
                "selected_epoch": receipt["selected_epoch"],
                "selected_dev_metrics": receipt["selected_dev_metrics"],
                "matcher_sha256": receipt["matcher_sha256"],
            }
            for receipt, _ in sorted(rows, key=lambda row: ORDER[row[0]["candidate"]])
        ],
    }
    (args.out / "selection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
