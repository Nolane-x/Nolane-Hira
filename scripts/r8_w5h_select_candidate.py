from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from nmd.semantic_balanced_binding import CANDIDATES, dev_selection_key

ORDER = {name: index for index, name in enumerate(CANDIDATES)}
CONTROL = "idf-maxsim-proj128"


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
        if receipt.get("schema_version") != "r8-w5h-candidate-receipt-v1":
            continue
        state_path = receipt_path.parent / "matcher.pt"
        if file_sha256(state_path) != receipt["matcher_sha256"]:
            raise RuntimeError("W5h matcher SHA mismatch")
        rows.append((receipt, state_path))

    names = {receipt["candidate"] for receipt, _ in rows}
    if names != set(CANDIDATES) or len(rows) != 3:
        raise RuntimeError(f"expected exactly three W5h candidates, got {names}")

    selected, selected_state = min(rows, key=lambda row: key(row[0]))
    control_receipt, control_state = next(
        row for row in rows if row[0]["candidate"] == CONTROL
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out_selected = args.out / "matcher.pt"
    out_control = args.out / "control-matcher.pt"
    shutil.copyfile(selected_state, out_selected)
    shutil.copyfile(control_state, out_control)

    summary = {
        "schema_version": "r8-w5h-selection-receipt-v1",
        "status": "PASS",
        "selected_candidate": selected["candidate"],
        "selected_epoch": selected["selected_epoch"],
        "selected_dev_metrics": selected["selected_dev_metrics"],
        "selected_matcher_sha256": file_sha256(out_selected),
        "selected_trainable_parameter_count": selected["trainable_parameter_count"],
        "control_candidate": CONTROL,
        "control_epoch": control_receipt["selected_epoch"],
        "control_dev_metrics": control_receipt["selected_dev_metrics"],
        "control_matcher_sha256": file_sha256(out_control),
        "control_trainable_parameter_count": control_receipt["trainable_parameter_count"],
        "all_candidate_parameter_counts_equal": (
            len({int(receipt["trainable_parameter_count"]) for receipt, _ in rows}) == 1
        ),
        "confirm_exposed": False,
        "candidates": [
            {
                "candidate": receipt["candidate"],
                "selected_epoch": receipt["selected_epoch"],
                "selected_dev_metrics": receipt["selected_dev_metrics"],
                "trainable_parameter_count": receipt["trainable_parameter_count"],
                "history": receipt["history"],
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
