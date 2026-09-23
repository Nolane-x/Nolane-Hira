from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from nmd.semantic_token_curriculum import dev_selection_key

EXPECTED_MODES = {
    "pooled",
    "option_tokens",
    "state_tokens",
    "dual_tokens",
}
TIE_RANK = {
    "dual_tokens": 0,
    "state_tokens": 1,
    "option_tokens": 2,
    "pooled": 3,
}


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_key(receipt: dict) -> tuple:
    metrics = receipt["selected_dev_metrics"]
    epoch = int(receipt["selected_epoch"])
    return (
        *dev_selection_key(metrics, epoch),
        TIE_RANK[receipt["relation_mode"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[dict, Path]] = []
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "r8-w5b-candidate-v1":
            continue
        if receipt.get("status") != "PASS":
            raise RuntimeError("W5b candidate receipt is not PASS")
        head = receipt_path.parent / "hira-head.pt"
        if not head.exists():
            raise RuntimeError("W5b candidate head missing")
        if file_sha256(head) != receipt["head_sha256"]:
            raise RuntimeError("W5b candidate head SHA mismatch")
        rows.append((receipt, head))

    modes = {receipt["relation_mode"] for receipt, _ in rows}
    if len(rows) != 4 or modes != EXPECTED_MODES:
        raise RuntimeError(
            f"expected four unique W5b modes, got {sorted(modes)}"
        )

    selected, selected_head = min(rows, key=lambda row: row_key(row[0]))
    pooled = next(
        receipt
        for receipt, _ in rows
        if receipt["relation_mode"] == "pooled"
    )
    selected_accuracy = float(
        selected["selected_dev_metrics"]["accuracy"]
    )
    pooled_accuracy = float(
        pooled["selected_dev_metrics"]["accuracy"]
    )
    dev_gain = selected_accuracy - pooled_accuracy
    mechanism_gate = (
        selected["relation_mode"] != "pooled"
        and dev_gain >= 0.10
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out_head = args.out / "hira-head.pt"
    shutil.copyfile(selected_head, out_head)

    summary = {
        "schema_version": "r8-w5b-selection-v1",
        "status": "PASS",
        "candidate_count": 4,
        "selection_authority": "DEV_ONLY",
        "selected_relation_mode": selected["relation_mode"],
        "selected_epoch": selected["selected_epoch"],
        "selected_head_sha256": file_sha256(out_head),
        "selected_dev_metrics": selected["selected_dev_metrics"],
        "pooled_dev_metrics": pooled["selected_dev_metrics"],
        "selected_vs_pooled_accuracy_gain": dev_gain,
        "mechanism_evidence_gate": mechanism_gate,
        "confirm_exposed": False,
        "forbidden_benchmark_data_used": False,
        "candidates": [
            {
                "relation_mode": receipt["relation_mode"],
                "selected_epoch": receipt["selected_epoch"],
                "selected_dev_metrics": receipt[
                    "selected_dev_metrics"
                ],
                "head_sha256": receipt["head_sha256"],
            }
            for receipt, _ in sorted(
                rows,
                key=lambda row: row[0]["relation_mode"],
            )
        ],
    }
    (args.out / "selection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
