from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from nmd.semantic_encoder_adaptation import dev_selection_key


EXPECTED = 4


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_key(receipt: dict) -> tuple:
    metrics = receipt["selected_dev_metrics"]
    epoch = int(receipt["selected_epoch"])
    return (
        *dev_selection_key(metrics, epoch),
        int(receipt["top_n"]),
        float(receipt["lr"]),
        str(receipt["candidate_name"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[dict, Path]] = []
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "r8-w5d-candidate-v1":
            continue
        if receipt.get("status") != "PASS":
            raise RuntimeError("W5d candidate is not PASS")
        if receipt.get("confirm_exposed") is not False:
            raise RuntimeError("W5d candidate exposed confirm before selection")
        checkpoint = receipt_path.parent / "adaptation.pt"
        if not checkpoint.exists():
            raise RuntimeError("W5d candidate checkpoint missing")
        if file_sha256(checkpoint) != receipt["adaptation_sha256"]:
            raise RuntimeError("W5d candidate checkpoint SHA mismatch")
        rows.append((receipt, checkpoint))

    names = [row[0]["candidate_name"] for row in rows]
    if len(rows) != EXPECTED or len(set(names)) != EXPECTED:
        raise RuntimeError("expected exactly four unique W5d candidates")

    selected, source = min(rows, key=lambda row: candidate_key(row[0]))
    args.out.mkdir(parents=True, exist_ok=True)
    out_checkpoint = args.out / "adaptation.pt"
    shutil.copyfile(source, out_checkpoint)

    summary = {
        "schema_version": "r8-w5d-selection-v1",
        "status": "PASS",
        "selection_authority": "DEV_ONLY",
        "candidate_count": EXPECTED,
        "selected_candidate": selected["candidate_name"],
        "selected_config": {
            "top_n": selected["top_n"],
            "lr": selected["lr"],
            "selected_epoch": selected["selected_epoch"],
            "global_seed": selected["global_seed"],
        },
        "frozen_dev_metrics": selected["frozen_dev_metrics"],
        "selected_dev_metrics": selected["selected_dev_metrics"],
        "selected_adaptation_sha256": file_sha256(out_checkpoint),
        "base_a13_revision": selected["base_a13_revision"],
        "base_a13_weight_sha256": selected["base_a13_weight_sha256"],
        "layer_count": selected["layer_count"],
        "trainable_parameter_names": selected["trainable_parameter_names"],
        "trainable_parameter_count": selected["trainable_parameter_count"],
        "frozen_parameter_count": selected["frozen_parameter_count"],
        "candidates": [
            {
                "candidate_name": receipt["candidate_name"],
                "top_n": receipt["top_n"],
                "lr": receipt["lr"],
                "selected_epoch": receipt["selected_epoch"],
                "selected_dev_metrics": receipt["selected_dev_metrics"],
                "adaptation_sha256": receipt["adaptation_sha256"],
            }
            for receipt, _ in sorted(rows, key=lambda row: row[0]["candidate_name"])
        ],
        "confirm_exposed": False,
        "forbidden_benchmark_data_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "selection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
