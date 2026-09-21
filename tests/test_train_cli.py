import json
from pathlib import Path
import subprocess
import sys


def test_training_cli_emits_checkpoint_and_receipt(tmp_path: Path):
    data = tmp_path / "tiny.jsonl"
    rows = [
        {
            "state": "my transfer is pending",
            "primitive": "choice",
            "question": "what is the issue?",
            "options": [
                {"id": "a", "text": "cash withdrawal issue"},
                {"id": "b", "text": "pending transfer"},
            ],
            "gold_index": 1,
            "teacher_probs": [0.05, 0.95],
        },
        {
            "state": "cash did not arrive from the atm",
            "primitive": "choice",
            "question": "what is the issue?",
            "options": [
                {"id": "a", "text": "cash withdrawal issue"},
                {"id": "b", "text": "pending transfer"},
            ],
            "gold_index": 0,
            "teacher_probs": [0.95, 0.05],
        },
    ]
    data.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/train_decisions.py",
            "--data", str(data),
            "--out", str(out),
            "--run-id", "test-cli",
            "--code-revision", "test",
            "--encoder", "toy",
            "--toy-vocab", "256",
            "--toy-layers", "1",
            "--epochs", "1",
            "--forced-budget", "2",
            "--teacher-kl", "0.2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["steps"] == 2
    assert (out / "receipt.json").exists()
    assert (out / "checkpoint" / "model.pt").exists()
    manifest = json.loads((out / "checkpoint" / "manifest.json").read_text())
    receipt = json.loads((out / "receipt.json").read_text())
    assert manifest["weights_sha256"] == receipt["checkpoint_revisions"]["weights_sha256"]
