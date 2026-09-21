from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch

from nmd.factorized import FactorizedBertEmbeddings, FactorizedEmbeddingConfig
from nmd.hira import HIRACore
from nmd.receipts import RunReceipt, canonical_json_hash


def main() -> None:
    hira = HIRACore(d_model=256, d_rel=128)
    hira_params = sum(p.numel() for p in hira.parameters())
    assert hira_params == 422_159, hira_params

    emb = FactorizedBertEmbeddings(FactorizedEmbeddingConfig(vocab_size=24_000, embedding_size=96, hidden_size=256))
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    y = emb(x)
    assert y.shape == (1, 3, 256)

    cfg = {"candidate": "A7-FE24", "vocab": 24_000, "embedding_dim": 96, "hidden": 256}
    receipt = RunReceipt(
        run_id="preflight",
        config_hash=canonical_json_hash(cfg),
        code_revision="uncommitted-preflight",
        dataset_revisions={},
        checkpoint_revisions={},
        seed=0,
        status="PASS",
        metrics={"hira_params": float(hira_params)},
    )
    with tempfile.TemporaryDirectory() as td:
        path = receipt.write_once(Path(td) / "receipt.json")
        parsed = json.loads(path.read_text())
        assert parsed["status"] == "PASS"
        try:
            receipt.write_once(path)
        except FileExistsError:
            pass
        else:
            raise AssertionError("receipt overwrite was not blocked")

    print(json.dumps({"status": "PASS", "hira_params": hira_params, "factorized_shape": list(y.shape)}))


if __name__ == "__main__":
    main()
