from pathlib import Path

import pytest

from nmd.receipts import RunReceipt, canonical_json_hash


def test_receipt_is_immutable(tmp_path: Path):
    r = RunReceipt(
        run_id="r1",
        config_hash=canonical_json_hash({"a": 1}),
        code_revision="abc",
        dataset_revisions={"d": "sha"},
        checkpoint_revisions={"m": "rev"},
        seed=7,
        status="FAIL",
    )
    p = r.write_once(tmp_path / "r1.json")
    assert p.exists()
    with pytest.raises(FileExistsError):
        r.write_once(p)
