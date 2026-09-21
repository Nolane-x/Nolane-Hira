from pathlib import Path
import torch
from nmd.checkpoint import save_checkpoint, load_state_checked
from nmd.hira import HIRACore


def test_checkpoint_roundtrip_and_hash_guard(tmp_path: Path):
    torch.manual_seed(1)
    a = HIRACore(dropout=0.0)
    manifest = save_checkpoint(
        a, tmp_path, config={"d_model": 256},
        code_revision="test", encoder_revision="test-encoder"
    )
    b = HIRACore(dropout=0.0)
    load_state_checked(b, tmp_path)
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x, y)
    assert len(manifest.weights_sha256) == 64

    p = tmp_path / "model.pt"
    data = bytearray(p.read_bytes())
    data[-1] ^= 1
    p.write_bytes(data)
    try:
        load_state_checked(b, tmp_path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("corrupted checkpoint was accepted")
