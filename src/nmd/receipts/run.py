from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(payload)


@dataclass(frozen=True)
class RunReceipt:
    run_id: str
    config_hash: str
    code_revision: str
    dataset_revisions: dict[str, str]
    checkpoint_revisions: dict[str, str]
    seed: int
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def write_once(self, path: str | Path) -> Path:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"immutable receipt already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target
