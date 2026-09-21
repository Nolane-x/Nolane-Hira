from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointManifest:
    format_version: str
    model_class: str
    code_revision: str
    encoder_revision: str
    config_hash: str
    weights_sha256: str


def canonical_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    model: nn.Module, directory: str | Path, *, config: dict[str, Any],
    code_revision: str, encoder_revision: str
) -> CheckpointManifest:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "model.pt"
    torch.save(model.state_dict(), weights)
    manifest = CheckpointManifest(
        format_version="hira-v1",
        model_class=type(model).__name__,
        code_revision=code_revision,
        encoder_revision=encoder_revision,
        config_hash=canonical_hash(config),
        weights_sha256=file_sha256(weights),
    )
    (directory / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    )
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_state_checked(model: nn.Module, directory: str | Path) -> CheckpointManifest:
    directory = Path(directory)
    manifest = CheckpointManifest(**json.loads((directory / "manifest.json").read_text()))
    weights = directory / "model.pt"
    if file_sha256(weights) != manifest.weights_sha256:
        raise ValueError("checkpoint weights hash mismatch")
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    return manifest
