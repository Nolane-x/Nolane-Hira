from __future__ import annotations

from pathlib import Path

import torch

from .hira import HIRACore
from .runtime import NolaneHira
from .semantic import TextSemanticEncoder
from .symmetric_semantic import SymmetricSemanticScorer
from .typed_competitive_cache import file_sha256

W28_T0_CHECKPOINT_SHA256 = (
    "1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f"
)
W28_T0_CHECKPOINT_SCHEMA = "r8-w28-candidate-checkpoint-v1"
W28_T0_CANDIDATE = "T0"


def load_rescued_projection_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_candidate: str = W28_T0_CANDIDATE,
) -> torch.Tensor:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "rescued projection checkpoint SHA mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != W28_T0_CHECKPOINT_SCHEMA:
        raise RuntimeError("rescued projection checkpoint schema mismatch")
    if checkpoint.get("candidate") != expected_candidate:
        raise RuntimeError("rescued projection candidate mismatch")
    if checkpoint.get("kind") != "projection":
        raise RuntimeError("rescued projection checkpoint kind mismatch")
    weight = checkpoint.get("projection_weight")
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError("rescued projection checkpoint has no tensor weight")
    if tuple(weight.shape) != (128, 256):
        raise RuntimeError("rescued projection weight shape must be [128,256]")
    if not bool(torch.isfinite(weight).all()):
        raise RuntimeError("rescued projection weight must be finite")
    return weight.detach().float().clone()


def build_hira_v0_semantic_core(
    encoder: TextSemanticEncoder,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = W28_T0_CHECKPOINT_SHA256,
    hira: HIRACore | None = None,
) -> NolaneHira:
    if int(encoder.d_model) != 256:
        raise ValueError("HIRA v0 rescued semantic core requires d_model=256")
    weight = load_rescued_projection_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
    )
    scorer = SymmetricSemanticScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(weight, freeze=True)
    if scorer.trainable_parameter_count != 0:
        raise RuntimeError("HIRA v0 rescued projection must remain frozen")
    model = NolaneHira(
        encoder,
        hira or HIRACore(d_model=256),
        symmetric_semantic_scorer=scorer,
    )
    return model


__all__ = [
    "W28_T0_CANDIDATE",
    "W28_T0_CHECKPOINT_SCHEMA",
    "W28_T0_CHECKPOINT_SHA256",
    "build_hira_v0_semantic_core",
    "load_rescued_projection_checkpoint",
]
