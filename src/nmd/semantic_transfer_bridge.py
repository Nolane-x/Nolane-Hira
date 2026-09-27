from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .symmetric_semantic import SymmetricSemanticScorer


class LowRankSemanticTransferBridge(nn.Module):
    """Tiny shared residual correction in rescued semantic space."""

    def __init__(self, d_rel: int = 128, rank: int = 8):
        super().__init__()
        self.d_rel = int(d_rel)
        self.rank = int(rank)
        if self.d_rel < 1 or self.rank < 1 or self.rank > self.d_rel:
            raise ValueError("bridge requires 1 <= rank <= d_rel")
        self.down = nn.Linear(self.d_rel, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.d_rel, bias=False)
        # Exact identity function at initialization while preserving a useful
        # random basis in the down projection for the first optimizer step.
        nn.init.zeros_(self.up.weight)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.d_rel:
            raise ValueError(f"bridge input must end in d_rel={self.d_rel}")
        return x + self.up(self.down(x))


class BridgedSymmetricSemanticScorer(SymmetricSemanticScorer):
    """W29 scorer plus one shared rank-limited transfer bridge.

    The rescued T0 projection remains a distinct parameter tensor and can stay
    permanently frozen. The same bridge transforms state and option tokens.
    """

    def __init__(self, d_model: int = 256, d_rel: int = 128, rank: int = 8):
        super().__init__(d_model=d_model, d_rel=d_rel)
        self.bridge = LowRankSemanticTransferBridge(d_rel=d_rel, rank=rank)

    @property
    def bridge_parameter_count(self) -> int:
        return self.bridge.parameter_count

    @property
    def bridge_trainable_parameter_count(self) -> int:
        return self.bridge.trainable_parameter_count

    def freeze_bridge(self) -> None:
        self.bridge.freeze()

    def load_bridge_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        expected = {"down.weight", "up.weight"}
        if set(state_dict) != expected:
            raise ValueError("bridge checkpoint keys changed")
        down = state_dict["down.weight"]
        up = state_dict["up.weight"]
        if tuple(down.shape) != (self.bridge.rank, self.d_rel):
            raise ValueError("bridge down weight shape mismatch")
        if tuple(up.shape) != (self.d_rel, self.bridge.rank):
            raise ValueError("bridge up weight shape mismatch")
        if not bool(torch.isfinite(down).all()) or not bool(torch.isfinite(up).all()):
            raise ValueError("bridge weights must be finite")
        self.bridge.load_state_dict(
            {
                "down.weight": down.detach().to(self.bridge.down.weight),
                "up.weight": up.detach().to(self.bridge.up.weight),
            }
        )
        if freeze:
            self.freeze_bridge()

    def _project(self, x: Tensor) -> Tensor:
        projected = self.projection(x)
        bridged = self.bridge(projected)
        return F.normalize(bridged, dim=-1)


__all__ = [
    "BridgedSymmetricSemanticScorer",
    "LowRankSemanticTransferBridge",
]
