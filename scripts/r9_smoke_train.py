from __future__ import annotations
import json

import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.losses import LossWeights
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.training import DecisionExample, optimizer_step


def main():
    torch.manual_seed(13)
    encoder = TrainableSemanticEncoder(vocab_size=1024, d_model=256, n_layers=1, n_heads=4)
    model = NolaneHira(encoder, HIRACore(dropout=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    example = DecisionExample(
        state_text="my bank transfer is still pending",
        primitive="choice",
        question_text="what issue is described?",
        options=(
            LogicalOption("opaque_1", "cash withdrawal missing"),
            LogicalOption("opaque_2", "bank transfer pending"),
            LogicalOption("opaque_3", "card payment declined"),
        ),
        gold_index=1,
        teacher_probs=(0.02, 0.96, 0.02),
    )
    loss = optimizer_step(
        model, example, optimizer,
        weights=LossWeights(hard_ce=1.0, teacher_kl=0.2, brier=0.1),
        forced_budget=2,
    )
    print(json.dumps({
        "status": "PASS",
        "loss": loss,
        "state_encode_calls": encoder.state_encode_calls,
        "hira_params": sum(p.numel() for p in model.hira.parameters())
    }, sort_keys=True))


if __name__ == "__main__":
    main()
