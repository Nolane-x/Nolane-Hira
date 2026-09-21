import torch
from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.training import DecisionExample, loss_example
from nmd.losses import LossWeights


def test_end_to_end_gradients_reach_encoder_and_hira():
    torch.manual_seed(11)
    enc = TrainableSemanticEncoder(vocab_size=256, d_model=256, n_layers=1, n_heads=4)
    model = NolaneHira(enc, HIRACore(dropout=0.0))
    model.train()
    ex = DecisionExample(
        state_text="my transfer has not arrived",
        primitive="choice",
        question_text="what is the issue?",
        options=(
            LogicalOption("id_a", "cash withdrawal issue"),
            LogicalOption("id_b", "pending bank transfer"),
            LogicalOption("id_c", "card payment declined"),
        ),
        gold_index=1,
        teacher_probs=(0.05, 0.9, 0.05),
    )
    loss, _ = loss_example(
        model, ex,
        weights=LossWeights(hard_ce=1.0, teacher_kl=0.2, brier=0.1),
        forced_budget=2,
    )
    loss.backward()
    assert enc.embedding.weight.grad is not None
    assert model.hira.cross_score[0].weight.grad is not None
    assert torch.isfinite(loss)
