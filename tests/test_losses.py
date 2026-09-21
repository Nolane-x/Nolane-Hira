import torch
from nmd.losses import LossWeights, typed_decision_loss, pairwise_margin_loss


def test_typed_loss_is_finite_and_backpropagates():
    logits = torch.randn(4, 5, requires_grad=True)
    gold = torch.tensor([0, 1, 2, 3])
    teacher = torch.softmax(torch.randn(4, 5), dim=-1)
    loss, parts = typed_decision_loss(
        logits,
        gold,
        teacher_probs=teacher,
        weights=LossWeights(hard_ce=1.0, teacher_kl=0.5, brier=0.1, soft_brier=0.2),
    )
    assert torch.isfinite(loss)
    assert {"hard_ce", "teacher_kl", "brier", "soft_brier"} <= set(parts)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_pairwise_margin():
    logits = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
    loss = pairwise_margin_loss(logits, torch.tensor([0]), torch.tensor([1]), margin=1.5)
    assert abs(loss.item() - 0.5) < 1e-6
    loss.backward()
