import torch

from nmd.v1_paired_view_margin import (
    gold_vs_hardest_wrong_margin,
    paired_both_view_margin_loss,
)


def test_s18_zero_when_both_views_clear_margin():
    canonical = torch.tensor([[1.0, 0.1, -0.3, -0.7]], requires_grad=True)
    paraphrase = torch.tensor([[0.8, 0.2, -0.2, -0.5]], requires_grad=True)
    gold = torch.tensor([0], dtype=torch.long)
    loss, cm, pm, diag = paired_both_view_margin_loss(
        canonical, paraphrase, gold, margin=0.20
    )
    assert float(loss) == 0.0
    assert float(cm[0]) >= 0.20
    assert float(pm[0]) >= 0.20
    assert diag.both_satisfied_rate == 1.0


def test_s18_only_violating_view_receives_hinge_gradient():
    canonical = torch.tensor([[1.0, 0.0, -0.5, -1.0]], requires_grad=True)
    paraphrase = torch.tensor([[0.10, 0.05, 0.0, -0.2]], requires_grad=True)
    gold = torch.tensor([0], dtype=torch.long)
    loss, _cm, _pm, _diag = paired_both_view_margin_loss(
        canonical, paraphrase, gold, margin=0.20
    )
    loss.backward()
    assert canonical.grad is not None
    assert float(canonical.grad.abs().sum()) == 0.0
    assert paraphrase.grad is not None
    assert float(paraphrase.grad.abs().sum()) > 0.0


def test_s18_both_violating_views_receive_gradient():
    canonical = torch.tensor([[0.10, 0.05, 0.0, -0.2]], requires_grad=True)
    paraphrase = torch.tensor([[0.12, 0.08, 0.0, -0.2]], requires_grad=True)
    gold = torch.tensor([0], dtype=torch.long)
    loss, _cm, _pm, _diag = paired_both_view_margin_loss(
        canonical, paraphrase, gold, margin=0.20
    )
    loss.backward()
    assert float(canonical.grad.abs().sum()) > 0.0
    assert float(paraphrase.grad.abs().sum()) > 0.0


def test_s18_option_permutation_equivariance():
    torch.manual_seed(31001)
    canonical = torch.randn(9, 4)
    paraphrase = torch.randn(9, 4)
    gold = torch.tensor([0,1,2,3,0,1,2,3,1], dtype=torch.long)
    permutation = torch.tensor([2,0,3,1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(4)

    loss_a, cm_a, pm_a, _ = paired_both_view_margin_loss(
        canonical, paraphrase, gold
    )
    permuted_gold = inverse[gold]
    loss_b, cm_b, pm_b, _ = paired_both_view_margin_loss(
        canonical[:, permutation], paraphrase[:, permutation], permuted_gold
    )
    assert torch.allclose(loss_a, loss_b, atol=0.0, rtol=0.0)
    assert torch.allclose(cm_a, cm_b, atol=0.0, rtol=0.0)
    assert torch.allclose(pm_a, pm_b, atol=0.0, rtol=0.0)


def test_s18_identical_views_reduce_to_single_view_hinge():
    logits = torch.tensor([
        [0.2, 0.1, -0.1, -0.2],
        [0.0, 0.4, 0.2, -0.3],
    ])
    gold = torch.tensor([0, 1], dtype=torch.long)
    margin = gold_vs_hardest_wrong_margin(logits, gold)
    expected = torch.relu(logits.new_tensor(0.20) - margin).mean()
    loss, cm, pm, _ = paired_both_view_margin_loss(logits, logits, gold)
    assert torch.allclose(loss, expected)
    assert torch.equal(cm, pm)
