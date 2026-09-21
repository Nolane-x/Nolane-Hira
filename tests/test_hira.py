import torch
from nmd.hira import HIRACore, count_parameters


def test_hira_relation_refinement_preserves_unselected_logits():
    torch.manual_seed(7)
    m = HIRACore(dropout=0.0)
    assert count_parameters(m) == 422_159
    q = torch.randn(2, 256)
    seg = torch.randn(2, 5, 256)
    opt = torch.randn(2, 11, 256)
    typ = torch.tensor([0, 2])
    out = m(q, seg, opt, typ, forced_budget=4)

    assert out.probabilities.shape == (2, 11)
    assert torch.allclose(out.probabilities.sum(-1), torch.ones(2), atol=1e-6)
    assert out.selected_indices.shape == (2, 4)
    assert out.selected_mask.all()
    assert torch.equal(out.candidate_budget, torch.tensor([4, 4]))
    assert (out.tail_mass >= 0).all()

    selected = torch.zeros_like(out.logits, dtype=torch.bool)
    selected.scatter_(1, out.selected_indices, True)
    assert torch.allclose(out.logits[~selected], out.coarse_logits[~selected], atol=0, rtol=0)
    assert out.relation_delta[selected].abs().sum() > 0


def test_hira_relation_path_gets_gradients():
    torch.manual_seed(3)
    m = HIRACore(dropout=0.0)
    q = torch.randn(2, 256, requires_grad=True)
    seg = torch.randn(2, 6, 256, requires_grad=True)
    opt = torch.randn(2, 9, 256, requires_grad=True)
    typ = torch.tensor([0, 1])
    out = m(q, seg, opt, typ, forced_budget=3)
    loss = -out.probabilities[:, 0].log().mean()
    loss.backward()
    assert m.cross_score[0].weight.grad is not None
    assert m.cross_attn.in_proj_weight.grad is not None
    assert q.grad is not None and seg.grad is not None and opt.grad is not None


def test_safe_default_is_all_k_before_budget_calibration():
    m = HIRACore(dropout=0.0)
    out = m(
        torch.randn(1, 256),
        torch.randn(1, 4, 256),
        torch.randn(1, 7, 256),
        torch.tensor([0]),
    )
    assert out.candidate_budget.item() == 7
    assert out.tail_mass.item() == 0
