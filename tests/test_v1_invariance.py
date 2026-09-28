import torch

from nmd.v1_invariance import (
    selected_choice_agreement,
    symmetric_js_divergence,
)


def test_s8_js_is_zero_for_identical_logits():
    torch.manual_seed(108001)
    logits = torch.randn(7, 4)
    js = symmetric_js_divergence(logits, logits)
    assert torch.isclose(js, torch.tensor(0.0), atol=1e-7)


def test_s8_js_is_symmetric_nonnegative_and_backpropagates():
    torch.manual_seed(108002)
    a = torch.randn(8, 4, requires_grad=True)
    b = torch.randn(8, 4, requires_grad=True)
    ab = symmetric_js_divergence(a, b)
    ba = symmetric_js_divergence(b, a)
    assert bool(torch.isfinite(ab))
    assert float(ab) >= 0.0
    assert torch.allclose(ab, ba, atol=1e-7, rtol=1e-7)
    ab.backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0.0
    assert b.grad is not None and float(b.grad.abs().sum()) > 0.0


def test_s8_selected_choice_agreement():
    a = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    b = torch.tensor([[2.0, 1.0, 0.0], [3.0, 2.0, 1.0]])
    assert selected_choice_agreement(a, b).item() == 0.5
