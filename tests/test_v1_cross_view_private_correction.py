import torch

from nmd.v1_cross_view_private_correction import symmetric_cross_view_js


def test_s45_js_identical_logits_are_zero():
    g = torch.Generator().manual_seed(45001)
    logits = torch.randn(7, 4, generator=g)
    js = symmetric_cross_view_js(logits, logits)
    assert float(js) == 0.0


def test_s45_js_is_symmetric_and_positive_for_different_views():
    g = torch.Generator().manual_seed(45002)
    a = torch.randn(9, 7, generator=g)
    b = torch.randn(9, 7, generator=g)
    ab = symmetric_cross_view_js(a, b)
    ba = symmetric_cross_view_js(b, a)
    assert float(ab) > 0.0
    assert torch.allclose(ab, ba, rtol=0.0, atol=1e-8)


def test_s45_js_has_finite_nonzero_gradients():
    g = torch.Generator().manual_seed(45003)
    a = torch.randn(5, 4, generator=g, requires_grad=True)
    b = torch.randn(5, 4, generator=g, requires_grad=True)
    js = symmetric_cross_view_js(a, b)
    ga, gb = torch.autograd.grad(js, (a, b))
    assert bool(torch.isfinite(ga).all())
    assert bool(torch.isfinite(gb).all())
    assert float(ga.abs().sum()) > 0.0
    assert float(gb.abs().sum()) > 0.0


def test_s45_js_rejects_unpaired_shapes():
    a = torch.zeros(2, 4)
    b = torch.zeros(3, 4)
    try:
        symmetric_cross_view_js(a, b)
    except ValueError as exc:
        assert "shapes must match" in str(exc)
    else:
        raise AssertionError("S45 JS accepted unpaired semantic views")
