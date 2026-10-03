import torch

from nmd.v1_cross_view_relational_anchor import cross_view_similarity_matrix
from nmd.v1_cross_view_relative_gap_anchor import (
    relative_gap_geometry_loss,
    relative_gap_signature_geometry_anchor,
    relative_gap_tensor,
)


def test_s43_relative_gap_tensor_shape_and_zero_diagonal():
    g = torch.Generator().manual_seed(43001)
    geometry = torch.randn(3, 5, 5, generator=g)
    gaps = relative_gap_tensor(geometry)
    assert gaps.shape == (3, 5, 5)
    assert torch.equal(
        torch.diagonal(gaps, dim1=-2, dim2=-1),
        torch.zeros(3, 5),
    )


def test_s43_anchor_is_exact_zero_for_identical_geometry():
    g = torch.Generator().manual_seed(43002)
    c = torch.randn(2, 4, 13, generator=g)
    p = torch.randn(2, 4, 13, generator=g)
    loss = relative_gap_signature_geometry_anchor(c, p, c.clone(), p.clone())
    assert loss.item() == 0.0


def test_s43_reference_is_detached_but_treatment_is_live():
    g = torch.Generator().manual_seed(43003)
    tc = torch.randn(2, 4, 9, generator=g, requires_grad=True)
    tp = torch.randn(2, 4, 9, generator=g, requires_grad=True)
    rc = torch.randn(2, 4, 9, generator=g, requires_grad=True)
    rp = torch.randn(2, 4, 9, generator=g, requires_grad=True)
    loss = relative_gap_signature_geometry_anchor(tc, tp, rc, rp)
    grads = torch.autograd.grad(loss, (tc, tp, rc, rp), allow_unused=True)
    assert grads[0] is not None and float(grads[0].abs().sum()) > 0.0
    assert grads[1] is not None and float(grads[1].abs().sum()) > 0.0
    assert grads[2] is None
    assert grads[3] is None


def test_s43_gap_loss_detects_same_option_diagonal_change():
    ref = torch.zeros(2, 4, 4)
    changed = ref.clone()
    changed[:, 1, 1] = 0.25
    loss = relative_gap_geometry_loss(changed, ref)
    assert loss.item() > 0.0


def test_s43_gap_loss_detects_wrong_option_change():
    ref = torch.zeros(2, 4, 4)
    changed = ref.clone()
    changed[:, 0, 3] = -0.4
    assert torch.equal(
        torch.diagonal(changed, dim1=-2, dim2=-1),
        torch.zeros(2, 4),
    )
    loss = relative_gap_geometry_loss(changed, ref)
    assert loss.item() > 0.0


def test_s43_gap_anchor_distinguishes_relative_distortion_with_equal_full_mse():
    ref = torch.zeros(1, 3, 3)

    # A uniform row shift changes absolute geometry but leaves every
    # diag-minus-offdiag gap unchanged.
    absolute_shift = ref.clone()
    absolute_shift[:, 0, :] = 0.2

    # Same number/magnitude of changed cells, but one diagonal and one wrong
    # option move in opposing directions, changing relative gaps.
    gap_distortion = ref.clone()
    gap_distortion[:, 0, 0] = 0.2
    gap_distortion[:, 0, 1] = -0.2
    gap_distortion[:, 0, 2] = 0.2

    full_a = torch.mean((absolute_shift-ref)**2)
    full_b = torch.mean((gap_distortion-ref)**2)
    assert torch.allclose(full_a, full_b, rtol=0.0, atol=1e-12)

    gap_a = relative_gap_geometry_loss(absolute_shift, ref)
    gap_b = relative_gap_geometry_loss(gap_distortion, ref)
    assert gap_a.item() == 0.0
    assert gap_b.item() > 0.0


def test_s43_anchor_scalar_is_invariant_to_shared_option_permutation():
    g = torch.Generator().manual_seed(43004)
    tc = torch.randn(3, 5, 17, generator=g)
    tp = torch.randn(3, 5, 17, generator=g)
    rc = torch.randn(3, 5, 17, generator=g)
    rp = torch.randn(3, 5, 17, generator=g)
    perm = torch.tensor([3, 0, 4, 1, 2])
    base = relative_gap_signature_geometry_anchor(tc, tp, rc, rp)
    moved = relative_gap_signature_geometry_anchor(
        tc[:, perm], tp[:, perm], rc[:, perm], rp[:, perm]
    )
    assert torch.allclose(base, moved, rtol=0.0, atol=1e-7)


def test_s43_similarity_geometry_still_conjugates_under_permutation():
    g = torch.Generator().manual_seed(43005)
    c = torch.randn(2, 4, 7, generator=g)
    p = torch.randn(2, 4, 7, generator=g)
    perm = torch.tensor([2, 0, 3, 1])
    base = cross_view_similarity_matrix(c, p)
    moved = cross_view_similarity_matrix(c[:, perm], p[:, perm])
    expected = base[:, perm][:, :, perm]
    assert torch.allclose(moved, expected, rtol=0.0, atol=1e-6)
