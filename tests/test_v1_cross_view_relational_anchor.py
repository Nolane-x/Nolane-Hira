import torch

from nmd.v1_cross_view_relational_anchor import (
    cross_view_similarity_matrix,
    relational_geometry_matrix_loss,
    relational_signature_geometry_anchor,
)


def test_s42_cross_view_matrix_has_full_kxk_shape():
    g = torch.Generator().manual_seed(42001)
    c = torch.randn(3, 7, 11, generator=g)
    p = torch.randn(3, 7, 11, generator=g)
    m = cross_view_similarity_matrix(c, p)
    assert m.shape == (3, 7, 7)
    assert torch.isfinite(m).all()


def test_s42_anchor_is_exact_zero_for_identical_geometry():
    g = torch.Generator().manual_seed(42002)
    c = torch.randn(2, 4, 13, generator=g)
    p = torch.randn(2, 4, 13, generator=g)
    loss = relational_signature_geometry_anchor(c, p, c.clone(), p.clone())
    assert loss.item() == 0.0


def test_s42_reference_is_detached_but_treatment_is_live():
    g = torch.Generator().manual_seed(42003)
    tc = torch.randn(2, 4, 9, generator=g, requires_grad=True)
    tp = torch.randn(2, 4, 9, generator=g, requires_grad=True)
    rc = torch.randn(2, 4, 9, generator=g, requires_grad=True)
    rp = torch.randn(2, 4, 9, generator=g, requires_grad=True)
    loss = relational_signature_geometry_anchor(tc, tp, rc, rp)
    grads = torch.autograd.grad(loss, (tc, tp, rc, rp), allow_unused=True)
    assert grads[0] is not None and float(grads[0].abs().sum()) > 0.0
    assert grads[1] is not None and float(grads[1].abs().sum()) > 0.0
    assert grads[2] is None
    assert grads[3] is None


def test_s42_full_matrix_loss_detects_diagonal_only_change():
    ref = torch.zeros(2, 4, 4)
    changed = ref.clone()
    changed[:, 1, 1] = 0.25
    loss = relational_geometry_matrix_loss(changed, ref)
    assert loss.item() > 0.0


def test_s42_full_matrix_loss_detects_offdiagonal_only_change():
    ref = torch.zeros(2, 4, 4)
    changed = ref.clone()
    changed[:, 0, 3] = -0.4
    changed[:, 2, 1] = 0.2
    assert torch.equal(torch.diagonal(changed, dim1=-2, dim2=-1), torch.zeros(2, 4))
    loss = relational_geometry_matrix_loss(changed, ref)
    assert loss.item() > 0.0


def test_s42_anchor_scalar_is_invariant_to_shared_option_permutation():
    g = torch.Generator().manual_seed(42004)
    tc = torch.randn(3, 5, 17, generator=g)
    tp = torch.randn(3, 5, 17, generator=g)
    rc = torch.randn(3, 5, 17, generator=g)
    rp = torch.randn(3, 5, 17, generator=g)
    perm = torch.tensor([3, 0, 4, 1, 2])
    base = relational_signature_geometry_anchor(tc, tp, rc, rp)
    moved = relational_signature_geometry_anchor(
        tc[:, perm],
        tp[:, perm],
        rc[:, perm],
        rp[:, perm],
    )
    assert torch.allclose(base, moved, rtol=0.0, atol=1e-7)


def test_s42_geometry_conjugates_under_option_permutation():
    g = torch.Generator().manual_seed(42005)
    c = torch.randn(2, 4, 7, generator=g)
    p = torch.randn(2, 4, 7, generator=g)
    perm = torch.tensor([2, 0, 3, 1])
    base = cross_view_similarity_matrix(c, p)
    moved = cross_view_similarity_matrix(c[:, perm], p[:, perm])
    expected = base[:, perm][:, :, perm]
    assert torch.allclose(moved, expected, rtol=0.0, atol=1e-6)
