import torch

from nmd.v1_reference_anchor_projection import (
    project_runtime_gradient_against_reference_anchor,
    signature_anchor_loss,
)


def test_s40_signature_anchor_zero_at_identity_and_reference_detached():
    torch.manual_seed(40001)
    reference = torch.randn(3, 4, 256, requires_grad=True)
    treatment = reference.detach().clone().requires_grad_(True)
    loss = signature_anchor_loss(treatment, reference)
    assert abs(float(loss.detach())) < 2e-7
    loss.backward()
    assert reference.grad is None


def test_s40_signature_anchor_detects_drift_and_gradients_treatment_only():
    torch.manual_seed(40002)
    reference = torch.randn(2, 5, 256, requires_grad=True)
    treatment = reference.detach().clone()
    treatment[..., :64] *= -1.0
    treatment.requires_grad_(True)
    loss = signature_anchor_loss(treatment, reference)
    assert float(loss.detach()) > 1e-4
    loss.backward()
    assert treatment.grad is not None
    assert float(treatment.grad.abs().sum()) > 0.0
    assert reference.grad is None


def test_s40_projection_removes_only_conflicting_anchor_component():
    g = [torch.tensor([-2.0, 1.0]), torch.tensor([0.5])]
    a = [torch.tensor([1.0, 0.0]), torch.tensor([0.0])]
    out, diag = project_runtime_gradient_against_reference_anchor(g, a)
    assert diag.projected is True
    assert diag.pre_dot < 0.0
    assert abs(diag.post_dot) < 1e-6
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0]), rtol=0.0, atol=1e-6)
    assert torch.equal(out[1], g[1])


def test_s40_projection_is_exact_identity_when_nonconflicting():
    g = [torch.tensor([2.0, 1.0]), torch.tensor([0.5])]
    a = [torch.tensor([1.0, 0.0]), torch.tensor([0.0])]
    out, diag = project_runtime_gradient_against_reference_anchor(g, a)
    assert diag.projected is False
    assert diag.pre_dot >= 0.0
    assert diag.post_dot == diag.pre_dot
    assert all(torch.equal(x, y) for x, y in zip(out, g))


def test_s40_projection_handles_zero_anchor_gradient_as_identity():
    g = [torch.tensor([-2.0, 1.0])]
    a = [torch.zeros(2)]
    out, diag = project_runtime_gradient_against_reference_anchor(g, a)
    assert diag.projected is False
    assert diag.anchor_gradient_norm == 0.0
    assert diag.pre_dot == 0.0
    assert diag.post_dot == 0.0
    assert torch.equal(out[0], g[0])


def test_s40_signature_anchor_mask():
    reference = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    treatment = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    mask = torch.tensor([[True, False]])
    loss = signature_anchor_loss(treatment, reference, valid_mask=mask)
    assert float(loss) == 0.0
