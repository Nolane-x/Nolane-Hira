import torch

from nmd.v1_evidence_fusion import SymmetricFullKEvidenceFusion
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion


def test_s15_protected_fusion_has_zero_parameters():
    module = GradientIsolatedFullKEvidenceFusion()
    assert module.parameter_count == 0
    assert sum(p.numel() for p in module.parameters()) == 0


def test_s15_forward_is_numerically_identical_to_s14():
    torch.manual_seed(21401)
    triadic = torch.randn(8, 4)
    relation = torch.randn(8, 4)
    s14, _ = SymmetricFullKEvidenceFusion()(triadic, relation)
    s15, _ = GradientIsolatedFullKEvidenceFusion()(triadic, relation)
    assert torch.equal(s14, s15)


def test_s15_primary_gradient_reaches_triadic_but_not_relation_logits():
    torch.manual_seed(21402)
    triadic = torch.randn(6, 4, requires_grad=True)
    relation = torch.randn(6, 4, requires_grad=True)
    fused, _ = GradientIsolatedFullKEvidenceFusion()(triadic, relation)
    gold = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(fused, gold)
    loss.backward()

    assert triadic.grad is not None
    assert float(triadic.grad.abs().sum()) > 0.0
    assert relation.grad is None or float(relation.grad.abs().sum()) == 0.0


def test_s15_relation_logits_remain_trainable_through_relation_auxiliary():
    torch.manual_seed(21403)
    relation = torch.randn(6, 4, requires_grad=True)
    gold = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(relation, gold)
    loss.backward()
    assert relation.grad is not None
    assert float(relation.grad.abs().sum()) > 0.0


def test_s15_forward_keeps_s14_option_permutation_equivariance():
    torch.manual_seed(21404)
    triadic = torch.randn(5, 4)
    relation = torch.randn(5, 4)
    permutation = torch.tensor([2, 0, 3, 1])
    module = GradientIsolatedFullKEvidenceFusion()
    direct, _ = module(triadic, relation)
    changed, _ = module(triadic[:, permutation], relation[:, permutation])
    assert torch.allclose(changed, direct[:, permutation], atol=1e-6, rtol=0.0)


def test_s15_diagnostics_remain_numerically_equal_to_s14():
    torch.manual_seed(21405)
    triadic = torch.randn(7, 4)
    relation = torch.randn(7, 4)
    _a, d14 = SymmetricFullKEvidenceFusion()(triadic, relation)
    _b, d15 = GradientIsolatedFullKEvidenceFusion()(triadic, relation)
    assert d14.to_dict() == d15.to_dict()
