import torch

from nmd.v1_evidence_fusion import (
    SymmetricFullKEvidenceFusion,
    fused_gold_vs_max_wrong_margin,
    standardize_full_k_evidence,
)


def test_s14_fusion_has_zero_parameters():
    module = SymmetricFullKEvidenceFusion()
    assert module.parameter_count == 0
    assert sum(p.numel() for p in module.parameters()) == 0


def test_s14_fusion_is_exactly_symmetric_under_expert_swap():
    triadic = torch.tensor([
        [2.0, 1.0, -1.0, 0.0],
        [0.0, 4.0, 1.0, 2.0],
    ])
    relation = torch.tensor([
        [0.5, -0.5, 3.0, 1.0],
        [2.0, 0.0, 1.0, -1.0],
    ])
    module = SymmetricFullKEvidenceFusion()
    left, _ = module(triadic, relation)
    right, _ = module(relation, triadic)
    assert torch.equal(left, right)


def test_s14_fusion_is_option_permutation_equivariant():
    torch.manual_seed(20401)
    triadic = torch.randn(5, 4)
    relation = torch.randn(5, 4)
    permutation = torch.tensor([2, 0, 3, 1])

    module = SymmetricFullKEvidenceFusion()
    direct, _ = module(triadic, relation)
    permuted, _ = module(
        triadic[:, permutation],
        relation[:, permutation],
    )
    assert torch.allclose(
        permuted,
        direct[:, permutation],
        atol=1e-7,
        rtol=0.0,
    )


def test_s14_flat_expert_is_neutral_after_standardization():
    relation = torch.tensor([
        [4.0, 1.0, -2.0, 0.0],
        [-1.0, 3.0, 2.0, 0.0],
    ])
    flat = torch.full_like(relation, 7.0)
    module = SymmetricFullKEvidenceFusion()
    fused, _ = module(flat, relation)
    standardized, _ = standardize_full_k_evidence(relation)

    assert torch.allclose(fused, 0.5 * standardized, atol=1e-7, rtol=0.0)
    assert torch.equal(fused.argmax(-1), relation.argmax(-1))


def test_s14_expert_affine_scale_and_shift_do_not_change_fusion():
    torch.manual_seed(20402)
    triadic = torch.randn(3, 4)
    relation = torch.randn(3, 4)
    module = SymmetricFullKEvidenceFusion()

    direct, _ = module(triadic, relation)
    changed, _ = module(
        triadic * 11.0 + 37.0,
        relation * 0.25 - 9.0,
    )
    assert torch.allclose(direct, changed, atol=1e-6, rtol=0.0)


def test_s14_fusion_backpropagates_to_both_experts():
    torch.manual_seed(20403)
    triadic = torch.randn(4, 4, requires_grad=True)
    relation = torch.randn(4, 4, requires_grad=True)
    fused, _ = SymmetricFullKEvidenceFusion()(triadic, relation)
    gold = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(fused, gold)
    loss.backward()

    assert triadic.grad is not None
    assert relation.grad is not None
    assert float(triadic.grad.abs().sum()) > 0.0
    assert float(relation.grad.abs().sum()) > 0.0


def test_s14_margin_is_signed():
    logits = torch.tensor([
        [3.0, 1.0, 0.0, -1.0],
        [0.0, 2.0, 3.0, 1.0],
    ])
    gold = torch.tensor([0, 1], dtype=torch.long)
    margin = fused_gold_vs_max_wrong_margin(logits, gold)
    assert torch.allclose(margin, torch.tensor([2.0, -1.0]))


def test_s14_rejects_invalid_contracts():
    try:
        SymmetricFullKEvidenceFusion(epsilon=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive S14 epsilon must fail")

    module = SymmetricFullKEvidenceFusion()
    try:
        module(torch.randn(2, 4), torch.randn(2, 3))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("mismatched S14 expert shapes must fail")

    bad = torch.tensor([[0.0, float("nan"), 1.0, 2.0]])
    try:
        standardize_full_k_evidence(bad)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite S14 evidence must fail")
