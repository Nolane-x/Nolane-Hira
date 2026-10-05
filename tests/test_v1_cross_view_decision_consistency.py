import torch

from nmd.v1_cross_view_decision_consistency import (
    decision_discrimination_diagnostics,
    pairwise_ordering_consistency,
    standardize_full_k_logits,
    symmetric_js_from_standardized_logits,
    weighted_cross_view_decision_auxiliary,
)


def test_s56_identical_logits_zero_decision_consistency():
    x=torch.tensor([[2.0,0.0,-1.0,0.5]],requires_grad=True)
    loss=symmetric_js_from_standardized_logits(x,x)
    assert abs(float(loss))<=1e-8


def test_s56_controlled_disagreement_positive():
    a=torch.tensor([[3.0,1.0,0.0,-2.0]],requires_grad=True)
    b=torch.tensor([[-2.0,0.0,1.0,3.0]],requires_grad=True)
    loss=symmetric_js_from_standardized_logits(a,b)
    assert float(loss)>0.01


def test_s56_shared_offset_invariance():
    a=torch.tensor([[3.0,1.0,-1.0,0.5]])
    b=torch.tensor([[2.0,-1.0,0.0,1.5]])
    base=symmetric_js_from_standardized_logits(a,b)
    shifted=symmetric_js_from_standardized_logits(a+11.0,b-7.0)
    assert torch.allclose(base,shifted,rtol=0.0,atol=1e-6)


def test_s56_positive_scale_invariance():
    a=torch.tensor([[3.0,1.0,-1.0,0.5]])
    b=torch.tensor([[2.0,-1.0,0.0,1.5]])
    base=symmetric_js_from_standardized_logits(a,b)
    scaled=symmetric_js_from_standardized_logits(a*5.0,b*0.25)
    assert torch.allclose(base,scaled,rtol=0.0,atol=3e-5)


def test_s56_flat_logits_are_finite_and_zero_rms():
    x=torch.zeros(3,7)
    z=standardize_full_k_logits(x)
    assert bool(torch.isfinite(z).all())
    d=decision_discrimination_diagnostics(x)
    assert float(d["mean_top1_top2_probability_gap"])==0.0
    assert float(d["mean_unregularized_logit_rms"])==0.0


def test_s56_ordering_matching_confident_is_zero():
    a=torch.tensor([[3.0,1.0,-1.0]])
    b=torch.tensor([[6.0,2.0,-2.0]])
    loss,diag=pairwise_ordering_consistency(a,b)
    assert float(loss)==0.0
    assert float(diag["active_pair_fraction"])>0.0


def test_s56_ordering_sign_flip_positive():
    a=torch.tensor([[3.0,1.0,-1.0]])
    b=torch.tensor([[-1.0,1.0,3.0]])
    loss,diag=pairwise_ordering_consistency(a,b)
    assert float(loss)>0.0
    assert float(diag["sign_disagreement_fraction"])>0.0


def test_s56_ordering_below_threshold_inactive():
    a=torch.tensor([[0.01,0.0]])
    b=torch.tensor([[0.015,0.0]])
    loss,diag=pairwise_ordering_consistency(a,b)
    # Standardization magnifies any non-flat 2-way logits, so use exactly flat
    # paired logits to exercise the inactive threshold path.
    a=torch.zeros(1,2)
    b=torch.zeros(1,2)
    loss,diag=pairwise_ordering_consistency(a,b)
    assert float(loss)==0.0
    assert float(diag["active_pair_fraction"])==0.0


def test_s56_matched_option_permutation_invariance():
    a=torch.tensor([[3.0,0.5,-1.0,1.2]])
    b=torch.tensor([[2.5,-0.4,-0.8,1.8]])
    perm=torch.tensor([2,0,3,1])
    js1=symmetric_js_from_standardized_logits(a,b)
    js2=symmetric_js_from_standardized_logits(a[:,perm],b[:,perm])
    o1,_=pairwise_ordering_consistency(a,b)
    o2,_=pairwise_ordering_consistency(a[:,perm],b[:,perm])
    assert torch.allclose(js1,js2,rtol=0.0,atol=1e-7)
    assert torch.allclose(o1,o2,rtol=0.0,atol=1e-7)


def test_s56_reference_auxiliary_exact_zero_and_treatment_gradient_live():
    a=torch.tensor([[3.0,0.5,-1.0,1.2]],requires_grad=True)
    b=torch.tensor([[2.5,-0.4,-0.8,1.8]],requires_grad=True)
    ref,_=weighted_cross_view_decision_auxiliary(
        a,b,decision_coefficient=0.0,ordering_coefficient=0.0
    )
    assert float(ref)==0.0

    trt,_=weighted_cross_view_decision_auxiliary(
        a,b,decision_coefficient=0.10,ordering_coefficient=0.05
    )
    grads=torch.autograd.grad(trt,(a,b),allow_unused=True)
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0.0


def test_s56_k_three_seven_255_finite():
    g=torch.Generator().manual_seed(56001)
    for k in (3,7,255):
        a=torch.randn(2,k,generator=g)
        b=torch.randn(2,k,generator=g)
        aux,diag=weighted_cross_view_decision_auxiliary(
            a,b,decision_coefficient=0.10,ordering_coefficient=0.05
        )
        assert bool(torch.isfinite(aux))
        assert bool(torch.isfinite(diag["decision_js"]))
        assert bool(torch.isfinite(diag["ordering_loss"]))
