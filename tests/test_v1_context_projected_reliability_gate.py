import math

import torch

from nmd.v1_learned_set_reliability_gate import LearnedSetReliabilityGate
from nmd.v1_context_projected_reliability_gate import (
    S64_CONTEXT_DIM,
    S64_CONTEXT_PROJECTION_DIM,
    S64_GATE_PARAMETER_COUNT,
    S64_PROJECTION_SEED,
    ContextProjectedReliabilityGate,
    context_projected_option_features,
    fixed_context_projection,
    projection_sha256,
)


def test_s64_projection_is_fixed_rademacher_and_deterministic():
    p1=fixed_context_projection()
    p2=fixed_context_projection()
    assert p1.shape==(4,512)
    assert torch.equal(p1,p2)
    expected=1.0/math.sqrt(512)
    assert torch.equal(p1.abs(),torch.full_like(p1,expected))
    assert not p1.requires_grad
    assert len(projection_sha256(p1))==64
    assert S64_CONTEXT_DIM==512
    assert S64_CONTEXT_PROJECTION_DIM==4
    assert S64_PROJECTION_SEED==64064


def test_s64_matched_capacity_and_learned_initialization():
    ref=LearnedSetReliabilityGate()
    trt=ContextProjectedReliabilityGate()
    assert ref.parameter_count==61
    assert trt.parameter_count==S64_GATE_PARAMETER_COUNT==61
    rs=ref.state_dict_exact()
    ts=trt.learned_state_dict_exact()
    assert set(rs)==set(ts)=={"W_phi","b_phi","w_out","b_out"}
    assert all(torch.equal(rs[k],ts[k]) for k in rs)
    assert "P_context" in trt.state_dict()
    assert trt.P_context.requires_grad is False


def test_s64_context_projection_invariances_and_permutation():
    g=torch.Generator().manual_seed(6406401)
    gate=ContextProjectedReliabilityGate()
    with torch.no_grad():
        gate.w_out.copy_(torch.linspace(-0.2,0.2,20))
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=g)
        pair=torch.randn(2,k,generator=g)
        context=torch.randn(2,k,512,generator=g)
        base=gate.alpha(fused,pair,context)
        offset=gate.alpha(fused,pair,3.0*context+7.0)
        assert float((base-offset).abs().max())<=1e-6

        perm=torch.randperm(k,generator=g)
        permuted=gate.alpha(
            fused[:,perm],pair[:,perm],context[:,perm,:]
        )
        assert float((base-permuted).abs().max())<=1e-6


def test_s64_context_features_detach_and_are_finite_when_flat():
    context=torch.zeros(2,7,512,requires_grad=True)
    features=context_projected_option_features(context)
    assert features.shape==(2,7,4)
    assert features.requires_grad is False
    assert torch.isfinite(features).all()


def test_s64_alpha_override_zero_is_exact_fused_identity():
    g=torch.Generator().manual_seed(6406402)
    gate=ContextProjectedReliabilityGate()
    fused=torch.randn(2,7,generator=g)
    pair=torch.randn(2,7,generator=g)
    context=torch.randn(2,7,512,generator=g)
    out=gate.compose(fused,pair,context,alpha_override=0.0)
    assert torch.equal(out,fused)
