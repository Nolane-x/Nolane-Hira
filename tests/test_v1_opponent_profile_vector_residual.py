import torch

from nmd.v1_multistat_pairwise_row_composer import MultiStatPairwiseRowComposer
from nmd.v1_opponent_profile_vector_residual import (
    bounded_direction_from_option_scores,
    compose_with_residual_direction,
    opponent_profile_confidence,
    opponent_profile_vector_direction,
    reference_pairwise_mean_direction,
)


def _random_inputs(batch=8,k=4,seed=72001):
    g=torch.Generator().manual_seed(seed)
    fused=torch.randn(batch,k,generator=g)
    raw=torch.randn(batch,k,k,generator=g)
    pair=raw-raw.transpose(-1,-2)
    pair=pair-torch.diag_embed(pair.diagonal(dim1=-2,dim2=-1))
    context=torch.randn(batch,k,512,generator=g)
    return fused,pair,context


def test_s72_composer_capacity_and_alpha_are_exactly_matched():
    fused,pair,context=_random_inputs()
    ref=MultiStatPairwiseRowComposer(use_multistat=False)
    trt=MultiStatPairwiseRowComposer(use_multistat=False)

    assert ref.parameter_count==80
    assert trt.parameter_count==80
    for (kr,vr),(kt,vt) in zip(
        ref.parameter_state_dict_exact().items(),
        trt.parameter_state_dict_exact().items(),
    ):
        assert kr==kt
        assert torch.equal(vr,vt)

    ar=ref.alpha(fused,pair,context)
    at=trt.alpha(fused,pair,context)
    assert torch.equal(ar,at)


def test_s72_profile_confidence_is_bounded_normalized_and_diagonal_free():
    fused,pair,_=_random_inputs(batch=16,k=7,seed=72002)
    confidence,diag=opponent_profile_confidence(fused,pair)

    assert confidence.shape==(16,7)
    assert float(confidence.abs().max())<=1.0+1e-6
    assert float((diag["opponent_mass"]-1.0).abs().max())<=1e-6
    assert float(diag["opponent_weights"].diagonal(dim1=-2,dim2=-1).abs().max())==0.0


def test_s72_neutral_uniform_prior_equal_strength_collapses_to_reference():
    # Total-order tournament: all off-diagonal magnitudes are exactly one.
    # Under uniform fused prior, confidence equals uniform row mean exactly.
    k=4
    pair=torch.zeros(1,k,k)
    for i in range(k):
        for j in range(k):
            if i<j:
                pair[0,i,j]=1.0
                pair[0,j,i]=-1.0
    fused=torch.zeros(1,k)

    ref=reference_pairwise_mean_direction(pair)
    trt,_=opponent_profile_vector_direction(fused,pair)
    assert float((ref-trt).abs().max())<=1e-6


def test_s72_nontrivial_profile_changes_direction():
    fused,pair,_=_random_inputs(batch=32,k=4,seed=72003)
    ref=reference_pairwise_mean_direction(pair)
    trt,_=opponent_profile_vector_direction(fused,pair)
    assert float((ref-trt).abs().max())>1e-4


def test_s72_direction_is_permutation_equivariant():
    fused,pair,_=_random_inputs(batch=8,k=7,seed=72004)
    perm=torch.tensor([3,0,6,2,5,1,4])

    ref=reference_pairwise_mean_direction(pair)
    trt,_=opponent_profile_vector_direction(fused,pair)

    pair_p=pair[:,perm][:,:,perm]
    fused_p=fused[:,perm]
    ref_p=reference_pairwise_mean_direction(pair_p)
    trt_p,_=opponent_profile_vector_direction(fused_p,pair_p)

    assert float((ref[:,perm]-ref_p).abs().max())<=2e-6
    assert float((trt[:,perm]-trt_p).abs().max())<=2e-6


def test_s72_compose_is_bounded_identity_safe_and_finite_for_arbitrary_k():
    for idx,k in enumerate((3,7,255)):
        fused,pair,context=_random_inputs(batch=2,k=k,seed=72010+idx)
        composer=MultiStatPairwiseRowComposer(use_multistat=False)

        identity=compose_with_residual_direction(
            composer,fused,pair,context,
            use_opponent_profile_direction=True,
            alpha_override=0.0,
        )
        assert torch.equal(identity,fused)

        for treatment in (False,True):
            out,diag=compose_with_residual_direction(
                composer,fused,pair,context,
                use_opponent_profile_direction=treatment,
                return_diagnostics=True,
            )
            assert torch.isfinite(out).all()
            assert float(
                ((out-fused).abs()-diag["residual_bound"]).clamp_min(0).max()
            )<=1e-7
            mass=torch.softmax(out,-1).sum(-1)
            assert float((mass-1.0).abs().max())<=1e-6


def test_s72_compose_blocks_upstream_gradients_but_composer_gradient_lives():
    fused,pair,context=_random_inputs(batch=16,k=4,seed=72020)
    fused.requires_grad_(True)
    pair.requires_grad_(True)
    context.requires_grad_(True)
    composer=MultiStatPairwiseRowComposer(use_multistat=False)

    out=compose_with_residual_direction(
        composer,fused,pair,context,
        use_opponent_profile_direction=True,
    )
    loss=out.square().mean()
    grads=torch.autograd.grad(
        loss,
        (composer.W_phi,composer.b_phi,composer.w_out,composer.b_out,
         fused,pair,context),
        allow_unused=True,
    )

    # Initial w_out=0, so output head is live first; phi becomes live after output warmup.
    assert grads[2] is not None and float(grads[2].abs().sum())>0
    assert grads[3] is not None and float(grads[3].abs().sum())>0
    assert grads[4:]==(None,None,None)


def test_s72_bounded_direction_from_scores_stays_within_unit_box():
    g=torch.Generator().manual_seed(72030)
    scores=1000*torch.randn(64,255,generator=g)
    d=bounded_direction_from_option_scores(scores)
    assert torch.isfinite(d).all()
    assert float(d.abs().max())<=1.0
