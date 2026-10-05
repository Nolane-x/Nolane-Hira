import torch

from nmd.v1_learned_pairwise_decision_head import (
    MatchedLearnedDecisionHeadFork,
    matched_head_initialization_exact,
)


def _inputs(k=4):
    g=torch.Generator().manual_seed(59001+k)
    identity=torch.randn(2,k,256,generator=g)
    identity=torch.nn.functional.normalize(identity,dim=-1)
    context=torch.randn(2,k,256,generator=g)
    context=torch.nn.functional.normalize(context,dim=-1)
    return identity,context


def test_s59_matched_capacity_and_zero_initial_residual():
    ref=MatchedLearnedDecisionHeadFork(mode="pointwise")
    trt=MatchedLearnedDecisionHeadFork(mode="pairwise")
    assert ref.correction_parameter_count==114688
    assert trt.correction_parameter_count==114688
    assert ref.decision_head_parameter_count==16384
    assert trt.decision_head_parameter_count==16384
    assert ref.total_private_parameter_count==131072
    assert trt.total_private_parameter_count==131072
    assert matched_head_initialization_exact(ref,trt)
    identity,context=_inputs()
    assert torch.equal(ref.pointwise_residual(identity,context),torch.zeros(2,4))
    assert torch.equal(trt.pairwise_residual(identity,context),torch.zeros(2,4))


def test_s59_pairwise_antisymmetry_and_diagonal():
    op=MatchedLearnedDecisionHeadFork(mode="pairwise",train_head=True)
    g=torch.Generator().manual_seed(59002)
    with torch.no_grad():
        op.decision_b.copy_(torch.randn(op.decision_b.shape,generator=g))
    identity,context=_inputs(7)
    p=op.pairwise_matrix(identity,context)
    assert float((p+p.transpose(-1,-2)).abs().max())<=1e-7
    assert float(torch.diagonal(p,dim1=-2,dim2=-1).abs().max())==0.0


def test_s59_pairwise_option_permutation_equivariance():
    op=MatchedLearnedDecisionHeadFork(mode="pairwise",train_head=True)
    g=torch.Generator().manual_seed(59003)
    with torch.no_grad():
        op.decision_b.copy_(torch.randn(op.decision_b.shape,generator=g))
    identity,context=_inputs(7)
    base=op.pairwise_residual(identity,context)
    perm=torch.tensor([4,0,6,1,3,5,2])
    got=op.pairwise_residual(identity[:,perm],context[:,perm])
    assert torch.allclose(got,base[:,perm],atol=1e-6,rtol=0.0)


def test_s59_arbitrary_k_and_finite_degenerate_context():
    op=MatchedLearnedDecisionHeadFork(mode="pairwise",train_head=True)
    with torch.no_grad():
        op.decision_b.fill_(0.01)
    for k in (3,7,255):
        identity,_context=_inputs(k)
        context=torch.zeros_like(identity)
        residual=op.pairwise_residual(identity,context)
        assert residual.shape==(2,k)
        assert bool(torch.isfinite(residual).all())


def test_s59_zero_b_warm_start_then_a_becomes_live():
    op=MatchedLearnedDecisionHeadFork(mode="pairwise",train_head=True)
    identity,context=_inputs(4)
    gold=torch.tensor([0,3],dtype=torch.long)

    residual=op.pairwise_residual(identity,context)
    loss=torch.nn.functional.cross_entropy(residual,gold)
    ga,gb=torch.autograd.grad(loss,(op.decision_a,op.decision_b))
    assert float(ga.abs().sum())==0.0
    assert float(gb.abs().sum())>0.0

    with torch.no_grad():
        op.decision_b.add_(-0.1*gb)
    residual2=op.pairwise_residual(identity,context)
    loss2=torch.nn.functional.cross_entropy(residual2,gold)
    ga2,gb2=torch.autograd.grad(loss2,(op.decision_a,op.decision_b))
    assert float(ga2.abs().sum())>0.0
    assert float(gb2.abs().sum())>0.0
