import torch
import torch.nn.functional as F

from nmd.v1_native_relation_geometry import NativeA13RelationCanonicalizer
from nmd.v1_native_signature_readout import NativeSignatureLinearCorrectnessReadout


def _inputs(*,batch=3,k=4,views=2,s=5,q=3,t=4,d=256):
    torch.manual_seed(36001 + k)
    state=torch.randn(batch,s,d)
    question=torch.randn(batch,q,d)
    options=torch.randn(batch,k,views,t,d)
    sm=torch.ones(batch,s,dtype=torch.bool)
    qm=torch.ones(batch,q,dtype=torch.bool)
    otm=torch.ones(batch,k,views,t,dtype=torch.bool)
    ovm=torch.ones(batch,k,views,dtype=torch.bool)
    return dict(
        state_tokens=state,
        state_mask=sm,
        question_tokens=question,
        question_mask=qm,
        option_view_tokens=options,
        option_view_token_mask=otm,
        option_view_mask=ovm,
    )


def _base():
    return NativeA13RelationCanonicalizer(
        role_temperature=0.10,
        pair_temperature=0.10,
        contrastive_temperature=0.10,
        native_dimension=256,
    )


def _readout(*,train=True):
    return NativeSignatureLinearCorrectnessReadout(
        role_temperature=0.10,
        pair_temperature=0.10,
        contrastive_temperature=0.10,
        native_dimension=256,
        residual_scale=1.0,
        train_readout=train,
    )


def test_s36_exact_parameter_surface_and_zero_init():
    op=_readout()
    assert op.added_parameter_count==256
    assert op.parameter_count==256
    assert tuple(op.readout_weight.shape)==(256,)
    assert torch.count_nonzero(op.readout_weight).item()==0
    assert {name for name,_ in op.named_parameters()}=={"readout_weight"}


def test_s36_zero_init_exact_identity_to_native_base():
    x=_inputs()
    base=_base()
    op=_readout()
    b_logits,b_sig,_=base(**x)
    t_logits,t_sig,_=op(**x)
    assert torch.equal(t_logits,b_logits)
    assert torch.equal(t_sig,b_sig)


def test_s36_arbitrary_k_three_and_seven():
    op=_readout()
    for k in (3,7):
        logits,sig,_=op(**_inputs(k=k))
        assert tuple(logits.shape)==(3,k)
        assert tuple(sig.shape)==(3,k,256)
        assert bool(torch.isfinite(logits).all())
        assert bool(torch.isfinite(sig).all())


def test_s36_logical_option_permutation_equivariance():
    x=_inputs(k=7)
    op=_readout()
    with torch.no_grad():
        op.readout_weight.copy_(torch.linspace(-0.1,0.1,256))
    logits,sig,_=op(**x)
    perm=torch.tensor([4,0,6,2,1,5,3])
    px=dict(x)
    px["option_view_tokens"]=x["option_view_tokens"][:,perm]
    px["option_view_token_mask"]=x["option_view_token_mask"][:,perm]
    px["option_view_mask"]=x["option_view_mask"][:,perm]
    plog,psig,_=op(**px)
    assert torch.allclose(plog,logits[:,perm],rtol=0.0,atol=2e-6)
    assert torch.allclose(psig,sig[:,perm],rtol=0.0,atol=2e-6)


def test_s36_readout_gradient_is_live_and_base_signature_unchanged():
    x=_inputs(batch=4,k=4)
    op=_readout()
    logits,sig,_=op(**x)
    baseline=sig.detach().clone()
    gold=torch.tensor([0,1,2,3])
    loss=F.cross_entropy(logits,gold)
    loss.backward()
    assert op.readout_weight.grad is not None
    assert float(op.readout_weight.grad.abs().sum())>0.0
    assert torch.equal(sig.detach(),baseline)


def test_s36_checkpoint_roundtrip():
    source=_readout()
    with torch.no_grad():
        source.readout_weight.copy_(torch.linspace(-0.25,0.25,256))
    state=source.readout_state_dict()
    target=_readout()
    target.load_readout_state_dict(state,freeze=True)
    replay=target.readout_state_dict()
    assert torch.equal(replay["readout.weight"],state["readout.weight"])
    assert target.readout_weight.requires_grad is False
