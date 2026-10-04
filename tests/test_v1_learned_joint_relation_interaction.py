import torch

from nmd.v1_learned_joint_relation_interaction import (
    LearnedJointRelationPrivateCorrectionFork,
    LearnedJointRelationTransform,
    learned_joint_initialization_exact,
)


def _inputs(batch=2,k=7,seed=55001):
    g=torch.Generator().manual_seed(seed+k)
    state=torch.randn(batch,6,256,generator=g,requires_grad=True)
    state_mask=torch.ones(batch,6,dtype=torch.bool)
    option=torch.randn(batch,k,2,4,256,generator=g,requires_grad=True)
    token_mask=torch.ones(batch,k,2,4,dtype=torch.bool)
    view_mask=torch.ones(batch,k,2,dtype=torch.bool)
    question=torch.randn(batch,5,256,generator=g,requires_grad=True)
    question_mask=torch.ones(batch,5,dtype=torch.bool)
    native=torch.randn(batch,k,generator=g,requires_grad=True)
    return dict(
        native_logits=native,
        state_tokens=state,
        state_mask=state_mask,
        option_view_tokens=option,
        option_view_token_mask=token_mask,
        option_view_mask=view_mask,
        question_tokens=question,
        question_mask=question_mask,
    )


def _arms():
    reference=LearnedJointRelationPrivateCorrectionFork(
        use_state_joint_context=False,
        train_correction=True,
    )
    treatment=LearnedJointRelationPrivateCorrectionFork(
        use_state_joint_context=True,
        train_correction=True,
    )
    return reference,treatment


def test_s55_parameter_surface_and_initialization_match():
    ref,trt=_arms()
    assert ref.correction_parameter_count==114688
    assert trt.correction_parameter_count==114688
    assert ref.learned_joint_parameter_count==65536
    assert trt.learned_joint_parameter_count==65536
    assert ref.private_trainable_parameter_count==180224
    assert trt.private_trainable_parameter_count==180224
    assert ref.identity_parameter_count==0
    assert trt.identity_parameter_count==0
    assert learned_joint_initialization_exact(ref,trt)


def test_s55_zero_init_warm_start_preserves_query_context():
    op=LearnedJointRelationTransform()
    g=torch.Generator().manual_seed(55100)
    q=torch.randn(2,4,256,generator=g)
    s=torch.randn(2,4,256,generator=g)
    o=torch.randn(2,4,256,generator=g)
    code=op(query_context=q,state_joint_context=s,option_identity=o)
    expected=torch.nn.functional.normalize(q,dim=-1)
    assert torch.allclose(code,expected,rtol=0.0,atol=1e-7)


def test_s55_k_three_seven_255_and_probability_mass():
    ref,trt=_arms()
    with torch.no_grad():
        for op in (ref,trt):
            op.adapter_b.normal_(generator=torch.Generator().manual_seed(55110),std=0.01)
            op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(55111),std=0.01)
            op.learned_joint_transform.adapter_b.normal_(
                generator=torch.Generator().manual_seed(55112),std=0.01
            )

    for k in (3,7,255):
        data=_inputs(k=k,seed=55120)
        for op in (ref,trt):
            logits,_=op.correction_logits_from_state_option(**data)
            assert tuple(logits.shape)==(2,k)
            assert bool(torch.isfinite(logits).all())
            p=torch.softmax(logits,dim=-1)
            assert float((p.sum(-1)-1.0).abs().max())<=1e-6


def test_s55_explicit_state_channel_is_controlled_variable():
    ref,trt=_arms()
    g=torch.Generator().manual_seed(55200)
    q=torch.nn.functional.normalize(torch.randn(2,4,256,generator=g),dim=-1)
    o=torch.nn.functional.normalize(torch.randn(2,4,256,generator=g),dim=-1)
    s1=torch.nn.functional.normalize(torch.randn(2,4,256,generator=g),dim=-1)
    s2=torch.nn.functional.normalize(torch.randn(2,4,256,generator=g),dim=-1)

    with torch.no_grad():
        for op in (ref,trt):
            op.learned_joint_transform.adapter_b.normal_(
                generator=torch.Generator().manual_seed(55201),std=0.02
            )

    zero=torch.zeros_like(s1)
    r1=ref.learned_joint_transform(query_context=q,state_joint_context=zero,option_identity=o)
    r2=ref.learned_joint_transform(query_context=q,state_joint_context=zero,option_identity=o)
    t1=trt.learned_joint_transform(query_context=q,state_joint_context=s1,option_identity=o)
    t2=trt.learned_joint_transform(query_context=q,state_joint_context=s2,option_identity=o)

    assert torch.equal(r1,r2)
    assert float((t1-t2).abs().max())>1e-6


def test_s55_learned_transform_and_correction_gradients_are_live_but_cache_isolated():
    ref,trt=_arms()
    data=_inputs(k=4,seed=55300)
    with torch.no_grad():
        for op in (ref,trt):
            op.adapter_b.normal_(generator=torch.Generator().manual_seed(55301),std=0.01)
            op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(55302),std=0.01)

    logits,_=trt.correction_logits_from_state_option(**data)
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1]))
    params=trt.private_parameters()
    grads=torch.autograd.grad(loss,params,allow_unused=True)
    correction_grads=grads[:len(trt.correction_parameters())]
    learned_grads=grads[len(trt.correction_parameters()):]

    assert any(g is not None and float(g.abs().sum())>0 for g in correction_grads)
    assert any(g is not None and float(g.abs().sum())>0 for g in learned_grads)

    # Cache/native inputs are detached by S51/S53/S54 operators and must not
    # receive a gradient path from private training.
    for key in (
        "native_logits",
        "state_tokens",
        "option_view_tokens",
        "question_tokens",
    ):
        assert data[key].grad is None


def test_s55_state_token_perturbation_changes_treatment_relation_code():
    _ref,trt=_arms()
    data=_inputs(k=4,seed=55400)
    with torch.no_grad():
        trt.learned_joint_transform.adapter_b.normal_(
            generator=torch.Generator().manual_seed(55401),std=0.02
        )
    out1=trt.correction_logits_from_state_option(**data,return_context=True)
    data2=dict(data)
    data2["state_tokens"]=data["state_tokens"].detach().clone()
    data2["state_tokens"][:,0,:]+=3.0
    out2=trt.correction_logits_from_state_option(**data2,return_context=True)
    code1=out1[2]
    code2=out2[2]
    assert float((code1-code2).abs().max())>1e-6


def test_s55_query_and_option_perturbations_are_live_in_both_arms():
    ref,trt=_arms()
    with torch.no_grad():
        for op in (ref,trt):
            op.learned_joint_transform.adapter_b.normal_(
                generator=torch.Generator().manual_seed(55501),std=0.02
            )
    data=_inputs(k=4,seed=55500)
    for op in (ref,trt):
        base=op.correction_logits_from_state_option(**data,return_context=True)[2]

        qdata=dict(data)
        qdata["question_tokens"]=data["question_tokens"].detach().clone()
        qdata["question_tokens"][:,0,:]+=2.0
        q=op.correction_logits_from_state_option(**qdata,return_context=True)[2]
        assert float((base-q).abs().max())>1e-6

        odata=dict(data)
        odata["option_view_tokens"]=data["option_view_tokens"].detach().clone()
        odata["option_view_tokens"][:,0,0,0,:]+=2.0
        o=op.correction_logits_from_state_option(**odata,return_context=True)[2]
        assert float((base-o).abs().max())>1e-6


def test_s55_deterministic_replay():
    ref,trt=_arms()
    data=_inputs(k=7,seed=55600)
    for op in (ref,trt):
        a=op.correction_logits_from_state_option(**data,return_context=True)
        b=op.correction_logits_from_state_option(**data,return_context=True)
        assert torch.equal(a[0],b[0])
        assert torch.equal(a[2],b[2])
