import torch

from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork
from nmd.v1_shared_native_private_readouts import (
    correction_initialization_exact,
    freeze_shared_native_evidence,
    fused_private_logits,
    reference_private_logits,
    treatment_private_logits,
)


def _inputs(*,batch=3,k=7):
    g=torch.Generator().manual_seed(50000+k)
    case_ids=tuple(f"s50-{i}" for i in range(batch))
    gold=torch.arange(batch,dtype=torch.long)%k
    triadic_logits=torch.randn(batch,k,generator=g,requires_grad=True)
    native_logits=torch.randn(batch,k,generator=g,requires_grad=True)
    native_signatures=torch.randn(batch,k,256,generator=g,requires_grad=True)
    state_tokens=torch.randn(batch,5,256,generator=g,requires_grad=True)
    state_mask=torch.ones(batch,5,dtype=torch.bool)
    option_view_tokens=torch.randn(batch,k,2,4,256,generator=g,requires_grad=True)
    option_view_token_mask=torch.ones(batch,k,2,4,dtype=torch.bool)
    option_view_mask=torch.ones(batch,k,2,dtype=torch.bool)
    question_tokens=torch.randn(batch,3,256,generator=g,requires_grad=True)
    question_mask=torch.ones(batch,3,dtype=torch.bool)
    return dict(
        case_ids=case_ids,
        gold=gold,
        triadic_logits=triadic_logits,
        native_logits=native_logits,
        native_signatures=native_signatures,
        state_tokens=state_tokens,
        state_mask=state_mask,
        option_view_tokens=option_view_tokens,
        option_view_token_mask=option_view_token_mask,
        option_view_mask=option_view_mask,
        question_tokens=question_tokens,
        question_mask=question_mask,
    )


def test_s50_cache_is_detached_contiguous_and_digest_stable():
    src=_inputs(batch=3,k=7)
    cache=freeze_shared_native_evidence(**src)
    digest=cache.digest()
    assert len(digest)==64
    assert all(not x.requires_grad for x in cache.tensors())
    assert all(x.is_contiguous() for x in cache.tensors())

    with torch.no_grad():
        src["triadic_logits"].sub_(900)
        src["native_logits"].add_(1000)
        src["native_signatures"].mul_(0)
        src["state_tokens"].add_(500)
        src["option_view_tokens"].sub_(700)
        src["question_tokens"].mul_(-3)
    assert cache.digest()==digest


def test_s50_cache_deterministic_roundtrip_from_equal_bytes():
    a=_inputs(batch=2,k=4)
    b={k:(v.detach().clone() if isinstance(v,torch.Tensor) else tuple(v)) for k,v in a.items()}
    ca=freeze_shared_native_evidence(**a)
    cb=freeze_shared_native_evidence(**b)
    assert ca.digest()==cb.digest()
    assert ca.case_ids==cb.case_ids
    for x,y in zip(ca.tensors(),cb.tensors()):
        assert torch.equal(x,y)


def test_s50_reference_treatment_initialization_is_bit_identical():
    ref=PrivateCorrectionRepresentationFork(train_correction=True)
    trt=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    assert correction_initialization_exact(ref,trt)
    assert ref.correction_parameter_count==114688
    assert trt.correction_parameter_count==114688
    assert trt.identity_parameter_count==0


def test_s50_private_losses_have_no_gradient_path_into_cache():
    cache=freeze_shared_native_evidence(**_inputs(batch=2,k=4))
    ref=PrivateCorrectionRepresentationFork(train_correction=True)
    trt=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)

    with torch.no_grad():
        b=torch.randn(ref.adapter_b.shape,generator=torch.Generator().manual_seed(50101))*0.01
        w=torch.randn(ref.bilinear_weight.shape,generator=torch.Generator().manual_seed(50102))*0.01
        ref.adapter_b.copy_(b); trt.adapter_b.copy_(b)
        ref.bilinear_weight.copy_(w); trt.bilinear_weight.copy_(w)

    rl=reference_private_logits(ref,cache).sum()
    tl,_=treatment_private_logits(trt,cache)
    loss=rl+tl.sum()
    grads=torch.autograd.grad(loss,ref.correction_parameters()+trt.correction_parameters(),allow_unused=False)
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    assert all(x.grad is None for x in cache.tensors())


def test_s50_branch_order_replay_is_exact_and_cache_immutable():
    cache=freeze_shared_native_evidence(**_inputs(batch=2,k=7))
    digest=cache.digest()
    ref=PrivateCorrectionRepresentationFork(train_correction=True)
    trt=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)

    with torch.no_grad():
        b=torch.randn(ref.adapter_b.shape,generator=torch.Generator().manual_seed(50201))*0.01
        w=torch.randn(ref.bilinear_weight.shape,generator=torch.Generator().manual_seed(50202))*0.01
        ref.adapter_b.copy_(b); trt.adapter_b.copy_(b)
        ref.bilinear_weight.copy_(w); trt.bilinear_weight.copy_(w)

    r1=reference_private_logits(ref,cache)
    t1,i1=treatment_private_logits(trt,cache)

    t2,i2=treatment_private_logits(trt,cache)
    r2=reference_private_logits(ref,cache)

    assert torch.equal(r1,r2)
    assert torch.equal(t1,t2)
    assert torch.equal(i1,i2)
    assert cache.digest()==digest


def test_s50_arbitrary_k_three_seven_255_and_probability_mass():
    ref=PrivateCorrectionRepresentationFork(train_correction=True)
    trt=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    with torch.no_grad():
        b=torch.randn(ref.adapter_b.shape,generator=torch.Generator().manual_seed(50301))*0.01
        w=torch.randn(ref.bilinear_weight.shape,generator=torch.Generator().manual_seed(50302))*0.01
        ref.adapter_b.copy_(b); trt.adapter_b.copy_(b)
        ref.bilinear_weight.copy_(w); trt.bilinear_weight.copy_(w)

    for k in (3,7,255):
        cache=freeze_shared_native_evidence(**_inputs(batch=2,k=k))
        r=reference_private_logits(ref,cache)
        t,_=treatment_private_logits(trt,cache)
        assert tuple(r.shape)==(2,k)
        assert tuple(t.shape)==(2,k)
        for logits in (r,t):
            assert bool(torch.isfinite(logits).all())
            probs=torch.softmax(logits,dim=-1)
            assert float((probs.sum(-1)-1.0).abs().max())<=1e-6


def test_s50_fused_replay_uses_cached_triadic_logits_only():
    cache=freeze_shared_native_evidence(**_inputs(batch=2,k=7))
    ref=PrivateCorrectionRepresentationFork(train_correction=True)
    with torch.no_grad():
        ref.adapter_b.normal_(generator=torch.Generator().manual_seed(50401),std=0.01)
        ref.bilinear_weight.normal_(generator=torch.Generator().manual_seed(50402),std=0.01)
    relation=reference_private_logits(ref,cache)
    a=fused_private_logits(cache,relation)
    b=fused_private_logits(cache,relation)
    assert torch.equal(a,b)
    assert tuple(a.shape)==(2,7)
    probs=torch.softmax(a,dim=-1)
    assert float((probs.sum(-1)-1.0).abs().max())<=1e-6


def test_s50_cache_created_inside_inference_mode_is_normal_autograd_compatible_tensor():
    # Reproduce the production cache-materialization path exactly: native
    # evidence is generated while inference mode is active.
    with torch.inference_mode():
        src=_inputs(batch=2,k=4)
        # _inputs creates tensors while inference mode is active, matching the
        # runtime evidence path rather than the ordinary unit-test path.
        cache=freeze_shared_native_evidence(**src)

    assert all(not torch.is_inference(x) for x in cache.tensors())
    assert all(not x.requires_grad for x in cache.tensors())

    ref=PrivateCorrectionRepresentationFork(train_correction=True)
    with torch.no_grad():
        ref.adapter_b.normal_(generator=torch.Generator().manual_seed(50501),std=0.01)
        ref.bilinear_weight.normal_(generator=torch.Generator().manual_seed(50502),std=0.01)

    relation=reference_private_logits(ref,cache)
    loss=torch.nn.functional.cross_entropy(relation,cache.gold)
    grads=torch.autograd.grad(loss,ref.correction_parameters(),allow_unused=False)
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
