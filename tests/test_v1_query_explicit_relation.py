import torch
from torch import nn
import torch.nn.functional as F

from nmd.v1_query_explicit_relation import QueryExplicitRelationCanonicalizer
from nmd.v1_relation_canonicalization import CrossViewRelationCanonicalizer


def _masks(batch=1,k=2,v=1,s=2,q=1,t=1):
    return {
        "state_mask":torch.ones(batch,s,dtype=torch.bool),
        "question_mask":torch.ones(batch,q,dtype=torch.bool),
        "option_view_token_mask":torch.ones(batch,k,v,t,dtype=torch.bool),
        "option_view_mask":torch.ones(batch,k,v,dtype=torch.bool),
    }


def _synthetic():
    projection=nn.Linear(4,4,bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(4))

    state=torch.tensor([[
        [1.0,0.0,0.0,0.0],
        [0.0,0.0,1.0,0.0],
    ]])
    q_a=torch.tensor([[[1.0,0.0,0.0,0.0]]])
    q_b=torch.tensor([[[0.0,1.0,0.0,0.0]]])
    options=torch.tensor([[
        [[[1.0,0.0,0.0,0.0]]],
        [[[0.0,1.0,0.0,0.0]]],
    ]])
    masks=_masks()
    return projection,state,q_a,q_b,options,masks


def _call(module,projection,state,question,options,masks):
    return module.forward_with_components(
        projection=projection,
        state_tokens=state,
        state_mask=masks["state_mask"],
        question_tokens=question,
        question_mask=masks["question_mask"],
        option_view_tokens=options,
        option_view_token_mask=masks["option_view_token_mask"],
        option_view_mask=masks["option_view_mask"],
    )


def test_s33_has_zero_parameters_and_uses_50_50_scores():
    module=QueryExplicitRelationCanonicalizer()
    assert module.parameter_count==0
    assert module.base_weight==0.5
    assert module.query_option_weight==0.5


def test_s33_preserves_exact_s13_base_block():
    projection,state,q_a,_q_b,options,masks=_synthetic()
    control=CrossViewRelationCanonicalizer()
    treatment=QueryExplicitRelationCanonicalizer()

    c_logits,c_sig,_=control(
        projection=projection,
        state_tokens=state,
        state_mask=masks["state_mask"],
        question_tokens=q_a,
        question_mask=masks["question_mask"],
        option_view_tokens=options,
        option_view_token_mask=masks["option_view_token_mask"],
        option_view_mask=masks["option_view_mask"],
    )
    _t_logits,t_sig,_diag,parts=_call(
        treatment,projection,state,q_a,options,masks
    )
    assert torch.equal(parts["base_logits"],c_logits)
    assert torch.equal(parts["base_signatures"],c_sig)
    assert t_sig.shape[-1]==2*c_sig.shape[-1]


def test_s33_query_anchor_matches_masked_projected_mean():
    projection=nn.Linear(4,4,bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(4))
    module=QueryExplicitRelationCanonicalizer()
    question=torch.tensor([[
        [1.0,0.0,0.0,0.0],
        [0.0,1.0,0.0,0.0],
        [0.0,0.0,1.0,0.0],
    ]])
    mask=torch.tensor([[True,True,False]])
    projected=F.normalize(projection(question),dim=-1)
    expected=F.normalize(projected[:,:2].mean(dim=1),dim=-1)
    got=module._masked_query_anchor(projected,mask)
    assert torch.allclose(got,expected,atol=0.0,rtol=0.0)


def test_s33_question_intervention_changes_explicit_option_geometry():
    projection,state,q_a,q_b,options,masks=_synthetic()
    module=QueryExplicitRelationCanonicalizer()
    _la,_sa,_da,a=_call(module,projection,state,q_a,options,masks)
    _lb,_sb,_db,b=_call(module,projection,state,q_b,options,masks)

    assert not torch.equal(a["query_anchor"],b["query_anchor"])
    assert not torch.equal(
        a["query_option_signature"],
        b["query_option_signature"],
    )
    assert not torch.equal(
        a["query_option_logits"],
        b["query_option_logits"],
    )

    # q_a directly matches option 0, q_b directly matches option 1.
    assert a["query_option_logits"][0,0] > a["query_option_logits"][0,1]
    assert b["query_option_logits"][0,1] > b["query_option_logits"][0,0]


def test_s33_option_permutation_is_equivariant():
    projection,state,q_a,_q_b,options,masks=_synthetic()
    module=QueryExplicitRelationCanonicalizer()
    logits,sig,_diag,parts=_call(
        module,projection,state,q_a,options,masks
    )

    perm=torch.tensor([1,0])
    p_options=options[:,perm]
    p_logits,p_sig,_pdiag,p_parts=_call(
        module,projection,state,q_a,p_options,masks
    )
    assert torch.allclose(p_logits,logits[:,perm],atol=0.0,rtol=0.0)
    assert torch.allclose(p_sig,sig[:,perm],atol=0.0,rtol=0.0)
    assert torch.allclose(
        p_parts["query_option_logits"],
        parts["query_option_logits"][:,perm],
        atol=0.0,
        rtol=0.0,
    )


def test_s33_degenerate_equal_query_option_geometry_is_finite():
    projection=nn.Linear(4,4,bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(4))
    module=QueryExplicitRelationCanonicalizer()
    state=torch.tensor([[
        [1.0,0.0,0.0,0.0],
        [0.0,1.0,0.0,0.0],
    ]])
    question=torch.tensor([[[1.0,0.0,0.0,0.0]]])
    options=torch.tensor([[
        [[[1.0,0.0,0.0,0.0]]],
        [[[1.0,0.0,0.0,0.0]]],
    ]])
    masks=_masks()
    logits,sig,_diag,parts=_call(
        module,projection,state,question,options,masks
    )
    assert torch.isfinite(logits).all()
    assert torch.isfinite(sig).all()
    assert torch.isfinite(parts["query_option_signature"]).all()
