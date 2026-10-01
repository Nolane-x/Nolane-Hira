import torch
import torch.nn.functional as F

from nmd.v1_anchor_factor_transport import (
    StateFactorAnchors,
    anchor_level_factor_transport_loss,
    anchor_transport_added_parameter_count,
    extract_state_factor_anchors,
)


def _identity_projection(d: int = 4) -> torch.nn.Linear:
    p = torch.nn.Linear(d, d, bias=False)
    with torch.no_grad():
        p.weight.copy_(torch.eye(d))
    return p


def _anchors(role: torch.Tensor, value: torch.Tensor) -> StateFactorAnchors:
    b = role.shape[0]
    return StateFactorAnchors(
        role=F.normalize(role, dim=-1),
        value=F.normalize(value, dim=-1),
        role_weights=torch.ones(b, 1),
        value_weights=torch.ones(b, 1),
    )


def _orthogonal(n: int = 4, d: int = 8):
    role = torch.eye(d)[:n]
    value = torch.eye(d)[n:n+n]
    return _anchors(role, value)


def test_s28_anchor_transport_adds_zero_params():
    assert anchor_transport_added_parameter_count() == 0


def test_s28_extractor_matches_s26_state_formula_reference():
    torch.manual_seed(28001)
    projection = _identity_projection(4)
    state_tokens = torch.randn(3, 5, 4)
    question_tokens = torch.randn(3, 2, 4)
    state_mask = torch.tensor(
        [[1,1,1,1,1],[1,1,1,0,0],[1,1,1,1,0]],
        dtype=torch.bool,
    )
    question_mask = torch.tensor(
        [[1,1],[1,0],[1,1]],
        dtype=torch.bool,
    )

    got = extract_state_factor_anchors(
        projection=projection,
        state_tokens=state_tokens,
        state_mask=state_mask,
        question_tokens=question_tokens,
        question_mask=question_mask,
    )

    state = F.normalize(projection(state_tokens), dim=-1)
    question = F.normalize(projection(question_tokens), dim=-1)
    score = torch.einsum("bqd,bsd->bqs", question, state)
    score = score.masked_fill(~question_mask[...,None], -1e4)
    score = score.amax(1).masked_fill(~state_mask, -1e4)
    w = torch.softmax(score/0.10, -1)
    w = w * state_mask.to(w.dtype)
    w = w / w.sum(-1,keepdim=True).clamp_min(1e-12)
    role = F.normalize(torch.einsum("bs,bsd->bd", w, state), dim=-1)
    coeff = torch.einsum("bsd,bd->bs", state, role)
    residual = F.normalize(state-coeff[...,None]*role[:,None,:], dim=-1)
    vw = (1.0-w)*state_mask.to(w.dtype)
    vw = vw/vw.sum(-1,keepdim=True).clamp_min(1e-12)
    value = F.normalize(torch.einsum("bs,bsd->bd", vw, residual), dim=-1)

    assert torch.allclose(got.role, role, atol=1e-7, rtol=0)
    assert torch.allclose(got.value, value, atol=1e-7, rtol=0)
    assert torch.allclose(got.role_weights, w, atol=1e-7, rtol=0)
    assert torch.allclose(got.value_weights, vw, atol=1e-7, rtol=0)


def test_s28_identical_orthogonal_anchors_have_zero_loss():
    x = _orthogonal()
    loss = anchor_level_factor_transport_loss(x, x)
    assert float(loss.total) < 1e-7
    assert float(loss.role_total) < 1e-7
    assert float(loss.value_total) < 1e-7


def test_s28_role_only_mismatch_localizes():
    c = _orthogonal()
    p = _orthogonal()
    p_role = p.role.clone()
    p_role[0] = p.role[1]
    p = _anchors(p_role, p.value)
    loss = anchor_level_factor_transport_loss(c, p)
    assert float(loss.role_total) > 0.1
    assert float(loss.value_total) < 1e-7


def test_s28_value_only_mismatch_localizes():
    c = _orthogonal()
    p = _orthogonal()
    p_value = p.value.clone()
    p_value[0] = p.value[1]
    p = _anchors(p.role, p_value)
    loss = anchor_level_factor_transport_loss(c, p)
    assert float(loss.value_total) > 0.1
    assert float(loss.role_total) < 1e-7


def test_s28_constant_anchor_collapse_is_penalized():
    role = torch.ones(4, 8)
    value = torch.ones(4, 8)
    collapsed = _anchors(role, value)
    loss = anchor_level_factor_transport_loss(collapsed, collapsed)
    assert float(loss.role_separation) >= 0.19
    assert float(loss.value_separation) >= 0.19
    assert float(loss.total) >= 0.19


def test_s28_batch_permutation_preserves_loss():
    c = _orthogonal()
    p = _orthogonal()
    p_role = p.role.clone()
    p_role[0] = p.role[1]
    p = _anchors(p_role, p.value)
    base = anchor_level_factor_transport_loss(c, p)
    perm = torch.tensor([2,0,3,1])
    moved = anchor_level_factor_transport_loss(
        _anchors(c.role[perm], c.value[perm]),
        _anchors(p.role[perm], p.value[perm]),
    )
    assert torch.allclose(base.total, moved.total, atol=1e-7, rtol=0)
    assert torch.allclose(base.role_total, moved.role_total, atol=1e-7, rtol=0)
    assert torch.allclose(base.value_total, moved.value_total, atol=1e-7, rtol=0)


def test_s28_degenerate_extractor_stays_finite():
    projection = _identity_projection(4)
    base = torch.tensor([1.0,0.0,0.0,0.0])
    state = base.reshape(1,1,4).repeat(2,3,1)
    question = base.reshape(1,1,4).repeat(2,2,1)
    got = extract_state_factor_anchors(
        projection=projection,
        state_tokens=state,
        state_mask=torch.ones(2,3,dtype=torch.bool),
        question_tokens=question,
        question_mask=torch.ones(2,2,dtype=torch.bool),
    )
    assert torch.isfinite(got.role).all()
    assert torch.isfinite(got.value).all()
