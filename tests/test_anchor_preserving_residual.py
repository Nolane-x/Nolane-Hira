from __future__ import annotations

import torch

from nmd.anchor_preserving_residual import (
    AnchorPreservingResidualMixer,
    INITIAL_BETA_FRACTION,
    MAX_SOURCE_FRACTION,
    count_anchor_residual_parameters,
    robust_anchor_spread,
)


def _fixture():
    anchor = torch.tensor(
        [
            [1.0, 0.7, 0.2, -0.1],
            [0.8, 0.4, 0.1, -0.4],
            [1.2, 0.9, 0.3, -0.2],
        ],
        dtype=torch.float32,
    )
    d0 = anchor + torch.tensor(
        [
            [0.03, -0.01, 0.02, -0.02],
            [-0.02, 0.04, -0.01, 0.01],
            [0.01, -0.03, 0.02, 0.00],
        ]
    )
    competitive = d0 + torch.tensor(
        [
            [4.0, -2.0, 1.0, -3.0],
            [1.5, -1.0, 3.0, -3.5],
            [-4.0, 2.0, 3.0, -1.0],
        ]
    )
    relation = torch.tensor(
        [
            [3.0, -1.0, 2.0, -4.0],
            [-2.0, 4.0, 1.0, -3.0],
            [5.0, -2.0, -1.0, -2.0],
        ],
        dtype=torch.float32,
    )
    qtype = torch.tensor([0, 1, 2], dtype=torch.long)
    mask = torch.ones_like(anchor, dtype=torch.bool)
    return anchor, d0, competitive, relation, qtype, mask


def test_w15_residual_parameter_budget_and_near_anchor_initialization():
    bounded = AnchorPreservingResidualMixer(bounded=True)
    unbounded = AnchorPreservingResidualMixer(bounded=False)

    assert count_anchor_residual_parameters(bounded) == 6
    assert count_anchor_residual_parameters(unbounded) == 6

    expected = torch.full((3,), INITIAL_BETA_FRACTION)
    assert torch.allclose(
        bounded.beta_competitive().detach(),
        expected,
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        bounded.beta_relation().detach(),
        expected,
        atol=1e-7,
        rtol=0.0,
    )
    assert bool((bounded.beta_competitive() < MAX_SOURCE_FRACTION).all())
    assert bool((bounded.beta_relation() < MAX_SOURCE_FRACTION).all())


def test_w15_zero_residual_is_exact_anchor():
    anchor, d0, _, _, qtype, mask = _fixture()
    mixer = AnchorPreservingResidualMixer(bounded=True)

    out = mixer(
        anchor_logits=anchor,
        d0_anchor_logits=d0,
        competitive_logits=d0.clone(),
        relation_delta=torch.zeros_like(anchor),
        qtype=qtype,
        option_mask=mask,
    )

    assert torch.equal(out.logits, anchor)
    assert torch.equal(out.competitive_correction, torch.zeros_like(anchor))
    assert torch.equal(out.relation_correction, torch.zeros_like(anchor))


def test_w15_residual_centering_is_shift_invariant():
    anchor, d0, competitive, relation, qtype, mask = _fixture()
    mixer = AnchorPreservingResidualMixer(bounded=True)

    a = mixer(
        anchor_logits=anchor,
        d0_anchor_logits=d0,
        competitive_logits=competitive,
        relation_delta=relation,
        qtype=qtype,
        option_mask=mask,
    )
    b = mixer(
        anchor_logits=anchor,
        d0_anchor_logits=d0,
        competitive_logits=competitive + 17.0,
        relation_delta=relation - 9.0,
        qtype=qtype,
        option_mask=mask,
    )

    assert torch.allclose(a.logits, b.logits, atol=1e-6, rtol=0.0)
    assert torch.allclose(
        a.competitive_residual,
        b.competitive_residual,
        atol=5e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        a.relation_residual,
        b.relation_residual,
        atol=5e-6,
        rtol=0.0,
    )


def test_w15_bounded_correction_respects_structural_cap():
    anchor, d0, competitive, relation, qtype, mask = _fixture()
    mixer = AnchorPreservingResidualMixer(bounded=True)

    with torch.no_grad():
        mixer.theta_competitive.fill_(20.0)
        mixer.theta_relation.fill_(20.0)

    out = mixer(
        anchor_logits=anchor,
        d0_anchor_logits=d0,
        competitive_logits=competitive * 100.0,
        relation_delta=relation * 100.0,
        qtype=qtype,
        option_mask=mask,
    )

    spread = out.anchor_spread[:, None]
    eps = 1e-6
    assert bool(
        (
            out.competitive_correction.abs()
            <= MAX_SOURCE_FRACTION * spread + eps
        ).all()
    )
    assert bool(
        (
            out.relation_correction.abs()
            <= MAX_SOURCE_FRACTION * spread + eps
        ).all()
    )
    assert bool(
        (
            (out.competitive_correction + out.relation_correction).abs()
            <= 2.0 * MAX_SOURCE_FRACTION * spread + eps
        ).all()
    )


def test_w15_unbounded_control_can_exceed_bounded_cap():
    anchor, d0, competitive, relation, qtype, mask = _fixture()
    mixer = AnchorPreservingResidualMixer(bounded=False)

    with torch.no_grad():
        mixer.theta_competitive.fill_(20.0)
        mixer.theta_relation.fill_(20.0)

    out = mixer(
        anchor_logits=anchor,
        d0_anchor_logits=d0,
        competitive_logits=competitive * 100.0,
        relation_delta=relation * 100.0,
        qtype=qtype,
        option_mask=mask,
    )
    spread = robust_anchor_spread(anchor, mask)[:, None]
    assert bool(
        (
            out.competitive_correction.abs()
            > MAX_SOURCE_FRACTION * spread
        ).any()
    )


def test_w15_candidate_permutation_equivariance():
    anchor, d0, competitive, relation, qtype, mask = _fixture()
    mixer = AnchorPreservingResidualMixer(bounded=True)
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)

    original = mixer(
        anchor_logits=anchor,
        d0_anchor_logits=d0,
        competitive_logits=competitive,
        relation_delta=relation,
        qtype=qtype,
        option_mask=mask,
    )
    permuted = mixer(
        anchor_logits=anchor[:, perm],
        d0_anchor_logits=d0[:, perm],
        competitive_logits=competitive[:, perm],
        relation_delta=relation[:, perm],
        qtype=qtype,
        option_mask=mask[:, perm],
    )

    assert torch.allclose(
        permuted.logits,
        original.logits[:, perm],
        atol=1e-6,
        rtol=0.0,
    )


def test_w15_gradients_reach_all_primitive_specific_scales():
    anchor, d0, competitive, relation, qtype, mask = _fixture()
    mixer = AnchorPreservingResidualMixer(bounded=True)

    out = mixer(
        anchor_logits=anchor,
        d0_anchor_logits=d0,
        competitive_logits=competitive,
        relation_delta=relation,
        qtype=qtype,
        option_mask=mask,
    )
    weights = torch.tensor(
        [
            [1.0, -0.5, 0.2, -0.1],
            [-0.3, 0.7, -0.2, 0.4],
            [0.2, -0.6, 0.9, -0.5],
        ]
    )
    loss = (out.logits * weights).sum()
    loss.backward()

    assert mixer.theta_competitive.grad is not None
    assert mixer.theta_relation.grad is not None
    assert bool((mixer.theta_competitive.grad.abs() > 0).all())
    assert bool((mixer.theta_relation.grad.abs() > 0).all())


def test_w15_masked_candidates_do_not_affect_valid_center_or_spread():
    anchor, d0, competitive, relation, qtype, mask = _fixture()
    mask = mask.clone()
    mask[:, -1] = False

    mixer = AnchorPreservingResidualMixer(bounded=True)
    out_a = mixer(
        anchor_logits=anchor,
        d0_anchor_logits=d0,
        competitive_logits=competitive,
        relation_delta=relation,
        qtype=qtype,
        option_mask=mask,
    )

    changed_anchor = anchor.clone()
    changed_d0 = d0.clone()
    changed_competitive = competitive.clone()
    changed_relation = relation.clone()
    changed_anchor[:, -1] = 999.0
    changed_d0[:, -1] = -999.0
    changed_competitive[:, -1] = 5000.0
    changed_relation[:, -1] = -5000.0

    out_b = mixer(
        anchor_logits=changed_anchor,
        d0_anchor_logits=changed_d0,
        competitive_logits=changed_competitive,
        relation_delta=changed_relation,
        qtype=qtype,
        option_mask=mask,
    )

    assert torch.allclose(
        out_a.logits[:, :-1],
        out_b.logits[:, :-1],
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.equal(
        out_a.logits[:, -1],
        torch.full_like(out_a.logits[:, -1], -1e4),
    )
    assert torch.equal(
        out_b.logits[:, -1],
        torch.full_like(out_b.logits[:, -1], -1e4),
    )
