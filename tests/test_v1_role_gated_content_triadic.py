from __future__ import annotations

import torch
import torch.nn.functional as F

from nmd.v1_role_gated_content_triadic import RoleGatedContentTriadicScorer


def _scorer() -> RoleGatedContentTriadicScorer:
    scorer = RoleGatedContentTriadicScorer(d_model=4, d_rel=4)
    scorer.load_projection_weight(torch.eye(4), freeze=True)
    return scorer


def _court():
    e = torch.eye(4)
    state = torch.stack(
        [
            e[0],
            F.normalize(torch.tensor([0.5, 0.0, 1.0, 0.0]), dim=0),
            e[1],
            F.normalize(torch.tensor([0.0, 0.5, 0.0, 1.0]), dim=0),
        ]
    )[None]
    options = torch.stack(
        [
            torch.stack([e[0], F.normalize(torch.tensor([0.5, 0.0, 1.0, 0.0]), dim=0)]),
            torch.stack([e[0], F.normalize(torch.tensor([0.5, 0.0, 0.0, 1.0]), dim=0)]),
            torch.stack([e[1], F.normalize(torch.tensor([0.0, 0.5, 0.0, 1.0]), dim=0)]),
            torch.stack([e[1], F.normalize(torch.tensor([0.0, 0.5, 1.0, 0.0]), dim=0)]),
        ]
    )[None, :, None]
    state_mask = torch.ones(1, 4, dtype=torch.bool)
    option_token_mask = torch.ones(1, 4, 1, 2, dtype=torch.bool)
    option_view_mask = torch.ones(1, 4, 1, dtype=torch.bool)
    return state, state_mask, options, option_token_mask, option_view_mask


def _run(question: torch.Tensor) -> torch.Tensor:
    scorer = _scorer()
    state, sm, options, otm, ovm = _court()
    return scorer(
        state_tokens=state,
        state_mask=sm,
        question_tokens=question[None, None],
        question_mask=torch.ones(1, 1, dtype=torch.bool),
        option_view_tokens=options,
        option_view_token_mask=otm,
        option_view_mask=ovm,
    )


def test_s21_adds_no_factorization_parameters() -> None:
    scorer = RoleGatedContentTriadicScorer(d_model=256, d_rel=128)
    assert scorer.factorization_added_parameter_count == 0
    assert scorer.projection_parameter_count == 32768


def test_s21_role_content_synthetic_court() -> None:
    e = torch.eye(4)
    logits_a = _run(e[0])[0]
    # correct A = 0; same-role wrong-content = 1; wrong-role same-content = 3
    assert logits_a[0] > logits_a[1]
    assert logits_a[0] > logits_a[3]

    logits_b = _run(e[1])[0]
    # changing only the role query must switch to the B-bound content.
    assert int(logits_a.argmax()) == 0
    assert int(logits_b.argmax()) == 2


def test_s21_option_permutation_is_equivariant() -> None:
    scorer = _scorer()
    state, sm, options, otm, ovm = _court()
    q = torch.eye(4)[0][None, None]
    qm = torch.ones(1, 1, dtype=torch.bool)
    base = scorer(
        state_tokens=state,
        state_mask=sm,
        question_tokens=q,
        question_mask=qm,
        option_view_tokens=options,
        option_view_token_mask=otm,
        option_view_mask=ovm,
    )
    p = torch.tensor([2, 0, 3, 1])
    perm = scorer(
        state_tokens=state,
        state_mask=sm,
        question_tokens=q,
        question_mask=qm,
        option_view_tokens=options[:, p],
        option_view_token_mask=otm[:, p],
        option_view_mask=ovm[:, p],
    )
    assert torch.allclose(perm, base[:, p], atol=1e-6, rtol=1e-6)


def test_s21_view_permutation_is_invariant() -> None:
    scorer = _scorer()
    state, sm, one, otm1, ovm1 = _court()
    options = torch.cat([one, one.flip(-2)], dim=2)
    otm = torch.cat([otm1, otm1], dim=2)
    ovm = torch.cat([ovm1, ovm1], dim=2)
    q = torch.eye(4)[0][None, None]
    qm = torch.ones(1, 1, dtype=torch.bool)
    base = scorer(
        state_tokens=state,
        state_mask=sm,
        question_tokens=q,
        question_mask=qm,
        option_view_tokens=options,
        option_view_token_mask=otm,
        option_view_mask=ovm,
    )
    swapped = scorer(
        state_tokens=state,
        state_mask=sm,
        question_tokens=q,
        question_mask=qm,
        option_view_tokens=options[:, :, [1, 0]],
        option_view_token_mask=otm[:, :, [1, 0]],
        option_view_mask=ovm[:, :, [1, 0]],
    )
    assert torch.allclose(base, swapped, atol=1e-6, rtol=1e-6)


def test_s21_inactive_view_does_not_change_score() -> None:
    scorer = _scorer()
    state, sm, one, otm1, ovm1 = _court()
    junk = torch.randn_like(one)
    options = torch.cat([one, junk], dim=2)
    otm = torch.cat([otm1, torch.zeros_like(otm1)], dim=2)
    ovm = torch.cat([ovm1, torch.zeros_like(ovm1)], dim=2)
    q = torch.eye(4)[0][None, None]
    qm = torch.ones(1, 1, dtype=torch.bool)
    base = scorer(
        state_tokens=state,
        state_mask=sm,
        question_tokens=q,
        question_mask=qm,
        option_view_tokens=one,
        option_view_token_mask=otm1,
        option_view_mask=ovm1,
    )
    with_junk = scorer(
        state_tokens=state,
        state_mask=sm,
        question_tokens=q,
        question_mask=qm,
        option_view_tokens=options,
        option_view_token_mask=otm,
        option_view_mask=ovm,
    )
    assert torch.allclose(base, with_junk, atol=1e-6, rtol=1e-6)


def test_s21_degenerate_residuals_remain_finite() -> None:
    scorer = _scorer()
    token = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    options = token[:, None, None].expand(1, 2, 1, 1, 4).clone()
    logits, diag = scorer.forward_with_diagnostics(
        state_tokens=token,
        state_mask=torch.ones(1, 1, dtype=torch.bool),
        question_tokens=token,
        question_mask=torch.ones(1, 1, dtype=torch.bool),
        option_view_tokens=options,
        option_view_token_mask=torch.ones(1, 2, 1, 1, dtype=torch.bool),
        option_view_mask=torch.ones(1, 2, 1, dtype=torch.bool),
    )
    assert bool(torch.isfinite(logits).all())
    assert all(torch.isfinite(torch.tensor(v)) for v in diag.to_dict().values())
