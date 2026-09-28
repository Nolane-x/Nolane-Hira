from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from nmd.v1_role_value_binding import (
    RoleValueFactorizedBinding,
    binding_gold_vs_max_wrong_margin,
)


def _identity_projection(d: int) -> nn.Linear:
    projection = nn.Linear(d, d, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(d))
    return projection


def _toy(question_axis: int):
    d = 6
    projection = _identity_projection(d)
    binding = RoleValueFactorizedBinding(
        role_temperature=0.10,
        state_temperature=0.10,
    )

    basis = torch.eye(d)
    field_a, field_b, value_a, value_b, wrong_a, wrong_b = basis

    state = torch.stack((field_a, value_a, field_b, value_b))[None, :, :]
    state_mask = torch.ones(1, 4, dtype=torch.bool)
    question = basis[question_axis][None, None, :]
    question_mask = torch.ones(1, 1, dtype=torch.bool)

    options = torch.stack(
        (
            torch.stack((field_a, value_a)),
            torch.stack((field_a, wrong_a)),
            torch.stack((field_b, value_b)),
            torch.stack((field_b, wrong_b)),
        )
    )[None, :, None, :, :]
    option_token_mask = torch.ones(1, 4, 1, 2, dtype=torch.bool)
    option_view_mask = torch.ones(1, 4, 1, dtype=torch.bool)

    return binding(
        projection=projection,
        state_tokens=state,
        state_mask=state_mask,
        question_tokens=question,
        question_mask=question_mask,
        option_view_tokens=options,
        option_view_token_mask=option_token_mask,
        option_view_mask=option_view_mask,
    )


def test_s11_operator_has_zero_parameters():
    binding = RoleValueFactorizedBinding()
    assert binding.parameter_count == 0
    assert list(binding.parameters()) == []


def test_s11_factorized_binding_selects_role_and_present_value():
    logits_a, role_a, state_a, _ = _toy(0)
    logits_b, role_b, state_b, _ = _toy(1)

    assert int(logits_a.argmax(-1).item()) == 0
    assert int(logits_b.argmax(-1).item()) == 2

    # Role expert alone cannot distinguish the same-field distractor.
    assert torch.allclose(role_a[0, 0], role_a[0, 1])
    assert torch.allclose(role_b[0, 2], role_b[0, 3])

    # State expert preserves both facts present in the state as the top-2.
    assert set(state_a[0].topk(2).indices.tolist()) == {0, 2}
    assert set(state_b[0].topk(2).indices.tolist()) == {0, 2}


def test_s11_option_permutation_is_equivariant():
    torch.manual_seed(7)
    b, s, q, k, v, t, d = 2, 5, 3, 4, 2, 4, 8
    projection = nn.Linear(d, 6, bias=False)
    binding = RoleValueFactorizedBinding()

    state = torch.randn(b, s, d)
    question = torch.randn(b, q, d)
    options = torch.randn(b, k, v, t, d)
    state_mask = torch.ones(b, s, dtype=torch.bool)
    question_mask = torch.ones(b, q, dtype=torch.bool)
    option_token_mask = torch.ones(b, k, v, t, dtype=torch.bool)
    option_view_mask = torch.ones(b, k, v, dtype=torch.bool)

    base, base_role, base_state, _ = binding(
        projection=projection,
        state_tokens=state,
        state_mask=state_mask,
        question_tokens=question,
        question_mask=question_mask,
        option_view_tokens=options,
        option_view_token_mask=option_token_mask,
        option_view_mask=option_view_mask,
    )

    perm = torch.tensor([2, 0, 3, 1])
    out, role, state_scores, _ = binding(
        projection=projection,
        state_tokens=state,
        state_mask=state_mask,
        question_tokens=question,
        question_mask=question_mask,
        option_view_tokens=options[:, perm],
        option_view_token_mask=option_token_mask[:, perm],
        option_view_mask=option_view_mask[:, perm],
    )

    assert torch.allclose(out, base[:, perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(role, base_role[:, perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(state_scores, base_state[:, perm], atol=1e-6, rtol=1e-6)


def test_s11_masked_junk_tokens_do_not_change_binding():
    logits, _, _, _ = _toy(0)

    d = 6
    projection = _identity_projection(d)
    binding = RoleValueFactorizedBinding()
    basis = torch.eye(d)
    field_a, field_b, value_a, value_b, wrong_a, wrong_b = basis

    state = torch.stack((field_a, value_a, field_b, value_b, 100 * wrong_b))[None]
    state_mask = torch.tensor([[True, True, True, True, False]])
    question = torch.stack((field_a, 100 * wrong_b))[None]
    question_mask = torch.tensor([[True, False]])
    options = torch.stack(
        (
            torch.stack((field_a, value_a, 100 * wrong_b)),
            torch.stack((field_a, wrong_a, 100 * value_a)),
            torch.stack((field_b, value_b, 100 * wrong_a)),
            torch.stack((field_b, wrong_b, 100 * value_b)),
        )
    )[None, :, None]
    option_token_mask = torch.tensor(
        [[[[True, True, False]]] * 4],
        dtype=torch.bool,
    )
    option_view_mask = torch.ones(1, 4, 1, dtype=torch.bool)

    masked, _, _, _ = binding(
        projection=projection,
        state_tokens=state,
        state_mask=state_mask,
        question_tokens=question,
        question_mask=question_mask,
        option_view_tokens=options,
        option_view_token_mask=option_token_mask,
        option_view_mask=option_view_mask,
    )

    assert torch.allclose(masked, logits, atol=1e-6, rtol=1e-6)


def test_s11_gradient_flows_only_through_shared_projection_surface():
    torch.manual_seed(11)
    d = 7
    projection = nn.Linear(d, 5, bias=False)
    binding = RoleValueFactorizedBinding()

    state = torch.randn(3, 6, d)
    question = torch.randn(3, 4, d)
    options = torch.randn(3, 4, 2, 5, d)
    state_mask = torch.ones(3, 6, dtype=torch.bool)
    question_mask = torch.ones(3, 4, dtype=torch.bool)
    option_token_mask = torch.ones(3, 4, 2, 5, dtype=torch.bool)
    option_view_mask = torch.ones(3, 4, 2, dtype=torch.bool)

    logits, _, _, _ = binding(
        projection=projection,
        state_tokens=state,
        state_mask=state_mask,
        question_tokens=question,
        question_mask=question_mask,
        option_view_tokens=options,
        option_view_token_mask=option_token_mask,
        option_view_mask=option_view_mask,
    )
    gold = torch.tensor([0, 1, 2])
    loss = F.cross_entropy(logits, gold)
    loss.backward()

    assert projection.weight.grad is not None
    assert bool(torch.isfinite(projection.weight.grad).all())
    assert float(projection.weight.grad.abs().sum()) > 0.0
    assert binding.parameter_count == 0


def test_s11_margin_contract():
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.5]])
    gold = torch.tensor([0, 2], dtype=torch.long)
    margin = binding_gold_vs_max_wrong_margin(logits, gold)
    assert torch.allclose(margin, torch.tensor([2.0, -0.5]))
