from __future__ import annotations

import torch

from nmd.competitive import (
    CompetitiveCoarseScorer,
    count_competitive_parameters,
)
from nmd.conjunctive_coarse import (
    CONJUNCTIVE_EXTRA_PARAMETER_COUNT,
    CONJUNCTIVE_TOTAL_PARAMETER_COUNT,
    ConjunctiveEvidenceScorer,
    conjunctive_extra_parameter_count,
    count_conjunctive_parameters,
)


def _random_inputs():
    torch.manual_seed(7101)
    return {
        "state_tokens": torch.randn(2, 7, 256),
        "state_mask": torch.ones(2, 7, dtype=torch.bool),
        "question_tokens": torch.randn(2, 3, 256),
        "question_mask": torch.ones(2, 3, dtype=torch.bool),
        "option_tokens": torch.randn(2, 5, 4, 256),
        "option_token_ids": torch.randint(
            10, 1000, (2, 5, 4), dtype=torch.long
        ),
        "option_mask": torch.ones(2, 5, 4, dtype=torch.bool),
    }


def _factor_inputs(batch=2, candidates=5, factors=4, tokens=3):
    torch.manual_seed(7103)
    return {
        "factor_tokens": torch.randn(
            batch, candidates, factors, tokens, 256
        ),
        "factor_token_mask": torch.ones(
            batch, candidates, factors, tokens, dtype=torch.bool
        ),
        "factor_present_mask": torch.ones(
            batch, candidates, factors, dtype=torch.bool
        ),
    }


def test_w7_parameter_budget_is_exactly_plus_three():
    base = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    model = ConjunctiveEvidenceScorer(base)
    assert count_competitive_parameters(base) == 32769
    assert CONJUNCTIVE_EXTRA_PARAMETER_COUNT == 3
    assert CONJUNCTIVE_TOTAL_PARAMETER_COUNT == 32772
    assert conjunctive_extra_parameter_count(model) == 3
    assert count_conjunctive_parameters(model) == 32772

    scalar_names = {
        "factor_threshold",
        "log_factor_temperature",
        "raw_conjunction_alpha",
    }
    direct = {
        name
        for name, _ in model.named_parameters(recurse=False)
    }
    assert direct == scalar_names


def test_freeform_forward_is_exact_base_fallback():
    torch.manual_seed(7107)
    base = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    model = ConjunctiveEvidenceScorer(base)
    inputs = _random_inputs()
    expected = base(**inputs)
    actual = model(**inputs)
    assert torch.equal(actual, expected)


def test_factor_order_is_permutation_invariant():
    torch.manual_seed(7109)
    model = ConjunctiveEvidenceScorer(
        CompetitiveCoarseScorer(d_model=256, d_rel=128)
    )
    base_inputs = _random_inputs()
    factor = _factor_inputs()
    original = model.forward_with_factors(
        **base_inputs,
        **factor,
    )

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model.forward_with_factors(
        **base_inputs,
        factor_tokens=factor["factor_tokens"][:, :, permutation],
        factor_token_mask=(
            factor["factor_token_mask"][:, :, permutation]
        ),
        factor_present_mask=(
            factor["factor_present_mask"][:, :, permutation]
        ),
    )
    assert torch.allclose(
        original.logits,
        permuted.logits,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        original.conjunction_log_evidence,
        permuted.conjunction_log_evidence,
        atol=1e-6,
        rtol=1e-6,
    )


def test_one_weak_factor_penalizes_otherwise_equal_candidate():
    base = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    with torch.no_grad():
        base.projection.weight.zero_()
        base.projection.weight[:, :128] = torch.eye(128)
        base.log_scale.fill_(torch.log(torch.tensor(10.0)))

    model = ConjunctiveEvidenceScorer(base)
    with torch.no_grad():
        model.factor_threshold.zero_()
        model.log_factor_temperature.zero_()
        model.raw_conjunction_alpha.fill_(0.0)

    e1 = torch.zeros(256)
    e1[0] = 1.0
    e2 = torch.zeros(256)
    e2[1] = 1.0

    state_tokens = e1.view(1, 1, 256)
    state_mask = torch.ones(1, 1, dtype=torch.bool)
    question_tokens = e1.view(1, 1, 256)
    question_mask = torch.ones(1, 1, dtype=torch.bool)

    # Identical free-form option artifacts force an equal base score.
    option_tokens = e1.view(1, 1, 1, 256).repeat(1, 2, 1, 1)
    option_ids = torch.tensor([[[17], [17]]], dtype=torch.long)
    option_mask = torch.ones(1, 2, 1, dtype=torch.bool)

    factor_tokens = e1.view(1, 1, 1, 1, 256).repeat(
        1, 2, 4, 1, 1
    )
    factor_tokens[:, 1, 0, 0] = e2
    factor_mask = torch.ones(1, 2, 4, 1, dtype=torch.bool)
    present = torch.ones(1, 2, 4, dtype=torch.bool)

    out = model.forward_with_factors(
        state_tokens=state_tokens,
        state_mask=state_mask,
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_tokens=option_tokens,
        option_token_ids=option_ids,
        option_mask=option_mask,
        factor_tokens=factor_tokens,
        factor_token_mask=factor_mask,
        factor_present_mask=present,
    )
    assert torch.allclose(
        out.base_logits[0, 0],
        out.base_logits[0, 1],
        atol=1e-7,
        rtol=0,
    )
    assert out.factor_evidence[0, 0, 0] > out.factor_evidence[0, 1, 0]
    assert out.conjunction_residual[0, 0] > out.conjunction_residual[0, 1]
    assert out.logits[0, 0] > out.logits[0, 1]


def test_variable_factor_count_uses_present_mask_only():
    torch.manual_seed(7111)
    model = ConjunctiveEvidenceScorer(
        CompetitiveCoarseScorer(d_model=256, d_rel=128)
    )
    base_inputs = _random_inputs()
    factor = _factor_inputs()

    factor["factor_present_mask"][:, :, 3] = False
    factor["factor_token_mask"][:, :, 3] = False

    out = model.forward_with_factors(
        **base_inputs,
        **factor,
    )
    assert out.logits.shape == (2, 5)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.factor_logits).all()


def test_factor_path_gradients_reach_projection_and_all_three_scalars():
    torch.manual_seed(7117)
    model = ConjunctiveEvidenceScorer(
        CompetitiveCoarseScorer(d_model=256, d_rel=128)
    )
    base_inputs = _random_inputs()
    factor = _factor_inputs()

    out = model.forward_with_factors(
        **base_inputs,
        **factor,
    )
    loss = (
        out.logits.square().mean()
        + 0.1 * out.factor_logits.square().mean()
    )
    loss.backward()

    assert model.base.projection.weight.grad is not None
    assert torch.isfinite(model.base.projection.weight.grad).all()
    for parameter in (
        model.factor_threshold,
        model.log_factor_temperature,
        model.raw_conjunction_alpha,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_temperature_positive_and_alpha_nonnegative():
    model = ConjunctiveEvidenceScorer()
    assert float(model.factor_temperature()) > 0.0
    assert float(model.conjunction_alpha()) >= 0.0
