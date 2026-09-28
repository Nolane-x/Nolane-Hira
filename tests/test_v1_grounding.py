import torch
from torch import nn

from nmd.v1_grounding import (
    QueryConditionedStateOptionGrounding,
    grounding_cross_entropy,
    grounding_gold_vs_max_wrong_margin,
)


def _identity_projection(dim=4):
    projection = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(dim))
    return projection


def _fixture():
    state = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]])
    state_mask = torch.tensor([[True, True, False]])

    question = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]])
    question_mask = torch.tensor([[True, True]])

    option = torch.zeros(1, 4, 2, 2, 4)
    option[0, 0, :, :, 0] = 1.0
    option[0, 1, :, :, 1] = 1.0
    option[0, 2, :, :, 0] = -1.0
    option[0, 3, :, :, 1] = -1.0
    option_token_mask = torch.ones(1, 4, 2, 2, dtype=torch.bool)
    option_view_mask = torch.ones(1, 4, 2, dtype=torch.bool)

    return {
        "state_tokens": state,
        "state_mask": state_mask,
        "question_tokens": question,
        "question_mask": question_mask,
        "option_view_tokens": option,
        "option_view_token_mask": option_token_mask,
        "option_view_mask": option_view_mask,
    }


def test_s10_grounding_has_zero_parameters():
    module = QueryConditionedStateOptionGrounding()
    assert module.parameter_count == 0
    assert sum(p.numel() for p in module.parameters()) == 0


def test_s10_grounding_selects_question_conditioned_state_fact():
    module = QueryConditionedStateOptionGrounding(
        attention_temperature=0.10,
        contrastive_temperature=0.10,
    )
    projection = _identity_projection()
    args = _fixture()

    logits, diagnostics = module(projection=projection, **args)
    assert logits.shape == (1, 4)
    assert int(logits.argmax(-1).item()) == 0
    assert float(diagnostics.max_attention_weight) > 0.99
    assert 0.0 <= float(diagnostics.normalized_attention_entropy) <= 1.0

    args["question_tokens"] = torch.tensor([[
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]])
    changed, _ = module(projection=projection, **args)
    assert int(changed.argmax(-1).item()) == 1


def test_s10_grounding_ignores_masked_state_tokens():
    module = QueryConditionedStateOptionGrounding()
    projection = _identity_projection()
    args = _fixture()

    baseline, _ = module(projection=projection, **args)
    changed_state = args["state_tokens"].clone()
    changed_state[0, 2] = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0])
    args["state_tokens"] = changed_state
    observed, _ = module(projection=projection, **args)

    assert torch.allclose(observed, baseline, atol=1e-7, rtol=0.0)


def test_s10_grounding_option_permutation_equivariance():
    module = QueryConditionedStateOptionGrounding()
    projection = _identity_projection()
    args = _fixture()
    direct, _ = module(projection=projection, **args)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_args = dict(args)
    permuted_args["option_view_tokens"] = args["option_view_tokens"][:, permutation]
    permuted_args["option_view_token_mask"] = args[
        "option_view_token_mask"
    ][:, permutation]
    permuted_args["option_view_mask"] = args["option_view_mask"][:, permutation]
    permuted, _ = module(projection=projection, **permuted_args)

    assert torch.allclose(
        permuted,
        direct[:, permutation],
        atol=1e-7,
        rtol=0.0,
    )


def test_s10_grounding_loss_and_margin_are_semantically_signed():
    logits = torch.tensor([
        [3.0, 1.0, 0.0, -1.0],
        [0.0, 2.0, 3.0, 1.0],
    ])
    gold = torch.tensor([0, 1], dtype=torch.long)

    loss = grounding_cross_entropy(logits, gold)
    assert float(loss) > 0.0

    margins = grounding_gold_vs_max_wrong_margin(logits, gold)
    assert torch.allclose(margins, torch.tensor([2.0, -1.0]))


def test_s10_grounding_backpropagates_into_shared_projection_and_tokens():
    torch.manual_seed(17101)
    module = QueryConditionedStateOptionGrounding()
    projection = nn.Linear(4, 4, bias=False)
    args = _fixture()
    args["state_tokens"] = args["state_tokens"].clone().requires_grad_(True)
    args["question_tokens"] = args["question_tokens"].clone().requires_grad_(True)
    args["option_view_tokens"] = (
        args["option_view_tokens"].clone().requires_grad_(True)
    )

    logits, _ = module(projection=projection, **args)
    loss = grounding_cross_entropy(logits, torch.tensor([0]))
    loss.backward()

    assert projection.weight.grad is not None
    assert float(projection.weight.grad.abs().sum()) > 0.0
    assert args["state_tokens"].grad is not None
    assert float(args["state_tokens"].grad.abs().sum()) > 0.0
    assert args["question_tokens"].grad is not None
    assert float(args["question_tokens"].grad.abs().sum()) > 0.0
    assert args["option_view_tokens"].grad is not None
    assert float(args["option_view_tokens"].grad.abs().sum()) > 0.0


def test_s10_grounding_rejects_bad_temperatures_and_shapes():
    try:
        QueryConditionedStateOptionGrounding(attention_temperature=0.0)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("zero S10 attention temperature must fail")

    module = QueryConditionedStateOptionGrounding()
    projection = _identity_projection()
    args = _fixture()
    args["state_mask"] = torch.ones(1, 2, dtype=torch.bool)
    try:
        module(projection=projection, **args)
    except ValueError as exc:
        assert "state_mask" in str(exc)
    else:
        raise AssertionError("bad S10 state mask must fail")
