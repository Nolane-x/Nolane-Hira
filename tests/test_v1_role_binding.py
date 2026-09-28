import torch
from torch import nn

from nmd.v1_role_binding import (
    RolePreservingEvidenceBinding,
    binding_cross_entropy,
    binding_gold_vs_max_wrong_margin,
)


def _identity_projection(dim=6):
    projection = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(dim))
    return projection


def _fixture():
    # role-A, value-A, filler, role-B, value-B
    state = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ]])
    state_mask = torch.tensor([[True, True, True, True, True]])

    question = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]])
    question_mask = torch.tensor([[True, True]])

    option = torch.zeros(1, 4, 2, 2, 6)
    # Each option has a field token and a value token. Options 0 and 2 share
    # role A; only option 0 carries the correct nearby value A.
    option[0, 0, :, 0, 0] = 1.0
    option[0, 0, :, 1, 1] = 1.0
    option[0, 1, :, 0, 2] = 1.0
    option[0, 1, :, 1, 3] = 1.0
    option[0, 2, :, 0, 0] = 1.0
    option[0, 2, :, 1, 5] = 1.0
    option[0, 3, :, 0, 2] = 1.0
    option[0, 3, :, 1, 5] = 1.0

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


def test_s11_role_binding_has_zero_parameters():
    module = RolePreservingEvidenceBinding()
    assert module.parameter_count == 0
    assert sum(p.numel() for p in module.parameters()) == 0


def test_s11_binds_role_to_nearby_value_instead_of_pooled_field_match():
    module = RolePreservingEvidenceBinding(
        role_temperature=0.05,
        contrastive_temperature=0.10,
        value_window=1,
    )
    args = _fixture()
    direct, diagnostics = module(
        projection=_identity_projection(),
        **args,
    )
    assert direct.shape == (1, 4)
    assert int(direct.argmax(-1).item()) == 0
    assert float(diagnostics.max_role_weight) > 0.99

    args["question_tokens"] = torch.tensor([[
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]])
    changed, _ = module(
        projection=_identity_projection(),
        **args,
    )
    assert int(changed.argmax(-1).item()) == 1


def test_s11_binding_ignores_masked_state_tokens():
    module = RolePreservingEvidenceBinding(value_window=1)
    projection = _identity_projection()
    args = _fixture()

    args["state_mask"] = torch.tensor([[True, True, True, True, False]])
    baseline, _ = module(projection=projection, **args)

    changed = args["state_tokens"].clone()
    changed[0, 4] = torch.tensor([1000.0] * 6)
    args["state_tokens"] = changed
    observed, _ = module(projection=projection, **args)
    assert torch.allclose(observed, baseline, atol=1e-7, rtol=0.0)


def test_s11_binding_option_permutation_equivariance():
    module = RolePreservingEvidenceBinding(value_window=1)
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

    assert torch.allclose(permuted, direct[:, permutation], atol=1e-7, rtol=0.0)


def test_s11_binding_loss_and_margin_are_signed():
    logits = torch.tensor([
        [3.0, 1.0, 0.0, -1.0],
        [0.0, 2.0, 3.0, 1.0],
    ])
    gold = torch.tensor([0, 1], dtype=torch.long)
    assert float(binding_cross_entropy(logits, gold)) > 0.0
    margins = binding_gold_vs_max_wrong_margin(logits, gold)
    assert torch.allclose(margins, torch.tensor([2.0, -1.0]))


def test_s11_binding_backpropagates_into_existing_projection_and_tokens():
    torch.manual_seed(18101)
    module = RolePreservingEvidenceBinding(value_window=1)
    projection = nn.Linear(6, 6, bias=False)
    args = _fixture()
    args["state_tokens"] = args["state_tokens"].clone().requires_grad_(True)
    args["question_tokens"] = args["question_tokens"].clone().requires_grad_(True)
    args["option_view_tokens"] = args["option_view_tokens"].clone().requires_grad_(True)

    logits, _ = module(projection=projection, **args)
    loss = binding_cross_entropy(logits, torch.tensor([0]))
    loss.backward()

    assert projection.weight.grad is not None
    assert float(projection.weight.grad.abs().sum()) > 0.0
    assert args["state_tokens"].grad is not None
    assert float(args["state_tokens"].grad.abs().sum()) > 0.0
    assert args["question_tokens"].grad is not None
    assert float(args["question_tokens"].grad.abs().sum()) > 0.0
    assert args["option_view_tokens"].grad is not None
    assert float(args["option_view_tokens"].grad.abs().sum()) > 0.0


def test_s11_binding_rejects_bad_contracts():
    for kwargs in (
        {"role_temperature": 0.0},
        {"contrastive_temperature": 0.0},
        {"value_window": 0},
    ):
        try:
            RolePreservingEvidenceBinding(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid S11 binding configuration must fail")

    module = RolePreservingEvidenceBinding()
    args = _fixture()
    args["state_mask"] = torch.ones(1, 2, dtype=torch.bool)
    try:
        module(projection=_identity_projection(), **args)
    except ValueError as exc:
        assert "state_mask" in str(exc)
    else:
        raise AssertionError("bad S11 state mask must fail")
