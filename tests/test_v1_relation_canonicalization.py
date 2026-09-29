import torch
from torch import nn

from nmd.v1_relation_canonicalization import (
    CrossViewRelationCanonicalizer,
    cross_view_relation_signature_loss,
    relation_signature_same_option_cosine,
)


def _identity_projection(dim=6):
    projection = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(dim))
    return projection


def _fixture():
    state = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]])
    state_mask = torch.ones(1, 5, dtype=torch.bool)
    question = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]])
    question_mask = torch.tensor([[True, True]])

    option = torch.zeros(1, 4, 2, 2, 6)
    option[0, 0, :, 0, 0] = 1.0
    option[0, 0, :, 1, 1] = 1.0
    option[0, 1, :, 0, 2] = 1.0
    option[0, 1, :, 1, 3] = 1.0
    option[0, 2, :, 0, 0] = 1.0
    option[0, 2, :, 1, 4] = 1.0
    option[0, 3, :, 0, 2] = 1.0
    option[0, 3, :, 1, 4] = 1.0
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


def test_s13_canonicalizer_has_zero_parameters():
    module = CrossViewRelationCanonicalizer()
    assert module.parameter_count == 0
    assert sum(p.numel() for p in module.parameters()) == 0


def test_s13_canonicalizer_returns_option_signatures_and_finite_logits():
    module = CrossViewRelationCanonicalizer()
    logits, signatures, diagnostics = module(
        projection=_identity_projection(),
        **_fixture(),
    )
    assert logits.shape == (1, 4)
    assert signatures.shape == (1, 4, 6)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(signatures).all()
    assert torch.allclose(
        signatures.norm(dim=-1),
        torch.ones(1, 4),
        atol=1e-6,
        rtol=0.0,
    )
    assert 0.0 <= float(diagnostics.state_role_normalized_entropy) <= 1.0
    assert 0.0 <= float(diagnostics.mean_pair_entropy) <= 1.0


def test_s13_signature_identity_has_perfect_same_option_cosine():
    module = CrossViewRelationCanonicalizer()
    _logits, signatures, _diag = module(
        projection=_identity_projection(),
        **_fixture(),
    )
    cosine = relation_signature_same_option_cosine(signatures, signatures)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6, rtol=0.0)


def test_s13_option_permutation_equivariance_for_logits_and_signatures():
    module = CrossViewRelationCanonicalizer()
    projection = _identity_projection()
    args = _fixture()
    logits, signatures, _ = module(projection=projection, **args)

    permutation = torch.tensor([2, 0, 3, 1])
    changed = dict(args)
    changed["option_view_tokens"] = args["option_view_tokens"][:, permutation]
    changed["option_view_token_mask"] = args[
        "option_view_token_mask"
    ][:, permutation]
    changed["option_view_mask"] = args["option_view_mask"][:, permutation]

    p_logits, p_signatures, _ = module(projection=projection, **changed)
    assert torch.allclose(p_logits, logits[:, permutation], atol=1e-6, rtol=0.0)
    assert torch.allclose(
        p_signatures,
        signatures[:, permutation],
        atol=1e-6,
        rtol=0.0,
    )


def test_s13_cross_view_loss_rewards_same_relation_identity():
    canonical = torch.eye(4).unsqueeze(0)
    paraphrase = canonical.clone()
    total, alignment, separation = cross_view_relation_signature_loss(
        canonical,
        paraphrase,
        separation_margin=0.20,
    )
    assert float(alignment) == 0.0
    assert float(separation) == 0.0
    assert float(total) == 0.0

    swapped = paraphrase[:, torch.tensor([1, 0, 3, 2])]
    changed, _, _ = cross_view_relation_signature_loss(
        canonical,
        swapped,
        separation_margin=0.20,
    )
    assert float(changed) > 0.0


def test_s13_canonicalizer_backpropagates_only_through_supplied_surface():
    torch.manual_seed(20101)
    module = CrossViewRelationCanonicalizer()
    projection = nn.Linear(6, 6, bias=False)
    args = _fixture()
    args["state_tokens"] = args["state_tokens"].clone().requires_grad_(True)
    args["question_tokens"] = args["question_tokens"].clone().requires_grad_(True)
    args["option_view_tokens"] = args["option_view_tokens"].clone().requires_grad_(True)

    _logits, signatures, _ = module(projection=projection, **args)
    loss, _, _ = cross_view_relation_signature_loss(
        signatures,
        signatures.roll(shifts=1, dims=1),
    )
    loss.backward()

    assert projection.weight.grad is not None
    assert float(projection.weight.grad.abs().sum()) > 0.0
    assert args["state_tokens"].grad is not None
    assert float(args["state_tokens"].grad.abs().sum()) > 0.0
    assert args["question_tokens"].grad is not None
    assert float(args["question_tokens"].grad.abs().sum()) > 0.0
    assert args["option_view_tokens"].grad is not None
    assert float(args["option_view_tokens"].grad.abs().sum()) > 0.0


def test_s13_rejects_invalid_temperatures_and_shapes():
    for kwargs in (
        {"role_temperature": 0.0},
        {"pair_temperature": 0.0},
        {"contrastive_temperature": 0.0},
    ):
        try:
            CrossViewRelationCanonicalizer(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid S13 temperature must fail")

    module = CrossViewRelationCanonicalizer()
    args = _fixture()
    args["state_mask"] = torch.ones(1, 2, dtype=torch.bool)
    try:
        module(projection=_identity_projection(), **args)
    except ValueError as exc:
        assert "state_mask" in str(exc)
    else:
        raise AssertionError("bad S13 state mask must fail")
