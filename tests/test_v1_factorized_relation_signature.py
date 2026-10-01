import torch

from nmd.v1_factorized_relation_signature import (
    FactorizedRoleValueRelationCanonicalizer,
)


def _identity_projection(d: int = 4) -> torch.nn.Linear:
    projection = torch.nn.Linear(d, d, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(d))
    return projection


def test_s26_factorized_relation_is_parameter_free_and_finite():
    op = FactorizedRoleValueRelationCanonicalizer(
        role_temperature=0.10,
        pair_temperature=0.10,
        contrastive_temperature=0.10,
    )
    projection = _identity_projection()
    state = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    question = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    options = torch.tensor(
        [[[
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ], [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ], [
            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ], [
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ]]]
    )
    state_mask = torch.ones(1, 2, dtype=torch.bool)
    question_mask = torch.ones(1, 1, dtype=torch.bool)
    option_token_mask = torch.ones(1, 4, 1, 2, dtype=torch.bool)
    option_view_mask = torch.ones(1, 4, 1, dtype=torch.bool)

    logits, signatures, diagnostics = op(
        projection=projection,
        state_tokens=state,
        state_mask=state_mask,
        question_tokens=question,
        question_mask=question_mask,
        option_view_tokens=options,
        option_view_token_mask=option_token_mask,
        option_view_mask=option_view_mask,
    )

    assert op.factorization_added_parameter_count == 0
    assert logits.shape == (1, 4)
    assert signatures.shape == (1, 4, 8)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(signatures).all()
    assert torch.isfinite(
        torch.tensor(list(diagnostics.to_dict().values()))
    ).all()


def test_s26_hard_negative_quadrants_separate_role_and_value():
    op = FactorizedRoleValueRelationCanonicalizer()
    projection = _identity_projection()
    state = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    question = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    options = torch.tensor(
        [[[
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ], [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ], [
            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ], [
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ]]]
    )
    logits, _sig, _diag = op(
        projection=projection,
        state_tokens=state,
        state_mask=torch.ones(1, 2, dtype=torch.bool),
        question_tokens=question,
        question_mask=torch.ones(1, 1, dtype=torch.bool),
        option_view_tokens=options,
        option_view_token_mask=torch.ones(1, 4, 1, 2, dtype=torch.bool),
        option_view_mask=torch.ones(1, 4, 1, dtype=torch.bool),
    )
    assert float(logits[0, 0] - logits[0, 1]) > 0.0
    assert float(logits[0, 0] - logits[0, 2]) > 0.0
    assert float(logits[0, 0] - logits[0, 3]) > 0.0


def test_s26_option_permutation_equivariance():
    torch.manual_seed(26001)
    op = FactorizedRoleValueRelationCanonicalizer()
    projection = _identity_projection()
    b, k, v, s, q, t, d = 2, 4, 2, 3, 2, 3, 4
    state = torch.randn(b, s, d)
    question = torch.randn(b, q, d)
    options = torch.randn(b, k, v, t, d)
    state_mask = torch.ones(b, s, dtype=torch.bool)
    question_mask = torch.ones(b, q, dtype=torch.bool)
    option_token_mask = torch.ones(b, k, v, t, dtype=torch.bool)
    option_view_mask = torch.ones(b, k, v, dtype=torch.bool)

    logits, signatures, _ = op(
        projection=projection,
        state_tokens=state,
        state_mask=state_mask,
        question_tokens=question,
        question_mask=question_mask,
        option_view_tokens=options,
        option_view_token_mask=option_token_mask,
        option_view_mask=option_view_mask,
    )
    p = torch.tensor([2, 0, 3, 1])
    moved_logits, moved_signatures, _ = op(
        projection=projection,
        state_tokens=state,
        state_mask=state_mask,
        question_tokens=question,
        question_mask=question_mask,
        option_view_tokens=options[:, p],
        option_view_token_mask=option_token_mask[:, p],
        option_view_mask=option_view_mask[:, p],
    )
    assert torch.allclose(moved_logits, logits[:, p], atol=1e-6, rtol=0.0)
    assert torch.allclose(
        moved_signatures,
        signatures[:, p],
        atol=1e-6,
        rtol=0.0,
    )


def test_s26_degenerate_role_content_geometry_stays_finite():
    op = FactorizedRoleValueRelationCanonicalizer()
    projection = _identity_projection()
    base = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state = base.reshape(1, 1, 4).repeat(1, 2, 1)
    question = base.reshape(1, 1, 4)
    options = base.reshape(1, 1, 1, 1, 4).repeat(1, 4, 2, 2, 1)
    logits, signatures, _ = op(
        projection=projection,
        state_tokens=state,
        state_mask=torch.ones(1, 2, dtype=torch.bool),
        question_tokens=question,
        question_mask=torch.ones(1, 1, dtype=torch.bool),
        option_view_tokens=options,
        option_view_token_mask=torch.ones(1, 4, 2, 2, dtype=torch.bool),
        option_view_mask=torch.ones(1, 4, 2, dtype=torch.bool),
    )
    assert torch.isfinite(logits).all()
    assert torch.isfinite(signatures).all()
