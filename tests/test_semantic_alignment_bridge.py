from __future__ import annotations

import copy

import torch

from nmd.competitive import CompetitiveCoarseScorer, count_competitive_parameters
from nmd.semantic_alignment_bridge import (
    ASYMMETRIC_BRIDGE_PARAMETER_COUNT,
    SHARED_BRIDGE_PARAMETER_COUNT,
    SemanticAlignmentBridgeScorer,
    expected_bridge_parameter_count,
    semantic_alignment_scores,
)


def _inputs(d_model: int = 8):
    torch.manual_seed(7)
    state = torch.randn(2, 5, d_model)
    question = torch.randn(2, 3, d_model)
    options = torch.randn(2, 4, 6, d_model)
    state_mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )
    question_mask = torch.tensor(
        [[1, 1, 0], [1, 1, 1]],
        dtype=torch.bool,
    )
    option_mask = torch.tensor(
        [
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0],
            ],
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0],
            ],
        ],
        dtype=torch.bool,
    )
    token_ids = torch.tensor(
        [
            [
                [11, 12, 13, 0, 0, 0],
                [11, 21, 22, 23, 0, 0],
                [31, 32, 0, 0, 0, 0],
                [41, 42, 43, 44, 45, 0],
            ],
            [
                [11, 14, 15, 0, 0, 0],
                [11, 24, 25, 26, 0, 0],
                [33, 34, 0, 0, 0, 0],
                [46, 47, 48, 49, 50, 0],
            ],
        ],
        dtype=torch.long,
    )
    return {
        "state_tokens": state,
        "state_mask": state_mask,
        "question_tokens": question,
        "question_mask": question_mask,
        "option_tokens": options,
        "option_token_ids": token_ids,
        "option_mask": option_mask,
    }


def test_default_bridge_parameter_contracts():
    base = CompetitiveCoarseScorer()
    assert count_competitive_parameters(base) == 32769

    shared = SemanticAlignmentBridgeScorer(
        copy.deepcopy(base),
        mode="shared",
    )
    asymmetric = SemanticAlignmentBridgeScorer(
        copy.deepcopy(base),
        mode="asymmetric",
    )

    assert shared.parameter_counts().bridge == SHARED_BRIDGE_PARAMETER_COUNT
    assert asymmetric.parameter_counts().bridge == ASYMMETRIC_BRIDGE_PARAMETER_COUNT
    assert expected_bridge_parameter_count("shared") == 4096
    assert expected_bridge_parameter_count("asymmetric") == 8192


def test_identity_initialization_reproduces_production_logits():
    torch.manual_seed(11)
    base = CompetitiveCoarseScorer(d_model=8, d_rel=4)
    inputs = _inputs(d_model=8)

    expected = base(**inputs)
    for mode in ("shared", "asymmetric"):
        wrapped = SemanticAlignmentBridgeScorer(
            copy.deepcopy(base),
            mode=mode,
            rank=2,
        )
        actual = wrapped(**inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_alignment_score_identity_matches_base_projection():
    torch.manual_seed(13)
    base = CompetitiveCoarseScorer(d_model=8, d_rel=4)
    inputs = _inputs(d_model=8)
    definitions = inputs["option_tokens"]
    definition_mask = inputs["option_mask"]

    expected = semantic_alignment_scores(
        base,
        state_tokens=inputs["state_tokens"],
        state_mask=inputs["state_mask"],
        definition_tokens=definitions,
        definition_mask=definition_mask,
    )
    wrapped = SemanticAlignmentBridgeScorer(
        copy.deepcopy(base),
        mode="asymmetric",
        rank=2,
    )
    actual = semantic_alignment_scores(
        wrapped,
        state_tokens=inputs["state_tokens"],
        state_mask=inputs["state_mask"],
        definition_tokens=definitions,
        definition_mask=definition_mask,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_candidate_permutation_equivariance():
    torch.manual_seed(17)
    base = CompetitiveCoarseScorer(d_model=8, d_rel=4)
    scorer = SemanticAlignmentBridgeScorer(
        base,
        mode="asymmetric",
        rank=2,
    )
    inputs = _inputs(d_model=8)
    original = scorer(**inputs)

    perm = torch.tensor([2, 0, 3, 1])
    permuted_inputs = dict(inputs)
    permuted_inputs["option_tokens"] = inputs["option_tokens"][:, perm]
    permuted_inputs["option_token_ids"] = inputs["option_token_ids"][:, perm]
    permuted_inputs["option_mask"] = inputs["option_mask"][:, perm]
    permuted = scorer(**permuted_inputs)

    torch.testing.assert_close(permuted, original[:, perm], rtol=1e-6, atol=1e-6)


def test_freeze_base_routes_gradient_only_to_bridge():
    torch.manual_seed(19)
    base = CompetitiveCoarseScorer(d_model=8, d_rel=4)
    scorer = SemanticAlignmentBridgeScorer(
        base,
        mode="asymmetric",
        rank=2,
    )
    scorer.freeze_base()

    inputs = _inputs(d_model=8)
    scores = semantic_alignment_scores(
        scorer,
        state_tokens=inputs["state_tokens"],
        state_mask=inputs["state_mask"],
        definition_tokens=inputs["option_tokens"],
        definition_mask=inputs["option_mask"],
    )
    loss = -scores[:, 0].mean() + scores[:, 1:].mean()
    loss.backward()

    assert all(parameter.grad is None for parameter in scorer.base.parameters())
    bridge_parameters = scorer.bridge_parameters()
    assert all(parameter.requires_grad for parameter in bridge_parameters)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in bridge_parameters
    )


def test_shared_bridge_really_shares_parameters():
    base = CompetitiveCoarseScorer(d_model=8, d_rel=4)
    scorer = SemanticAlignmentBridgeScorer(
        base,
        mode="shared",
        rank=2,
    )
    assert scorer.shared_bridge is not None
    assert scorer.context_bridge is None
    assert scorer.schema_bridge is None
    assert len(scorer.bridge_parameters()) == 2
