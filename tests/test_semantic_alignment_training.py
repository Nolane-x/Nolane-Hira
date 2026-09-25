from __future__ import annotations

import copy

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.semantic_alignment_bridge import SemanticAlignmentBridgeScorer
from nmd.semantic_alignment_training import (
    CANDIDATES,
    PROJECTION_PARAMETER_COUNT,
    configure_trainability,
    expected_trainable_parameters,
    semantic_alignment_loss,
)


def _base_case(d_model: int = 8):
    torch.manual_seed(41)
    return {
        "state_content_tokens": torch.randn(5, d_model),
        "views": [
            {
                "view_id": "definition",
                "diagnosis_k": 16,
                "gold_index": 3,
                "option_tokens": torch.randn(16, 6, d_model),
                "option_content_mask": torch.ones(
                    16, 6, dtype=torch.bool
                ),
            }
        ],
    }


def test_w9_candidate_parameter_contracts():
    assert CANDIDATES == (
        "frozen-w6e-control",
        "projection-semantic-control",
        "shared-bridge-semantic-control",
        "asymmetric-bridge-primary",
        "asymmetric-bridge-replica",
    )
    assert expected_trainable_parameters("frozen-w6e-control") == 0
    assert expected_trainable_parameters("projection-semantic-control") == 32768
    assert expected_trainable_parameters("shared-bridge-semantic-control") == 4096
    assert expected_trainable_parameters("asymmetric-bridge-primary") == 8192
    assert expected_trainable_parameters("asymmetric-bridge-replica") == 8192
    assert PROJECTION_PARAMETER_COUNT == 32768


def test_projection_control_freezes_log_scale():
    hira = HIRACore()
    scorer = CompetitiveCoarseScorer()
    params = configure_trainability(
        "projection-semantic-control",
        hira,
        scorer,
    )
    assert params == [scorer.projection.weight]
    assert scorer.projection.weight.requires_grad
    assert not scorer.log_scale.requires_grad
    assert all(not p.requires_grad for p in hira.parameters())


def test_bridge_controls_freeze_base_and_route_trainability():
    hira = HIRACore()
    base = CompetitiveCoarseScorer()
    shared = SemanticAlignmentBridgeScorer(
        copy.deepcopy(base),
        mode="shared",
    )
    params = configure_trainability(
        "shared-bridge-semantic-control",
        hira,
        shared,
    )
    assert sum(p.numel() for p in params) == 4096
    assert all(not p.requires_grad for p in shared.base.parameters())
    assert all(p.requires_grad for p in params)

    asym = SemanticAlignmentBridgeScorer(
        copy.deepcopy(base),
        mode="asymmetric",
    )
    params = configure_trainability(
        "asymmetric-bridge-primary",
        hira,
        asym,
    )
    assert sum(p.numel() for p in params) == 8192
    assert all(not p.requires_grad for p in asym.base.parameters())
    assert all(p.requires_grad for p in params)


def test_semantic_alignment_loss_is_finite_and_differentiable():
    torch.manual_seed(43)
    base = CompetitiveCoarseScorer(d_model=8, d_rel=4)
    scorer = SemanticAlignmentBridgeScorer(
        base,
        mode="asymmetric",
        rank=2,
    )
    for p in scorer.base.parameters():
        p.requires_grad_(False)
    case = _base_case(d_model=8)
    loss = semantic_alignment_loss(scorer, case)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        and float(p.grad.abs().sum()) > 0
        for p in scorer.bridge_parameters()
    )
