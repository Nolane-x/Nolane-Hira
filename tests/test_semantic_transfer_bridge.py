import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.semantic_transfer_bridge import (
    BridgedSymmetricSemanticScorer,
    LowRankSemanticTransferBridge,
)
from nmd.symmetric_semantic import SymmetricSemanticScorer


def _inputs():
    torch.manual_seed(30001)
    return dict(
        state_tokens=torch.randn(2, 7, 256),
        state_mask=torch.tensor([
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False],
        ]),
        option_view_tokens=torch.randn(2, 3, 2, 5, 256),
        option_view_token_mask=torch.tensor([
            [
                [[True, True, True, False, False], [True, True, False, False, False]],
                [[True, True, True, True, False], [True, True, True, False, False]],
                [[True, True, False, False, False], [True, True, True, False, False]],
            ],
            [
                [[True, True, True, False, False], [True, True, True, False, False]],
                [[True, True, False, False, False], [True, True, True, True, False]],
                [[True, True, True, False, False], [True, True, False, False, False]],
            ],
        ]),
        option_view_mask=torch.ones(2, 3, 2, dtype=torch.bool),
    )


def test_bridge_has_exact_frozen_capacity_contract():
    bridge = LowRankSemanticTransferBridge(d_rel=128, rank=8)
    assert bridge.parameter_count == 2048
    assert bridge.trainable_parameter_count == 2048
    x = torch.randn(4, 128)
    assert torch.equal(bridge(x), x)
    bridge.freeze()
    assert bridge.trainable_parameter_count == 0


def test_zero_bridge_is_exact_w29_functional_baseline():
    torch.manual_seed(30002)
    weight = torch.randn(128, 256)
    base = SymmetricSemanticScorer(d_model=256, d_rel=128)
    base.load_projection_weight(weight, freeze=True)

    bridged = BridgedSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=8,
    )
    bridged.load_projection_weight(weight, freeze=True)

    args = _inputs()
    expected = base(**args)
    actual = bridged(**args)
    assert torch.equal(actual, expected)
    assert bridged.projection.weight.requires_grad is False
    assert bridged.bridge_parameter_count == 2048
    assert bridged.trainable_parameter_count == 2048


def _model() -> NolaneHira:
    torch.manual_seed(30003)
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = BridgedSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=8,
    )
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    return NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        bridged_symmetric_semantic_scorer=scorer,
    ).eval()


def _options():
    return (
        LogicalOption(
            "no",
            "the important service capability remains available",
            aliases=("the central function remains usable",),
            exemplars=("the key task can still be completed",),
            value=0.0,
        ),
        LogicalOption(
            "yes",
            "the important service capability is unavailable",
            aliases=("the central function has been lost",),
            exemplars=("the key task is blocked by functional loss",),
            value=1.0,
        ),
    )


def test_bridged_runtime_is_coarse_only_and_state_once():
    model = _model()
    memory = model.compile_state(
        "the central service capability has failed and the key task cannot be completed"
    )
    before = model.state_encode_calls
    selected = []
    for primitive in ("choice", "score", "noul"):
        schema, _ = model.compile_schema(
            primitive=primitive,
            question_text="Is major functional capability lost?",
            options=_options(),
            include_token_artifacts=True,
            use_cache=True,
        )
        out = model.forward_compiled(
            memory,
            schema,
            coarse_mode="bridged_symmetric_semantic",
        )
        selected.append(int(out.probabilities.argmax()))
        assert torch.equal(out.hira.relation_delta, torch.zeros_like(out.hira.relation_delta))
        assert int(out.hira.candidate_budget.item()) == 2
        assert bool(out.hira.selected_mask.all())
        assert torch.isclose(out.probabilities.sum(), out.probabilities.new_tensor(1.0), atol=1e-6)

    assert model.state_encode_calls == before
    assert selected[0] == selected[1] == selected[2]


def test_bridge_checkpoint_load_and_freeze():
    scorer = BridgedSymmetricSemanticScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    down = torch.randn(8, 128)
    up = torch.randn(128, 8)
    scorer.load_bridge_state_dict(
        {"down.weight": down, "up.weight": up},
        freeze=True,
    )
    assert scorer.bridge_trainable_parameter_count == 0
    assert scorer.trainable_parameter_count == 0
    assert torch.equal(scorer.bridge.down.weight, down)
    assert torch.equal(scorer.bridge.up.weight, up)
