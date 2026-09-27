import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.symmetric_semantic import SymmetricSemanticScorer
from nmd.w31_dual_semantic_adapter import DualAdapterSymmetricSemanticScorer


def _semantic_inputs():
    torch.manual_seed(31101)
    return dict(
        state_tokens=torch.randn(2, 6, 256),
        state_mask=torch.tensor([
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
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


def test_w31_dual_adapter_has_exact_capacity_contract():
    scorer = DualAdapterSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=8,
    )
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    assert scorer.state_adapter.parameter_count == 2048
    assert scorer.schema_adapter.parameter_count == 2048
    assert scorer.adapter_parameter_count == 4096
    assert scorer.adapter_trainable_parameter_count == 4096
    assert scorer.trainable_parameter_count == 4096
    assert scorer.projection.weight.requires_grad is False

    scorer.freeze_adapters()
    assert scorer.adapter_trainable_parameter_count == 0
    assert scorer.trainable_parameter_count == 0


def test_w31_zero_init_is_exact_unbridged_t0_baseline():
    torch.manual_seed(31102)
    weight = torch.randn(128, 256)

    base = SymmetricSemanticScorer(d_model=256, d_rel=128)
    base.load_projection_weight(weight, freeze=True)

    dual = DualAdapterSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=8,
    )
    dual.load_projection_weight(weight, freeze=True)

    args = _semantic_inputs()
    expected = base(**args)
    actual = dual(**args)
    assert torch.equal(actual, expected)


def test_w31_state_and_schema_adapters_are_independent():
    torch.manual_seed(31103)
    scorer = DualAdapterSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=8,
    )
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    args = _semantic_inputs()
    baseline = scorer(**args)

    with torch.no_grad():
        scorer.state_adapter.up.weight.normal_(mean=0.0, std=0.03)
    state_changed = scorer(**args)
    assert not torch.equal(state_changed, baseline)

    with torch.no_grad():
        scorer.state_adapter.up.weight.zero_()
        scorer.schema_adapter.up.weight.normal_(mean=0.0, std=0.03)
    schema_changed = scorer(**args)
    assert not torch.equal(schema_changed, baseline)


def _options():
    return (
        LogicalOption(
            "retained",
            "the pivotal service capability remains available",
            aliases=("the main task retains its required function",),
            exemplars=("the core objective can still be completed",),
            value=0.0,
        ),
        LogicalOption(
            "lost",
            "the pivotal service capability is unavailable",
            aliases=("the main task has lost its required function",),
            exemplars=("the core objective cannot be completed",),
            value=1.0,
        ),
    )


def _model():
    torch.manual_seed(31104)
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = DualAdapterSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=8,
    )
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    return NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        dual_symmetric_semantic_scorer=scorer,
    ).eval()


def test_w31_runtime_is_coarse_only_typed_and_state_once():
    model = _model()
    memory = model.compile_state(
        "the pivotal function has failed so the main task cannot be completed"
    )
    after_compile = model.state_encode_calls
    preds = []

    for primitive in ("choice", "score", "noul"):
        schema, _ = model.compile_schema(
            primitive=primitive,
            question_text="Is the pivotal service capability lost?",
            options=_options(),
            include_token_artifacts=True,
            use_cache=True,
        )
        out = model.forward_compiled(
            memory,
            schema,
            coarse_mode="dual_symmetric_semantic",
        )
        preds.append(int(out.probabilities.argmax()))
        assert torch.equal(
            out.hira.relation_delta,
            torch.zeros_like(out.hira.relation_delta),
        )
        assert torch.equal(out.hira.logits, out.hira.coarse_logits)
        assert int(out.hira.candidate_budget.item()) == 2
        assert bool(out.hira.selected_mask.all())
        assert torch.isclose(
            out.probabilities.sum(),
            out.probabilities.new_tensor(1.0),
            atol=1e-6,
        )

    assert model.state_encode_calls == after_compile
    assert preds[0] == preds[1] == preds[2]


def test_w31_adapter_checkpoint_roundtrip():
    torch.manual_seed(31105)
    scorer = DualAdapterSymmetricSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=8,
    )
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    state = {
        "state_adapter.down.weight": torch.randn(8, 128),
        "state_adapter.up.weight": torch.randn(128, 8),
        "schema_adapter.down.weight": torch.randn(8, 128),
        "schema_adapter.up.weight": torch.randn(128, 8),
    }
    scorer.load_adapter_state_dict(state, freeze=True)

    assert scorer.trainable_parameter_count == 0
    own = scorer.state_dict()
    for key, expected in state.items():
        assert torch.equal(own[key], expected)
