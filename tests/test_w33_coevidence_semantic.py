import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.symmetric_semantic import SymmetricSemanticScorer
from nmd.w33_coevidence_semantic import CoEvidenceSemanticScorer


def _inputs():
    torch.manual_seed(33101)
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


def test_w33_candidate_has_exact_capacity():
    scorer = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    assert scorer.adapter_parameter_count == 4096
    assert scorer.interaction_parameter_count == 2048
    assert scorer.composition_parameter_count == 2048
    assert scorer.candidate_parameter_count == 8192
    assert scorer.candidate_trainable_parameter_count == 8192
    assert scorer.trainable_parameter_count == 8192
    assert scorer.projection.weight.requires_grad is False

    scorer.freeze_candidate()
    assert scorer.candidate_trainable_parameter_count == 0
    assert scorer.trainable_parameter_count == 0


def test_w33_zero_init_is_exact_unbridged_t0():
    torch.manual_seed(33102)
    weight = torch.randn(128, 256)

    baseline = SymmetricSemanticScorer(d_model=256, d_rel=128)
    baseline.load_projection_weight(weight, freeze=True)

    candidate = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    candidate.load_projection_weight(weight, freeze=True)

    args = _inputs()
    assert torch.equal(candidate(**args), baseline(**args))


def test_w33_coevidence_uses_second_distinct_state_match():
    # [B=1,K=1,V=1,T=2,S=4]
    composition = torch.tensor([[[[[
        0.90, 0.40, 0.10, -0.20
    ], [
        0.80, 0.70, 0.20, -0.10
    ]]]]])
    state_mask = torch.tensor([[True, True, True, False]])
    option_token_mask = torch.tensor([[[[True, True]]]])
    view_mask = torch.tensor([[[True]]])

    second = CoEvidenceSemanticScorer._second_distinct_state_support(
        composition,
        state_mask=state_mask,
        option_view_token_mask=option_token_mask,
        option_view_mask=view_mask,
    )

    # Per-state support after max over option tokens is [0.90, 0.70, 0.20].
    # The second strongest distinct state-token support is 0.70.
    assert torch.allclose(second, torch.tensor([[[0.70]]]))


def test_w33_composition_path_changes_logits_after_activation():
    torch.manual_seed(33103)
    scorer = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    args = _inputs()
    baseline = scorer(**args)

    with torch.no_grad():
        scorer.composition_state.weight.normal_(mean=0.0, std=0.03)

    changed = scorer(**args)
    assert not torch.equal(changed, baseline)


def _options():
    return (
        LogicalOption(
            "not-critical",
            "the combined immediate condition is absent",
            aliases=("multiple required evidence pieces are not jointly present",),
            exemplars=("the case does not satisfy the complete conjunction",),
            value=0.0,
        ),
        LogicalOption(
            "critical",
            "the combined immediate condition is present",
            aliases=("multiple required evidence pieces are jointly present",),
            exemplars=("the case satisfies the complete conjunction",),
            value=1.0,
        ),
    )


def _model():
    torch.manual_seed(33104)
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    return NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        coevidence_symmetric_semantic_scorer=scorer,
    ).eval()


def test_w33_runtime_is_typed_full_k_state_once_and_relation_free():
    model = _model()
    memory = model.compile_state(
        "both the immediate timing requirement and the severe delay consequence are present"
    )
    after_compile = model.state_encode_calls
    preds = []

    for primitive in ("choice", "score", "noul"):
        schema, _ = model.compile_schema(
            primitive=primitive,
            question_text="Is the complete critical conjunction present?",
            options=_options(),
            include_token_artifacts=True,
            use_cache=True,
        )
        out = model.forward_compiled(
            memory,
            schema,
            coarse_mode="coevidence_symmetric_semantic",
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


def test_w33_candidate_checkpoint_roundtrip():
    torch.manual_seed(33105)
    scorer = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    state = {
        "state_adapter.down.weight": torch.randn(8, 128),
        "state_adapter.up.weight": torch.randn(128, 8),
        "schema_adapter.down.weight": torch.randn(8, 128),
        "schema_adapter.up.weight": torch.randn(128, 8),
        "interaction_state.weight": torch.randn(8, 128),
        "interaction_schema.weight": torch.randn(8, 128),
        "composition_state.weight": torch.randn(8, 128),
        "composition_schema.weight": torch.randn(8, 128),
    }
    scorer.load_candidate_state_dict(state, freeze=True)
    assert scorer.trainable_parameter_count == 0
    own = scorer.state_dict()
    for key, expected in state.items():
        assert torch.equal(own[key], expected)
