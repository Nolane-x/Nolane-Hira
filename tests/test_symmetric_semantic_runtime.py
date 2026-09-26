import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.semantic_core import (
    build_hira_v0_semantic_core,
    load_rescued_projection_checkpoint,
)
from nmd.symmetric_semantic import (
    SymmetricSemanticScorer,
    count_symmetric_parameters,
)
from nmd.typed_competitive_cache import file_sha256


def _model() -> NolaneHira:
    torch.manual_seed(29001)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=96,
    )
    scorer = SymmetricSemanticScorer(d_model=256, d_rel=128)
    weight = torch.randn(128, 256) * 0.02
    scorer.load_projection_weight(weight, freeze=True)
    model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        symmetric_semantic_scorer=scorer,
    )
    model.eval()
    return model


def _binary_options():
    return (
        LogicalOption(
            "negative",
            "normal workflow remains intact",
            aliases=("ordinary operation continues",),
            exemplars=("users can complete the central task normally",),
            value=0.0,
        ),
        LogicalOption(
            "positive",
            "normal workflow is materially disrupted",
            aliases=(
                "users require a substantive workaround",
                "the expected operational route no longer works",
            ),
            exemplars=("users must materially adapt the activity",),
            value=1.0,
        ),
    )


def test_symmetric_scorer_has_only_rescued_projection_capacity():
    scorer = SymmetricSemanticScorer(d_model=256, d_rel=128)
    assert scorer.projection.bias is None
    assert count_symmetric_parameters(scorer) == 32768
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    assert scorer.trainable_parameter_count == 0


def test_symmetric_scorer_single_view_matches_reference_operator():
    torch.manual_seed(29002)
    scorer = SymmetricSemanticScorer(d_model=256, d_rel=128)
    weight = torch.randn(128, 256)
    scorer.load_projection_weight(weight, freeze=True)

    state = torch.randn(1, 5, 256)
    state_mask = torch.tensor([[True, True, True, False, False]])
    options = torch.randn(1, 2, 1, 4, 256)
    option_token_mask = torch.tensor(
        [[
            [[True, True, True, False]],
            [[True, True, False, False]],
        ]]
    )
    view_mask = torch.ones(1, 2, 1, dtype=torch.bool)

    out = scorer(
        state_tokens=state,
        state_mask=state_mask,
        option_view_tokens=options,
        option_view_token_mask=option_token_mask,
        option_view_mask=view_mask,
    )[0]

    def reference_one(option, mask):
        st = torch.nn.functional.normalize(state[0] @ weight.T, dim=-1)
        op = torch.nn.functional.normalize(option @ weight.T, dim=-1)
        sim = op @ st.T
        sim_valid = sim[:, state_mask[0]]
        o2s = sim_valid.max(dim=1).values[mask].mean()
        s2o = sim_valid[mask].max(dim=0).values.mean()
        return 0.5 * (o2s + s2o)

    expected = torch.stack(
        [
            reference_one(options[0, i, 0], option_token_mask[0, i, 0])
            for i in range(2)
        ]
    )
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_schema_compiles_positive_multiview_artifacts_and_cache_hits():
    model = _model()
    options = _binary_options()
    schema1, receipt1 = model.compile_schema(
        primitive="choice",
        question_text="Is normal workflow materially disrupted?",
        options=options,
        include_token_artifacts=True,
        use_cache=True,
    )
    schema2, receipt2 = model.compile_schema(
        primitive="choice",
        question_text="Is normal workflow materially disrupted?",
        options=options,
        include_token_artifacts=True,
        use_cache=True,
    )

    assert not receipt1.cache_hit
    assert receipt2.cache_hit
    assert schema1.schema_hash == schema2.schema_hash
    assert schema1.option_view_token_embeddings is not None
    assert schema1.option_view_token_mask is not None
    assert schema1.option_view_mask is not None
    assert schema1.option_view_mask.shape == (2, 4)
    assert schema1.option_view_mask[0].sum().item() == 3
    assert schema1.option_view_mask[1].sum().item() == 4
    assert (
        schema1.option_view_token_mask[
            schema1.option_view_mask
        ].sum(-1) >= 1
    ).all()

    # Legacy single-view artifacts remain criterion-only and available.
    assert schema1.option_token_embeddings is not None
    assert schema1.option_content_token_mask is not None
    assert schema1.option_token_embeddings.shape[0] == 2


def test_symmetric_runtime_is_coarse_only_and_state_once_across_typed_queries():
    model = _model()
    options = _binary_options()

    before = model.state_encode_calls
    memory = model.compile_state(
        "normal operation is materially disrupted and users require a substantive workaround"
    )
    assert model.state_encode_calls - before == 1

    argmaxes = []
    for primitive in ("choice", "score", "noul"):
        schema, _ = model.compile_schema(
            primitive=primitive,
            question_text="Is normal workflow materially disrupted?",
            options=options,
            include_token_artifacts=True,
            use_cache=True,
        )
        out = model.forward_compiled(
            memory,
            schema,
            coarse_mode="symmetric_semantic",
        )
        argmaxes.append(int(out.probabilities.argmax()))
        assert torch.equal(
            out.hira.relation_delta,
            torch.zeros_like(out.hira.relation_delta),
        )
        assert torch.equal(out.hira.logits, out.hira.coarse_logits)
        assert int(out.hira.candidate_budget.item()) == 2
        assert torch.isclose(
            out.probabilities.sum(),
            out.probabilities.new_tensor(1.0),
            atol=1e-6,
        )

    assert model.state_encode_calls - before == 1
    assert argmaxes[0] == argmaxes[1] == argmaxes[2]


def test_symmetric_runtime_option_identity_is_order_invariant():
    model = _model()
    memory = model.compile_state(
        "normal operation is materially disrupted and users require a substantive workaround"
    )
    base = _binary_options()
    flipped = (base[1], base[0])

    schema_a, _ = model.compile_schema(
        primitive="choice",
        question_text="Is normal workflow materially disrupted?",
        options=base,
        include_token_artifacts=True,
        use_cache=False,
    )
    schema_b, _ = model.compile_schema(
        primitive="choice",
        question_text="Is normal workflow materially disrupted?",
        options=flipped,
        include_token_artifacts=True,
        use_cache=False,
    )
    out_a = model.forward_compiled(
        memory,
        schema_a,
        coarse_mode="symmetric_semantic",
    )
    out_b = model.forward_compiled(
        memory,
        schema_b,
        coarse_mode="symmetric_semantic",
    )

    selected_a = schema_a.options[int(out_a.probabilities.argmax())].criterion_text
    selected_b = schema_b.options[int(out_b.probabilities.argmax())].criterion_text
    assert selected_a == selected_b
    assert torch.allclose(
        out_b.probabilities,
        out_a.probabilities.flip(0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_semantic_core_loader_verifies_checkpoint_and_freezes_projection(tmp_path):
    checkpoint_path = tmp_path / "candidate.pt"
    weight = torch.randn(128, 256)
    torch.save(
        {
            "schema_version": "r8-w28-candidate-checkpoint-v1",
            "candidate": "T0",
            "kind": "projection",
            "projection_weight": weight,
        },
        checkpoint_path,
    )
    digest = file_sha256(checkpoint_path)
    loaded = load_rescued_projection_checkpoint(
        checkpoint_path,
        expected_sha256=digest,
    )
    assert torch.equal(loaded, weight.float())

    encoder = TrainableSemanticEncoder(
        vocab_size=512,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    model = build_hira_v0_semantic_core(
        encoder,
        checkpoint_path,
        expected_sha256=digest,
        hira=HIRACore(d_model=256, dropout=0.0),
    )
    assert model.symmetric_semantic_scorer is not None
    assert model.symmetric_semantic_scorer.trainable_parameter_count == 0
    assert torch.equal(
        model.symmetric_semantic_scorer.projection.weight.detach().cpu(),
        weight.float(),
    )
