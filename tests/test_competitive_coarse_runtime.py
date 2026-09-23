import math

import pytest
import torch

from nmd.competitive import (
    CompetitiveCoarseScorer,
    candidate_relative_idf,
    count_competitive_parameters,
)
from nmd.contracts import LogicalOption
from nmd.hira import HIRACore, count_parameters
from nmd.losses import LossWeights
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.training import DecisionExample, loss_example
from nmd.typed_decisions import (
    TypedDecision,
    TypedDecisionCase,
    execute_typed_case,
)


def make_model(*, scorer: bool = True) -> NolaneHira:
    torch.manual_seed(7001)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=96,
    )
    hira = HIRACore(d_model=256, dropout=0.0)
    competitive = (
        CompetitiveCoarseScorer(d_model=256)
        if scorer
        else None
    )
    return NolaneHira(
        encoder,
        hira,
        coarse_scorer=competitive,
    )


def options():
    return (
        LogicalOption(
            "opaque-a",
            "amber pilot inspect garden",
        ),
        LogicalOption(
            "opaque-b",
            "violet baker measure plaza",
        ),
        LogicalOption(
            "opaque-c",
            "teal mason sort bridge",
        ),
    )


def test_w6_does_not_change_default_hira_parameter_contract():
    legacy = HIRACore(d_model=256, dropout=0.0)
    assert count_parameters(legacy) == 422_159

    state = legacy.state_dict()
    restored = HIRACore(d_model=256, dropout=0.0)
    restored.load_state_dict(state, strict=True)
    assert count_parameters(restored) == 422_159


def test_competitive_scorer_has_promoted_w5h_capacity():
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    assert count_competitive_parameters(scorer) == 256 * 128 + 1
    assert scorer.projection.bias is None


def test_candidate_relative_idf_is_candidate_relative_and_mean_normalized():
    ids = torch.tensor([[
        [10, 20, 101, 0],
        [10, 20, 102, 0],
        [10, 20, 103, 0],
    ]], dtype=torch.long)
    mask = torch.tensor([[
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False],
    ]])
    weights = candidate_relative_idf(ids, mask)
    assert weights.shape == ids.shape
    for row in range(3):
        valid = mask[0, row]
        assert torch.isclose(
            weights[0, row][valid].mean(),
            torch.tensor(1.0),
            atol=1e-6,
        )
        assert weights[0, row, 2] > weights[0, row, 0]
        assert weights[0, row, 2] > weights[0, row, 1]
        assert weights[0, row, 3] == 0


def test_schema_exposes_competitive_artifacts_only_when_requested():
    model = make_model()
    model.eval()

    pooled, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches every field?",
        options=options(),
        include_token_artifacts=False,
        use_cache=False,
    )
    competitive, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches every field?",
        options=options(),
        include_token_artifacts=True,
        use_cache=False,
    )

    assert pooled.question_token_embeddings is None
    assert pooled.option_token_ids is None
    assert pooled.option_content_token_mask is None

    assert competitive.question_token_embeddings is not None
    assert competitive.question_content_token_mask is not None
    assert competitive.option_token_embeddings is not None
    assert competitive.option_token_ids is not None
    assert competitive.option_content_token_mask is not None
    assert competitive.option_token_ids.dtype == torch.long
    assert competitive.option_content_token_mask.dtype == torch.bool
    assert (
        competitive.option_content_token_mask.sum(-1) >= 1
    ).all()


def test_state_memory_preserves_legacy_tokens_and_adds_content_tokens():
    model = make_model()
    memory = model.compile_state(
        "the amber pilot must inspect the garden"
    )
    assert memory.token_embeddings is not None
    assert memory.content_token_embeddings is not None
    assert memory.content_token_embeddings.shape[-1] == 256
    assert (
        memory.content_token_embeddings.shape[0]
        < memory.token_embeddings.shape[0]
    )


def test_default_runtime_remains_exact_legacy_with_scorer_attached():
    model = make_model()
    model.eval()
    memory = model.compile_state(
        "the amber pilot must inspect the garden"
    )
    schema, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches every field?",
        options=options(),
        include_token_artifacts=True,
        use_cache=False,
    )
    default = model.forward_compiled(
        memory,
        schema,
        forced_budget=255,
    )
    explicit = model.forward_compiled(
        memory,
        schema,
        forced_budget=255,
        coarse_mode="legacy",
    )
    assert torch.equal(default.logits, explicit.logits)
    assert torch.equal(default.probabilities, explicit.probabilities)
    assert torch.equal(
        default.hira.coarse_logits,
        explicit.hira.coarse_logits,
    )


def test_competitive_mode_injects_exact_full_k_coarse_override_state_once():
    model = make_model()
    model.eval()
    before = model.state_encode_calls
    memory = model.compile_state(
        "the amber pilot must inspect the garden"
    )
    schema, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches every field?",
        options=options(),
        include_token_artifacts=True,
        use_cache=False,
    )
    expected = model._competitive_coarse(memory, schema)[0]
    out = model.forward_compiled(
        memory,
        schema,
        forced_budget=255,
        coarse_mode="competitive",
    )

    assert model.state_encode_calls - before == 1
    assert torch.allclose(out.hira.coarse_logits, expected, atol=0, rtol=0)
    assert torch.isfinite(out.probabilities).all()
    assert torch.isclose(
        out.probabilities.sum(),
        out.probabilities.new_tensor(1.0),
        atol=1e-6,
    )
    assert int(out.hira.candidate_budget.item()) == 3


def test_decide_text_competitive_is_state_once_and_auto_compiles_artifacts():
    model = make_model()
    model.eval()
    before = model.state_encode_calls
    out = model.decide_text(
        "the amber pilot must inspect the garden",
        primitive="choice",
        question_text="Which route matches every field?",
        options=options(),
        forced_budget=255,
        coarse_mode="competitive",
    )
    assert model.state_encode_calls - before == 1
    assert out.probabilities.shape == (3,)
    assert torch.isclose(
        out.probabilities.sum(),
        out.probabilities.new_tensor(1.0),
        atol=1e-6,
    )


def test_competitive_mode_fails_closed_without_scorer_or_artifacts():
    no_scorer = make_model(scorer=False)
    no_scorer.eval()
    memory = no_scorer.compile_state(
        "the amber pilot must inspect the garden"
    )
    schema, _ = no_scorer.compile_schema(
        primitive="choice",
        question_text="Which route matches?",
        options=options(),
        include_token_artifacts=True,
        use_cache=False,
    )
    with pytest.raises(ValueError, match="requires CompetitiveCoarseScorer"):
        no_scorer.forward_compiled(
            memory,
            schema,
            coarse_mode="competitive",
        )

    model = make_model()
    model.eval()
    memory = model.compile_state(
        "the amber pilot must inspect the garden"
    )
    tokenless_schema, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches?",
        options=options(),
        include_token_artifacts=False,
        use_cache=False,
    )
    with pytest.raises(ValueError, match="requires schema token artifacts"):
        model.forward_compiled(
            memory,
            tokenless_schema,
            coarse_mode="competitive",
        )


def test_competitive_mode_fails_closed_without_state_content_tokens():
    model = make_model()
    model.eval()
    memory = model.compile_state(
        "the amber pilot must inspect the garden"
    )
    tokenless = type(memory)(
        model_hash=memory.model_hash,
        tokenizer_hash=memory.tokenizer_hash,
        state_hash=memory.state_hash,
        global_embedding=memory.global_embedding,
        segment_embeddings=memory.segment_embeddings,
        token_embeddings=memory.token_embeddings,
        content_token_embeddings=None,
    )
    schema, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches?",
        options=options(),
        include_token_artifacts=True,
        use_cache=False,
    )
    with pytest.raises(ValueError, match="requires state content tokens"):
        model.forward_compiled(
            tokenless,
            schema,
            coarse_mode="competitive",
        )


def test_opaque_option_ids_do_not_change_competitive_semantics():
    model = make_model()
    model.eval()
    memory = model.compile_state(
        "the amber pilot must inspect the garden"
    )
    a = options()
    b = tuple(
        LogicalOption(f"renamed-{i}", option.criterion_text)
        for i, option in enumerate(a)
    )
    sa, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches?",
        options=a,
        include_token_artifacts=True,
        use_cache=False,
    )
    sb, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches?",
        options=b,
        include_token_artifacts=True,
        use_cache=False,
    )
    oa = model.forward_compiled(
        memory,
        sa,
        forced_budget=255,
        coarse_mode="competitive",
    )
    ob = model.forward_compiled(
        memory,
        sb,
        forced_budget=255,
        coarse_mode="competitive",
    )
    assert sa.schema_hash != sb.schema_hash
    assert torch.equal(oa.hira.coarse_logits, ob.hira.coarse_logits)
    assert torch.equal(oa.probabilities, ob.probabilities)


def test_option_order_only_permutes_competitive_outputs():
    model = make_model()
    model.eval()
    memory = model.compile_state(
        "the amber pilot must inspect the garden"
    )
    base = options()
    permutation = (2, 0, 1)
    permuted = tuple(base[i] for i in permutation)
    sa, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches?",
        options=base,
        include_token_artifacts=True,
        use_cache=False,
    )
    sb, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches?",
        options=permuted,
        include_token_artifacts=True,
        use_cache=False,
    )
    oa = model.forward_compiled(
        memory,
        sa,
        forced_budget=255,
        coarse_mode="competitive",
    )
    ob = model.forward_compiled(
        memory,
        sb,
        forced_budget=255,
        coarse_mode="competitive",
    )
    expected_coarse = oa.hira.coarse_logits[
        torch.tensor(permutation)
    ]
    expected_probabilities = oa.probabilities[
        torch.tensor(permutation)
    ]
    assert torch.allclose(
        ob.hira.coarse_logits,
        expected_coarse,
        atol=1e-6,
    )
    assert torch.allclose(
        ob.probabilities,
        expected_probabilities,
        atol=1e-6,
    )


def test_competitive_training_reaches_scorer_encoder_and_relation_head():
    model = make_model()
    model.train()
    example = DecisionExample(
        state_text="the amber pilot must inspect the garden",
        primitive="choice",
        question_text="Which route matches every field?",
        options=options(),
        gold_index=0,
    )
    loss, _ = loss_example(
        model,
        example,
        weights=LossWeights(hard_ce=1.0, brier=0.1),
        forced_budget=255,
        coarse_mode="competitive",
    )
    loss.backward()

    assert math.isfinite(float(loss.detach()))
    assert model.coarse_scorer is not None
    assert model.coarse_scorer.projection.weight.grad is not None
    assert model.coarse_scorer.log_scale.grad is not None
    assert model.encoder.embedding.weight.grad is not None
    assert model.hira.cross_score[0].weight.grad is not None


def test_typed_competitive_execution_reuses_one_state_encode():
    model = make_model()
    model.eval()
    opts = options()
    case = TypedDecisionCase(
        case_id="w6-contract",
        workflow="w6",
        state_text="the amber pilot must inspect the garden",
        decisions=(
            TypedDecision(
                question_id="route",
                primitive="choice",
                question_text="Which route matches every field?",
                options=opts,
                gold_index=0,
                gold_probabilities=(1.0, 0.0, 0.0),
            ),
            TypedDecision(
                question_id="alternate",
                primitive="choice",
                question_text="Which route describes the teal mason?",
                options=opts,
                gold_index=2,
                gold_probabilities=(0.0, 0.0, 1.0),
            ),
        ),
    )
    before = model.state_encode_calls
    execution = execute_typed_case(
        model,
        case,
        forced_budget=255,
        coarse_mode="competitive",
        use_schema_cache=False,
    )
    assert model.state_encode_calls - before == 1
    assert execution.receipt.state_encode_calls == 1
    assert execution.receipt.decision_count == 2
    for out in execution.outputs:
        assert torch.isclose(
            out.probabilities.sum(),
            out.probabilities.new_tensor(1.0),
            atol=1e-6,
        )
