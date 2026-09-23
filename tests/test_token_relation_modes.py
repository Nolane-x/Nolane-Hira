import pytest
import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore, count_parameters
from nmd.runtime import NolaneHira, RELATION_MODES
from nmd.semantic import TrainableSemanticEncoder


def make_model():
    torch.manual_seed(7)
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    encoder.eval()
    hira = HIRACore(d_model=256, dropout=0.0)
    hira.eval()
    return NolaneHira(encoder, hira)


def options():
    return (
        LogicalOption(
            "route-a",
            "color amber; role pilot; operation inspect; site garden",
        ),
        LogicalOption(
            "route-b",
            "color violet; role baker; operation measure; site plaza",
        ),
        LogicalOption(
            "route-c",
            "color teal; role mason; operation sort; site bridge",
        ),
    )


def test_schema_token_artifacts_are_opt_in_and_cache_separated():
    model = make_model()
    kwargs = dict(
        primitive="choice",
        question_text="Which route matches every requested field?",
        options=options(),
        use_cache=True,
    )

    pooled, pooled_receipt = model.compile_schema(
        **kwargs,
        include_token_artifacts=False,
    )
    tokenized, token_receipt = model.compile_schema(
        **kwargs,
        include_token_artifacts=True,
    )
    pooled_again, pooled_again_receipt = model.compile_schema(
        **kwargs,
        include_token_artifacts=False,
    )
    tokenized_again, tokenized_again_receipt = model.compile_schema(
        **kwargs,
        include_token_artifacts=True,
    )

    assert pooled.schema_hash == tokenized.schema_hash
    assert pooled_receipt.schema_hash == token_receipt.schema_hash
    assert pooled.option_token_embeddings is None
    assert pooled.option_token_mask is None

    assert tokenized.option_token_embeddings is not None
    assert tokenized.option_token_mask is not None
    assert tokenized.option_token_embeddings.ndim == 3
    assert tokenized.option_token_embeddings.shape[0] == 3
    assert tokenized.option_token_embeddings.shape[-1] == 256
    assert tokenized.option_token_mask.shape == (
        tokenized.option_token_embeddings.shape[0],
        tokenized.option_token_embeddings.shape[1],
    )
    assert tokenized.option_token_mask.dtype == torch.bool

    assert pooled_again is pooled
    assert tokenized_again is tokenized
    assert pooled_again_receipt.cache_hit is True
    assert tokenized_again_receipt.cache_hit is True


def test_default_runtime_is_exactly_explicit_pooled_mode():
    model = make_model()
    state = (
        "Dispatch says the pilot must inspect the garden "
        "while using the amber color marker."
    )
    memory = model.compile_state(state)
    schema, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches every requested field?",
        options=options(),
        include_token_artifacts=True,
        use_cache=False,
    )

    default = model.forward_compiled(
        memory,
        schema,
        forced_budget=255,
    )
    pooled = model.forward_compiled(
        memory,
        schema,
        forced_budget=255,
        relation_mode="pooled",
    )

    assert torch.equal(default.logits, pooled.logits)
    assert torch.equal(default.probabilities, pooled.probabilities)
    assert int(default.hira.candidate_budget.item()) == 3


def test_all_relation_modes_execute_full_k_without_new_parameters():
    model = make_model()
    assert count_parameters(model.hira) == 422_159

    memory = model.compile_state(
        "At the garden, the pilot will inspect under the amber marker."
    )
    schema, _ = model.compile_schema(
        primitive="choice",
        question_text="Select the route matching all four attributes.",
        options=options(),
        include_token_artifacts=True,
        use_cache=False,
    )

    seen = {}
    for mode in RELATION_MODES:
        out = model.forward_compiled(
            memory,
            schema,
            forced_budget=255,
            relation_mode=mode,
        )
        assert out.probabilities.shape == (3,)
        assert torch.isfinite(out.probabilities).all()
        assert torch.isclose(
            out.probabilities.sum(),
            out.probabilities.new_tensor(1.0),
            atol=1e-6,
        )
        assert int(out.hira.candidate_budget.item()) == 3
        seen[mode] = out.logits.detach().clone()

    assert set(seen) == set(RELATION_MODES)


def test_option_token_modes_fail_closed_without_token_artifacts():
    model = make_model()
    memory = model.compile_state(
        "The amber pilot inspects the garden."
    )
    schema, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches?",
        options=options(),
        include_token_artifacts=False,
        use_cache=False,
    )

    for mode in ("option_tokens", "dual_tokens"):
        with pytest.raises(
            ValueError,
            match="requires schema token artifacts",
        ):
            model.forward_compiled(
                memory,
                schema,
                relation_mode=mode,
            )


def test_state_token_modes_require_state_token_memory():
    model = make_model()
    memory = model.compile_state(
        "The amber pilot inspects the garden."
    )
    tokenless = type(memory)(
        model_hash=memory.model_hash,
        tokenizer_hash=memory.tokenizer_hash,
        state_hash=memory.state_hash,
        global_embedding=memory.global_embedding,
        segment_embeddings=memory.segment_embeddings,
        token_embeddings=None,
    )
    schema, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches?",
        options=options(),
        include_token_artifacts=True,
        use_cache=False,
    )

    for mode in ("state_tokens", "dual_tokens"):
        with pytest.raises(
            ValueError,
            match="StateMemory.token_embeddings is unavailable",
        ):
            model.forward_compiled(
                tokenless,
                schema,
                relation_mode=mode,
            )


def test_decide_text_compiles_option_tokens_only_when_needed():
    model = make_model()

    pooled = model.decide_text(
        "The amber pilot inspects the garden.",
        primitive="choice",
        question_text="Which route matches?",
        options=options(),
        relation_mode="pooled",
        forced_budget=255,
    )
    dual = model.decide_text(
        "The amber pilot inspects the garden.",
        primitive="choice",
        question_text="Which route matches?",
        options=options(),
        relation_mode="dual_tokens",
        forced_budget=255,
    )

    assert pooled.probabilities.shape == dual.probabilities.shape == (3,)
    assert torch.isfinite(dual.probabilities).all()
