import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.w33_coevidence_semantic import CoEvidenceSemanticScorer
from nmd.v1_query_conditioned_semantic import QueryConditionedCoEvidenceScorer


def _inputs():
    torch.manual_seed(41001)
    return dict(
        state_tokens=torch.randn(2, 6, 256),
        state_mask=torch.tensor([
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]),
        question_tokens=torch.randn(2, 4, 256),
        question_mask=torch.tensor([
            [True, True, True, False],
            [True, True, True, True],
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


def _base_args(args):
    return {
        "state_tokens": args["state_tokens"],
        "state_mask": args["state_mask"],
        "option_view_tokens": args["option_view_tokens"],
        "option_view_token_mask": args["option_view_token_mask"],
        "option_view_mask": args["option_view_mask"],
    }


def test_v1_qcce_candidate_has_bounded_capacity():
    scorer = QueryConditionedCoEvidenceScorer(
        d_model=256,
        d_rel=128,
        rank=8,
        query_rank=8,
    )
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    assert scorer.adapter_parameter_count == 4096
    assert scorer.interaction_parameter_count == 2048
    assert scorer.composition_parameter_count == 2048
    assert scorer.query_parameter_count == 3072
    assert scorer.candidate_parameter_count == 11264
    assert scorer.candidate_trainable_parameter_count == 11264
    assert scorer.trainable_parameter_count == 11264
    assert scorer.projection.weight.requires_grad is False

    scorer.freeze_candidate()
    assert scorer.candidate_trainable_parameter_count == 0
    assert scorer.trainable_parameter_count == 0


def test_v1_qcce_zero_init_is_exact_w34_identity():
    torch.manual_seed(41002)
    weight = torch.randn(128, 256)

    w34 = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    w34.load_projection_weight(weight, freeze=True)

    v1 = QueryConditionedCoEvidenceScorer(
        d_model=256,
        d_rel=128,
        rank=8,
        query_rank=8,
    )
    v1.load_projection_weight(weight, freeze=True)

    args = _inputs()
    expected = w34(**_base_args(args))
    observed = v1(**args)
    assert torch.equal(observed, expected)


def test_v1_qcce_question_changes_logits_after_query_activation():
    torch.manual_seed(41003)
    scorer = QueryConditionedCoEvidenceScorer(
        d_model=256,
        d_rel=128,
        rank=8,
        query_rank=8,
    )
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    with torch.no_grad():
        scorer.query_state.weight.normal_(mean=0.0, std=0.05)
        scorer.query_schema.weight.normal_(mean=0.0, std=0.05)

    args = _inputs()
    first = scorer(**args)

    changed = dict(args)
    changed["question_tokens"] = args["question_tokens"].flip(1).clone()
    # A pure token-order flip can preserve masked mean. Change the actual query
    # content too, while leaving state and options exactly fixed.
    changed["question_tokens"][:, 0] += 1.75
    second = scorer(**changed)

    assert not torch.equal(first, second)
    assert float((first - second).abs().max()) > 1e-7


def test_v1_qcce_option_permutation_is_equivariant():
    torch.manual_seed(41004)
    scorer = QueryConditionedCoEvidenceScorer(
        d_model=256,
        d_rel=128,
        rank=8,
        query_rank=8,
    )
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    with torch.no_grad():
        scorer.query_state.weight.normal_(mean=0.0, std=0.03)
        scorer.query_schema.weight.normal_(mean=0.0, std=0.03)

    args = _inputs()
    direct = scorer(**args)

    permutation = torch.tensor([2, 0, 1])
    permuted_args = dict(args)
    permuted_args["option_view_tokens"] = args["option_view_tokens"][:, permutation]
    permuted_args["option_view_token_mask"] = args["option_view_token_mask"][:, permutation]
    permuted_args["option_view_mask"] = args["option_view_mask"][:, permutation]
    permuted = scorer(**permuted_args)

    assert torch.allclose(
        permuted,
        direct[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )


def test_v1_qcce_rejects_empty_question():
    scorer = QueryConditionedCoEvidenceScorer()
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    args = _inputs()
    args["question_mask"] = torch.zeros_like(args["question_mask"])
    try:
        scorer(**args)
    except ValueError as exc:
        assert "question" in str(exc)
    else:
        raise AssertionError("empty question mask must fail closed")


def _options():
    return (
        LogicalOption(
            "red",
            "the answer is red",
            aliases=("red is the requested attribute",),
            value=0.0,
        ),
        LogicalOption(
            "tuesday",
            "the answer is Tuesday",
            aliases=("Tuesday is the requested attribute",),
            value=1.0,
        ),
        LogicalOption(
            "unknown",
            "the requested fact is unknown",
            aliases=("the state does not determine the requested fact",),
            value=2.0,
        ),
    )


def _runtime():
    torch.manual_seed(41005)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = QueryConditionedCoEvidenceScorer(
        d_model=256,
        d_rel=128,
        rank=8,
        query_rank=8,
    )
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    with torch.no_grad():
        scorer.query_state.weight.normal_(mean=0.0, std=0.03)
        scorer.query_schema.weight.normal_(mean=0.0, std=0.03)
    return NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        query_conditioned_coevidence_scorer=scorer,
    ).eval()


def test_v1_runtime_consumes_question_tokens_full_k_and_state_once():
    model = _runtime()
    before = model.state_encode_calls
    memory = model.compile_state(
        "Mira bought a red bicycle on Tuesday."
    )
    assert model.state_encode_calls - before == 1

    schemas = []
    for question in (
        "What color was the bicycle?",
        "On which day was the bicycle bought?",
    ):
        schema, _ = model.compile_schema(
            primitive="choice",
            question_text=question,
            options=_options(),
            include_token_artifacts=True,
            use_cache=False,
        )
        schemas.append(schema)

    first = model.forward_compiled(
        memory,
        schemas[0],
        coarse_mode="query_conditioned_coevidence",
    )
    second = model.forward_compiled(
        memory,
        schemas[1],
        coarse_mode="query_conditioned_coevidence",
    )

    assert model.state_encode_calls - before == 1
    assert int(first.hira.candidate_budget.item()) == 3
    assert int(second.hira.candidate_budget.item()) == 3
    assert bool(first.hira.selected_mask.all())
    assert bool(second.hira.selected_mask.all())
    assert torch.equal(
        first.hira.relation_delta,
        torch.zeros_like(first.hira.relation_delta),
    )
    assert torch.isclose(first.probabilities.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(second.probabilities.sum(), torch.tensor(1.0), atol=1e-6)
    assert not torch.equal(first.logits, second.logits)


def test_v1_runtime_option_identity_is_order_invariant():
    model = _runtime()
    memory = model.compile_state("Mira bought a red bicycle on Tuesday.")
    base = _options()
    flipped = tuple(reversed(base))

    schema_a, _ = model.compile_schema(
        primitive="choice",
        question_text="What color was the bicycle?",
        options=base,
        include_token_artifacts=True,
        use_cache=False,
    )
    schema_b, _ = model.compile_schema(
        primitive="choice",
        question_text="What color was the bicycle?",
        options=flipped,
        include_token_artifacts=True,
        use_cache=False,
    )
    out_a = model.forward_compiled(
        memory,
        schema_a,
        coarse_mode="query_conditioned_coevidence",
    )
    out_b = model.forward_compiled(
        memory,
        schema_b,
        coarse_mode="query_conditioned_coevidence",
    )

    selected_a = schema_a.options[int(out_a.probabilities.argmax())].option_id
    selected_b = schema_b.options[int(out_b.probabilities.argmax())].option_id
    assert selected_a == selected_b
    assert torch.allclose(
        out_b.probabilities,
        out_a.probabilities.flip(0),
        atol=1e-6,
        rtol=1e-6,
    )
