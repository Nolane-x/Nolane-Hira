import random

import torch

from nmd.hira import HIRACore, count_parameters
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.semantic_routing_curriculum import (
    TRAIN_MATERIALS,
    TRAIN_ENTITIES,
    TRAIN_ACTIONS as W5A_TRAIN_ACTIONS,
    TRAIN_LOCATIONS,
    CONFIRM_SYMBOLS,
    CONFIRM_ITEMS,
    CONFIRM_ACTIONS as W5A_CONFIRM_ACTIONS,
    CONFIRM_LOCATIONS,
)
from nmd.semantic_token_curriculum import (
    CONFIRM_ACTIONS,
    CONFIRM_K_COUNTS,
    CONFIRM_ROLES,
    CONFIRM_SITES,
    CONFIRM_TAGS,
    DEV_K_COUNTS,
    RELATION_MODES,
    TRAIN_ACTIONS,
    TRAIN_K_COUNTS,
    TRAIN_ROLES,
    TRAIN_SITES,
    TRAIN_TAGS,
    compile_token_cache,
    evaluate_token_cache,
    generate_token_authority,
    generate_token_case,
    token_case_loss,
    validate_token_cache,
)


def make_model():
    torch.manual_seed(17)
    encoder = TrainableSemanticEncoder(
        vocab_size=4096,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=96,
    )
    encoder.eval()
    hira = HIRACore(d_model=256, dropout=0.0)
    hira.eval()
    return NolaneHira(encoder, hira)


def test_w5b_authority_counts_and_k_distributions_are_frozen():
    expected = {
        "train": (TRAIN_K_COUNTS, 512),
        "dev": (DEV_K_COUNTS, 160),
        "confirm": (CONFIRM_K_COUNTS, 192),
    }
    for split, (counts, total) in expected.items():
        rows = generate_token_authority(split)
        assert len(rows) == total
        assert len({row.case_id for row in rows}) == total
        observed = {
            k: sum(row.k == k for row in rows)
            for k in counts
        }
        assert observed == counts
        assert all(row.split == split for row in rows)
        assert all(len(row.options) == row.k for row in rows)
        assert all(
            0 <= row.gold_index < row.k
            for row in rows
        )


def test_w5b_vocab_is_disjoint_from_w5a_vocab():
    w5a = set().union(
        TRAIN_MATERIALS,
        TRAIN_ENTITIES,
        W5A_TRAIN_ACTIONS,
        TRAIN_LOCATIONS,
        CONFIRM_SYMBOLS,
        CONFIRM_ITEMS,
        W5A_CONFIRM_ACTIONS,
        CONFIRM_LOCATIONS,
    )
    w5b_train = set().union(
        TRAIN_TAGS,
        TRAIN_ROLES,
        TRAIN_ACTIONS,
        TRAIN_SITES,
    )
    w5b_confirm = set().union(
        CONFIRM_TAGS,
        CONFIRM_ROLES,
        CONFIRM_ACTIONS,
        CONFIRM_SITES,
    )

    assert not (w5a & w5b_train)
    assert not (w5a & w5b_confirm)
    assert not (w5b_train & w5b_confirm)


def test_generated_case_uses_opaque_ids_and_unique_signatures_at_k255():
    rng = random.Random(61001)
    case = generate_token_case(
        split="train",
        k=255,
        case_index=0,
        rng=rng,
        template_id="train-record",
    )

    assert len(case.options) == 255
    assert len({option.option_id for option in case.options}) == 255
    assert len({option.criterion_text for option in case.options}) == 255
    assert all(
        option.option_id.startswith("route-")
        for option in case.options
    )
    assert all(
        option.option_id not in option.criterion_text
        for option in case.options
    )


def test_token_cache_contains_state_and_option_token_artifacts():
    model = make_model()
    rng = random.Random(123)
    cases = [
        generate_token_case(
            split="train",
            k=8,
            case_index=i,
            rng=rng,
            template_id="train-fields",
        )
        for i in range(2)
    ]

    cache = compile_token_cache(model, cases)
    validate_token_cache(cache, expected_split="train")

    assert cache["case_count"] == 2
    assert cache["state_encode_calls"] == 2
    assert cache["state_encode_calls_per_case"] == 1.0

    row = cache["cases"][0]
    assert row["state_segments"].shape[-1] == 256
    assert row["state_tokens"].shape[-1] == 256
    assert row["option_embeddings"].shape == (8, 256)
    assert row["option_tokens"].ndim == 3
    assert row["option_tokens"].shape[0] == 8
    assert row["option_tokens"].shape[-1] == 256
    assert row["option_token_mask"].shape == row["option_tokens"].shape[:2]
    assert row["option_token_mask"].dtype == torch.bool


def test_all_four_modes_are_full_k_finite_and_parameter_neutral():
    model = make_model()
    assert count_parameters(model.hira) == 422_159

    rng = random.Random(321)
    cases = [
        generate_token_case(
            split="dev",
            k=8,
            case_index=0,
            rng=rng,
            template_id="dev-note",
        )
    ]
    cache = compile_token_cache(model, cases)

    for mode in RELATION_MODES:
        metrics = evaluate_token_cache(
            model.hira,
            cache,
            relation_mode=mode,
        )
        assert metrics["case_count"] == 1
        assert metrics["candidate_budget_min"] == 8
        assert metrics["candidate_budget_max"] == 8
        assert metrics["probability_mass_max_error"] <= 1e-6
        assert torch.isfinite(
            torch.tensor(metrics["hard_brier"])
        )


def test_token_modes_reach_their_unique_parameters_with_gradients():
    model = make_model()
    model.hira.train()
    rng = random.Random(777)
    case = generate_token_case(
        split="train",
        k=8,
        case_index=0,
        rng=rng,
        template_id="train-order",
    )
    cache = compile_token_cache(model, [case])
    row = cache["cases"][0]

    for mode in ("option_tokens", "dual_tokens"):
        model.hira.zero_grad(set_to_none=True)
        loss = token_case_loss(
            model.hira,
            row,
            relation_mode=mode,
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert model.hira.token_proj.weight.grad is not None
        first = model.hira.option_token_weight[0].weight
        assert first.grad is not None


def test_pooled_mode_does_not_use_option_token_parameters():
    model = make_model()
    model.hira.train()
    rng = random.Random(888)
    case = generate_token_case(
        split="train",
        k=8,
        case_index=0,
        rng=rng,
        template_id="train-report",
    )
    cache = compile_token_cache(model, [case])
    row = cache["cases"][0]

    model.hira.zero_grad(set_to_none=True)
    loss = token_case_loss(
        model.hira,
        row,
        relation_mode="pooled",
    )
    loss.backward()

    assert model.hira.token_proj.weight.grad is None
    assert model.hira.option_token_weight[0].weight.grad is None
