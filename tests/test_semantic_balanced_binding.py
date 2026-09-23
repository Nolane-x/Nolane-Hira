import math

import pytest
import torch

from nmd.semantic_balanced_binding import (
    CANDIDATES,
    CONFIRM_K_COUNTS,
    CONFIRM_SEED,
    DEV_K_COUNTS,
    DEV_SEED,
    TRAIN_K_COUNTS,
    TRAIN_SEED,
    BalancedBindingMatcher,
    _option_salience,
    all_w5h_vocab,
    competence_gates,
    confirm_verdict,
    dev_selection_key,
    evaluate_matcher,
    generate_binding_authority,
    matcher_case_loss,
    validate_binding_cache,
)


def synthetic_case(*, gold_index: int = 0, k: int = 3):
    d = 256
    state = torch.zeros(4, d)
    question = torch.zeros(3, d)
    state[1, 0] = 1.0
    state[2, 1] = 1.0
    question[1, 2] = 1.0
    state_mask = torch.tensor([False, True, True, False])
    question_mask = torch.tensor([False, True, False])

    options = torch.zeros(k, 5, d)
    option_mask = torch.tensor([[False, True, True, True, False]] * k)
    option_ids = torch.zeros(k, 5, dtype=torch.long)
    for i in range(k):
        option_ids[i, 1] = 50
        option_ids[i, 2] = 60
        option_ids[i, 3] = 100 + i
        options[i, 1, 5] = 1.0
        options[i, 2, 6] = 1.0
        options[i, 3, 20 + i] = 1.0

    options[gold_index, 3] = 0
    options[gold_index, 3, 0] = 1.0
    options[:, 1] = 0
    options[:, 1, 1] = 1.0

    salience = _option_salience(option_ids, option_mask)
    pooled = options[:, 1:4].mean(1)
    return {
        "case_id": "synthetic",
        "split": "dev",
        "template_id": "fixture",
        "k": k,
        "gold_index": gold_index,
        "state_tokens": state.half(),
        "state_mask": state_mask,
        "question_tokens": question.half(),
        "question_mask": question_mask,
        "option_tokens": options.half(),
        "option_mask": option_mask,
        "option_input_ids": option_ids,
        "option_salience": salience,
        "state_pooled": state[1:3].mean(0).half(),
        "question_pooled": question[1].half(),
        "option_pooled": pooled.half(),
    }


def synthetic_cache(case):
    return {
        "schema_version": "r8-w5h-balanced-binding-cache-v1",
        "case_count": 1,
        "encoder_calls": 1,
        "state_text_encodes": 1,
        "state_text_encodes_per_case": 1.0,
        "cases": [case],
    }


def test_w5h_authority_counts_are_frozen():
    assert sum(TRAIN_K_COUNTS.values()) == 512
    assert sum(DEV_K_COUNTS.values()) == 176
    assert sum(CONFIRM_K_COUNTS.values()) == 192
    train = generate_binding_authority("train")
    dev = generate_binding_authority("dev")
    confirm = generate_binding_authority("confirm")
    assert len(train) == 512
    assert len(dev) == 176
    assert len(confirm) == 192
    assert {c.case_id for c in train}.isdisjoint({c.case_id for c in dev})
    assert all(c.split == "confirm" for c in confirm)


def test_w5h_repaired_authority_is_fresh_and_avoids_w5g_template_scaffolds():
    assert (TRAIN_SEED, DEV_SEED, CONFIRM_SEED) == (141109, 142211, 143313)
    rendered = []
    for split in ("train", "dev", "confirm"):
        for case in generate_binding_authority(split):
            rendered.extend((case.state_text, case.question_text))
    text = "\n".join(rendered)
    forbidden_w5g_scaffolds = (
        "Ledger fields:",
        "Dossier records",
        "Card links",
        "Index ->",
        "Placard shows",
        "Memo pairs",
        "Registry fields:",
        "Capsule ->",
        "Bulletin links",
    )
    for phrase in forbidden_w5g_scaffolds:
        assert phrase not in text


def test_w5h_vocab_is_disjoint_from_w5a_through_w5g():
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab
    from nmd import semantic_alignment_probes as w5c
    from nmd import semantic_capacity_control as w5e
    from nmd import semantic_encoder_adaptation as w5d
    from nmd import semantic_routing_curriculum as w5a
    from nmd import semantic_token_curriculum as w5b

    prior = set(all_w5g_vocab()) | set(all_w5f_vocab())
    for module in (w5a, w5b, w5c, w5d, w5e):
        for name, value in vars(module).items():
            if not (name.startswith("TRAIN_") or name.startswith("CONFIRM_")):
                continue
            if isinstance(value, tuple) and all(
                isinstance(item, str) for item in value
            ):
                prior.update(value)
    overlap = all_w5h_vocab() & prior
    assert not overlap, overlap


def test_option_salience_downweights_common_tokens_and_normalizes_mean():
    ids = torch.tensor([
        [0, 10, 20, 101, 0],
        [0, 10, 20, 102, 0],
        [0, 10, 20, 103, 0],
        [0, 10, 20, 104, 0],
    ])
    mask = torch.tensor([[False, True, True, True, False]] * 4)
    weights = _option_salience(ids, mask)
    for row in range(4):
        assert torch.isclose(weights[row][mask[row]].mean(), torch.tensor(1.0))
        assert weights[row, 3] > weights[row, 1]
        assert weights[row, 3] > weights[row, 2]


def test_all_three_candidates_have_exact_same_trainable_capacity():
    counts = {}
    for name in CANDIDATES:
        matcher = BalancedBindingMatcher(name)
        counts[name] = sum(p.numel() for p in matcher.parameters())
    assert len(set(counts.values())) == 1
    assert next(iter(counts.values())) == 256 * 128 + 1


def test_competitive_binding_subtracts_sibling_common_mode():
    matcher = BalancedBindingMatcher("idf-competitive-proj128")
    similarity = torch.tensor([[
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
    ]])
    option_mask = torch.tensor([[True, True, True]])
    context_mask = torch.tensor([True, True])
    salience = torch.ones(1, 3)
    coverage = matcher._competitive_coverage(
        similarity, option_mask, context_mask, salience
    )
    expected_common = similarity.mean(dim=1, keepdim=True)
    expected = (similarity - expected_common).max(dim=-1).values
    assert torch.allclose(coverage, expected)


def test_greedy_unique_avoids_context_reuse_until_exhausted():
    matcher = BalancedBindingMatcher("idf-greedy-unique-proj128")
    similarity = torch.tensor([[
        [0.95, 0.20, 0.10],
        [0.90, 0.80, 0.10],
        [0.85, 0.30, 0.70],
    ]])
    option_mask = torch.tensor([[True, True, True]])
    context_mask = torch.tensor([True, True, True])
    salience = torch.tensor([[3.0, 2.0, 1.0]])
    coverage = matcher._greedy_unique_coverage(
        similarity, option_mask, context_mask, salience
    )
    assert torch.allclose(coverage, torch.tensor([[0.95, 0.80, 0.70]]))


def test_matcher_backprop_and_probability_integrity():
    torch.manual_seed(503)
    case = synthetic_case(gold_index=1, k=3)
    cache = synthetic_cache(case)
    validate_binding_cache(cache, expected_split="dev")
    for mode in CANDIDATES:
        matcher = BalancedBindingMatcher(mode)
        loss = matcher_case_loss(matcher, case)
        loss.backward()
        assert math.isfinite(float(loss.detach()))
        assert matcher.projection.weight.grad is not None
        assert matcher.log_scale.grad is not None
        metrics = evaluate_matcher(matcher, cache)
        assert metrics["probability_mass_max_error"] < 1e-6


def test_dev_selection_key_prioritizes_accuracy_then_k255_top5():
    base = {
        "accuracy": 0.5,
        "top5_recall": 0.6,
        "mrr": 0.2,
        "hard_brier": 0.4,
        "per_k": {
            "128": {"accuracy": 0.2},
            "255": {"accuracy": 0.1, "top5_recall": 0.3},
        },
    }
    higher_acc = {**base, "accuracy": 0.51}
    assert dev_selection_key(higher_acc, 6) < dev_selection_key(base, 1)
    higher_k255 = {**base, "per_k": {
        "128": {"accuracy": 0.2},
        "255": {"accuracy": 0.1, "top5_recall": 0.4},
    }}
    assert dev_selection_key(higher_k255, 6) < dev_selection_key(base, 1)


def test_rescue_requires_competence_and_mechanism_gates():
    control = {
        "accuracy": 0.54,
        "mrr": 0.60,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "128": {"accuracy": 0.40},
            "255": {"accuracy": 0.31, "top5_recall": 0.64},
        },
    }
    selected = {
        "accuracy": 0.62,
        "mrr": 0.70,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "128": {"accuracy": 0.50},
            "255": {"accuracy": 0.38, "top5_recall": 0.75},
        },
    }
    verdict, gates = confirm_verdict(selected, control)
    assert verdict == "BALANCED_BINDING_RESCUE"
    assert all(gates.values())

    no_mechanism = {
        **selected,
        "accuracy": 0.58,
        "per_k": {
            "128": {"accuracy": 0.50},
            "255": {"accuracy": 0.38, "top5_recall": 0.68},
        },
    }
    verdict, _ = confirm_verdict(no_mechanism, control)
    assert verdict in {"BALANCED_BINDING_PARTIAL", "BALANCED_BINDING_FAIL"}


def test_cache_validator_rejects_non_normalized_salience():
    case = synthetic_case()
    case["option_salience"][0, 1] *= 3
    with pytest.raises(ValueError, match="salience must have mean one"):
        validate_binding_cache(synthetic_cache(case))
