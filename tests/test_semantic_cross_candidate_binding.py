import math

import pytest
import torch

from nmd.semantic_cross_candidate_binding import (
    CANDIDATES,
    CONFIRM_K_COUNTS,
    CONFIRM_SEED,
    CONFIRM_TEMPLATES,
    DEV_K_COUNTS,
    DEV_SEED,
    TRAIN_K_COUNTS,
    TRAIN_SEED,
    CrossCandidateBindingMatcher,
    _option_salience,
    _render,
    all_w5i_vocab,
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
        "schema_version": "r8-w5i-balanced-binding-cache-v1",
        "case_count": 1,
        "encoder_calls": 1,
        "state_text_encodes": 1,
        "state_text_encodes_per_case": 1.0,
        "cases": [case],
    }


def test_w5i_authority_counts_and_confirm_seal():
    assert sum(TRAIN_K_COUNTS.values()) == 512
    assert sum(DEV_K_COUNTS.values()) == 176
    assert sum(CONFIRM_K_COUNTS.values()) == 192
    assert (TRAIN_SEED, DEV_SEED, CONFIRM_SEED) == (151117, 152219, 153321)
    train = generate_binding_authority("train")
    dev = generate_binding_authority("dev")
    assert len(train) == 512
    assert len(dev) == 176
    assert {c.case_id for c in train}.isdisjoint({c.case_id for c in dev})
    with pytest.raises(RuntimeError, match="sealed until post-selection"):
        generate_binding_authority("confirm")


def test_w5i_templates_avoid_known_w5a_through_w5h_scaffolds():
    rendered = []
    for split in ("train", "dev"):
        for case in generate_binding_authority(split):
            rendered.extend((case.state_text, case.question_text))
    placeholder = ("alpha", "beta", "gamma", "delta")
    for template_id in CONFIRM_TEMPLATES:
        rendered.extend(_render(placeholder, template_id))
    text = "\n".join(rendered)
    forbidden = (
        "The active route uses", "Route attributes:", "Dispatch record says",
        "Manifest: [", "Ledger entry ->", "Brief: at",
        "Record: color=", "The report names", "Note for ", "Dossier says",
        "Envelope:", "Message from", "Index entry", "Briefing names",
        "Summary:", "Register places", "Memo fields:", "Sheet records", "Entry ->",
        "Card fields:", "Log says", "Note links", "Ledger ->",
        "Ticket contains", "Form fields:", "Dossier records", "Slip ->",
        "Manifest fields:", "Index lists", "Catalog row:", "Register links",
        "Sheet ->", "Entry contains", "Brief fields:", "Panel records", "File ->",
        "Roster fields:", "Table row lists", "Mapping ->", "Slip records",
        "Grid cells:", "Page fields:", "Tag ->", "Report links",
        "Ledger fields:", "Card links", "Index ->", "Placard shows",
        "Memo pairs", "Registry fields:", "Capsule ->", "Bulletin links",
        "Observation matrix has", "Courier packet lists", "Reference strip reads",
        "Keymap:", "Beacon record encodes", "Notebook trace:",
        "Specimen matrix contains", "Dispatch packet records", "Quadrant note assigns",
    )
    for phrase in forbidden:
        assert phrase not in text


def test_w5i_vocab_is_disjoint_from_w5a_through_w5h():
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab
    from nmd import semantic_alignment_probes as w5c
    from nmd import semantic_capacity_control as w5e
    from nmd import semantic_encoder_adaptation as w5d
    from nmd import semantic_routing_curriculum as w5a
    from nmd import semantic_token_curriculum as w5b

    prior = set(all_w5h_vocab()) | set(all_w5g_vocab()) | set(all_w5f_vocab())
    for module in (w5a, w5b, w5c, w5d, w5e):
        for name, value in vars(module).items():
            if not (name.startswith("TRAIN_") or name.startswith("CONFIRM_")):
                continue
            if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
                prior.update(value)
    overlap = all_w5i_vocab() & prior
    assert not overlap, overlap


def test_option_salience_normalizes_and_downweights_common_tokens():
    ids = torch.tensor([
        [0, 10, 20, 101, 0],
        [0, 10, 20, 102, 0],
        [0, 10, 20, 103, 0],
    ])
    mask = torch.tensor([[False, True, True, True, False]] * 3)
    weights = _option_salience(ids, mask)
    for row in range(3):
        assert torch.isclose(weights[row][mask[row]].mean(), torch.tensor(1.0))
        assert weights[row, 3] > weights[row, 1]


def test_all_candidates_have_equal_capacity():
    counts = {}
    for name in CANDIDATES:
        matcher = CrossCandidateBindingMatcher(name)
        counts[name] = sum(p.numel() for p in matcher.parameters())
    assert len(set(counts.values())) == 1
    assert next(iter(counts.values())) == 256 * 128 + 1


def test_competitive_adjustment_subtracts_within_option_common_mode():
    matcher = CrossCandidateBindingMatcher("idf-competitive-forward-proj128")
    similarity = torch.tensor([[
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
    ]])
    option_mask = torch.tensor([[True, True, True]])
    context_mask = torch.tensor([True, True])
    salience = torch.ones(1, 3)
    adjusted = matcher._competitive_adjusted_similarity(
        similarity, option_mask, context_mask, salience
    )
    expected = similarity - similarity.mean(dim=1, keepdim=True)
    assert torch.allclose(adjusted, expected)


def test_reverse_mean_evidence_is_candidate_relative():
    matcher = CrossCandidateBindingMatcher("idf-competitive-bidir-mean-proj128")
    adjusted = torch.tensor([
        [[0.9, 0.9], [0.1, 0.2]],
        [[0.4, 0.2], [0.1, 0.1]],
    ])
    option_mask = torch.tensor([[True, True], [True, True]])
    context_mask = torch.tensor([True, True])
    evidence = matcher._reverse_evidence(
        adjusted, option_mask, context_mask, baseline="mean"
    )
    assert torch.allclose(evidence, torch.tensor([0.3, -0.3]), atol=1e-6)


def test_reverse_logmeanexp_matches_frozen_formula():
    matcher = CrossCandidateBindingMatcher("idf-competitive-bidir-logmeanexp-proj128")
    adjusted = torch.tensor([
        [[0.9, 0.8], [0.2, 0.1]],
        [[0.3, 0.2], [0.1, 0.0]],
        [[0.5, 0.4], [0.2, 0.2]],
    ])
    option_mask = torch.ones(3, 2, dtype=torch.bool)
    context_mask = torch.tensor([True, True])
    support = adjusted.max(dim=1).values
    baseline = torch.logsumexp(support, dim=0) - math.log(3)
    expected = (support - baseline).mean(dim=1)
    actual = matcher._reverse_evidence(
        adjusted, option_mask, context_mask, baseline="logmeanexp"
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_matcher_backprop_and_probability_integrity():
    torch.manual_seed(607)
    case = synthetic_case(gold_index=1, k=3)
    cache = synthetic_cache(case)
    validate_binding_cache(cache, expected_split="dev")
    for mode in CANDIDATES:
        matcher = CrossCandidateBindingMatcher(mode)
        loss = matcher_case_loss(matcher, case)
        loss.backward()
        assert math.isfinite(float(loss.detach()))
        assert matcher.projection.weight.grad is not None
        assert matcher.log_scale.grad is not None
        metrics = evaluate_matcher(matcher, cache)
        assert metrics["probability_mass_max_error"] < 1e-6


def test_dev_selection_key_prioritizes_accuracy_then_k255_accuracy():
    base = {
        "accuracy": 0.5,
        "top5_recall": 0.8,
        "mrr": 0.6,
        "hard_brier": 0.4,
        "per_k": {
            "128": {"accuracy": 0.4},
            "255": {"accuracy": 0.2, "top5_recall": 0.7},
        },
    }
    higher_acc = {**base, "accuracy": 0.51}
    assert dev_selection_key(higher_acc, 6) < dev_selection_key(base, 1)
    higher_k255 = {
        **base,
        "per_k": {
            "128": {"accuracy": 0.4},
            "255": {"accuracy": 0.21, "top5_recall": 0.7},
        },
    }
    assert dev_selection_key(higher_k255, 6) < dev_selection_key(base, 1)


def test_cross_candidate_rescue_requires_absolute_and_mechanism_gates():
    control = {
        "accuracy": 0.50,
        "mrr": 0.62,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "128": {"accuracy": 0.42},
            "255": {"accuracy": 0.28, "top5_recall": 0.71},
        },
    }
    selected = {
        "accuracy": 0.61,
        "mrr": 0.73,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "128": {"accuracy": 0.50},
            "255": {"accuracy": 0.36, "top5_recall": 0.78},
        },
    }
    verdict, gates = confirm_verdict(selected, control)
    assert verdict == "CROSS_CANDIDATE_BINDING_RESCUE"
    assert all(gates.values())


def test_control_already_rescues_is_not_misattributed():
    control = {
        "accuracy": 0.61,
        "mrr": 0.70,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "128": {"accuracy": 0.45},
            "255": {"accuracy": 0.34, "top5_recall": 0.72},
        },
    }
    selected = {
        "accuracy": 0.67,
        "mrr": 0.75,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "128": {"accuracy": 0.50},
            "255": {"accuracy": 0.40, "top5_recall": 0.80},
        },
    }
    verdict, _ = confirm_verdict(selected, control)
    assert verdict == "CROSS_CANDIDATE_CONTROL_ALREADY_RESCUES"


def test_material_gain_below_competence_is_partial():
    control = {
        "accuracy": 0.44,
        "mrr": 0.58,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "128": {"accuracy": 0.38},
            "255": {"accuracy": 0.20, "top5_recall": 0.66},
        },
    }
    selected = {
        "accuracy": 0.50,
        "mrr": 0.65,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "128": {"accuracy": 0.43},
            "255": {"accuracy": 0.27, "top5_recall": 0.72},
        },
    }
    verdict, _ = confirm_verdict(selected, control)
    assert verdict == "CROSS_CANDIDATE_BINDING_PARTIAL"


def test_cache_validator_rejects_non_normalized_salience():
    case = synthetic_case()
    case["option_salience"][0, 1] *= 3
    with pytest.raises(ValueError, match="salience must have mean one"):
        validate_binding_cache(synthetic_cache(case))
