import torch

from nmd.conjunctive_authority import (
    generate_w7_dev,
)
from nmd.conjunctive_cache import (
    factor_identity_sha256,
    factor_target_mask,
    one_field_negative_indices,
    semantic_signature_sha256,
)


def test_factor_target_mask_matches_signature_identity():
    case = generate_w7_dev()[0]
    target = factor_target_mask(case)
    diagnosis = case.typed.decisions[0]
    gold_index = int(diagnosis.gold_index)
    gold = case.diagnosis_signatures[gold_index]
    assert target.shape == (case.diagnosis_k, 4)
    assert target.dtype == torch.bool
    assert bool(target[gold_index].all())
    for index, signature in enumerate(case.diagnosis_signatures):
        expected = torch.tensor(
            [signature[role] == gold[role] for role in range(4)],
            dtype=torch.bool,
        )
        assert torch.equal(target[index], expected)


def test_one_field_labels_match_exact_hamming_distance():
    rows = generate_w7_dev()
    for case in rows[:64]:
        labels = one_field_negative_indices(case)
        diagnosis = case.typed.decisions[0]
        gold_index = int(diagnosis.gold_index)
        gold = case.diagnosis_signatures[gold_index]
        labelled = {
            int(index)
            for values in labels.values()
            for index in values
        }
        expected = {
            index
            for index, signature in enumerate(case.diagnosis_signatures)
            if index != gold_index
            and sum(a != b for a, b in zip(gold, signature)) == 1
        }
        assert labelled == expected


def test_factor_and_signature_hashes_are_deterministic_and_distinct():
    rows = generate_w7_dev()
    factor_a = factor_identity_sha256(rows)
    factor_b = factor_identity_sha256(list(reversed(rows)))
    semantic_a = semantic_signature_sha256(rows)
    semantic_b = semantic_signature_sha256(list(reversed(rows)))
    assert factor_a == factor_b
    assert semantic_a == semantic_b
    assert factor_a != semantic_a
    assert len(factor_a) == 64
    assert len(semantic_a) == 64
