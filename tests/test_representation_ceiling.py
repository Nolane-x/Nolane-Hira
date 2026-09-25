import torch

from nmd.hira import HIRACore
from nmd.representation_ceiling import (
    A13_REPRESENTATION_CEILING,
    OUTCOME_STABLE,
    classify_domain,
    cross_domain_outcome,
)
from nmd.representation_ceiling_authority import (
    BASES_PER_DOMAIN,
    generate_w10_domain,
)
from nmd.representation_ceiling_cache import compile_w10_cache
from nmd.representation_ceiling_eval import _symmetric_maxsim
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w10_generator_has_paired_nested_identity():
    rows = generate_w10_domain("BF")
    assert len(rows) == BASES_PER_DOMAIN * 3 * 2
    assert len({row.base_id for row in rows}) == BASES_PER_DOMAIN

    base_id = rows[0].base_id
    base = [row for row in rows if row.base_id == base_id]
    assert len(base) == 6

    by_k = {}
    for k in (4, 8, 16):
        pair = [row for row in base if row.diagnosis_k == k]
        assert {row.view_id for row in pair} == {"definition", "label"}
        assert pair[0].option_ids == pair[1].option_ids
        assert pair[0].gold_index == pair[1].gold_index
        by_k[k] = pair[0].option_ids

    assert set(by_k[4]) < set(by_k[8]) < set(by_k[16])
    assert [x for x in by_k[8] if x in by_k[4]] == list(by_k[4])
    assert [x for x in by_k[16] if x in by_k[8]] == list(by_k[8])


def test_w10_classifier_a13_ceiling_and_stability():
    metrics = {
        "a0_k4_top1": .50,
        "a1_k4_top1": .55,
        "a1_k16_top1": .30,
        "p0_k4_top1": .45,
        "p1_k4_top1": .62,
        "p1_k16_top1": .35,
        "s1_k4_top1": .55,
        "s1_k16_top1": .30,
        "r0_k4_top1": .82,
    }
    row = classify_domain(metrics)
    assert row["classification"] == A13_REPRESENTATION_CEILING

    per = {
        "BF": row,
        "BG": row,
        "BH": row,
        "BI": classify_domain({**metrics, "a1_k4_top1": .62}),
    }
    outcome = cross_domain_outcome(per)
    assert outcome["outcome"] == OUTCOME_STABLE
    assert outcome["stable_classification"] == A13_REPRESENTATION_CEILING


def test_symmetric_maxsim_uses_both_directions_correctly():
    state = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    options = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    mask = torch.ones(2, 2, dtype=torch.bool)
    scores = _symmetric_maxsim(state, options, mask)
    assert scores.shape == (2,)
    assert float(scores[0]) > float(scores[1])


def test_w10_cache_preserves_state_once_for_one_base():
    rows = generate_w10_domain("BF")
    base_id = rows[0].base_id
    rows = [row for row in rows if row.base_id == base_id]

    encoder = TrainableSemanticEncoder(
        vocab_size=512,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    cache = compile_w10_cache(model, rows)
    assert cache["metadata"]["base_count"] == 1
    assert cache["metadata"]["view_count"] == 6
    assert cache["metadata"]["state_encode_calls"] == 1
    assert cache["metadata"]["state_encode_calls_per_base"] == 1.0
