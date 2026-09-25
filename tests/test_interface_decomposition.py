import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.interface_decomposition import (
    DIRECTIONALITY_LOSS,
    OUTCOME_STABLE,
    classify_domain,
    cross_domain_outcome,
)
from nmd.interface_decomposition_authority import (
    BASES_PER_DOMAIN,
    generate_w11_domain,
)
from nmd.interface_decomposition_cache import compile_w11_cache
from nmd.interface_decomposition_eval import (
    _manual_q1_q5,
    _order,
    _q6_actual,
)
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w11_generator_has_paired_nested_identity():
    rows = generate_w11_domain("BJ")
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


def _classifier_fixture():
    stages = {
        name: {
            "4": {"top1": .65},
            "8": {"top1": .55},
            "16": {"top1": .48},
        }
        for name in ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7")
    }
    stages["Q0"]["4"]["top1"] = .82
    stages["Q0"]["16"]["top1"] = .62

    transitions = {}
    for source, target in (
        ("Q0", "Q1"),
        ("Q1", "Q2"),
        ("Q2", "Q3"),
        ("Q3", "Q4"),
        ("Q4", "Q5"),
        ("Q6", "Q7"),
    ):
        transitions[f"{source}->{target}"] = {
            "gate_correct_to_wrong_rate": .02,
            "gate_wrong_to_correct_rate": .01,
        }
    transitions["Q0->Q1"] = {
        "gate_correct_to_wrong_rate": .20,
        "gate_wrong_to_correct_rate": .04,
    }
    return {
        "stages": stages,
        "transitions": transitions,
        "reference": {
            "4": {"top1": .90},
            "8": {"top1": .80},
            "16": {"top1": .70},
        },
    }


def test_w11_classifier_directionality_and_cross_domain_stability():
    row = classify_domain(_classifier_fixture())
    assert row["classification"] == DIRECTIONALITY_LOSS

    outcome = cross_domain_outcome({
        "BJ": row,
        "BK": row,
        "BL": row,
        "BM": {
            "classification": "INTERFACE_DECOMPOSITION_UNRESOLVED",
        },
    })
    assert outcome["outcome"] == OUTCOME_STABLE
    assert outcome["stable_classification"] == DIRECTIONALITY_LOSS


def test_w11_cache_preserves_state_once_for_one_base():
    rows = generate_w11_domain("BJ")
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
    cache = compile_w11_cache(model, rows)
    assert cache["metadata"]["base_count"] == 1
    assert cache["metadata"]["view_count"] == 6
    assert cache["metadata"]["state_encode_calls"] == 1
    assert cache["metadata"]["state_encode_calls_per_base"] == 1.0


def test_w11_manual_q5_and_actual_q6_have_identical_rank_order():
    torch.manual_seed(110)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.eval()

    state = torch.randn(7, 256)
    question = torch.randn(5, 256)
    options = torch.randn(4, 6, 256)
    option_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    option_ids = torch.tensor(
        [
            [11, 12, 13, 14, 0, 0],
            [11, 22, 23, 24, 25, 0],
            [31, 32, 33, 0, 0, 0],
            [41, 42, 43, 44, 45, 46],
        ],
        dtype=torch.long,
    )
    question_mask = torch.ones(5, dtype=torch.bool)

    manual = _manual_q1_q5(
        state_tokens=state,
        question_tokens=question,
        question_mask=question_mask,
        option_tokens=options,
        option_token_ids=option_ids,
        option_mask=option_mask,
        projection=scorer.projection.weight.detach(),
    )["Q5"]

    actual = _q6_actual(
        scorer,
        {"state_content_tokens": state},
        {
            "question_tokens": question,
            "question_content_mask": question_mask,
            "option_tokens": options,
            "option_token_ids": option_ids,
            "option_content_mask": option_mask,
        },
    )
    assert _order(manual) == _order(actual)
