import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.field_semantic_rescue import (
    ADAPTER_PARAMETER_COUNT,
    SemanticAdaptedCompetitiveScorer,
    SemanticResidualAdapter,
    absolute_gates,
    adapter_parameter_count,
    causal_gates,
    configure_trainability,
    one_field_negative_indices,
    rescue_verdict,
)
from nmd.hira import HIRACore


def _batch():
    torch.manual_seed(9)
    state = torch.randn(2, 7, 256)
    question = torch.randn(2, 5, 256)
    options = torch.randn(2, 8, 6, 256)
    state_mask = torch.ones(2, 7, dtype=torch.bool)
    question_mask = torch.ones(2, 5, dtype=torch.bool)
    option_mask = torch.ones(2, 8, 6, dtype=torch.bool)
    ids = torch.arange(2 * 8 * 6, dtype=torch.long).reshape(2, 8, 6) + 10
    return state, state_mask, question, question_mask, options, ids, option_mask


def test_adapter_has_exact_frozen_parameter_budget_and_zero_identity():
    base = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    adapter = SemanticResidualAdapter()
    wrapped = SemanticAdaptedCompetitiveScorer(base, adapter)
    assert adapter_parameter_count(adapter) == ADAPTER_PARAMETER_COUNT == 8192

    args = _batch()
    native = base(
        state_tokens=args[0],
        state_mask=args[1],
        question_tokens=args[2],
        question_mask=args[3],
        option_tokens=args[4],
        option_token_ids=args[5],
        option_mask=args[6],
    )
    adapted = wrapped(
        state_tokens=args[0],
        state_mask=args[1],
        question_tokens=args[2],
        question_mask=args[3],
        option_tokens=args[4],
        option_token_ids=args[5],
        option_mask=args[6],
    )
    assert torch.allclose(native, adapted, atol=1e-6, rtol=1e-6)


def test_adapter_path_trains_only_8192_parameters():
    hira = HIRACore(d_model=256, dropout=0.0)
    base = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    wrapped = SemanticAdaptedCompetitiveScorer(base)
    params = configure_trainability("semantic-residual-adapter", hira, wrapped)
    assert sum(parameter.numel() for parameter in params) == 8192
    assert not any(parameter.requires_grad for parameter in hira.parameters())
    assert not any(parameter.requires_grad for parameter in wrapped.base.parameters())
    assert all(parameter.requires_grad for parameter in wrapped.adapter.parameters())


def test_projection_control_keeps_hira_frozen_and_trains_exact_scorer():
    hira = HIRACore(d_model=256, dropout=0.0)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    params = configure_trainability("projection-retune-control", hira, scorer)
    assert sum(parameter.numel() for parameter in params) == 32769
    assert not any(parameter.requires_grad for parameter in hira.parameters())
    assert all(parameter.requires_grad for parameter in scorer.parameters())


def test_one_field_labeler_never_supplies_role_identity_to_model():
    options = [
        "entity a; location b; anomaly c; channel d",
        "entity x; location b; anomaly c; channel d",
        "entity a; location y; anomaly c; channel d",
        "entity a; location b; anomaly z; channel d",
        "entity a; location b; anomaly c; channel q",
        "entity x; location y; anomaly c; channel d",
    ]
    labels = one_field_negative_indices(options, 0)
    assert labels == {0: [1], 1: [2], 2: [3], 3: [4]}


def _metrics(*, overall=.75, choice=.72, score=.70, noul=.75, k64=.62, pair=.85):
    return {
        "accuracy": overall,
        "primitive_accuracy": {
            "choice": choice,
            "score": score,
            "noul": noul,
        },
        "diagnosis_per_k": {"64": {"accuracy": k64}},
        "mean_pair_accuracy": pair,
        "role_pair_accuracy": {
            "0": pair,
            "1": pair,
            "2": pair,
            "3": pair,
        },
        "probability_mass_max_error": 1e-7,
        "source_state_encodes_per_case": 1.0,
        "hard_brier": .3,
        "score_mae": .2,
    }


def test_absolute_and_causal_gate_contracts():
    adapter = _metrics(k64=.62, pair=.85)
    frozen = _metrics(k64=.48, pair=.70)
    projection = _metrics(k64=.64, pair=.87)
    assert all(absolute_gates(adapter).values())
    assert all(causal_gates(adapter, frozen, projection).values())


def test_rescue_partial_and_fail_are_not_conflated():
    strong = {
        "frozen-joint-control": _metrics(k64=.48, pair=.70),
        "projection-retune-control": _metrics(k64=.64, pair=.87),
        "semantic-residual-adapter": _metrics(k64=.62, pair=.85),
    }
    verdict, details = rescue_verdict(confirm_y=strong, confirm_z=strong)
    assert verdict == "FIELD_SEMANTIC_RESCUE"
    assert all(details["full_pass"].values())

    partial_z = {
        **strong,
        "semantic-residual-adapter": _metrics(k64=.54, pair=.77),
    }
    verdict, _ = rescue_verdict(confirm_y=strong, confirm_z=partial_z)
    assert verdict == "FIELD_SEMANTIC_PARTIAL"

    fail = {
        **strong,
        "semantic-residual-adapter": _metrics(k64=.49, pair=.72),
    }
    verdict, _ = rescue_verdict(confirm_y=fail, confirm_z=fail)
    assert verdict == "FIELD_SEMANTIC_FAIL"
