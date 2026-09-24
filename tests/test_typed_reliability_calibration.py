import math

import torch

from nmd.calibration import (
    TypedReliabilityCalibrator,
    count_calibrator_parameters,
)
from nmd.competitive import CompetitiveCoarseScorer
from nmd.contracts import LogicalOption
from nmd.hira import HIRACore, count_parameters
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def make_model(*, calibrator=None):
    torch.manual_seed(8101)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=96,
    )
    hira = HIRACore(d_model=256, dropout=0.0)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    return NolaneHira(
        encoder,
        hira,
        coarse_scorer=scorer,
        reliability_calibrator=calibrator,
    )


def choice_options():
    return (
        LogicalOption("a", "amber pilot inspect garden"),
        LogicalOption("b", "violet baker measure plaza"),
        LogicalOption("c", "teal mason sort bridge"),
    )


def noul_options(*, reversed_order=False):
    rows = (
        LogicalOption(
            "false",
            "Human review is not required.",
            value=0.0,
        ),
        LogicalOption(
            "true",
            "Human review is required.",
            value=1.0,
        ),
    )
    return tuple(reversed(rows)) if reversed_order else rows


def test_calibrator_parameter_counts_are_exact():
    t = TypedReliabilityCalibrator("primitive-temperature")
    tb = TypedReliabilityCalibrator(
        "primitive-temperature-noul-bias"
    )
    assert count_calibrator_parameters(t) == 3
    assert count_calibrator_parameters(tb) == 4
    assert t.trainable_parameter_count == 3
    assert tb.trainable_parameter_count == 4


def test_identity_calibrator_is_exact_for_logits_and_probabilities():
    model = make_model()
    model.eval()
    memory = model.compile_state(
        "the amber pilot must inspect the garden"
    )
    schema, _ = model.compile_schema(
        primitive="choice",
        question_text="Which route matches every field?",
        options=choice_options(),
        include_token_artifacts=True,
        use_cache=False,
    )
    base = model.forward_compiled(
        memory,
        schema,
        forced_budget=255,
        coarse_mode="competitive",
    )

    model.reliability_calibrator = TypedReliabilityCalibrator(
        "primitive-temperature-noul-bias"
    )
    calibrated = model.forward_compiled(
        memory,
        schema,
        forced_budget=255,
        coarse_mode="competitive",
    )

    assert torch.equal(base.logits, calibrated.logits)
    assert torch.equal(base.probabilities, calibrated.probabilities)
    assert torch.equal(base.hira.logits, calibrated.hira.logits)
    assert torch.equal(
        base.hira.probabilities,
        calibrated.hira.probabilities,
    )


def test_default_runtime_and_hiracore_contract_remain_unchanged():
    model = make_model(calibrator=None)
    assert model.reliability_calibrator is None
    assert count_parameters(model.hira) == 422_159


def test_temperature_only_cannot_change_argmax():
    calibrator = TypedReliabilityCalibrator(
        "primitive-temperature"
    )
    with torch.no_grad():
        calibrator.log_temperature.copy_(
            torch.tensor([
                math.log(0.5),
                math.log(2.0),
                math.log(5.0),
            ])
        )
    logits = torch.tensor([
        [0.1, 0.9, -0.3],
        [2.0, 1.0, 0.0],
        [-1.0, 0.5, 0.3],
    ])
    qtype = torch.tensor([0, 1, 2])
    out = calibrator(logits, qtype)
    assert torch.equal(logits.argmax(-1), out.argmax(-1))


def test_noul_bias_follows_semantic_true_not_option_position():
    calibrator = TypedReliabilityCalibrator(
        "primitive-temperature-noul-bias"
    )
    with torch.no_grad():
        calibrator.noul_true_bias.fill_(1.25)

    logits = torch.tensor([[0.4, 0.4]])
    qtype = torch.tensor([2])
    true_second = torch.tensor([[False, True]])
    true_first = torch.tensor([[True, False]])

    a = calibrator(
        logits,
        qtype,
        noul_true_mask=true_second,
    )
    b = calibrator(
        logits.flip(-1),
        qtype,
        noul_true_mask=true_first,
    )
    assert a[0, 1] > a[0, 0]
    assert b[0, 0] > b[0, 1]
    assert torch.allclose(a.flip(-1), b)


def test_noul_bias_does_not_change_choice_or_score_rows():
    calibrator = TypedReliabilityCalibrator(
        "primitive-temperature-noul-bias"
    )
    with torch.no_grad():
        calibrator.noul_true_bias.fill_(3.0)
    logits = torch.tensor([
        [0.2, 0.8],
        [0.1, 0.9],
        [0.4, 0.4],
    ])
    qtype = torch.tensor([0, 1, 2])
    mask = torch.tensor([
        [False, False],
        [False, False],
        [False, True],
    ])
    out = calibrator(
        logits,
        qtype,
        noul_true_mask=mask,
    )
    assert torch.equal(out[0], logits[0])
    assert torch.equal(out[1], logits[1])
    assert out[2, 1] > logits[2, 1]


def test_runtime_noul_permutation_equivariance_with_bias():
    calibrator = TypedReliabilityCalibrator(
        "primitive-temperature-noul-bias"
    )
    with torch.no_grad():
        calibrator.noul_true_bias.fill_(0.75)
    model = make_model(calibrator=calibrator)
    model.eval()
    memory = model.compile_state(
        "evidence is uncertain and severity is serious"
    )

    a_schema, _ = model.compile_schema(
        primitive="noul",
        question_text="Does this require human review?",
        options=noul_options(reversed_order=False),
        use_cache=False,
    )
    b_schema, _ = model.compile_schema(
        primitive="noul",
        question_text="Does this require human review?",
        options=noul_options(reversed_order=True),
        use_cache=False,
    )
    a = model.forward_compiled(
        memory,
        a_schema,
        forced_budget=255,
    )
    b = model.forward_compiled(
        memory,
        b_schema,
        forced_budget=255,
    )
    assert torch.allclose(
        a.probabilities.flip(-1),
        b.probabilities,
        atol=1e-6,
    )
    assert torch.allclose(a.value, b.value, atol=1e-6)


def test_calibration_preserves_probability_mass_and_state_once():
    calibrator = TypedReliabilityCalibrator(
        "primitive-temperature-noul-bias"
    )
    with torch.no_grad():
        calibrator.log_temperature.copy_(
            torch.tensor([0.2, -0.1, 0.4])
        )
        calibrator.noul_true_bias.fill_(0.3)
    model = make_model(calibrator=calibrator)
    model.eval()
    before = model.state_encode_calls
    out = model.decide_text(
        "evidence is uncertain and severity is critical",
        primitive="noul",
        question_text="Does this require human review?",
        options=noul_options(),
        forced_budget=255,
    )
    assert model.state_encode_calls - before == 1
    assert torch.isclose(
        out.probabilities.sum(),
        out.probabilities.new_tensor(1.0),
        atol=1e-6,
    )


def test_calibrator_gradients_do_not_require_hira_or_scorer_gradients():
    calibrator = TypedReliabilityCalibrator(
        "primitive-temperature-noul-bias"
    )
    model = make_model(calibrator=calibrator)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.hira.parameters():
        parameter.requires_grad_(False)
    assert model.coarse_scorer is not None
    for parameter in model.coarse_scorer.parameters():
        parameter.requires_grad_(False)

    out = model.decide_text(
        "evidence is uncertain and severity is serious",
        primitive="noul",
        question_text="Does this require human review?",
        options=noul_options(),
        forced_budget=255,
        coarse_mode="competitive",
    )
    loss = -torch.log(out.probabilities[1].clamp_min(1e-8))
    loss.backward()

    assert calibrator.log_temperature.grad is not None
    assert calibrator.noul_true_bias is not None
    assert calibrator.noul_true_bias.grad is not None
    assert all(
        parameter.grad is None
        for parameter in model.hira.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.coarse_scorer.parameters()
    )
