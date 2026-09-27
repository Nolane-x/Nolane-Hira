import math

import pytest
import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.mainline import (
    HIRA_V0_MAINLINE_M2_VERSION,
    HIRA_V0_MAX_K,
    HiraV0Mainline,
    HiraV0Manifest,
)
from nmd.mainline_high_k import (
    M2_K_LADDER,
    profile_high_k_query,
    run_high_k_mechanics_suite,
)
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.w33_coevidence_semantic import CoEvidenceSemanticScorer


def _model() -> HiraV0Mainline:
    torch.manual_seed(17201)
    encoder = TrainableSemanticEncoder(
        vocab_size=4096,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=48,
    )
    scorer = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    scorer.freeze_candidate()
    runtime = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        coevidence_symmetric_semantic_scorer=scorer,
    )
    return HiraV0Mainline(
        runtime,
        manifest=HiraV0Manifest.m2_mechanics_provisional(),
    )


def _options(k: int) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=f"m2-route-{index:03d}",
            criterion_text=f"route token {index:03d} applies",
            aliases=(f"select route token {index:03d}",),
        )
        for index in range(k)
    )


def test_m2_manifest_separates_mechanics_from_semantic_quality():
    manifest = HiraV0Manifest.m2_mechanics_provisional().to_dict()
    assert manifest["version"] == HIRA_V0_MAINLINE_M2_VERSION
    assert manifest["reliability_ood_abstention"] == "provisional"
    assert manifest["high_k"] == "provisional"
    assert manifest["high_k_mechanics"] == "provisional"
    assert manifest["production_ready"] is False
    assert manifest["transfer_core"] == "provisional"


def test_v0_mainline_explicitly_bounds_dynamic_schema_at_255():
    assert HIRA_V0_MAX_K == 255
    model = _model()
    session = model.open_session("route token 001 applies")

    with pytest.raises(ValueError, match="at most 255"):
        session.decide(
            primitive="choice",
            question_text="Which route applies?",
            options=_options(256),
            use_schema_cache=False,
        )


def test_profiled_high_k_query_preserves_state_once_and_full_k():
    model = _model()
    before = model.runtime.state_encode_calls
    session = model.open_session("route token 031 applies")
    assert model.runtime.state_encode_calls == before + 1

    report = profile_high_k_query(
        session,
        question_text="Which route applies?",
        options=_options(32),
    )

    assert model.runtime.state_encode_calls == before + 1
    assert report.requested_k == 32
    assert report.actual_candidate_budget == 32
    assert report.state_encode_delta == 0
    assert report.full_k is True
    assert report.finite is True
    assert report.probability_mass_error <= 1e-6
    assert report.relation_delta_max == 0.0
    assert report.schema_tensor_bytes > 0
    assert math.isfinite(report.schema_compile_ms)
    assert math.isfinite(report.decision_ms)
    assert report.schema_compile_ms >= 0.0
    assert report.decision_ms >= 0.0
    assert report.mechanics_pass is True


def test_full_m2_k_ladder_passes_inside_one_state_once_session():
    model = _model()
    suite = run_high_k_mechanics_suite(
        model,
        state_text="route token 003 applies",
        question_text="Which route applies?",
        option_factory=_options,
    )

    assert suite.k_values == M2_K_LADDER
    assert suite.k_values == (4, 8, 16, 32, 64, 128, 255)
    assert suite.state_encode_delta == 1
    assert suite.query_count == 14
    assert len(suite.query_reports) == 7
    assert len(suite.permutation_reports) == 7
    assert suite.max_probability_mass_error <= 1e-6
    assert suite.max_permutation_error <= 2e-6
    assert suite.max_schema_tensor_bytes > 0
    assert suite.mechanics_pass is True

    previous_bytes = 0
    for expected_k, report, permutation in zip(
        M2_K_LADDER,
        suite.query_reports,
        suite.permutation_reports,
    ):
        assert report.requested_k == expected_k
        assert report.actual_candidate_budget == expected_k
        assert report.state_encode_delta == 0
        assert report.full_k is True
        assert report.finite is True
        assert report.relation_delta_max == 0.0
        assert report.mechanics_pass is True
        assert report.schema_tensor_bytes >= previous_bytes
        previous_bytes = report.schema_tensor_bytes

        assert permutation.k == expected_k
        assert permutation.probability_max_error <= 2e-6
        assert permutation.selected_option_invariant is True
        assert permutation.permutation_pass is True


def test_high_k_profile_does_not_enable_training_or_reliability_promotion():
    model = _model()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    assert model.manifest.reliability_ood_abstention == "provisional"
    assert model.manifest.production_ready is False
