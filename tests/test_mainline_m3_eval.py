import copy

import torch

from nmd.hira import HIRACore
from nmd.mainline import (
    HIRA_V0_MAINLINE_M3_VERSION,
    M2_MECHANICS_AUTHORITY,
    HiraV0Mainline,
    HiraV0Manifest,
)
from nmd.mainline_m3_authority import generate_m3_paired_authority
from nmd.mainline_m3_eval import (
    evaluate_m3_paired_cases,
    m3_multilingual_qualification,
)
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.w33_coevidence_semantic import CoEvidenceSemanticScorer


def _model() -> HiraV0Mainline:
    torch.manual_seed(18301)
    encoder = TrainableSemanticEncoder(
        vocab_size=4096,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=96,
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
        manifest=HiraV0Manifest.m3_multilingual_provisional(),
    )


def test_m3_manifest_preserves_m2_mechanics_and_marks_multilingual_provisional():
    manifest = HiraV0Manifest.m3_multilingual_provisional().to_dict()
    assert manifest["version"] == HIRA_V0_MAINLINE_M3_VERSION
    assert manifest["multilingual"] == "provisional"
    assert manifest["high_k"] == "provisional"
    assert manifest["high_k_mechanics"] == "available"
    assert manifest["high_k_mechanics_authority"] == M2_MECHANICS_AUTHORITY
    assert manifest["reliability_ood_abstention"] == "provisional"
    assert manifest["production_ready"] is False


def test_m3_evaluator_preserves_state_once_full_k_and_language_pairing():
    model = _model()
    rows = generate_m3_paired_authority("dev")[:4]
    before = model.runtime.state_encode_calls
    report = evaluate_m3_paired_cases(model, rows)

    assert model.runtime.state_encode_calls - before == 8
    assert report["pair_count"] == 4
    assert report["language_case_count"] == 8
    assert report["state_encode_count"] == 8
    assert report["state_encodes_per_language_case"] == 1.0
    assert report["full_k"] == 1.0
    assert report["finite"] == 1.0
    assert report["probability_mass_max_error"] <= 1e-6
    assert report["relation_delta_max"] == 0.0
    assert report["gradient_updates_used"] is False
    assert report["candidate_pruning_used"] is False
    assert len(report["pairs"]) == 4


def _passing_report():
    return {
        "en": {"top1": 0.80, "mrr": 0.86},
        "vi": {"top1": 0.75, "mrr": 0.80},
        "vi_en_top1_ratio": 0.9375,
        "vi_en_mrr_ratio": 0.9302,
        "paired_prediction_agreement": 0.90,
        "state_encodes_per_language_case": 1.0,
        "full_k": 1.0,
        "finite": 1.0,
        "probability_mass_max_error": 1e-7,
        "relation_delta_max": 0.0,
        "gradient_updates_used": False,
        "candidate_pruning_used": False,
        "relation_refinement_used": False,
        "adaptive_budget_used": False,
    }


def test_m3_qualification_requires_absolute_and_relative_language_quality():
    report = _passing_report()
    result = m3_multilingual_qualification(report)
    assert result["pass"] is True
    assert all(result["gates"].values())

    broken = copy.deepcopy(report)
    broken["vi"]["top1"] = 0.60
    broken["vi_en_top1_ratio"] = 0.75
    failed = m3_multilingual_qualification(broken)
    assert failed["pass"] is False
    assert failed["gates"]["vi_top1"] is False
    assert failed["gates"]["vi_en_top1_ratio"] is False


def test_m3_qualification_rejects_pair_disagreement_even_when_marginals_pass():
    report = _passing_report()
    report["paired_prediction_agreement"] = 0.70
    failed = m3_multilingual_qualification(report)
    assert failed["pass"] is False
    assert failed["gates"]["paired_prediction_agreement"] is False
