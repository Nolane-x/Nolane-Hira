from pathlib import Path

import pytest
import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.mainline import (
    HIRA_V0_MAINLINE_VERSION,
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    HiraV0Mainline,
    HiraV0Manifest,
    build_hira_v0_mainline,
)
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.w33_coevidence_semantic import CoEvidenceSemanticScorer


def _options_binary():
    return (
        LogicalOption(
            "wait-ok",
            "a short wait is acceptable",
            aliases=("brief delay remains allowed",),
            exemplars=("the task may pause briefly",),
            value=0.0,
        ),
        LogicalOption(
            "act-now",
            "action must begin immediately",
            aliases=("no short delay is allowed",),
            exemplars=("the task must start now",),
            value=1.0,
        ),
    )


def _options_three():
    return (
        LogicalOption(
            "low",
            "low urgency",
            aliases=("minor time pressure",),
            exemplars=("ordinary handling remains sufficient",),
            value=0.0,
        ),
        LogicalOption(
            "medium",
            "medium urgency",
            aliases=("meaningful time pressure",),
            exemplars=("handling should be accelerated",),
            value=0.5,
        ),
        LogicalOption(
            "high",
            "high urgency",
            aliases=("immediate time pressure",),
            exemplars=("handling must begin now",),
            value=1.0,
        ),
    )


def _synthetic_mainline() -> HiraV0Mainline:
    torch.manual_seed(15901)
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = CoEvidenceSemanticScorer(
        d_model=256,
        d_rel=128,
        rank=8,
    )
    scorer.load_projection_weight(
        torch.randn(128, 256) * 0.02,
        freeze=True,
    )
    scorer.freeze_candidate()
    runtime = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        coevidence_symmetric_semantic_scorer=scorer,
    )
    return HiraV0Mainline(runtime)


def test_m0_manifest_is_machine_readable_and_cannot_claim_ready():
    model = _synthetic_mainline()
    manifest = model.manifest.to_dict()

    assert manifest["version"] == HIRA_V0_MAINLINE_VERSION
    assert manifest["semantic_frontend"] == "provisional"
    assert manifest["projection"] == "frozen_research_base"
    assert manifest["transfer_core"] == "provisional"
    assert manifest["typed_runtime"] == "available"
    assert manifest["reliability_ood_abstention"] == "pending"
    assert manifest["high_k"] == "pending"
    assert manifest["multilingual"] == "pending"
    assert manifest["production_ready"] is False
    assert manifest["transfer_core_promoted"] is False
    assert manifest["relation_refinement"] is False
    assert manifest["adaptive_budget"] is False
    assert manifest["t0_checkpoint_sha256"] == W28_T0_CHECKPOINT_SHA256
    assert (
        manifest["transfer_checkpoint_sha256"]
        == W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256
    )
    assert manifest["transfer_candidate_parameter_count"] == 8192


def test_one_state_encode_serves_multiple_typed_dynamic_schema_queries():
    model = _synthetic_mainline()
    before = model.runtime.state_encode_calls
    session = model.open_session(
        "The request can wait briefly, but a later review may require faster handling."
    )
    assert model.runtime.state_encode_calls == before + 1

    choice = session.decide(
        primitive="choice",
        question_text="Which timing status applies?",
        options=_options_binary(),
    )
    score = session.decide(
        primitive="score",
        question_text="What urgency score best applies?",
        options=_options_three(),
    )
    noul = session.decide(
        primitive="noul",
        question_text="Must action begin immediately?",
        options=_options_binary(),
    )

    assert model.runtime.state_encode_calls == before + 1
    assert session.query_count == 3
    assert choice.primitive == "choice"
    assert score.primitive == "score"
    assert noul.primitive == "noul"
    assert choice.probabilities.numel() == 2
    assert score.probabilities.numel() == 3
    assert noul.probabilities.numel() == 2
    assert torch.isfinite(score.value)
    assert 0.0 <= float(noul.value) <= 1.0

    for out, k in ((choice, 2), (score, 3), (noul, 2)):
        assert int(out.hira.candidate_budget.item()) == k
        assert bool(out.hira.selected_mask.all())
        assert torch.equal(
            out.hira.relation_delta,
            torch.zeros_like(out.hira.relation_delta),
        )
        assert torch.isclose(
            out.probabilities.sum(),
            out.probabilities.new_tensor(1.0),
            atol=1e-6,
        )


def test_option_order_preserves_semantic_choice_inside_one_session():
    model = _synthetic_mainline()
    session = model.open_session(
        "The action must start immediately because even a brief delay is unacceptable."
    )
    options = _options_binary()

    forward = session.decide(
        primitive="choice",
        question_text="Which timing status applies?",
        options=options,
        use_schema_cache=False,
    )
    reversed_out = session.decide(
        primitive="choice",
        question_text="Which timing status applies?",
        options=tuple(reversed(options)),
        use_schema_cache=False,
    )

    assert forward.selected_option_id == reversed_out.selected_option_id
    assert session.query_count == 2
    assert model.runtime.state_encode_calls == 1


def test_m0_parameter_report_separates_resident_and_candidate_surfaces():
    model = _synthetic_mainline()
    report = model.parameter_report().to_dict()

    assert report["resident_total"] > 0
    assert report["semantic_frontend_resident"] > 0
    assert report["relation_core_resident"] > 0
    assert report["transfer_scorer_resident"] > 0
    assert report["projection_parameters"] == 128 * 256
    assert report["transfer_candidate_parameters"] == 8192
    assert report["trainable_total"] == 0


def test_m0_is_frozen_and_rejects_train_mode():
    model = _synthetic_mainline()
    with pytest.raises(RuntimeError, match="frozen inference shell"):
        model.train(True)


def test_manifest_rejects_accidental_production_promotion():
    model = _synthetic_mainline()
    manifest = HiraV0Manifest.m0_provisional()
    bad = HiraV0Manifest(
        **{
            **manifest.to_dict(),
            "production_ready": True,
        }
    )
    with pytest.raises(RuntimeError, match="production readiness"):
        HiraV0Mainline(model.runtime, manifest=bad)


def test_exact_builder_fails_closed_on_wrong_checkpoint_bytes(tmp_path: Path):
    encoder = TrainableSemanticEncoder(
        vocab_size=512,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=32,
    )
    t0 = tmp_path / "wrong-t0.pt"
    transfer = tmp_path / "wrong-transfer.pt"
    t0.write_bytes(b"not-the-frozen-t0")
    transfer.write_bytes(b"not-the-frozen-w34")

    with pytest.raises(RuntimeError, match="SHA mismatch"):
        build_hira_v0_mainline(
            encoder,
            t0,
            transfer,
        )
