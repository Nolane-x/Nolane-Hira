from __future__ import annotations

from pathlib import Path

from .hira import HIRACore
from .runtime import NolaneHira
from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256, load_rescued_projection_checkpoint
from .v1_query_token_fusion import QuestionAsEvidenceScorer, QueryTokenResidualFusionScorer
from .w34_transfer_core import W34_CANDIDATE_PARAMETER_COUNT, W34_CANDIDATE_RANK, load_w34_candidate_checkpoint

HIRA_V1_S2_FUSION_PARAMETER_COUNT = 8192
HIRA_V1_S2_CANDIDATE_PARAMETER_COUNT = W34_CANDIDATE_PARAMETER_COUNT + HIRA_V1_S2_FUSION_PARAMETER_COUNT


def _freeze_encoder_hira(encoder: TextSemanticEncoder) -> HIRACore:
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    hira = HIRACore(d_model=256, dropout=0.0)
    for p in hira.parameters():
        p.requires_grad_(False)
    hira.eval()
    return hira


def _load_base(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    w34_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str,
    expected_w34_sha256: str | None,
):
    if int(encoder.d_model) != 256:
        raise ValueError("Hira v1 S2 requires d_model=256")
    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path, expected_sha256=expected_t0_sha256
    )
    state, meta = load_w34_candidate_checkpoint(
        w34_checkpoint_path, expected_sha256=expected_w34_sha256
    )
    return projection, state, meta


def build_hira_v1_s2_question_as_evidence_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    w34_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_w34_sha256: str | None = None,
) -> NolaneHira:
    projection, state, _ = _load_base(
        encoder, t0_checkpoint_path, w34_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        expected_w34_sha256=expected_w34_sha256,
    )
    hira = _freeze_encoder_hira(encoder)
    scorer = QuestionAsEvidenceScorer(d_model=256, d_rel=128, rank=W34_CANDIDATE_RANK)
    scorer.load_projection_weight(projection, freeze=True)
    scorer.load_candidate_state_dict(state, freeze=True)
    if scorer.added_parameter_count != 0 or scorer.trainable_parameter_count != 0:
        raise RuntimeError("S2 A0 must remain zero-parameter and frozen")
    runtime = NolaneHira(encoder, hira=hira, question_as_evidence_scorer=scorer)
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad) != 0:
        raise RuntimeError("S2 A0 runtime unexpectedly trainable")
    return runtime


def build_hira_v1_s2_fusion_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    w34_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_w34_sha256: str | None = None,
    train_fusion: bool = True,
) -> NolaneHira:
    projection, state, _ = _load_base(
        encoder, t0_checkpoint_path, w34_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        expected_w34_sha256=expected_w34_sha256,
    )
    hira = _freeze_encoder_hira(encoder)
    scorer = QueryTokenResidualFusionScorer(d_model=256, d_rel=128, rank=W34_CANDIDATE_RANK)
    scorer.load_projection_weight(projection, freeze=True)
    scorer.load_w34_base_state_dict(state, freeze_base=True)
    if scorer.fusion_parameter_count != HIRA_V1_S2_FUSION_PARAMETER_COUNT:
        raise RuntimeError("S2 fusion parameter count changed")
    if scorer.candidate_parameter_count != HIRA_V1_S2_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("S2 candidate parameter count changed")
    if not train_fusion:
        scorer.freeze_fusion()
    runtime = NolaneHira(encoder, hira=hira, query_token_residual_fusion_scorer=scorer)
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    expected = HIRA_V1_S2_FUSION_PARAMETER_COUNT if train_fusion else 0
    if trainable != expected:
        raise RuntimeError(f"S2 optimization surface changed: {trainable} != {expected}")
    return runtime


__all__ = [
    "HIRA_V1_S2_FUSION_PARAMETER_COUNT",
    "HIRA_V1_S2_CANDIDATE_PARAMETER_COUNT",
    "build_hira_v1_s2_question_as_evidence_core",
    "build_hira_v1_s2_fusion_core",
]
