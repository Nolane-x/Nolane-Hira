from __future__ import annotations

from pathlib import Path

from .hira import HIRACore
from .runtime import NolaneHira
from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256, load_rescued_projection_checkpoint
from .v1_query_keyed_evidence import (
    ParameterFreeQueryEvidenceScorer,
    QueryKeyedEvidenceScorer,
)
from .w34_transfer_core import (
    W34_CANDIDATE_PARAMETER_COUNT,
    W34_CANDIDATE_RANK,
    load_w34_candidate_checkpoint,
)

HIRA_V1_S1_EXTRACTOR_PARAMETER_COUNT = 8192
HIRA_V1_S1_CANDIDATE_PARAMETER_COUNT = (
    W34_CANDIDATE_PARAMETER_COUNT + HIRA_V1_S1_EXTRACTOR_PARAMETER_COUNT
)


def _freeze_encoder_and_hira(encoder: TextSemanticEncoder) -> HIRACore:
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()

    hira = HIRACore(d_model=256, dropout=0.0)
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
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
        raise ValueError("Hira v1 S1 requires d_model=256")
    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path,
        expected_sha256=expected_t0_sha256,
    )
    w34_state, metadata = load_w34_candidate_checkpoint(
        w34_checkpoint_path,
        expected_sha256=expected_w34_sha256,
    )
    return projection, w34_state, metadata


def build_hira_v1_s1_parameter_free_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    w34_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_w34_sha256: str | None = None,
) -> NolaneHira:
    projection, w34_state, _ = _load_base(
        encoder,
        t0_checkpoint_path,
        w34_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        expected_w34_sha256=expected_w34_sha256,
    )
    hira = _freeze_encoder_and_hira(encoder)

    scorer = ParameterFreeQueryEvidenceScorer(
        d_model=256,
        d_rel=128,
        rank=W34_CANDIDATE_RANK,
    )
    scorer.load_projection_weight(projection, freeze=True)
    scorer.load_candidate_state_dict(w34_state, freeze=True)

    if scorer.added_parameter_count != 0:
        raise RuntimeError("S1 A0 parameter-free baseline added parameters")
    if scorer.trainable_parameter_count != 0:
        raise RuntimeError("S1 A0 baseline must be fully frozen")

    runtime = NolaneHira(
        encoder,
        hira=hira,
        parameter_free_query_evidence_scorer=scorer,
    )
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad) != 0:
        raise RuntimeError("S1 A0 runtime unexpectedly has trainable parameters")
    return runtime


def build_hira_v1_s1_query_keyed_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    w34_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_w34_sha256: str | None = None,
    train_extractor: bool = True,
) -> NolaneHira:
    projection, w34_state, _ = _load_base(
        encoder,
        t0_checkpoint_path,
        w34_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        expected_w34_sha256=expected_w34_sha256,
    )
    hira = _freeze_encoder_and_hira(encoder)

    scorer = QueryKeyedEvidenceScorer(
        d_model=256,
        d_rel=128,
        rank=W34_CANDIDATE_RANK,
    )
    scorer.load_projection_weight(projection, freeze=True)
    scorer.load_w34_base_state_dict(w34_state, freeze_base=True)

    if scorer.extractor_parameter_count != HIRA_V1_S1_EXTRACTOR_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S1 extractor parameter count changed")
    if scorer.candidate_parameter_count != HIRA_V1_S1_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S1 candidate parameter count changed")

    if not train_extractor:
        scorer.freeze_extractor()

    runtime = NolaneHira(
        encoder,
        hira=hira,
        query_keyed_evidence_scorer=scorer,
    )
    trainable = sum(
        p.numel()
        for p in runtime.parameters()
        if p.requires_grad
    )
    expected_trainable = HIRA_V1_S1_EXTRACTOR_PARAMETER_COUNT if train_extractor else 0
    if trainable != expected_trainable:
        raise RuntimeError(
            f"Hira v1 S1 optimization surface changed: {trainable} != {expected_trainable}"
        )
    return runtime


__all__ = [
    "HIRA_V1_S1_CANDIDATE_PARAMETER_COUNT",
    "HIRA_V1_S1_EXTRACTOR_PARAMETER_COUNT",
    "build_hira_v1_s1_parameter_free_core",
    "build_hira_v1_s1_query_keyed_core",
]
