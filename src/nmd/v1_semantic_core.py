from __future__ import annotations

from pathlib import Path

from .hira import HIRACore
from .runtime import NolaneHira
from .semantic import TextSemanticEncoder
from .semantic_core import (
    W28_T0_CHECKPOINT_SHA256,
    load_rescued_projection_checkpoint,
)
from .v1_query_conditioned_semantic import QueryConditionedCoEvidenceScorer
from .w34_transfer_core import (
    W34_CANDIDATE_PARAMETER_COUNT,
    W34_CANDIDATE_RANK,
    load_w34_candidate_checkpoint,
)

HIRA_V1_S0_QUERY_RANK = 8
HIRA_V1_S0_QUERY_PARAMETER_COUNT = 3072
HIRA_V1_S0_CANDIDATE_PARAMETER_COUNT = (
    W34_CANDIDATE_PARAMETER_COUNT + HIRA_V1_S0_QUERY_PARAMETER_COUNT
)


def build_hira_v1_s0_query_core(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    w34_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_w34_sha256: str | None = None,
    train_query_binding: bool = True,
) -> NolaneHira:
    """Build the first Hira v1 query-conditioned semantic candidate.

    Frozen:
    - semantic encoder;
    - HIRACore;
    - exact W28 projection;
    - all eight W34 residual/interaction/composition tensors.

    Trainable when train_query_binding=True:
    - query_basis;
    - query_state;
    - query_schema.

    This keeps the first v1 optimization surface exactly 3,072 parameters.
    """
    if int(encoder.d_model) != 256:
        raise ValueError("Hira v1 S0 requires d_model=256")

    projection = load_rescued_projection_checkpoint(
        t0_checkpoint_path,
        expected_sha256=expected_t0_sha256,
    )
    w34_state, _ = load_w34_candidate_checkpoint(
        w34_checkpoint_path,
        expected_sha256=expected_w34_sha256,
    )

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()

    hira = HIRACore(d_model=256, dropout=0.0)
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    hira.eval()

    scorer = QueryConditionedCoEvidenceScorer(
        d_model=256,
        d_rel=128,
        rank=W34_CANDIDATE_RANK,
        query_rank=HIRA_V1_S0_QUERY_RANK,
    )
    scorer.load_projection_weight(projection, freeze=True)
    scorer.load_w34_base_state_dict(w34_state, freeze_base=True)

    if scorer.candidate_parameter_count != HIRA_V1_S0_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S0 candidate parameter count changed")
    if scorer.query_parameter_count != HIRA_V1_S0_QUERY_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S0 query parameter count changed")

    if not train_query_binding:
        scorer.freeze_candidate()

    runtime = NolaneHira(
        encoder,
        hira=hira,
        query_conditioned_coevidence_scorer=scorer,
    )

    trainable = sum(
        parameter.numel()
        for parameter in runtime.parameters()
        if parameter.requires_grad
    )
    expected_trainable = (
        HIRA_V1_S0_QUERY_PARAMETER_COUNT
        if train_query_binding
        else 0
    )
    if trainable != expected_trainable:
        raise RuntimeError(
            "Hira v1 S0 optimization surface changed: "
            f"{trainable} != {expected_trainable}"
        )

    return runtime


__all__ = [
    "HIRA_V1_S0_CANDIDATE_PARAMETER_COUNT",
    "HIRA_V1_S0_QUERY_PARAMETER_COUNT",
    "HIRA_V1_S0_QUERY_RANK",
    "build_hira_v1_s0_query_core",
]
