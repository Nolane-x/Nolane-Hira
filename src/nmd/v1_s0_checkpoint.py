from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .typed_competitive_cache import file_sha256
from .v1_semantic_core import (
    HIRA_V1_S0_CANDIDATE_PARAMETER_COUNT,
    HIRA_V1_S0_QUERY_PARAMETER_COUNT,
    HIRA_V1_S0_QUERY_RANK,
    build_hira_v1_s0_query_core,
)
from .w34_transfer_core import W34_CANDIDATE_PARAMETER_COUNT

S0_QUERY_CHECKPOINT_SCHEMA = "hira-v1-s0-query-checkpoint-v1"
S0_QUERY_CHECKPOINT_KIND = "query-conditioned-coevidence"
S0_QUERY_KEYS = {
    "query_basis.weight",
    "query_state.weight",
    "query_schema.weight",
}


def load_hira_v1_s0_query_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_w34_sha256: str,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha = file_sha256(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RuntimeError(
            f"Hira v1 S0 query checkpoint SHA mismatch: "
            f"{actual_sha} != {expected_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != S0_QUERY_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected Hira v1 S0 query checkpoint schema")
    if payload.get("kind") != S0_QUERY_CHECKPOINT_KIND:
        raise RuntimeError("unexpected Hira v1 S0 query checkpoint kind")
    if int(payload.get("query_rank", -1)) != HIRA_V1_S0_QUERY_RANK:
        raise RuntimeError("Hira v1 S0 query rank changed")
    if (
        int(payload.get("query_parameter_count", -1))
        != HIRA_V1_S0_QUERY_PARAMETER_COUNT
    ):
        raise RuntimeError("Hira v1 S0 query parameter count changed")
    if (
        int(payload.get("candidate_parameter_count", -1))
        != HIRA_V1_S0_CANDIDATE_PARAMETER_COUNT
    ):
        raise RuntimeError("Hira v1 S0 candidate parameter count changed")
    if payload.get("t0_checkpoint_sha256") != expected_t0_sha256:
        raise RuntimeError("Hira v1 S0 T0 identity changed")
    if payload.get("w34_checkpoint_sha256") != expected_w34_sha256:
        raise RuntimeError("Hira v1 S0 W34 identity changed")

    state = payload.get("query_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Hira v1 S0 query checkpoint missing state dict")
    state = {str(key): value for key, value in state.items()}
    if set(state) != S0_QUERY_KEYS:
        raise RuntimeError("Hira v1 S0 query checkpoint keys changed")

    shapes = {
        "query_basis.weight": (HIRA_V1_S0_QUERY_RANK, 128),
        "query_state.weight": (HIRA_V1_S0_QUERY_RANK, 128),
        "query_schema.weight": (HIRA_V1_S0_QUERY_RANK, 128),
    }
    for key, shape in shapes.items():
        value = state[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(f"Hira v1 S0 query tensor shape changed: {key}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Hira v1 S0 query tensor is non-finite: {key}")

    selected_epoch = int(payload.get("selected_dev_epoch", -1))
    if selected_epoch < 1:
        raise RuntimeError("Hira v1 S0 selected DEV epoch is invalid")

    metadata = {
        "sha256": actual_sha,
        "selected_dev_epoch": selected_epoch,
        "query_parameter_count": HIRA_V1_S0_QUERY_PARAMETER_COUNT,
        "candidate_parameter_count": HIRA_V1_S0_CANDIDATE_PARAMETER_COUNT,
        "frozen_w34_parameter_count": W34_CANDIDATE_PARAMETER_COUNT,
    }
    return (
        {key: value.detach().float().clone() for key, value in state.items()},
        metadata,
    )


def build_frozen_hira_v1_s0_candidate(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    w34_checkpoint_path: str | Path,
    query_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_w34_sha256: str,
    expected_query_sha256: str | None = None,
):
    runtime = build_hira_v1_s0_query_core(
        encoder,
        t0_checkpoint_path,
        w34_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        expected_w34_sha256=expected_w34_sha256,
        train_query_binding=True,
    )
    query_state, metadata = load_hira_v1_s0_query_checkpoint(
        query_checkpoint_path,
        expected_sha256=expected_query_sha256,
        expected_t0_sha256=expected_t0_sha256,
        expected_w34_sha256=expected_w34_sha256,
    )
    scorer = runtime.query_conditioned_coevidence_scorer
    if scorer is None:
        raise RuntimeError("Hira v1 S0 scorer missing during frozen replay")
    scorer.load_query_state_dict(query_state, freeze=True)

    trainable = sum(
        parameter.numel()
        for parameter in runtime.parameters()
        if parameter.requires_grad
    )
    if trainable != 0:
        raise RuntimeError(
            f"Hira v1 S0 frozen replay has trainable parameters: {trainable}"
        )
    runtime.eval()
    return runtime, metadata


__all__ = [
    "S0_QUERY_CHECKPOINT_KIND",
    "S0_QUERY_CHECKPOINT_SCHEMA",
    "S0_QUERY_KEYS",
    "build_frozen_hira_v1_s0_candidate",
    "load_hira_v1_s0_query_checkpoint",
]
