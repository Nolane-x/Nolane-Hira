from __future__ import annotations

from pathlib import Path

from .semantic import HFAutoSemanticEncoder
from .v1_s26_semantic_core import (
    HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S26_RELATION_ADDED_PARAMETER_COUNT,
    HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S26_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s26_factorized_relation_core,
    build_s26_factorized_relation_operator,
    enforce_s26_eval,
)

HIRA_V1_S27_SHARED_LORA_PARAMETER_COUNT = HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT
HIRA_V1_S27_PRIMARY_PROJECTION_PARAMETER_COUNT = (
    HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT
)
HIRA_V1_S27_RELATION_PROJECTION_PARAMETER_COUNT = (
    HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT
)
HIRA_V1_S27_TOTAL_PARAMETER_COUNT = HIRA_V1_S26_TOTAL_PARAMETER_COUNT
HIRA_V1_S27_CANONICALIZATION_ADDED_PARAMETER_COUNT = 0
HIRA_V1_S27_INFERENCE_RELATION_ADDED_PARAMETER_COUNT = (
    HIRA_V1_S26_RELATION_ADDED_PARAMETER_COUNT
)


def build_hira_v1_s27_blockwise_canonicalization_core(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str,
    train_lora: bool = True,
    train_primary_projection: bool = True,
    train_relation_projection: bool = True,
):
    return build_hira_v1_s26_factorized_relation_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=train_lora,
        train_primary_projection=train_primary_projection,
        train_relation_projection=train_relation_projection,
    )


def build_s27_factorized_relation_operator():
    return build_s26_factorized_relation_operator()


def enforce_s27_eval(runtime) -> None:
    enforce_s26_eval(runtime)


__all__ = [
    "HIRA_V1_S27_SHARED_LORA_PARAMETER_COUNT",
    "HIRA_V1_S27_PRIMARY_PROJECTION_PARAMETER_COUNT",
    "HIRA_V1_S27_RELATION_PROJECTION_PARAMETER_COUNT",
    "HIRA_V1_S27_TOTAL_PARAMETER_COUNT",
    "HIRA_V1_S27_CANONICALIZATION_ADDED_PARAMETER_COUNT",
    "HIRA_V1_S27_INFERENCE_RELATION_ADDED_PARAMETER_COUNT",
    "build_hira_v1_s27_blockwise_canonicalization_core",
    "build_s27_factorized_relation_operator",
    "enforce_s27_eval",
]
