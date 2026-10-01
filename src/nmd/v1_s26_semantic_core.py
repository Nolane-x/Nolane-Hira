from __future__ import annotations

from pathlib import Path

from .semantic import HFAutoSemanticEncoder
from .v1_factorized_relation_signature import (
    FactorizedRoleValueRelationCanonicalizer,
)
from .v1_s25_semantic_core import (
    HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S25_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s25_decoupled_projection_core,
    enforce_s25_eval,
    get_s25_relation_projection,
)

HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT = HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT
HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT = (
    HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT
)
HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT = (
    HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT
)
HIRA_V1_S26_TOTAL_PARAMETER_COUNT = HIRA_V1_S25_TOTAL_PARAMETER_COUNT
HIRA_V1_S26_RELATION_ADDED_PARAMETER_COUNT = 0


def build_hira_v1_s26_factorized_relation_core(
    encoder: HFAutoSemanticEncoder,
    t0_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str,
    train_lora: bool = True,
    train_primary_projection: bool = True,
    train_relation_projection: bool = True,
):
    runtime = build_hira_v1_s25_decoupled_projection_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=train_lora,
        train_primary_projection=train_primary_projection,
        train_relation_projection=train_relation_projection,
    )
    # S26 changes no learned runtime state.  The relation operator is
    # instantiated per relation court and consumes S25's private relation
    # projection explicitly.
    operator = FactorizedRoleValueRelationCanonicalizer(
        role_temperature=0.10,
        pair_temperature=0.10,
        contrastive_temperature=0.10,
    )
    if operator.factorization_added_parameter_count != 0:
        raise RuntimeError("Hira v1 S26 relation operator unexpectedly added params")
    get_s25_relation_projection(runtime)
    return runtime


def build_s26_factorized_relation_operator() -> FactorizedRoleValueRelationCanonicalizer:
    operator = FactorizedRoleValueRelationCanonicalizer(
        role_temperature=0.10,
        pair_temperature=0.10,
        contrastive_temperature=0.10,
    )
    if operator.factorization_added_parameter_count != 0:
        raise RuntimeError("Hira v1 S26 relation operator unexpectedly added params")
    return operator


def enforce_s26_eval(runtime) -> None:
    enforce_s25_eval(runtime)


__all__ = [
    "HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT",
    "HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT",
    "HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT",
    "HIRA_V1_S26_TOTAL_PARAMETER_COUNT",
    "HIRA_V1_S26_RELATION_ADDED_PARAMETER_COUNT",
    "build_hira_v1_s26_factorized_relation_core",
    "build_s26_factorized_relation_operator",
    "enforce_s26_eval",
]
