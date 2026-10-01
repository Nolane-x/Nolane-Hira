from nmd.v1_s27_semantic_core import (
    HIRA_V1_S27_CANONICALIZATION_ADDED_PARAMETER_COUNT,
    HIRA_V1_S27_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S27_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S27_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S27_TOTAL_PARAMETER_COUNT,
    build_s27_factorized_relation_operator,
)


def test_s27_capacity_is_exact_s26_capacity():
    assert HIRA_V1_S27_SHARED_LORA_PARAMETER_COUNT == 16384
    assert HIRA_V1_S27_PRIMARY_PROJECTION_PARAMETER_COUNT == 32768
    assert HIRA_V1_S27_RELATION_PROJECTION_PARAMETER_COUNT == 32768
    assert HIRA_V1_S27_TOTAL_PARAMETER_COUNT == 81920
    assert HIRA_V1_S27_CANONICALIZATION_ADDED_PARAMETER_COUNT == 0


def test_s27_inference_operator_adds_no_params():
    operator = build_s27_factorized_relation_operator()
    assert operator.factorization_added_parameter_count == 0
