from nmd.v1_s28_semantic_core import (
    HIRA_V1_S28_ANCHOR_TRANSPORT_ADDED_PARAMETER_COUNT,
    HIRA_V1_S28_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S28_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S28_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S28_TOTAL_PARAMETER_COUNT,
    build_s28_factorized_relation_operator,
)


def test_s28_capacity_is_exact_s27_capacity():
    assert HIRA_V1_S28_SHARED_LORA_PARAMETER_COUNT == 16384
    assert HIRA_V1_S28_PRIMARY_PROJECTION_PARAMETER_COUNT == 32768
    assert HIRA_V1_S28_RELATION_PROJECTION_PARAMETER_COUNT == 32768
    assert HIRA_V1_S28_TOTAL_PARAMETER_COUNT == 81920
    assert HIRA_V1_S28_ANCHOR_TRANSPORT_ADDED_PARAMETER_COUNT == 0


def test_s28_inference_operator_adds_no_params():
    operator = build_s28_factorized_relation_operator()
    assert operator.factorization_added_parameter_count == 0
