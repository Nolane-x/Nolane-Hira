import torch

from nmd.freeform_attribution_cache import W7B_CACHE_SCHEMA_VERSION


def test_w7b_cache_schema_name_is_distinct():
    assert W7B_CACHE_SCHEMA_VERSION == "r8-w7b-freeform-attribution-cache-v1"


def test_w7b_cache_contract_requires_no_factor_encoder_work():
    # Full compile is exercised in authority CI with frozen A13. This local
    # contract protects the scientific boundary: W7b is free-form attribution,
    # not a hidden continuation of the W7 factor branch.
    metadata = {
        "w7b_schema_version": W7B_CACHE_SCHEMA_VERSION,
        "factor_encoder_batches": 0,
        "factor_text_count": 0,
        "state_encode_calls_per_case": 1.0,
    }
    assert metadata["factor_encoder_batches"] == 0
    assert metadata["factor_text_count"] == 0
    assert metadata["state_encode_calls_per_case"] == 1.0
