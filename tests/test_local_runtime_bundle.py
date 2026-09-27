import json

import pytest

from nmd.local_runtime import (
    A13_MODEL,
    A13_REVISION,
    A13_WEIGHT_SHA256,
    M4_RUNTIME_BUNDLE_SCHEMA,
    default_runtime_cache_limits,
    read_runtime_bundle_manifest,
    validate_runtime_bundle_manifest,
)
from nmd.mainline import (
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    HiraV0Manifest,
)
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256


def _manifest():
    return {
        "schema_version": M4_RUNTIME_BUNDLE_SCHEMA,
        "hira_manifest": HiraV0Manifest.m4_runtime_provisional().to_dict(),
        "semantic_model": A13_MODEL,
        "semantic_revision": A13_REVISION,
        "semantic_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint": "checkpoints/w28-t0.pt",
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint": "checkpoints/w34-transfer.pt",
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "cache_max_entries": 16,
        "cache_max_bytes": 64 * 1024 * 1024,
        "m4a_authority": {
            "outcome": "HIRA_V0_M4_SCHEMA_BATCHING_READY",
        },
        "m4b_authority": {
            "outcome": "HIRA_V0_M4_BOUNDED_CACHE_READY",
        },
    }


def test_m4_runtime_bundle_manifest_accepts_exact_provenance():
    payload = _manifest()
    validate_runtime_bundle_manifest(payload)
    assert default_runtime_cache_limits() == {
        "max_entries": 16,
        "max_bytes": 64 * 1024 * 1024,
    }


def test_m4_runtime_bundle_manifest_rejects_production_overclaim():
    payload = _manifest()
    payload["hira_manifest"]["production_ready"] = True
    with pytest.raises(ValueError, match="production readiness"):
        validate_runtime_bundle_manifest(payload)


def test_m4_runtime_bundle_manifest_rejects_provenance_changes():
    payload = _manifest()
    payload["t0_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="T0 identity"):
        validate_runtime_bundle_manifest(payload)

    payload = _manifest()
    payload["semantic_revision"] = "floating-main"
    with pytest.raises(ValueError, match="semantic revision"):
        validate_runtime_bundle_manifest(payload)


def test_m4_runtime_bundle_manifest_rejects_negative_cache_limits():
    payload = _manifest()
    payload["cache_max_bytes"] = -1
    with pytest.raises(ValueError, match="cache limits"):
        validate_runtime_bundle_manifest(payload)


def test_read_runtime_bundle_manifest_is_deterministic(tmp_path):
    payload = _manifest()
    path = tmp_path / "runtime-manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = read_runtime_bundle_manifest(tmp_path)
    assert loaded == payload
