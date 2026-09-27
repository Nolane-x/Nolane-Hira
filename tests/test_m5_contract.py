from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "hira_v0_m5_contract_preflight.py"
    spec = importlib.util.spec_from_file_location("hira_v0_m5_contract_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_contract() -> dict:
    return json.loads(
        (ROOT / "benchmarks" / "m5_contract.json").read_text(encoding="utf-8")
    )


def exact_m4_manifest() -> dict:
    return {
        "schema_version": "hira-v0-mainline-m4-runtime-bundle-v1",
        "hira_manifest": {
            "version": "0.0-m4a",
            "production_ready": False,
        },
        "production_ready_claimed": False,
        "t0_checkpoint_sha256": "1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f",
        "transfer_checkpoint_sha256": "d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c",
        "semantic_model": "microsoft/xtremedistil-l6-h256-uncased",
        "semantic_revision": "4226d9e4d2c08703e5cb0491b479bfc6a1607181",
        "semantic_weight_sha256": "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880",
        "m4a_authority": {
            "outcome": "HIRA_V0_M4_SCHEMA_BATCHING_READY",
            "artifact_id": 10934506746,
        },
        "m4b1_authority": {
            "outcome": "HIRA_V0_M4_BOUNDED_CACHE_FAIL",
            "artifact_id": 10934986918,
        },
        "m4b_authority": {
            "outcome": "HIRA_V0_M4_BOUNDED_CACHE_READY",
            "artifact_id": 10934788096,
        },
    }


def test_m5_contract_is_frozen_and_valid():
    module = load_module()
    assert module.validate_contract(load_contract()) == []


def test_m5_contract_binds_exact_m4_runtime_authority():
    module = load_module()
    errors = module.validate_m4_manifest(load_contract(), exact_m4_manifest())
    assert errors == []


def test_m5_contract_rejects_checkpoint_drift():
    module = load_module()
    manifest = exact_m4_manifest()
    manifest["t0_checkpoint_sha256"] = "0" * 64
    errors = module.validate_m4_manifest(load_contract(), manifest)
    assert "M4 T0 identity mismatch" in errors


def test_m5_contract_rejects_global_majority_vote_claim():
    module = load_module()
    contract = load_contract()
    contract["claim_policy"]["no_global_majority_vote_superiority"] = False
    errors = module.validate_contract(contract)
    assert "claim policy not frozen: no_global_majority_vote_superiority" in errors


def test_m5_contract_preserves_missing_and_unsupported_semantics():
    contract = load_contract()
    assert contract["score_statuses"] == [
        "WIN",
        "TIE",
        "LOSS",
        "MISSING",
        "UNSUPPORTED",
        "NOT_COMPARABLE",
    ]
    assert contract["claim_policy"]["missing_is_not_zero"] is True
    assert contract["claim_policy"]["unsupported_is_not_loss"] is True
