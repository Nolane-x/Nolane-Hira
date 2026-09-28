from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request


CONTRACT_SCHEMA = "hira-v0-mainline-m5-contract-v1"
RECEIPT_SCHEMA = "hira-v0-mainline-m5-contract-receipt-v1"
M4_BUNDLE_SCHEMA = "hira-v0-mainline-m4-runtime-bundle-v1"

EXPECTED_BASE = "97840f88b592ddfa10ae66e156fab781a4f18af7"
EXPECTED_T0 = "1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f"
EXPECTED_W34 = "d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c"
EXPECTED_A13_REV = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
EXPECTED_A13_SHA = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_contract(contract: dict) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        errors.append("contract schema changed")
    if contract.get("base_main") != EXPECTED_BASE:
        errors.append("M5 base main changed")

    hira = contract.get("frozen_hira")
    if not isinstance(hira, dict):
        return errors + ["frozen_hira missing"]

    expected = {
        "m4_merge_sha": EXPECTED_BASE,
        "m4_package_run_id": 36357825580,
        "m4_package_artifact_id": 10944174953,
        "m4_package_artifact_digest": (
            "sha256:5fa30c93aca4110bc8602f347c2992b7eb034056c01c432d26c84acbfd619f9a"
        ),
        "m4_closure_run_id": 36358061105,
        "m4_closure_artifact_id": 10944248235,
        "m4_closure_artifact_digest": (
            "sha256:753dd5073bec2440f9ede5a3e074e157ed9a31a1eddc170ee425dfe5703324ce"
        ),
        "runtime_manifest_version": "0.0-m4a",
        "w28_t0_sha256": EXPECTED_T0,
        "w34_sha256": EXPECTED_W34,
        "semantic_model": "microsoft/xtremedistil-l6-h256-uncased",
        "semantic_revision": EXPECTED_A13_REV,
        "semantic_weight_sha256": EXPECTED_A13_SHA,
        "state_once": True,
        "full_k": True,
        "relation_refinement": False,
        "adaptive_budget": False,
        "trainable_parameters": 0,
        "production_ready": False,
    }
    for key, value in expected.items():
        if hira.get(key) != value:
            errors.append(f"frozen_hira mismatch: {key}")

    authorities = contract.get("external_authorities")
    if not isinstance(authorities, dict) or set(authorities) != {
        "laya",
        "jev_zero_shot",
        "jev_banking77_adapted",
        "banking77_official",
    }:
        errors.append("external authority set changed")

    claims = contract.get("claim_policy")
    if not isinstance(claims, dict):
        errors.append("claim_policy missing")
    else:
        required_true = {
            "no_cross_protocol_winner_claims",
            "no_global_majority_vote_superiority",
            "missing_is_not_zero",
            "unsupported_is_not_loss",
            "final_test_selection_forbidden",
            "final_labels_forbidden_for_training",
            "public_targets_not_sufficient_for_release_claim",
            "confirmatory_suite_required",
            "systems_and_quality_reported_separately",
        }
        for key in sorted(required_true):
            if claims.get(key) is not True:
                errors.append(f"claim policy not frozen: {key}")

    statuses = contract.get("score_statuses")
    if statuses != [
        "WIN",
        "TIE",
        "LOSS",
        "MISSING",
        "UNSUPPORTED",
        "NOT_COMPARABLE",
    ]:
        errors.append("score status semantics changed")

    return errors


def validate_m4_manifest(contract: dict, manifest: dict) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != M4_BUNDLE_SCHEMA:
        errors.append("M4 bundle schema changed")

    frozen = contract["frozen_hira"]
    hira = manifest.get("hira_manifest")
    if not isinstance(hira, dict):
        return errors + ["M4 Hira manifest missing"]

    if hira.get("version") != frozen["runtime_manifest_version"]:
        errors.append("M4 Hira manifest version mismatch")
    if hira.get("production_ready") is not False:
        errors.append("M4 bundle production-ready overclaim")
    if manifest.get("production_ready_claimed") is not False:
        errors.append("M4 package production-ready overclaim")
    if manifest.get("t0_checkpoint_sha256") != frozen["w28_t0_sha256"]:
        errors.append("M4 T0 identity mismatch")
    if manifest.get("transfer_checkpoint_sha256") != frozen["w34_sha256"]:
        errors.append("M4 W34 identity mismatch")
    if manifest.get("semantic_model") != frozen["semantic_model"]:
        errors.append("M4 semantic model mismatch")
    if manifest.get("semantic_revision") != frozen["semantic_revision"]:
        errors.append("M4 semantic revision mismatch")
    if manifest.get("semantic_weight_sha256") != frozen["semantic_weight_sha256"]:
        errors.append("M4 semantic weight identity mismatch")

    m4a = manifest.get("m4a_authority") or {}
    m4b = manifest.get("m4b_authority") or {}
    m4b1 = manifest.get("m4b1_authority") or {}
    if m4a.get("outcome") != "HIRA_V0_M4_SCHEMA_BATCHING_READY":
        errors.append("M4-A authority changed")
    if m4a.get("artifact_id") != 10934506746:
        errors.append("M4-A artifact changed")
    if m4b.get("outcome") != "HIRA_V0_M4_BOUNDED_CACHE_READY":
        errors.append("M4-B authority changed")
    if m4b.get("artifact_id") != 10934788096:
        errors.append("M4-B artifact changed")
    if m4b1.get("outcome") != "HIRA_V0_M4_BOUNDED_CACHE_FAIL":
        errors.append("M4-B1 immutable failure evidence changed")
    if m4b1.get("artifact_id") != 10934986918:
        errors.append("M4-B1 artifact changed")
    return errors


def verify_github_commit(repo: str, commit: str, token: str | None) -> dict:
    url = f"https://api.github.com/repos/{repo}/commits/{commit}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "nolane-hira-m5-contract",
        },
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    observed = str(payload.get("sha", ""))
    if observed != commit:
        raise RuntimeError(f"remote commit mismatch for {repo}: {observed}")
    return {
        "repository": repo,
        "commit": observed,
        "resolved": True,
    }


def verify_external_authorities(contract: dict) -> list[dict]:
    token = os.environ.get("GITHUB_TOKEN")
    out = []
    for key, row in contract["external_authorities"].items():
        result = verify_github_commit(
            str(row["repository"]),
            str(row["commit"]),
            token,
        )
        result["authority"] = key
        out.append(result)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--m4-runtime", type=Path, required=True)
    parser.add_argument("--verify-github", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    contract = load_json(args.contract)
    manifest = load_json(args.m4_runtime / "runtime-manifest.json")

    errors = validate_contract(contract)
    errors.extend(validate_m4_manifest(contract, manifest))

    remote: list[dict] = []
    if args.verify_github:
        try:
            remote = verify_external_authorities(contract)
        except Exception as exc:
            errors.append(f"external authority resolution failed: {exc}")

    ready = not errors and (not args.verify_github or len(remote) == 4)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS" if ready else "FAIL",
        "outcome": (
            "HIRA_V0_M5_CONTRACT_READY"
            if ready
            else "HIRA_V0_M5_CONTRACT_FAIL"
        ),
        "base_main": contract.get("base_main"),
        "m4_runtime_bound": not validate_m4_manifest(contract, manifest),
        "contract_valid": not validate_contract(contract),
        "remote_verification_requested": args.verify_github,
        "external_authorities": remote,
        "claim_policy": contract.get("claim_policy"),
        "score_statuses": contract.get("score_statuses"),
        "errors": errors,
        "final_scores_exposed": False,
        "production_ready_claimed": False,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))

    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
