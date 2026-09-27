from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

M4_RUNTIME_BUNDLE_SCHEMA = "hira-v0-mainline-m4-runtime-bundle-v1"
A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
T0_SHA = "1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f"
W34_SHA = "d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_checkpoint(
    source_dir: Path,
    destination: Path,
    *,
    expected_schema: str,
    expected_sha: str,
    receipt_sha_key: str = "checkpoint_sha256",
) -> None:
    receipt_path = source_dir / "receipt.json"
    checkpoint = source_dir / "candidate.pt"
    if not receipt_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint artifact incomplete: {source_dir}")
    receipt = _read(receipt_path)
    if receipt.get("schema_version") != expected_schema:
        raise RuntimeError(f"unexpected checkpoint receipt schema: {source_dir}")
    if receipt.get(receipt_sha_key) != expected_sha:
        raise RuntimeError(f"checkpoint receipt SHA changed: {source_dir}")
    if file_sha256(checkpoint) != expected_sha:
        raise RuntimeError(f"checkpoint bytes changed: {source_dir}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)


def _validate_m4a(directory: Path) -> dict[str, object]:
    row = _read(directory / "receipt.json")
    if row.get("schema_version") != "hira-v0-mainline-m4-schema-benchmark-v1":
        raise RuntimeError("unexpected M4-A receipt schema")
    if row.get("outcome") != "HIRA_V0_M4_SCHEMA_BATCHING_READY":
        raise RuntimeError("M4-C requires qualified M4-A")
    if not all(row.get("gates", {}).values()):
        raise RuntimeError("M4-C M4-A gates changed")
    return row


def _validate_m4b1(directory: Path) -> dict[str, object]:
    row = _read(directory / "receipt.json")
    if row.get("schema_version") != "hira-v0-mainline-m4-cache-stress-v1":
        raise RuntimeError("unexpected M4-B1 receipt schema")
    if row.get("outcome") != "HIRA_V0_M4_BOUNDED_CACHE_FAIL":
        raise RuntimeError("M4-C requires immutable M4-B1 failure evidence")
    return row


def _validate_m4b2(directory: Path) -> dict[str, object]:
    row = _read(directory / "receipt.json")
    if row.get("schema_version") != "hira-v0-mainline-m4-cache-witness-v2":
        raise RuntimeError("unexpected M4-B2 receipt schema")
    if row.get("outcome") != "HIRA_V0_M4_BOUNDED_CACHE_READY":
        raise RuntimeError("M4-C requires qualified M4-B2")
    if not all(row.get("gates", {}).values()):
        raise RuntimeError("M4-C M4-B2 gates changed")
    return row


def _single_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"M4-C expected exactly one wheel, got {len(wheels)}")
    return wheels[0]


def _manifest_lines(root: Path, *, skip_name: str) -> list[str]:
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == skip_name:
            continue
        lines.append(f"{file_sha256(path)}  {path.relative_to(root).as_posix()}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--m4a-dir", type=Path, required=True)
    parser.add_argument("--m4b1-dir", type=Path, required=True)
    parser.add_argument("--m4b2-dir", type=Path, required=True)
    parser.add_argument("--m4a-artifact-id", type=int, required=True)
    parser.add_argument("--m4a-artifact-digest", required=True)
    parser.add_argument("--m4b1-artifact-id", type=int, required=True)
    parser.add_argument("--m4b1-artifact-digest", required=True)
    parser.add_argument("--m4b2-run-id", type=int, required=True)
    parser.add_argument("--m4b2-artifact-id", type=int, required=True)
    parser.add_argument("--m4b2-artifact-digest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    m4a = _validate_m4a(args.m4a_dir.resolve())
    m4b1 = _validate_m4b1(args.m4b1_dir.resolve())
    m4b2 = _validate_m4b2(args.m4b2_dir.resolve())

    checkpoint_dir = out / "checkpoints"
    t0_out = checkpoint_dir / "w28-t0.pt"
    w34_out = checkpoint_dir / "w34-transfer.pt"
    _copy_checkpoint(
        args.t0_dir.resolve(),
        t0_out,
        expected_schema="r8-w28-candidate-receipt-v1",
        expected_sha=T0_SHA,
    )
    _copy_checkpoint(
        args.w34_dir.resolve(),
        w34_out,
        expected_schema="r8-w34-coevidence-training-receipt-v1",
        expected_sha=W34_SHA,
    )

    wheel = _single_wheel(args.dist_dir.resolve())
    wheel_out = out / "python" / wheel.name
    wheel_out.parent.mkdir(parents=True)
    shutil.copy2(wheel, wheel_out)

    evidence = out / "evidence"
    evidence.mkdir()
    shutil.copy2(args.m4a_dir.resolve() / "receipt.json", evidence / "m4a-schema.json")
    shutil.copy2(args.m4b1_dir.resolve() / "receipt.json", evidence / "m4b1-stress-fail.json")
    shutil.copy2(args.m4b2_dir.resolve() / "receipt.json", evidence / "m4b2-cache-ready.json")

    hira_manifest = m4a.get("manifest")
    if not isinstance(hira_manifest, dict):
        raise RuntimeError("M4-A receipt missing Hira manifest")
    if hira_manifest.get("version") != "0.0-m4a":
        raise RuntimeError("M4-C Hira manifest version changed")
    if hira_manifest.get("production_ready") is not False:
        raise RuntimeError("M4-C cannot package production-ready overclaim")

    manifest = {
        "schema_version": M4_RUNTIME_BUNDLE_SCHEMA,
        "hira_manifest": hira_manifest,
        "semantic_model": A13_MODEL,
        "semantic_revision": A13_REVISION,
        "semantic_weight_sha256": A13_WEIGHT_SHA256,
        "semantic_snapshot_packaged": False,
        "semantic_snapshot_requirement": (
            "Provide an exact local snapshot containing model.safetensors with "
            "the pinned SHA; the loader also supports fetching the pinned revision."
        ),
        "t0_checkpoint": "checkpoints/w28-t0.pt",
        "t0_checkpoint_sha256": T0_SHA,
        "transfer_checkpoint": "checkpoints/w34-transfer.pt",
        "transfer_checkpoint_sha256": W34_SHA,
        "python_wheel": f"python/{wheel.name}",
        "python_wheel_sha256": file_sha256(wheel_out),
        "cache_max_entries": 16,
        "cache_max_bytes": 67108864,
        "m4a_authority": {
            "run_id": 36327904288,
            "artifact_id": int(args.m4a_artifact_id),
            "artifact_digest": args.m4a_artifact_digest,
            "outcome": m4a["outcome"],
        },
        "m4b1_authority": {
            "run_id": 36328918665,
            "artifact_id": int(args.m4b1_artifact_id),
            "artifact_digest": args.m4b1_artifact_digest,
            "outcome": m4b1["outcome"],
            "role": "immutable failed stress-harness evidence",
        },
        "m4b_authority": {
            "run_id": int(args.m4b2_run_id),
            "artifact_id": int(args.m4b2_artifact_id),
            "artifact_digest": args.m4b2_artifact_digest,
            "outcome": m4b2["outcome"],
        },
        "production_ready_claimed": False,
    }
    manifest_path = out / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readme = out / "README.txt"
    readme.write_text(
        "Nolane HIRA v0 M4 local runtime bundle\n"
        "\n"
        "This package is a frozen research runtime, not a production-ready claim.\n"
        "Install the wheel under python/, provide the exact pinned A13 local snapshot,\n"
        "then call nmd.local_runtime.load_hira_v0_m4_bundle().\n"
        "\n"
        "Scientific maturity remains provisional for semantic quality, reliability,\n"
        "high-K semantic quality, and multilingual capability.\n",
        encoding="utf-8",
    )

    integrity = out / "INTEGRITY.sha256"
    integrity.write_text(
        "\n".join(_manifest_lines(out, skip_name=integrity.name)) + "\n",
        encoding="utf-8",
    )

    print("M4_RUNTIME_BUNDLE=" + str(out))
    print("M4_RUNTIME_MANIFEST_SHA256=" + file_sha256(manifest_path))
    print("M4_RUNTIME_WHEEL_SHA256=" + file_sha256(wheel_out))


if __name__ == "__main__":
    main()
