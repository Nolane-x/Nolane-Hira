from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .mainline import (
    HIRA_V0_MAINLINE_M4_VERSION,
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    HiraV0Mainline,
    build_hira_v0_m4_runtime,
)
from .schema import (
    DEFAULT_SCHEMA_CACHE_MAX_BYTES,
    DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
)
from .semantic import HFAutoSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256

M4_RUNTIME_BUNDLE_SCHEMA = "hira-v0-mainline-m4-runtime-bundle-v1"

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class M4RuntimeBundleManifest:
    schema_version: str
    hira_manifest: dict[str, Any]
    semantic_model: str
    semantic_revision: str
    semantic_weight_sha256: str
    t0_checkpoint: str
    t0_checkpoint_sha256: str
    transfer_checkpoint: str
    transfer_checkpoint_sha256: str
    cache_max_entries: int
    cache_max_bytes: int
    m4a_authority: dict[str, Any]
    m4b_authority: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hira_manifest": self.hira_manifest,
            "semantic_model": self.semantic_model,
            "semantic_revision": self.semantic_revision,
            "semantic_weight_sha256": self.semantic_weight_sha256,
            "t0_checkpoint": self.t0_checkpoint,
            "t0_checkpoint_sha256": self.t0_checkpoint_sha256,
            "transfer_checkpoint": self.transfer_checkpoint,
            "transfer_checkpoint_sha256": self.transfer_checkpoint_sha256,
            "cache_max_entries": self.cache_max_entries,
            "cache_max_bytes": self.cache_max_bytes,
            "m4a_authority": self.m4a_authority,
            "m4b_authority": self.m4b_authority,
        }


def validate_runtime_bundle_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != M4_RUNTIME_BUNDLE_SCHEMA:
        raise ValueError("unexpected M4 runtime bundle schema")

    hira = payload.get("hira_manifest")
    if not isinstance(hira, dict):
        raise ValueError("M4 runtime bundle missing Hira manifest")
    if hira.get("version") != HIRA_V0_MAINLINE_M4_VERSION:
        raise ValueError("M4 runtime bundle version changed")
    if hira.get("production_ready") is not False:
        raise ValueError("M4 runtime bundle cannot claim production readiness")
    if hira.get("typed_runtime") != "available":
        raise ValueError("M4 runtime bundle typed runtime maturity changed")
    if hira.get("high_k_mechanics") != "available":
        raise ValueError("M4 runtime bundle high-K mechanics maturity changed")
    for key in (
        "semantic_frontend",
        "transfer_core",
        "reliability_ood_abstention",
        "high_k",
        "multilingual",
    ):
        if hira.get(key) != "provisional":
            raise ValueError(f"M4 runtime bundle maturity changed: {key}")

    if payload.get("semantic_model") != A13_MODEL:
        raise ValueError("M4 runtime bundle semantic model changed")
    if payload.get("semantic_revision") != A13_REVISION:
        raise ValueError("M4 runtime bundle semantic revision changed")
    if payload.get("semantic_weight_sha256") != A13_WEIGHT_SHA256:
        raise ValueError("M4 runtime bundle semantic weight identity changed")
    if payload.get("t0_checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise ValueError("M4 runtime bundle T0 identity changed")
    if (
        payload.get("transfer_checkpoint_sha256")
        != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256
    ):
        raise ValueError("M4 runtime bundle transfer identity changed")

    cache_entries = int(payload.get("cache_max_entries", -1))
    cache_bytes = int(payload.get("cache_max_bytes", -1))
    if cache_entries < 0 or cache_bytes < 0:
        raise ValueError("M4 runtime bundle cache limits are invalid")

    for authority_key in ("m4a_authority", "m4b_authority"):
        authority = payload.get(authority_key)
        if not isinstance(authority, dict):
            raise ValueError(f"M4 runtime bundle missing {authority_key}")
        if authority.get("outcome") is None:
            raise ValueError(f"M4 runtime bundle {authority_key} lacks outcome")


def read_runtime_bundle_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir)
    path = root / "runtime-manifest.json"
    if not path.is_file():
        raise FileNotFoundError("M4 runtime bundle manifest missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_runtime_bundle_manifest(payload)
    return payload


def _resolve_checkpoint(
    root: Path,
    relative_path: str,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    path = (root / relative_path).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"M4 runtime bundle {label} path escapes bundle root")
    if not path.is_file():
        raise FileNotFoundError(f"M4 runtime bundle {label} checkpoint missing")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"M4 runtime bundle {label} SHA mismatch: {actual} != {expected_sha256}"
        )
    return path


def load_hira_v0_m4_bundle(
    bundle_dir: str | Path,
    *,
    semantic_snapshot_dir: str | Path | None = None,
    cache_max_entries: int | None = None,
    cache_max_bytes: int | None = None,
) -> HiraV0Mainline:
    """Load the frozen M4 CPU runtime from a deterministic local bundle.

    The bundle contains Hira checkpoints and provenance. The external A13
    semantic snapshot may be supplied explicitly for fully local/offline use;
    when omitted it is fetched at the exact pinned revision and then loaded
    locally.
    """
    root = Path(bundle_dir).resolve()
    payload = read_runtime_bundle_manifest(root)

    t0 = _resolve_checkpoint(
        root,
        str(payload["t0_checkpoint"]),
        str(payload["t0_checkpoint_sha256"]),
        label="T0",
    )
    transfer = _resolve_checkpoint(
        root,
        str(payload["transfer_checkpoint"]),
        str(payload["transfer_checkpoint_sha256"]),
        label="transfer",
    )

    from transformers import AutoModel, AutoTokenizer

    if semantic_snapshot_dir is None:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=str(payload["semantic_model"]),
                revision=str(payload["semantic_revision"]),
            )
        )
    else:
        snapshot = Path(semantic_snapshot_dir).resolve()

    weight = snapshot / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError("M4 semantic snapshot model.safetensors missing")
    actual_weight_sha = file_sha256(weight)
    if actual_weight_sha != str(payload["semantic_weight_sha256"]):
        raise RuntimeError("M4 semantic snapshot weight SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=str(payload["semantic_revision"]),
        max_length=256,
    )
    model = build_hira_v0_m4_runtime(
        encoder,
        t0,
        transfer,
        expected_t0_sha256=str(payload["t0_checkpoint_sha256"]),
        expected_transfer_sha256=str(payload["transfer_checkpoint_sha256"]),
        schema_cache_max_entries=(
            int(payload["cache_max_entries"])
            if cache_max_entries is None
            else int(cache_max_entries)
        ),
        schema_cache_max_bytes=(
            int(payload["cache_max_bytes"])
            if cache_max_bytes is None
            else int(cache_max_bytes)
        ),
    )
    if model.manifest.to_dict() != payload["hira_manifest"]:
        raise RuntimeError("M4 loaded runtime manifest differs from bundle manifest")
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M4 local bundle loaded trainable parameters")
    return model


def default_runtime_cache_limits() -> dict[str, int]:
    return {
        "max_entries": DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
        "max_bytes": DEFAULT_SCHEMA_CACHE_MAX_BYTES,
    }


__all__ = [
    "A13_MODEL",
    "A13_REVISION",
    "A13_WEIGHT_SHA256",
    "M4_RUNTIME_BUNDLE_SCHEMA",
    "M4RuntimeBundleManifest",
    "default_runtime_cache_limits",
    "file_sha256",
    "load_hira_v0_m4_bundle",
    "read_runtime_bundle_manifest",
    "validate_runtime_bundle_manifest",
]
