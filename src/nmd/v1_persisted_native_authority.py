from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping, Any

import torch
from torch import Tensor

from .local_runtime import load_hira_v0_m4_bundle
from .v1_a13_lora import a13_lora_state_dict, load_a13_lora_state_dict
from .v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)

S51_NATIVE_AUTHORITY_SCHEMA="hira-v1-s51-native-authority-v1"
S51_NATIVE_FIXED_EPOCH=24
S51_NATIVE_TRAINABLE_PARAMETER_COUNT=49_152


def file_sha256(path: Path) -> str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def verify_file_sha256(path: Path, expected: str) -> None:
    actual=file_sha256(path)
    if actual!=str(expected):
        raise RuntimeError(
            f"S51 native authority file SHA mismatch: {actual} != {expected}"
        )


def _digest_tensor(d, name: str, tensor: Tensor) -> None:
    if not isinstance(tensor,Tensor):
        raise TypeError(f"S51 authority tensor missing: {name}")
    value=tensor.detach().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"S51 authority tensor non-finite: {name}")
    d.update(name.encode("utf-8"))
    d.update(str(tuple(value.shape)).encode("ascii"))
    d.update(str(value.dtype).encode("ascii"))
    d.update(value.numpy().tobytes())


def native_tensor_digest(
    lora_state_dict: Mapping[str, Tensor],
    projection_weight: Tensor,
) -> str:
    d=sha256()
    d.update(b"hira-v1-s51-native-authority-tensors-v1")
    for name in sorted(lora_state_dict):
        _digest_tensor(d,name,lora_state_dict[name])
    _digest_tensor(d,"projection.weight",projection_weight)
    return d.hexdigest()


def runtime_state_sha256(runtime) -> str:
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S51 projection scorer missing")
    return native_tensor_digest(
        a13_lora_state_dict(runtime.encoder),
        scorer.projection.weight,
    )


def make_native_authority_payload(
    runtime,
    *,
    seed: int,
    epoch: int,
    semantic_revision: str,
    initialization_t0_sha256: str,
    train_manifest_sha256: str,
) -> dict[str, Any]:
    if int(epoch)!=S51_NATIVE_FIXED_EPOCH:
        raise ValueError("S51 native authority epoch changed")
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S51 projection scorer missing")
    lora={
        name:tensor.detach().cpu().clone().contiguous()
        for name,tensor in a13_lora_state_dict(runtime.encoder).items()
    }
    projection=scorer.projection.weight.detach().cpu().clone().contiguous()
    digest=native_tensor_digest(lora,projection)
    runtime_digest=runtime_state_sha256(runtime)
    if digest!=runtime_digest:
        raise RuntimeError("S51 payload/runtime tensor digest mismatch")
    return {
        "schema_version":S51_NATIVE_AUTHORITY_SCHEMA,
        "kind":"persisted-native-train-only-fixed-epoch24",
        "seed":int(seed),
        "epoch":int(epoch),
        "dev_encoded":False,
        "dev_scored":False,
        "semantic_revision":str(semantic_revision),
        "initialization_t0_sha256":str(initialization_t0_sha256),
        "train_manifest_sha256":str(train_manifest_sha256),
        "native_trainable_parameter_count":S51_NATIVE_TRAINABLE_PARAMETER_COUNT,
        "projection_parameter_count":HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
        "lora_state_dict":lora,
        "projection_state_dict":{"projection.weight":projection},
        "native_tensor_digest":digest,
        "runtime_state_sha256":runtime_digest,
    }


def validate_native_authority_payload(
    payload: Mapping[str, Any],
    *,
    expected_seed: int | None = None,
    expected_semantic_revision: str | None = None,
    expected_t0_sha256: str | None = None,
) -> None:
    if payload.get("schema_version")!=S51_NATIVE_AUTHORITY_SCHEMA:
        raise RuntimeError("S51 native authority schema changed")
    if payload.get("kind")!="persisted-native-train-only-fixed-epoch24":
        raise RuntimeError("S51 native authority kind changed")
    if int(payload.get("epoch",-1))!=S51_NATIVE_FIXED_EPOCH:
        raise RuntimeError("S51 native authority epoch changed")
    if payload.get("dev_encoded") is not False or payload.get("dev_scored") is not False:
        raise RuntimeError("S51 native authority exposed DEV")
    if int(payload.get("native_trainable_parameter_count",-1))!=S51_NATIVE_TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError("S51 native parameter surface changed")
    if int(payload.get("projection_parameter_count",-1))!=HIRA_V1_S17_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S51 projection parameter surface changed")
    if expected_seed is not None and int(payload.get("seed",-1))!=int(expected_seed):
        raise RuntimeError("S51 native authority seed changed")
    if expected_semantic_revision is not None and str(payload.get("semantic_revision"))!=str(expected_semantic_revision):
        raise RuntimeError("S51 native authority semantic revision changed")
    if expected_t0_sha256 is not None and str(payload.get("initialization_t0_sha256"))!=str(expected_t0_sha256):
        raise RuntimeError("S51 native authority T0 binding changed")

    lora=payload.get("lora_state_dict")
    projection_state=payload.get("projection_state_dict")
    if not isinstance(lora,dict):
        raise RuntimeError("S51 native LoRA state missing")
    expected_keys={
        "lora.0.a","lora.0.b","lora.1.a","lora.1.b",
        "lora.2.a","lora.2.b","lora.3.a","lora.3.b",
    }
    if set(lora)!=expected_keys:
        raise RuntimeError("S51 native LoRA checkpoint keys changed")
    if not isinstance(projection_state,dict) or set(projection_state)!={"projection.weight"}:
        raise RuntimeError("S51 native projection checkpoint keys changed")
    projection=projection_state["projection.weight"]
    if tuple(projection.shape)!=(128,256):
        raise RuntimeError("S51 native projection shape changed")
    for name,tensor in lora.items():
        if not isinstance(tensor,Tensor) or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"S51 invalid LoRA tensor: {name}")
    if not isinstance(projection,Tensor) or not bool(torch.isfinite(projection).all()):
        raise RuntimeError("S51 invalid projection tensor")

    digest=native_tensor_digest(lora,projection)
    if digest!=str(payload.get("native_tensor_digest")):
        raise RuntimeError("S51 native logical tensor digest mismatch")
    if digest!=str(payload.get("runtime_state_sha256")):
        raise RuntimeError("S51 native payload runtime digest mismatch")


def save_native_authority(path: Path, payload: Mapping[str, Any]) -> str:
    validate_native_authority_payload(payload)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(payload),path)
    return file_sha256(path)


def load_native_authority(
    *,
    bundle: Path,
    manifest: Mapping[str, Any],
    checkpoint: Path,
    expected_file_sha256: str | None = None,
    expected_seed: int | None = None,
):
    if expected_file_sha256 is not None:
        verify_file_sha256(checkpoint,expected_file_sha256)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if not isinstance(payload,dict):
        raise RuntimeError("S51 native authority payload must be a dict")
    validate_native_authority_payload(
        payload,
        expected_seed=expected_seed,
        expected_semantic_revision=str(manifest["semantic_revision"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
    )

    frozen=load_hira_v0_m4_bundle(bundle)
    runtime=build_hira_v1_s17_norm_balanced_core(
        frozen.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,
        train_projection=True,
    )
    del frozen

    load_a13_lora_state_dict(
        runtime.encoder,
        payload["lora_state_dict"],
        freeze=True,
    )
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S51 projection scorer missing after build")
    projection=payload["projection_state_dict"]["projection.weight"]
    if tuple(projection.shape)!=tuple(scorer.projection.weight.shape):
        raise RuntimeError("S51 loaded projection shape mismatch")
    with torch.no_grad():
        scorer.projection.weight.copy_(
            projection.to(
                device=scorer.projection.weight.device,
                dtype=scorer.projection.weight.dtype,
            )
        )

    for p in runtime.parameters():
        p.requires_grad_(False)
        p.grad=None
    runtime.clear_schema_cache()
    enforce_s17_eval(runtime)

    if any(p.requires_grad for p in runtime.parameters()):
        raise RuntimeError("S51 loaded native runtime remained trainable")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S51 loaded native optimizer surface nonzero")

    loaded_hash=runtime_state_sha256(runtime)
    if loaded_hash!=payload["runtime_state_sha256"]:
        raise RuntimeError(
            f"S51 loaded runtime hash mismatch: {loaded_hash} != {payload['runtime_state_sha256']}"
        )
    return runtime,payload


__all__=[
    "S51_NATIVE_AUTHORITY_SCHEMA",
    "S51_NATIVE_FIXED_EPOCH",
    "S51_NATIVE_TRAINABLE_PARAMETER_COUNT",
    "file_sha256",
    "verify_file_sha256",
    "native_tensor_digest",
    "runtime_state_sha256",
    "make_native_authority_payload",
    "validate_native_authority_payload",
    "save_native_authority",
    "load_native_authority",
]
