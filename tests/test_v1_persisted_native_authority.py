from pathlib import Path
import copy

import pytest
import torch

from nmd.v1_persisted_native_authority import (
    S51_NATIVE_AUTHORITY_SCHEMA,
    file_sha256,
    native_tensor_digest,
    save_native_authority,
    validate_native_authority_payload,
    verify_file_sha256,
)


def _payload():
    g=torch.Generator().manual_seed(51001)
    lora={}
    for i in range(4):
        lora[f"lora.{i}.a"]=torch.randn(8,256,generator=g)
        lora[f"lora.{i}.b"]=torch.randn(256,8,generator=g)
    projection=torch.randn(128,256,generator=g)
    digest=native_tensor_digest(lora,projection)
    return {
        "schema_version":S51_NATIVE_AUTHORITY_SCHEMA,
        "kind":"persisted-native-train-only-fixed-epoch24",
        "seed":72001,
        "epoch":24,
        "dev_encoded":False,
        "dev_scored":False,
        "semantic_revision":"test-revision",
        "initialization_t0_sha256":"a"*64,
        "train_manifest_sha256":"b"*64,
        "native_trainable_parameter_count":49152,
        "projection_parameter_count":32768,
        "lora_state_dict":lora,
        "projection_state_dict":{"projection.weight":projection},
        "native_tensor_digest":digest,
        "runtime_state_sha256":digest,
    }


def test_s51_logical_digest_is_stable_across_tensor_clones():
    a=_payload()
    b=copy.deepcopy(a)
    assert native_tensor_digest(
        a["lora_state_dict"],a["projection_state_dict"]["projection.weight"]
    )==native_tensor_digest(
        b["lora_state_dict"],b["projection_state_dict"]["projection.weight"]
    )
    validate_native_authority_payload(a,expected_seed=72001)


def test_s51_save_file_integrity_detects_tamper(tmp_path):
    payload=_payload()
    path=tmp_path/"authority.pt"
    digest=save_native_authority(path,payload)
    assert digest==file_sha256(path)
    verify_file_sha256(path,digest)
    data=bytearray(path.read_bytes())
    data[len(data)//2]^=0x01
    path.write_bytes(bytes(data))
    with pytest.raises(RuntimeError,match="file SHA mismatch"):
        verify_file_sha256(path,digest)


def test_s51_payload_rejects_wrong_revision_t0_and_dev_exposure():
    payload=_payload()
    with pytest.raises(RuntimeError,match="semantic revision"):
        validate_native_authority_payload(payload,expected_semantic_revision="other")
    with pytest.raises(RuntimeError,match="T0 binding"):
        validate_native_authority_payload(payload,expected_t0_sha256="c"*64)
    bad=copy.deepcopy(payload)
    bad["dev_encoded"]=True
    with pytest.raises(RuntimeError,match="exposed DEV"):
        validate_native_authority_payload(bad)


def test_s51_payload_rejects_missing_lora_key_projection_shape_and_nonfinite():
    payload=_payload()
    bad=copy.deepcopy(payload)
    del bad["lora_state_dict"]["lora.3.b"]
    with pytest.raises(RuntimeError,match="LoRA checkpoint keys"):
        validate_native_authority_payload(bad)

    bad=copy.deepcopy(payload)
    bad["projection_state_dict"]["projection.weight"]=torch.zeros(127,256)
    with pytest.raises(RuntimeError,match="projection shape"):
        validate_native_authority_payload(bad)

    bad=copy.deepcopy(payload)
    bad["lora_state_dict"]["lora.0.a"][0,0]=float("nan")
    with pytest.raises(RuntimeError,match="invalid LoRA tensor"):
        validate_native_authority_payload(bad)


def test_s51_payload_rejects_logical_digest_tamper():
    payload=_payload()
    bad=copy.deepcopy(payload)
    bad["lora_state_dict"]["lora.0.a"][0,0]+=1.0
    with pytest.raises(RuntimeError,match="logical tensor digest mismatch"):
        validate_native_authority_payload(bad)
