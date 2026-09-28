from types import SimpleNamespace

import torch
from torch import nn

from nmd.semantic import HFAutoSemanticEncoder
from nmd.v1_a13_lora import iter_a13_lora_modules


class _FakeSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(256, 256)
        self.key = nn.Linear(256, 256)
        self.value = nn.Linear(256, 256)


class _FakeAttentionOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(256, 256)


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = _FakeSelfAttention()
        self.output = _FakeAttentionOutput()


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _FakeAttention()
        self.ffn = nn.Linear(256, 256)


class _FakeEncoderBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([_FakeLayer() for _ in range(6)])


class _FakeBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="bert",
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            _name_or_path="fake-a13",
        )
        self.encoder = _FakeEncoderBody()
        self.embeddings = nn.Embedding(32, 256)


def _encoder():
    return HFAutoSemanticEncoder(
        _FakeBertModel(),
        tokenizer=SimpleNamespace(name_or_path="fake-tokenizer"),
        revision="fake-revision",
        max_length=64,
    )


def _write_t0(tmp_path):
    from nmd.typed_competitive_cache import file_sha256

    path = tmp_path / "t0.pt"
    weight = torch.randn(128, 256)
    torch.save(
        {
            "schema_version": "r8-w28-candidate-checkpoint-v1",
            "candidate": "T0",
            "kind": "projection",
            "projection_weight": weight,
        },
        path,
    )
    return path, file_sha256(path), weight


def test_s7_builder_exposes_exact_joint_surface(tmp_path):
    from nmd.v1_s7_semantic_core import (
        HIRA_V1_S7_PROJECTION_PARAMETER_COUNT,
        HIRA_V1_S7_TOTAL_PARAMETER_COUNT,
        build_hira_v1_s7_coadapt_core,
    )

    torch.manual_seed(97001)
    t0, t0_sha, weight = _write_t0(tmp_path)
    runtime = build_hira_v1_s7_coadapt_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=True,
        train_projection=True,
    )

    trainable = {
        name: p.numel()
        for name, p in runtime.named_parameters()
        if p.requires_grad
    }
    assert sum(trainable.values()) == HIRA_V1_S7_TOTAL_PARAMETER_COUNT
    assert trainable["projection_triadic_scorer.projection.weight"] == (
        HIRA_V1_S7_PROJECTION_PARAMETER_COUNT
    )

    lora_trainable = sum(
        count for name, count in trainable.items() if ".lora_" in name
    )
    assert lora_trainable == 16384
    assert len([name for name in trainable if ".lora_" in name]) == 8

    scorer = runtime.projection_triadic_scorer
    assert scorer is not None
    assert torch.equal(scorer.projection.weight, weight)
    assert scorer.projection_trainable_parameter_count == 32768

    modules = iter_a13_lora_modules(runtime.encoder)
    assert all(
        torch.equal(module.lora_b, torch.zeros_like(module.lora_b))
        for module in modules
    )

    original_a13 = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    assert original_a13 == 0
    assert not any(p.requires_grad for p in runtime.hira.parameters())


def test_s7_builder_supports_identity_frozen_candidate(tmp_path):
    from nmd.v1_s7_semantic_core import build_hira_v1_s7_coadapt_core

    t0, t0_sha, _ = _write_t0(tmp_path)
    runtime = build_hira_v1_s7_coadapt_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=False,
        train_projection=False,
    )
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 0


def test_s7_can_isolate_each_surface(tmp_path):
    from nmd.v1_s7_semantic_core import build_hira_v1_s7_coadapt_core

    t0, t0_sha, _ = _write_t0(tmp_path)

    lora_only = build_hira_v1_s7_coadapt_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=True,
        train_projection=False,
    )
    assert sum(p.numel() for p in lora_only.parameters() if p.requires_grad) == 16384

    projection_only = build_hira_v1_s7_coadapt_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=False,
        train_projection=True,
    )
    assert sum(
        p.numel() for p in projection_only.parameters() if p.requires_grad
    ) == 32768


def test_s7_eval_enforcement_keeps_trainable_surface(tmp_path):
    from nmd.v1_s7_semantic_core import (
        build_hira_v1_s7_coadapt_core,
        enforce_s7_eval,
    )

    t0, t0_sha, _ = _write_t0(tmp_path)
    runtime = build_hira_v1_s7_coadapt_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=True,
        train_projection=True,
    )
    runtime.train()
    enforce_s7_eval(runtime)

    assert runtime.encoder.training is False
    assert runtime.hira.training is False
    assert runtime.projection_triadic_scorer.training is False
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 49152
