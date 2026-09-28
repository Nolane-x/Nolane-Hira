from types import SimpleNamespace

import torch
from torch import nn

from nmd.semantic import HFAutoSemanticEncoder
from nmd.v1_a13_lora import (
    LoRALinear,
    a13_lora_state_dict,
    inject_a13_last_attention_lora,
    iter_a13_lora_modules,
    load_a13_lora_state_dict,
)


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


def test_s6_lora_zero_init_identity_and_parameter_surface():
    torch.manual_seed(86001)
    encoder = _encoder()
    layer = encoder.model.encoder.layer[-1]
    x = torch.randn(3, 5, 256)

    before = {
        "query": layer.attention.self.query(x).detach().clone(),
        "key": layer.attention.self.key(x).detach().clone(),
        "value": layer.attention.self.value(x).detach().clone(),
        "out": layer.attention.output.dense(x).detach().clone(),
    }

    receipt = inject_a13_last_attention_lora(encoder)

    assert receipt.layer_index == 5
    assert receipt.trainable_parameters == 16384
    assert receipt.original_trainable_parameters == 0
    assert receipt.wrapped_modules == (
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
    )

    modules = iter_a13_lora_modules(encoder)
    assert len(modules) == 4
    assert all(isinstance(module, LoRALinear) for module in modules)
    assert all(module.lora_parameter_count == 4096 for module in modules)

    after = {
        "query": layer.attention.self.query(x),
        "key": layer.attention.self.key(x),
        "value": layer.attention.self.value(x),
        "out": layer.attention.output.dense(x),
    }
    for name in before:
        assert torch.equal(after[name], before[name])

    trainable = {
        name: p.numel()
        for name, p in encoder.model.named_parameters()
        if p.requires_grad
    }
    assert sum(trainable.values()) == 16384
    assert len(trainable) == 8
    assert all(".lora_a" in name or ".lora_b" in name for name in trainable)


def test_s6_lora_update_changes_wrapped_linear_output():
    torch.manual_seed(86002)
    encoder = _encoder()
    inject_a13_last_attention_lora(encoder)
    module = iter_a13_lora_modules(encoder)[0]
    x = torch.randn(2, 4, 256)

    first = module(x).detach().clone()
    with torch.no_grad():
        module.lora_b.normal_(mean=0.0, std=0.02)
    second = module(x).detach().clone()

    assert not torch.equal(first, second)
    assert float((first - second).abs().max()) > 1e-7


def test_s6_lora_checkpoint_roundtrip():
    torch.manual_seed(86003)
    source = _encoder()
    inject_a13_last_attention_lora(source)
    for module in iter_a13_lora_modules(source):
        with torch.no_grad():
            module.lora_a.normal_(mean=0.0, std=0.03)
            module.lora_b.normal_(mean=0.0, std=0.02)

    state = a13_lora_state_dict(source)

    target = _encoder()
    inject_a13_last_attention_lora(target)
    load_a13_lora_state_dict(target, state, freeze=True)

    replay = a13_lora_state_dict(target)
    assert set(replay) == set(state)
    for key in state:
        assert torch.equal(replay[key], state[key])
    assert sum(p.numel() for p in target.parameters() if p.requires_grad) == 0


def test_s6_lora_rejects_duplicate_injection():
    encoder = _encoder()
    inject_a13_last_attention_lora(encoder)
    try:
        inject_a13_last_attention_lora(encoder)
    except RuntimeError as exc:
        assert "already injected" in str(exc)
    else:
        raise AssertionError("duplicate S6 LoRA injection must fail closed")


def test_s6_lora_rejects_wrong_pinned_architecture():
    encoder = _encoder()
    encoder.model.config.num_hidden_layers = 7
    try:
        inject_a13_last_attention_lora(encoder)
    except RuntimeError as exc:
        assert "num_hidden_layers=6" in str(exc)
    else:
        raise AssertionError("wrong A13 architecture must fail closed")


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


def test_s6_builder_exposes_only_last_attention_lora(tmp_path):
    from nmd.v1_s6_semantic_core import (
        HIRA_V1_S6_LORA_PARAMETER_COUNT,
        build_hira_v1_s6_a13_lora_core,
    )

    t0, t0_sha, weight = _write_t0(tmp_path)
    encoder = _encoder()
    runtime = build_hira_v1_s6_a13_lora_core(
        encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=True,
    )

    assert runtime.parameter_free_triadic_scorer is not None
    assert torch.equal(
        runtime.parameter_free_triadic_scorer.projection.weight,
        weight,
    )
    assert runtime.encoder.training is False
    assert runtime.hira.training is False

    trainable = {
        name: p.numel()
        for name, p in runtime.named_parameters()
        if p.requires_grad
    }
    assert sum(trainable.values()) == HIRA_V1_S6_LORA_PARAMETER_COUNT
    assert len(trainable) == 8
    assert all(".lora_a" in name or ".lora_b" in name for name in trainable)
    assert runtime.parameter_free_triadic_scorer.trainable_parameter_count == 0
    assert not any(p.requires_grad for p in runtime.hira.parameters())


def test_s6_builder_can_freeze_complete_candidate(tmp_path):
    from nmd.v1_s6_semantic_core import build_hira_v1_s6_a13_lora_core

    t0, t0_sha, _ = _write_t0(tmp_path)
    runtime = build_hira_v1_s6_a13_lora_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=False,
    )
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 0


def test_s6_enforce_encoder_eval_after_runtime_train(tmp_path):
    from nmd.v1_s6_semantic_core import (
        build_hira_v1_s6_a13_lora_core,
        enforce_s6_encoder_eval,
    )

    t0, t0_sha, _ = _write_t0(tmp_path)
    runtime = build_hira_v1_s6_a13_lora_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=True,
    )
    runtime.train()
    assert runtime.encoder.training is True

    enforce_s6_encoder_eval(runtime)
    assert runtime.encoder.training is False
    assert runtime.hira.training is False
    assert runtime.parameter_free_triadic_scorer.training is False


def test_s6_checkpoint_roundtrip_freezes_replay(tmp_path):
    from nmd.typed_competitive_cache import file_sha256
    from nmd.v1_s6_checkpoint import (
        build_frozen_hira_v1_s6_candidate,
        load_hira_v1_s6_lora_checkpoint,
    )
    from nmd.v1_s6_semantic_core import build_hira_v1_s6_a13_lora_core

    torch.manual_seed(86006)
    t0, t0_sha, _ = _write_t0(tmp_path)

    source = build_hira_v1_s6_a13_lora_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=True,
    )
    for module in iter_a13_lora_modules(source.encoder):
        with torch.no_grad():
            module.lora_a.normal_(mean=0.0, std=0.03)
            module.lora_b.normal_(mean=0.0, std=0.02)
    state = a13_lora_state_dict(source.encoder)

    checkpoint = tmp_path / "s6-lora.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s6-a13-lora-checkpoint-v1",
            "kind": "a13-last-attention-lora",
            "lora_parameter_count": 16384,
            "lora_rank": 8,
            "selected_dev_epoch": 7,
            "semantic_revision": "fake-revision",
            "initialization_t0_sha256": t0_sha,
            "lora_state_dict": state,
        },
        checkpoint,
    )
    checkpoint_sha = file_sha256(checkpoint)

    loaded, metadata = load_hira_v1_s6_lora_checkpoint(
        checkpoint,
        expected_sha256=checkpoint_sha,
        initialization_t0_sha256=t0_sha,
        semantic_revision="fake-revision",
    )
    assert metadata["selected_dev_epoch"] == 7
    assert set(loaded) == set(state)
    for key in state:
        assert torch.equal(loaded[key], state[key])

    replay, replay_meta = build_frozen_hira_v1_s6_candidate(
        _encoder(),
        t0,
        checkpoint,
        expected_t0_sha256=t0_sha,
        expected_lora_sha256=checkpoint_sha,
        semantic_revision="fake-revision",
    )
    assert replay_meta["sha256"] == checkpoint_sha
    assert sum(p.numel() for p in replay.parameters() if p.requires_grad) == 0
    replay_state = a13_lora_state_dict(replay.encoder)
    for key in state:
        assert torch.equal(replay_state[key], state[key])


def test_s6_checkpoint_rejects_wrong_semantic_revision(tmp_path):
    from nmd.v1_s6_checkpoint import load_hira_v1_s6_lora_checkpoint

    checkpoint = tmp_path / "bad-s6-lora.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s6-a13-lora-checkpoint-v1",
            "kind": "a13-last-attention-lora",
            "lora_parameter_count": 16384,
            "lora_rank": 8,
            "selected_dev_epoch": 1,
            "semantic_revision": "wrong",
            "initialization_t0_sha256": "t0",
            "lora_state_dict": {
                **{f"lora.{i}.a": torch.randn(8, 256) for i in range(4)},
                **{f"lora.{i}.b": torch.randn(256, 8) for i in range(4)},
            },
        },
        checkpoint,
    )
    try:
        load_hira_v1_s6_lora_checkpoint(
            checkpoint,
            initialization_t0_sha256="t0",
            semantic_revision="expected",
        )
    except RuntimeError as exc:
        assert "revision" in str(exc)
    else:
        raise AssertionError("wrong S6 semantic revision must fail closed")
