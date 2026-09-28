from types import SimpleNamespace

import torch
from torch import nn

from nmd.semantic import HFAutoSemanticEncoder
from nmd.v1_a13_lora import a13_lora_state_dict, iter_a13_lora_modules


class _SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(256, 256)
        self.key = nn.Linear(256, 256)
        self.value = nn.Linear(256, 256)


class _AttentionOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(256, 256)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = _SelfAttention()
        self.output = _AttentionOutput()


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()
        self.ffn = nn.Linear(256, 256)


class _EncoderBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([_Layer() for _ in range(6)])


class _FakeBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="bert",
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            _name_or_path="fake-a13",
        )
        self.encoder = _EncoderBody()
        self.embeddings = nn.Embedding(32, 256)


def _encoder():
    return HFAutoSemanticEncoder(
        _FakeBert(),
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
    return path, file_sha256(path)


def test_s8_checkpoint_roundtrip_freezes_candidate(tmp_path):
    from nmd.typed_competitive_cache import file_sha256
    from nmd.v1_s8_checkpoint import (
        build_frozen_hira_v1_s8_candidate,
        load_hira_v1_s8_checkpoint,
    )
    from nmd.v1_s8_semantic_core import build_hira_v1_s8_invariant_core

    torch.manual_seed(108001)
    t0, t0_sha = _write_t0(tmp_path)
    source = build_hira_v1_s8_invariant_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=True,
        train_projection=True,
    )

    for module in iter_a13_lora_modules(source.encoder):
        with torch.no_grad():
            module.lora_a.normal_(0.0, 0.03)
            module.lora_b.normal_(0.0, 0.02)

    scorer = source.projection_triadic_scorer
    assert scorer is not None
    with torch.no_grad():
        scorer.projection.weight.add_(
            0.01 * torch.randn_like(scorer.projection.weight)
        )

    lora = a13_lora_state_dict(source.encoder)
    projection = {
        "projection.weight": scorer.projection.weight.detach().cpu().clone(),
    }
    checkpoint = tmp_path / "s8.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s8-invariant-checkpoint-v1",
            "kind": "paraphrase-invariant-a13-w28",
            "lora_parameter_count": 16384,
            "projection_parameter_count": 32768,
            "total_parameter_count": 49152,
            "lora_rank": 8,
            "selected_dev_epoch": 6,
            "semantic_revision": "fake-revision",
            "initialization_t0_sha256": t0_sha,
            "lora_state_dict": lora,
            "projection_state_dict": projection,
        },
        checkpoint,
    )
    checkpoint_sha = file_sha256(checkpoint)

    loaded_lora, loaded_projection, metadata = load_hira_v1_s8_checkpoint(
        checkpoint,
        expected_sha256=checkpoint_sha,
        initialization_t0_sha256=t0_sha,
        semantic_revision="fake-revision",
    )
    assert metadata["selected_dev_epoch"] == 6
    for key in lora:
        assert torch.equal(loaded_lora[key], lora[key])
    assert torch.equal(
        loaded_projection["projection.weight"],
        projection["projection.weight"],
    )

    replay, replay_meta = build_frozen_hira_v1_s8_candidate(
        _encoder(),
        t0,
        checkpoint,
        expected_t0_sha256=t0_sha,
        expected_invariant_sha256=checkpoint_sha,
        semantic_revision="fake-revision",
    )
    assert replay_meta["sha256"] == checkpoint_sha
    assert sum(p.numel() for p in replay.parameters() if p.requires_grad) == 0
    replay_scorer = replay.projection_triadic_scorer
    assert replay_scorer is not None
    assert torch.equal(
        replay_scorer.projection.weight,
        projection["projection.weight"],
    )


def test_s8_checkpoint_rejects_wrong_revision(tmp_path):
    from nmd.v1_s8_checkpoint import load_hira_v1_s8_checkpoint

    t0, t0_sha = _write_t0(tmp_path)
    checkpoint = tmp_path / "bad-s8.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s8-invariant-checkpoint-v1",
            "kind": "paraphrase-invariant-a13-w28",
            "lora_parameter_count": 16384,
            "projection_parameter_count": 32768,
            "total_parameter_count": 49152,
            "lora_rank": 8,
            "selected_dev_epoch": 1,
            "semantic_revision": "wrong",
            "initialization_t0_sha256": t0_sha,
            "lora_state_dict": {
                **{f"lora.{i}.a": torch.randn(8, 256) for i in range(4)},
                **{f"lora.{i}.b": torch.randn(256, 8) for i in range(4)},
            },
            "projection_state_dict": {
                "projection.weight": torch.randn(128, 256),
            },
        },
        checkpoint,
    )
    try:
        load_hira_v1_s8_checkpoint(
            checkpoint,
            initialization_t0_sha256=t0_sha,
            semantic_revision="expected",
        )
    except RuntimeError as exc:
        assert "revision" in str(exc)
    else:
        raise AssertionError("wrong S8 semantic revision must fail closed")
