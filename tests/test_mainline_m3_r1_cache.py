import torch

from nmd.mainline_m3_alignment import (
    AlignedSemanticEncoder,
    MultilingualAlignmentAdapter,
)
from nmd.mainline_m3_r1_authority import generate_m3_r1_authority
from nmd.mainline_m3_r1_cache import (
    align_compiled_schema,
    align_state_memory,
    compile_m3_r1_base_cache,
    english_anchor_loss,
    paired_alignment_loss,
    validate_m3_r1_base_cache,
)
from nmd.semantic import TrainableSemanticEncoder


def _base():
    torch.manual_seed(19101)
    encoder = TrainableSemanticEncoder(
        vocab_size=4096,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=96,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return encoder


def test_m3_r1_base_cache_is_frozen_state_once_and_bilingual():
    base = _base()
    rows = generate_m3_r1_authority("dev")[:3]
    cache = compile_m3_r1_base_cache(base, rows, partition="dev")

    validate_m3_r1_base_cache(cache, expected_partition="dev")
    metadata = cache["metadata"]
    assert metadata["pair_count"] == 3
    assert metadata["language_case_count"] == 6
    assert metadata["state_encode_count"] == 6
    assert metadata["state_encodes_per_language_case"] == 1.0
    assert metadata["base_encoder_frozen"] is True
    assert metadata["gradient_updates_used"] is False

    for pair in cache["pairs"]:
        assert pair["en"]["memory"].global_embedding.requires_grad is False
        assert pair["vi"]["memory"].global_embedding.requires_grad is False
        assert pair["en"]["schema"].option_view_token_embeddings.requires_grad is False
        assert pair["vi"]["schema"].option_view_token_embeddings.requires_grad is False


def test_m3_r1_aligned_cache_hash_matches_runtime_encoder_identity():
    base = _base()
    row = generate_m3_r1_authority("dev")[:1]
    cache = compile_m3_r1_base_cache(base, row, partition="dev")
    adapter = MultilingualAlignmentAdapter()
    identity = "unit-identity"
    wrapped = AlignedSemanticEncoder(
        base,
        adapter,
        alignment_identity=identity,
    )

    payload = cache["pairs"][0]["en"]
    memory = align_state_memory(
        payload["memory"],
        adapter,
        alignment_identity=identity,
    )
    schema = align_compiled_schema(
        payload["schema"],
        adapter,
        alignment_identity=identity,
    )

    assert memory.model_hash == wrapped.encoder_hash
    assert schema.encoder_hash == wrapped.encoder_hash
    assert memory.model_hash == schema.encoder_hash
    assert memory.tokenizer_hash == wrapped.tokenizer_hash


def test_m3_r1_identity_adapter_keeps_anchor_zero_but_pair_loss_defined():
    base = _base()
    row = generate_m3_r1_authority("dev")[:1]
    cache = compile_m3_r1_base_cache(base, row, partition="dev")
    adapter = MultilingualAlignmentAdapter()
    identity = "identity-init"
    pair = cache["pairs"][0]

    aligned = {}
    for language in ("en", "vi"):
        payload = pair[language]
        aligned[language] = (
            align_state_memory(
                payload["memory"],
                adapter,
                alignment_identity=identity,
            ),
            align_compiled_schema(
                payload["schema"],
                adapter,
                alignment_identity=identity,
            ),
        )

    anchor = english_anchor_loss(
        pair["en"]["memory"],
        pair["en"]["schema"],
        aligned["en"][0],
        aligned["en"][1],
    )
    pair_loss = paired_alignment_loss(
        aligned["en"][0],
        aligned["en"][1],
        aligned["vi"][0],
        aligned["vi"][1],
    )

    assert float(anchor) == 0.0
    assert torch.isfinite(pair_loss)
    assert float(pair_loss) >= 0.0


def test_m3_r1_activated_adapter_backpropagates_without_touching_base_cache():
    base = _base()
    row = generate_m3_r1_authority("train")[:1]
    cache = compile_m3_r1_base_cache(base, row, partition="train")
    adapter = MultilingualAlignmentAdapter()
    pair = cache["pairs"][0]

    aligned_en_memory = align_state_memory(
        pair["en"]["memory"],
        adapter,
        alignment_identity="grad",
    )
    aligned_en_schema = align_compiled_schema(
        pair["en"]["schema"],
        adapter,
        alignment_identity="grad",
    )
    loss = (
        aligned_en_memory.global_embedding.square().mean()
        + aligned_en_schema.question_embedding.square().mean()
    )
    loss.backward()

    assert adapter.down.weight.grad is not None
    assert adapter.up.weight.grad is not None
    assert all(parameter.grad is None for parameter in base.parameters())
    assert pair["en"]["memory"].global_embedding.grad is None
