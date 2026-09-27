import torch

from nmd.mainline_m3_alignment import (
    AlignedSemanticEncoder,
    M3_ALIGNMENT_PARAMETER_COUNT,
    M3_ALIGNMENT_RANK,
    MultilingualAlignmentAdapter,
    alignment_identity_error,
)
from nmd.semantic import TrainableSemanticEncoder


def _base():
    torch.manual_seed(19001)
    return TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )


def test_m3_alignment_capacity_and_identity_init_are_exact():
    base = _base()
    adapter = MultilingualAlignmentAdapter()
    encoder = AlignedSemanticEncoder(base, adapter)

    assert adapter.rank == M3_ALIGNMENT_RANK == 16
    assert adapter.candidate_parameter_count == M3_ALIGNMENT_PARAMETER_COUNT == 8192
    assert adapter.trainable_parameter_count == 8192

    error = alignment_identity_error(
        encoder,
        (
            "The request follows the river route.",
            "Yêu cầu đi theo tuyến sông.",
        ),
    )
    assert error == 0.0


def test_m3_alignment_freezes_base_but_keeps_adapter_trainable():
    base = _base()
    encoder = AlignedSemanticEncoder(base)
    assert all(not p.requires_grad for p in encoder.base.parameters())
    assert encoder.trainable_parameter_count == 8192

    encoder.train(True)
    assert encoder.base.training is False
    assert encoder.adapter.training is True


def test_m3_alignment_updates_tokens_and_pooled_embeddings_after_activation():
    base = _base()
    encoder = AlignedSemanticEncoder(base)

    with torch.no_grad():
        encoder.adapter.up.weight.normal_(mean=0.0, std=0.01)

    base_batch = encoder.encode_base_texts(["paired semantic statement"])
    aligned = encoder.encode_texts(["paired semantic statement"])

    assert aligned.token_embeddings.shape == base_batch.token_embeddings.shape
    assert aligned.pooled_embeddings.shape == base_batch.pooled_embeddings.shape
    assert not torch.equal(aligned.token_embeddings, base_batch.token_embeddings)
    assert not torch.equal(aligned.pooled_embeddings, base_batch.pooled_embeddings)
    assert torch.equal(aligned.attention_mask, base_batch.attention_mask)


def test_m3_alignment_checkpoint_roundtrip_and_freeze():
    torch.manual_seed(19003)
    source = MultilingualAlignmentAdapter()
    with torch.no_grad():
        source.up.weight.normal_(mean=0.0, std=0.02)

    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    target = MultilingualAlignmentAdapter()
    target.load_candidate_state_dict(state, freeze=True)

    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value)
    assert target.trainable_parameter_count == 0


def test_m3_alignment_encoder_hash_tracks_alignment_identity_not_tokenizer():
    base = _base()
    a = AlignedSemanticEncoder(base, alignment_identity="a")
    b = AlignedSemanticEncoder(base, alignment_identity="b")

    assert a.encoder_hash != b.encoder_hash
    assert a.tokenizer_hash == b.tokenizer_hash == base.tokenizer_hash
