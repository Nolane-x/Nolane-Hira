import torch
import torch.nn.functional as F

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.schema import SchemaCompiler
from nmd.semantic import TrainableSemanticEncoder


class CountingEncoder(TrainableSemanticEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_encode_calls = 0

    def encode_texts(self, texts):
        self.text_encode_calls += 1
        return super().encode_texts(texts)


def _encoder(*, d_model=64):
    enc = CountingEncoder(
        vocab_size=4096,
        d_model=d_model,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    enc.eval()
    return enc


def _options(k=4):
    return tuple(
        LogicalOption(
            option_id=f"o-{i}",
            criterion_text=f"route {i} is explicitly selected",
            aliases=(f"use processing route {i}",),
            exemplars=(f"example routed through lane {i}",),
        )
        for i in range(k)
    )


@torch.inference_mode()
def _legacy_question_and_logical(encoder, question, options):
    q = encoder.encode_texts([question]).pooled_embeddings[0]
    logical = []
    for option in options:
        texts = [
            option.criterion_text,
            *option.aliases,
            *option.exemplars,
        ]
        texts = [text for text in texts if text and text.strip()]
        proto = encoder.encode_texts(texts).pooled_embeddings
        emb = F.normalize(proto, dim=-1).mean(0)
        logical.append(F.normalize(emb, dim=-1))
    return q, torch.stack(logical)


def test_batched_schema_matches_legacy_logical_embeddings():
    torch.manual_seed(41001)
    optimized = _encoder()
    legacy = _encoder()
    legacy.load_state_dict(optimized.state_dict())
    question = "which processing route applies?"
    options = _options(8)

    compiler = SchemaCompiler(optimized)
    schema, receipt = compiler.compile(
        primitive="choice",
        question_text=question,
        options=options,
        use_cache=False,
        include_token_artifacts=False,
    )
    legacy_q, legacy_options = _legacy_question_and_logical(
        legacy,
        question,
        options,
    )

    assert receipt.prototype_count == 24
    assert optimized.text_encode_calls == 2
    assert legacy.text_encode_calls == 1 + len(options)
    assert torch.allclose(schema.question_embedding, legacy_q, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        schema.option_embeddings,
        legacy_options,
        atol=2e-5,
        rtol=2e-5,
    )


def test_token_artifact_compile_reuses_single_positive_view_batch():
    torch.manual_seed(41003)
    encoder = _encoder()
    compiler = SchemaCompiler(encoder)
    options = _options(12)

    schema, receipt = compiler.compile(
        primitive="choice",
        question_text="which route applies?",
        options=options,
        use_cache=False,
        include_token_artifacts=True,
    )

    assert receipt.prototype_count == 36
    assert encoder.text_encode_calls == 3
    assert schema.option_embeddings.shape == (12, encoder.d_model)
    assert schema.question_token_embeddings is not None
    assert schema.option_token_embeddings is not None
    assert schema.option_view_token_embeddings is not None
    assert schema.option_view_token_mask is not None
    assert schema.option_view_mask is not None
    assert schema.option_view_mask.shape == (12, 3)
    assert bool(schema.option_view_mask.all())


def test_k255_cold_schema_compile_is_constant_encoder_calls():
    torch.manual_seed(41007)
    encoder = _encoder(d_model=32)
    compiler = SchemaCompiler(encoder)
    options = tuple(
        LogicalOption(
            option_id=f"id-{i}",
            criterion_text=f"semantic route {i}",
        )
        for i in range(255)
    )

    schema, receipt = compiler.compile(
        primitive="choice",
        question_text="which semantic route?",
        options=options,
        use_cache=True,
        include_token_artifacts=True,
    )

    assert receipt.option_count == 255
    assert receipt.prototype_count == 255
    assert encoder.text_encode_calls == 3
    assert schema.option_embeddings.shape == (255, 32)
    assert schema.option_view_mask.shape == (255, 1)

    before = encoder.text_encode_calls
    cached, cached_receipt = compiler.compile(
        primitive="choice",
        question_text="which semantic route?",
        options=options,
        use_cache=True,
        include_token_artifacts=True,
    )
    assert cached_receipt.cache_hit is True
    assert encoder.text_encode_calls == before
    assert cached is schema


def test_no_token_artifacts_k255_uses_two_encoder_calls():
    torch.manual_seed(41009)
    encoder = _encoder(d_model=32)
    compiler = SchemaCompiler(encoder)
    options = tuple(
        LogicalOption(f"id-{i}", f"route {i}")
        for i in range(255)
    )

    schema, _ = compiler.compile(
        primitive="choice",
        question_text="which route?",
        options=options,
        use_cache=False,
        include_token_artifacts=False,
    )

    assert encoder.text_encode_calls == 2
    assert schema.option_embeddings.shape == (255, 32)


def test_batched_schema_preserves_downstream_decision():
    torch.manual_seed(41011)
    encoder = _encoder(d_model=256)
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    options = _options(6)
    memory = model.compile_state("processing route 4 is explicitly selected")
    schema, _ = model.compile_schema(
        primitive="choice",
        question_text="which processing route applies?",
        options=options,
        use_cache=False,
        include_token_artifacts=True,
    )

    out_a = model.forward_compiled(memory, schema, forced_budget=6)
    out_b = model.forward_compiled(memory, schema, forced_budget=6)

    assert out_a.selected_option_id == out_b.selected_option_id
    assert torch.allclose(out_a.probabilities, out_b.probabilities, atol=1e-7)
    assert abs(float(out_a.probabilities.sum()) - 1.0) <= 1e-6
