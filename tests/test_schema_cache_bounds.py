import torch

from nmd.contracts import LogicalOption
from nmd.schema import (
    DEFAULT_SCHEMA_CACHE_MAX_BYTES,
    DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
    SchemaCompiler,
)
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


class CountingEncoder(TrainableSemanticEncoder):
    def __init__(self):
        super().__init__(
            vocab_size=4096,
            d_model=32,
            n_layers=1,
            n_heads=4,
            max_length=48,
        )
        self.calls = 0

    def encode_texts(self, texts):
        self.calls += 1
        return super().encode_texts(texts)


def _options(tag: str, k: int = 4):
    return tuple(
        LogicalOption(
            option_id=f"{tag}-{index}",
            criterion_text=f"{tag} semantic route {index}",
        )
        for index in range(k)
    )


def _compile(compiler: SchemaCompiler, tag: str):
    return compiler.compile(
        primitive="choice",
        question_text=f"which {tag} route?",
        options=_options(tag),
        use_cache=True,
        include_token_artifacts=True,
    )


def test_schema_cache_defaults_are_bounded():
    encoder = CountingEncoder()
    compiler = SchemaCompiler(encoder)
    info = compiler.cache_info()

    assert info["max_entries"] == DEFAULT_SCHEMA_CACHE_MAX_ENTRIES == 16
    assert info["max_bytes"] == DEFAULT_SCHEMA_CACHE_MAX_BYTES == 64 * 1024 * 1024
    assert info["entries"] == 0
    assert info["bytes"] == 0
    assert info["evictions"] == 0


def test_schema_cache_lru_eviction_respects_recent_hit():
    torch.manual_seed(42001)
    encoder = CountingEncoder()
    compiler = SchemaCompiler(
        encoder,
        max_cache_entries=2,
        max_cache_bytes=64 * 1024 * 1024,
    )

    a, ar = _compile(compiler, "a")
    b, br = _compile(compiler, "b")
    assert ar.cache_stored and br.cache_stored
    assert compiler.cache_info()["entries"] == 2

    calls_before_hit = encoder.calls
    a_hit, hit_receipt = _compile(compiler, "a")
    assert a_hit is a
    assert hit_receipt.cache_hit is True
    assert encoder.calls == calls_before_hit

    _c, cr = _compile(compiler, "c")
    assert cr.cache_evictions == 1
    assert compiler.cache_info()["entries"] == 2

    # A was touched most recently, so B must be the evicted LRU entry.
    calls_before_b = encoder.calls
    b2, b2r = _compile(compiler, "b")
    assert b2r.cache_hit is False
    assert b2 is not b
    assert encoder.calls > calls_before_b


def test_schema_cache_byte_cap_can_reject_oversized_entry_without_changing_output():
    torch.manual_seed(42003)
    encoder = CountingEncoder()
    compiler = SchemaCompiler(
        encoder,
        max_cache_entries=16,
        max_cache_bytes=1,
    )

    schema, receipt = _compile(compiler, "oversized")
    assert schema.option_embeddings.shape == (4, 32)
    assert receipt.cache_hit is False
    assert receipt.cache_stored is False
    assert receipt.cache_entry_bytes > 1
    assert receipt.cache_entries == 0
    assert receipt.cache_bytes == 0

    calls_before = encoder.calls
    schema2, receipt2 = _compile(compiler, "oversized")
    assert receipt2.cache_hit is False
    assert schema2 is not schema
    assert encoder.calls > calls_before


def test_schema_cache_byte_budget_evicts_until_within_limit():
    torch.manual_seed(42005)
    encoder = CountingEncoder()

    probe = SchemaCompiler(encoder, max_cache_entries=8, max_cache_bytes=64 * 1024 * 1024)
    _probe_schema, probe_receipt = _compile(probe, "probe")
    entry_bytes = probe_receipt.cache_entry_bytes
    assert entry_bytes > 0

    compiler = SchemaCompiler(
        encoder,
        max_cache_entries=8,
        max_cache_bytes=entry_bytes * 2,
    )
    _compile(compiler, "x")
    _compile(compiler, "y")
    _compile(compiler, "z")

    info = compiler.cache_info()
    assert info["entries"] <= 2
    assert info["bytes"] <= info["max_bytes"]
    assert info["evictions"] >= 1


def test_schema_cache_clear_resets_resident_bytes_and_eviction_counter():
    torch.manual_seed(42007)
    encoder = CountingEncoder()
    compiler = SchemaCompiler(
        encoder,
        max_cache_entries=1,
        max_cache_bytes=64 * 1024 * 1024,
    )
    _compile(compiler, "first")
    _compile(compiler, "second")
    assert compiler.cache_info()["evictions"] == 1
    assert compiler.cache_info()["bytes"] > 0

    compiler.clear()
    assert compiler.cache_info() == {
        "entries": 0,
        "bytes": 0,
        "max_entries": 1,
        "max_bytes": 64 * 1024 * 1024,
        "evictions": 0,
    }


def test_runtime_exposes_schema_cache_controls_without_model_mutation():
    torch.manual_seed(42011)
    encoder = CountingEncoder()
    runtime = NolaneHira(
        encoder,
        HIRACore(d_model=32, dropout=0.0),
        schema_cache_max_entries=3,
        schema_cache_max_bytes=1024 * 1024,
    )
    runtime.eval()

    assert runtime.schema_cache_info()["max_entries"] == 3
    assert runtime.schema_cache_info()["max_bytes"] == 1024 * 1024

    schema, first = runtime.compile_schema(
        primitive="choice",
        question_text="which runtime route?",
        options=_options("runtime"),
        use_cache=True,
        include_token_artifacts=False,
    )
    assert first.cache_stored is True
    before = schema.option_embeddings.detach().clone()

    runtime.configure_schema_cache(
        max_entries=1,
        max_bytes=512 * 1024,
        clear=True,
    )
    assert runtime.schema_cache_info()["entries"] == 0
    assert runtime.schema_cache_info()["max_entries"] == 1

    schema2, second = runtime.compile_schema(
        primitive="choice",
        question_text="which runtime route?",
        options=_options("runtime"),
        use_cache=True,
        include_token_artifacts=False,
    )
    assert second.cache_hit is False
    assert torch.allclose(schema2.option_embeddings, before, atol=1e-7, rtol=1e-6)


def test_cache_reconfiguration_rejects_shrinking_below_residency_without_clear():
    torch.manual_seed(42013)
    encoder = CountingEncoder()
    compiler = SchemaCompiler(encoder)
    _compile(compiler, "resident")
    assert compiler.cache_info()["entries"] == 1

    try:
        compiler.configure_cache(max_entries=0, max_bytes=0, clear=False)
    except ValueError as exc:
        assert "clear is required" in str(exc)
    else:
        raise AssertionError("shrinking below resident cache must fail without clear")
