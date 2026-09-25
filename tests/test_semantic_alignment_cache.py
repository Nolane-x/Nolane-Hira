from __future__ import annotations

import copy

import torch

from nmd.contracts import CompiledSchema, StateMemory
from nmd.schema import SchemaCompileReceipt
from nmd.semantic_alignment_authority import generate_w9_domain
from nmd.semantic_alignment_cache import compile_w9_cache, validate_w9_cache


class FakeCompileModel:
    def __init__(self):
        self.state_encode_calls = 0

    def eval(self):
        return self

    def compile_state(self, text, *, segment_tokens=32):
        del text, segment_tokens
        self.state_encode_calls += 1
        return StateMemory(
            model_hash="fake-encoder",
            tokenizer_hash="fake-tokenizer",
            state_hash=f"state-{self.state_encode_calls}",
            global_embedding=torch.zeros(256),
            segment_embeddings=torch.arange(
                2 * 256, dtype=torch.float32
            ).reshape(2, 256) / 1000,
            token_embeddings=None,
            content_token_embeddings=torch.arange(
                5 * 256, dtype=torch.float32
            ).reshape(5, 256) / 1000,
        )

    def compile_schema(
        self,
        *,
        primitive,
        question_text,
        options,
        use_cache,
        include_token_artifacts,
    ):
        assert primitive == "choice"
        assert use_cache is False
        assert include_token_artifacts is True
        options = tuple(options)
        k = len(options)
        qtokens = torch.ones(3, 256)
        otokens = torch.zeros(k, 4, 256)
        for index in range(k):
            otokens[index, :, index % 256] = 1.0
        schema = CompiledSchema(
            schema_hash=f"schema-{k}-{question_text}-{options[0].option_id}",
            encoder_hash="fake-encoder",
            primitive="choice",
            question_text=question_text,
            options=options,
            question_embedding=torch.ones(256),
            option_embeddings=otokens.mean(dim=1),
            option_token_embeddings=otokens,
            option_token_mask=torch.ones(k, 4, dtype=torch.bool),
            question_token_embeddings=qtokens,
            question_token_mask=torch.ones(3, dtype=torch.bool),
            question_content_token_mask=torch.ones(3, dtype=torch.bool),
            option_token_ids=torch.arange(k * 4).reshape(k, 4).long(),
            option_content_token_mask=torch.ones(k, 4, dtype=torch.bool),
        )
        receipt = SchemaCompileReceipt(
            schema_hash=schema.schema_hash,
            encoder_hash="fake-encoder",
            cache_hit=False,
            option_count=k,
            prototype_count=k,
        )
        return schema, receipt


def _first_ay_base_rows():
    all_rows = generate_w9_domain("AY")
    first_base = all_rows[0].base_id
    return [row for row in all_rows if row.base_id == first_base]


def test_w9_cache_encodes_state_once_for_six_paired_views():
    rows = _first_ay_base_rows()
    assert len(rows) == 6

    model = FakeCompileModel()
    cache = compile_w9_cache(model, rows)
    validate_w9_cache(cache)

    assert cache["metadata"]["base_count"] == 1
    assert cache["metadata"]["view_count"] == 6
    assert cache["metadata"]["state_encode_calls"] == 1
    assert cache["metadata"]["state_encode_calls_per_base"] == 1.0
    assert cache["metadata"]["domains"] == ["AY"]
    assert model.state_encode_calls == 1


def test_w9_cache_rejects_label_definition_identity_change():
    cache = compile_w9_cache(FakeCompileModel(), _first_ay_base_rows())
    base = cache["bases"][0]
    definition_k4 = next(
        view for view in base["views"]
        if view["view_id"] == "definition" and view["diagnosis_k"] == 4
    )
    ids = list(definition_k4["option_ids"])
    ids[0], ids[1] = ids[1], ids[0]
    definition_k4["option_ids"] = tuple(ids)

    try:
        validate_w9_cache(cache)
    except ValueError as exc:
        assert "label/definition views changed candidate identity" in str(exc)
    else:
        raise AssertionError("W9 cache accepted mismatched paired identity")


def test_w9_cache_rejects_nested_order_change():
    cache = compile_w9_cache(FakeCompileModel(), _first_ay_base_rows())
    broken = copy.deepcopy(cache)
    base = broken["bases"][0]
    for view in base["views"]:
        if view["diagnosis_k"] == 8:
            ids = list(view["option_ids"])
            ids[0], ids[1] = ids[1], ids[0]
            view["option_ids"] = tuple(ids)

    try:
        validate_w9_cache(broken)
    except ValueError as exc:
        assert "relative order changed" in str(exc)
    else:
        raise AssertionError("W9 cache accepted changed nested order")
