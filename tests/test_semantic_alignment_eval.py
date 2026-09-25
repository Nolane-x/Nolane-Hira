from __future__ import annotations

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.contracts import CompiledSchema, StateMemory
from nmd.hira import HIRACore
from nmd.schema import SchemaCompileReceipt
from nmd.semantic_alignment_authority import generate_w9_domain
from nmd.semantic_alignment_cache import compile_w9_cache
from nmd.semantic_alignment_eval import (
    alignment_metrics,
    evaluate_w9_checkpoint,
)


class FakeCompileModel:
    def __init__(self):
        self.state_encode_calls = 0

    def eval(self):
        return self

    def compile_state(self, text, *, segment_tokens=32):
        del text, segment_tokens
        self.state_encode_calls += 1
        torch.manual_seed(100 + self.state_encode_calls)
        return StateMemory(
            model_hash="fake-encoder",
            tokenizer_hash="fake-tokenizer",
            state_hash=f"state-{self.state_encode_calls}",
            global_embedding=torch.randn(256),
            segment_embeddings=torch.randn(2, 256),
            token_embeddings=None,
            content_token_embeddings=torch.randn(7, 256),
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
        generator = torch.Generator().manual_seed(
            2000 + k + sum(ord(ch) for ch in question_text) % 997
        )
        qtokens = torch.randn(4, 256, generator=generator)
        otokens = torch.randn(k, 5, 256, generator=generator)
        schema = CompiledSchema(
            schema_hash=f"schema-{k}-{options[0].option_id}",
            encoder_hash="fake-encoder",
            primitive="choice",
            question_text=question_text,
            options=options,
            question_embedding=qtokens.mean(dim=0),
            option_embeddings=otokens.mean(dim=1),
            option_token_embeddings=otokens,
            option_token_mask=torch.ones(k, 5, dtype=torch.bool),
            question_token_embeddings=qtokens,
            question_token_mask=torch.ones(4, dtype=torch.bool),
            question_content_token_mask=torch.ones(4, dtype=torch.bool),
            option_token_ids=(
                torch.arange(k * 5).reshape(k, 5).long() + 10
            ),
            option_content_token_mask=torch.ones(k, 5, dtype=torch.bool),
        )
        return schema, SchemaCompileReceipt(
            schema_hash=schema.schema_hash,
            encoder_hash="fake-encoder",
            cache_hit=False,
            option_count=k,
            prototype_count=k,
        )


def _one_base_cache():
    rows = generate_w9_domain("AY")
    base_id = rows[0].base_id
    rows = [row for row in rows if row.base_id == base_id]
    return compile_w9_cache(FakeCompileModel(), rows)


def test_alignment_metrics_cover_definition_k16():
    torch.manual_seed(51)
    cache = _one_base_cache()
    scorer = CompetitiveCoarseScorer()
    metrics = alignment_metrics(scorer, cache)
    assert set(metrics["per_domain"]) == {"AY"}
    pooled = metrics["pooled"]
    assert pooled["n"] == 1.0
    assert 0.0 <= pooled["top1"] <= 1.0
    assert 0.0 < pooled["mrr"] <= 1.0
    assert pooled["ce"] >= 0.0
    assert torch.isfinite(torch.tensor(pooled["mean_margin"]))


def test_full_w9_evaluator_preserves_probability_and_state_once_contract():
    torch.manual_seed(53)
    cache = _one_base_cache()
    hira = HIRACore(d_model=256, dropout=0.0)
    scorer = CompetitiveCoarseScorer()
    metrics = evaluate_w9_checkpoint(hira, scorer, cache)

    assert set(metrics["per_domain"]) == {"AY"}
    assert metrics["cached_base_count"] == 1
    assert metrics["cached_view_count"] == 6
    assert metrics["state_encodes_per_base"] == 1.0
    assert metrics["probability_mass_max_error"] <= 1e-6

    views = metrics["per_domain"]["AY"]["views"]
    assert set(views) == {"label", "definition"}
    assert set(views["definition"]) == {"4", "8", "16"}
    for view in ("label", "definition"):
        for k in ("4", "8", "16"):
            row = views[view][k]
            assert 0.0 <= row["coarse_top1"] <= 1.0
            assert 0.0 <= row["final_top1"] <= 1.0
            assert 0.0 < row["final_mrr"] <= 1.0
