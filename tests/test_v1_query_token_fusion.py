import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256
from nmd.v1_query_token_fusion import (
    QuestionAsEvidenceScorer,
    QueryTokenResidualFusionScorer,
)
from nmd.v1_s2_semantic_core import (
    HIRA_V1_S2_CANDIDATE_PARAMETER_COUNT,
    HIRA_V1_S2_FUSION_PARAMETER_COUNT,
    build_hira_v1_s2_fusion_core,
    build_hira_v1_s2_question_as_evidence_core,
)
from nmd.w33_coevidence_semantic import CoEvidenceSemanticScorer


def _inputs():
    torch.manual_seed(62001)
    return dict(
        state_tokens=torch.randn(2, 6, 256),
        state_mask=torch.tensor([
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]),
        question_tokens=torch.randn(2, 4, 256),
        question_mask=torch.tensor([
            [True, True, True, False],
            [True, True, True, True],
        ]),
        option_view_tokens=torch.randn(2, 4, 2, 5, 256),
        option_view_token_mask=torch.tensor([
            [
                [[True, True, True, False, False], [True, True, False, False, False]],
                [[True, True, True, True, False], [True, True, True, False, False]],
                [[True, True, False, False, False], [True, True, True, False, False]],
                [[True, True, True, False, False], [True, True, True, False, False]],
            ],
            [
                [[True, True, True, False, False], [True, True, True, False, False]],
                [[True, True, False, False, False], [True, True, True, True, False]],
                [[True, True, True, False, False], [True, True, False, False, False]],
                [[True, True, True, True, False], [True, True, True, False, False]],
            ],
        ]),
        option_view_mask=torch.ones(2, 4, 2, dtype=torch.bool),
    )


def _w34_args(args):
    return {
        "state_tokens": args["state_tokens"],
        "state_mask": args["state_mask"],
        "option_view_tokens": args["option_view_tokens"],
        "option_view_token_mask": args["option_view_token_mask"],
        "option_view_mask": args["option_view_mask"],
    }


def test_s2_question_as_evidence_adds_zero_parameters_and_uses_question():
    torch.manual_seed(62002)
    scorer = QuestionAsEvidenceScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    scorer.freeze_candidate()
    assert scorer.added_parameter_count == 0
    assert scorer.trainable_parameter_count == 0

    args = _inputs()
    first = scorer(**args)
    changed = dict(args)
    changed["question_tokens"] = args["question_tokens"].clone()
    changed["question_tokens"][:, 0] += 2.0
    second = scorer(**changed)
    assert not torch.equal(first, second)


def test_s2_fusion_has_exact_8192_new_parameters():
    scorer = QueryTokenResidualFusionScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    assert scorer.state_query.weight.shape == (16, 128)
    assert scorer.question_key.weight.shape == (16, 128)
    assert scorer.question_value.weight.shape == (16, 128)
    assert scorer.fusion_up.weight.shape == (128, 16)
    assert scorer.fusion_parameter_count == 8192
    assert scorer.candidate_parameter_count == 16384
    assert scorer.candidate_parameter_count == HIRA_V1_S2_CANDIDATE_PARAMETER_COUNT


def test_s2_zero_init_matches_w34_within_float_tolerance():
    torch.manual_seed(62003)
    weight = torch.randn(128, 256)
    base = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    base.load_projection_weight(weight, freeze=True)
    s2 = QueryTokenResidualFusionScorer(d_model=256, d_rel=128, rank=8)
    s2.load_projection_weight(weight, freeze=True)

    args = _inputs()
    expected = base(**_w34_args(args))
    observed = s2(**args)
    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)


def test_s2_fusion_activation_makes_question_change_logits():
    torch.manual_seed(62004)
    scorer = QueryTokenResidualFusionScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    with torch.no_grad():
        scorer.fusion_up.weight.normal_(mean=0.0, std=0.03)

    args = _inputs()
    first = scorer(**args)
    changed = dict(args)
    changed["question_tokens"] = args["question_tokens"].clone()
    changed["question_tokens"][:, 0] += 2.0
    second = scorer(**changed)
    assert not torch.equal(first, second)
    assert float((first - second).abs().max()) > 1e-7


def test_s2_option_permutation_equivariance():
    torch.manual_seed(62005)
    scorer = QueryTokenResidualFusionScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    with torch.no_grad():
        scorer.fusion_up.weight.normal_(mean=0.0, std=0.02)

    args = _inputs()
    direct = scorer(**args)
    permutation = torch.tensor([2, 0, 3, 1])
    perm = dict(args)
    perm["option_view_tokens"] = args["option_view_tokens"][:, permutation]
    perm["option_view_token_mask"] = args["option_view_token_mask"][:, permutation]
    perm["option_view_mask"] = args["option_view_mask"][:, permutation]
    observed = scorer(**perm)
    assert torch.allclose(observed, direct[:, permutation], atol=1e-6, rtol=1e-6)


def _options():
    return (
        LogicalOption("alpha", "the requested answer is alpha", value=0.0),
        LogicalOption("beta", "the requested answer is beta", value=1.0),
        LogicalOption("gamma", "the requested answer is gamma", value=2.0),
        LogicalOption("unknown", "the requested fact is unknown", value=3.0),
    )


def test_s2_runtime_state_once_full_k_relation_free():
    torch.manual_seed(62006)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048, d_model=256, n_layers=1, n_heads=4, max_length=64
    )
    scorer = QueryTokenResidualFusionScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    with torch.no_grad():
        scorer.fusion_up.weight.normal_(mean=0.0, std=0.02)
    runtime = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        query_token_residual_fusion_scorer=scorer,
    ).eval()

    before = runtime.state_encode_calls
    memory = runtime.compile_state("Record R1 stores alpha in field one and beta in field two.")
    assert runtime.state_encode_calls - before == 1

    for question in ("What is field one?", "What is field two?"):
        schema, _ = runtime.compile_schema(
            primitive="choice",
            question_text=question,
            options=_options(),
            include_token_artifacts=True,
            use_cache=False,
        )
        out = runtime.forward_compiled(
            memory,
            schema,
            coarse_mode="query_token_residual_fusion",
        )
        assert int(out.hira.candidate_budget.item()) == 4
        assert bool(out.hira.selected_mask.all())
        assert torch.equal(out.hira.relation_delta, torch.zeros_like(out.hira.relation_delta))
        assert torch.isclose(out.probabilities.sum(), out.probabilities.new_tensor(1.0), atol=1e-6)

    assert runtime.state_encode_calls - before == 1


def _write_checkpoints(tmp_path):
    t0 = tmp_path / "t0.pt"
    projection = torch.randn(128, 256)
    torch.save(
        {
            "schema_version": "r8-w28-candidate-checkpoint-v1",
            "candidate": "T0",
            "kind": "projection",
            "projection_weight": projection,
        },
        t0,
    )
    state = {
        "state_adapter.down.weight": torch.randn(8, 128),
        "state_adapter.up.weight": torch.randn(128, 8),
        "schema_adapter.down.weight": torch.randn(8, 128),
        "schema_adapter.up.weight": torch.randn(128, 8),
        "interaction_state.weight": torch.randn(8, 128),
        "interaction_schema.weight": torch.randn(8, 128),
        "composition_state.weight": torch.randn(8, 128),
        "composition_schema.weight": torch.randn(8, 128),
    }
    w34 = tmp_path / "w34.pt"
    torch.save(
        {
            "schema_version": "r8-w34-coevidence-semantic-checkpoint-v1",
            "kind": "coevidence-semantic",
            "rank": 8,
            "parameter_count": 8192,
            "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
            "selected_dev_epoch": 20,
            "candidate_state_dict": state,
        },
        w34,
    )
    return t0, file_sha256(t0), w34, file_sha256(w34), state


def _encoder():
    return TrainableSemanticEncoder(
        vocab_size=1024, d_model=256, n_layers=1, n_heads=4, max_length=64
    )


def test_s2_builder_exposes_only_fusion_params(tmp_path):
    torch.manual_seed(62007)
    t0, t0_sha, w34, w34_sha, state = _write_checkpoints(tmp_path)
    runtime = build_hira_v1_s2_fusion_core(
        _encoder(),
        t0,
        w34,
        expected_t0_sha256=t0_sha,
        expected_w34_sha256=w34_sha,
        train_fusion=True,
    )
    trainable = {
        name: p.numel()
        for name, p in runtime.named_parameters()
        if p.requires_grad
    }
    assert sum(trainable.values()) == HIRA_V1_S2_FUSION_PARAMETER_COUNT
    assert set(trainable) == {
        "query_token_residual_fusion_scorer.state_query.weight",
        "query_token_residual_fusion_scorer.question_key.weight",
        "query_token_residual_fusion_scorer.question_value.weight",
        "query_token_residual_fusion_scorer.fusion_up.weight",
    }
    scorer = runtime.query_token_residual_fusion_scorer
    assert scorer is not None
    own = scorer.state_dict()
    for key, value in state.items():
        assert torch.equal(own[key], value)


def test_s2_parameter_free_builder_is_frozen(tmp_path):
    torch.manual_seed(62008)
    t0, t0_sha, w34, w34_sha, _ = _write_checkpoints(tmp_path)
    runtime = build_hira_v1_s2_question_as_evidence_core(
        _encoder(),
        t0,
        w34,
        expected_t0_sha256=t0_sha,
        expected_w34_sha256=w34_sha,
    )
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 0
    assert runtime.question_as_evidence_scorer is not None


def test_s2_qtrf_checkpoint_roundtrip_freezes_replay(tmp_path):
    from nmd.v1_s2_checkpoint import (
        build_frozen_hira_v1_s2_candidate,
        load_hira_v1_s2_qtrf_checkpoint,
    )

    torch.manual_seed(62009)
    t0, t0_sha, w34, w34_sha, _ = _write_checkpoints(tmp_path)
    checkpoint = tmp_path / "qtrf.pt"
    fusion_state = {
        "state_query.weight": torch.randn(16, 128),
        "question_key.weight": torch.randn(16, 128),
        "question_value.weight": torch.randn(16, 128),
        "fusion_up.weight": torch.randn(128, 16),
    }
    torch.save(
        {
            "schema_version": "hira-v1-s2-qtrf-checkpoint-v1",
            "kind": "query-token-residual-fusion",
            "fusion_parameter_count": 8192,
            "candidate_parameter_count": 16384,
            "selected_dev_epoch": 6,
            "t0_checkpoint_sha256": t0_sha,
            "w34_checkpoint_sha256": w34_sha,
            "fusion_state_dict": fusion_state,
        },
        checkpoint,
    )
    qtrf_sha = file_sha256(checkpoint)

    loaded, metadata = load_hira_v1_s2_qtrf_checkpoint(
        checkpoint,
        expected_sha256=qtrf_sha,
        expected_t0_sha256=t0_sha,
        expected_w34_sha256=w34_sha,
    )
    assert metadata["selected_dev_epoch"] == 6
    for key, value in fusion_state.items():
        assert torch.equal(loaded[key], value.float())

    runtime, replay = build_frozen_hira_v1_s2_candidate(
        _encoder(),
        t0,
        w34,
        checkpoint,
        expected_t0_sha256=t0_sha,
        expected_w34_sha256=w34_sha,
        expected_qtrf_sha256=qtrf_sha,
    )
    assert replay["sha256"] == qtrf_sha
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 0
    scorer = runtime.query_token_residual_fusion_scorer
    assert scorer is not None
    own = scorer.state_dict()
    for key, value in fusion_state.items():
        assert torch.equal(own[key], value.float())


def test_s2_qtrf_checkpoint_rejects_wrong_base_identity(tmp_path):
    from nmd.v1_s2_checkpoint import load_hira_v1_s2_qtrf_checkpoint

    checkpoint = tmp_path / "bad-qtrf.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s2-qtrf-checkpoint-v1",
            "kind": "query-token-residual-fusion",
            "fusion_parameter_count": 8192,
            "candidate_parameter_count": 16384,
            "selected_dev_epoch": 1,
            "t0_checkpoint_sha256": "wrong-t0",
            "w34_checkpoint_sha256": "wrong-w34",
            "fusion_state_dict": {
                "state_query.weight": torch.randn(16, 128),
                "question_key.weight": torch.randn(16, 128),
                "question_value.weight": torch.randn(16, 128),
                "fusion_up.weight": torch.randn(128, 16),
            },
        },
        checkpoint,
    )

    try:
        load_hira_v1_s2_qtrf_checkpoint(
            checkpoint,
            expected_t0_sha256="expected-t0",
            expected_w34_sha256="expected-w34",
        )
    except RuntimeError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("wrong S2 base identity must fail closed")
