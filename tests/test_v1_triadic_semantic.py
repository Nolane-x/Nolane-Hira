import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.typed_competitive_cache import file_sha256
from nmd.v1_s3_semantic_core import (
    HIRA_V1_S3_FACTOR_PARAMETER_COUNT,
    build_hira_v1_s3_parameter_free_core,
    build_hira_v1_s3_triadic_core,
)
from nmd.v1_triadic_semantic import (
    ParameterFreeTriadicScorer,
    TriadicCPSemanticScorer,
)


def _inputs():
    torch.manual_seed(93101)
    return dict(
        state_tokens=torch.randn(2, 5, 256),
        state_mask=torch.tensor([
            [True, True, True, True, False],
            [True, True, True, True, True],
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


def test_s3_parameter_free_has_zero_added_params_and_finite_logits():
    scorer = ParameterFreeTriadicScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    assert scorer.added_parameter_count == 0
    assert scorer.trainable_parameter_count == 0

    logits = scorer(**_inputs())
    assert logits.shape == (2, 4)
    assert bool(torch.isfinite(logits).all())


def test_s3_triadic_exact_factor_budget():
    scorer = TriadicCPSemanticScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    assert scorer.state_factor.weight.shape == (32, 128)
    assert scorer.question_factor.weight.shape == (32, 128)
    assert scorer.option_factor.weight.shape == (32, 128)
    assert scorer.factor_parameter_count == 12288
    assert scorer.factor_parameter_count == HIRA_V1_S3_FACTOR_PARAMETER_COUNT
    assert scorer.factor_trainable_parameter_count == 12288
    assert scorer.trainable_parameter_count == 12288
    assert scorer.projection.weight.requires_grad is False

    scorer.freeze_factors()
    assert scorer.factor_trainable_parameter_count == 0
    assert scorer.trainable_parameter_count == 0


def test_s3_triadic_question_changes_logits():
    torch.manual_seed(93102)
    scorer = TriadicCPSemanticScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    args = _inputs()

    first = scorer(**args)
    changed = dict(args)
    changed["question_tokens"] = args["question_tokens"].clone()
    changed["question_tokens"][:, 0] += 1.75
    second = scorer(**changed)

    assert not torch.equal(first, second)
    assert float((first - second).abs().max()) > 1e-7


def test_s3_triadic_option_permutation_equivariance():
    torch.manual_seed(93103)
    scorer = TriadicCPSemanticScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    args = _inputs()
    direct = scorer(**args)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_args = dict(args)
    permuted_args["option_view_tokens"] = args["option_view_tokens"][:, permutation]
    permuted_args["option_view_token_mask"] = args["option_view_token_mask"][:, permutation]
    permuted_args["option_view_mask"] = args["option_view_mask"][:, permutation]
    permuted = scorer(**permuted_args)

    assert torch.allclose(
        permuted,
        direct[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )


def test_s3_parameter_free_option_permutation_equivariance():
    torch.manual_seed(93104)
    scorer = ParameterFreeTriadicScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    args = _inputs()
    direct = scorer(**args)

    permutation = torch.tensor([3, 1, 0, 2])
    permuted_args = dict(args)
    permuted_args["option_view_tokens"] = args["option_view_tokens"][:, permutation]
    permuted_args["option_view_token_mask"] = args["option_view_token_mask"][:, permutation]
    permuted_args["option_view_mask"] = args["option_view_mask"][:, permutation]
    permuted = scorer(**permuted_args)

    assert torch.allclose(
        permuted,
        direct[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )


def _options():
    return (
        LogicalOption("alpha", "the requested answer is alpha", value=0.0),
        LogicalOption("bravo", "the requested answer is bravo", value=1.0),
        LogicalOption("charlie", "the requested answer is charlie", value=2.0),
        LogicalOption("delta", "the requested answer is delta", value=3.0),
    )


def _runtime():
    torch.manual_seed(93105)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = TriadicCPSemanticScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    return NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        triadic_cp_scorer=scorer,
    ).eval()


def test_s3_runtime_state_once_full_k_relation_free():
    runtime = _runtime()
    before = runtime.state_encode_calls
    memory = runtime.compile_state(
        "The alpha module is mounted in bay bravo on Tuesday."
    )
    assert runtime.state_encode_calls - before == 1

    for question in (
        "Which module is mentioned?",
        "Which bay is mentioned?",
    ):
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
            coarse_mode="triadic_cp",
        )
        assert int(out.hira.candidate_budget.item()) == 4
        assert bool(out.hira.selected_mask.all())
        assert torch.equal(
            out.hira.relation_delta,
            torch.zeros_like(out.hira.relation_delta),
        )
        assert torch.isclose(
            out.probabilities.sum(),
            out.probabilities.new_tensor(1.0),
            atol=1e-6,
        )

    assert runtime.state_encode_calls - before == 1


def _write_t0(tmp_path):
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


def _encoder():
    return TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )


def test_s3_builder_exposes_only_12288_factor_params(tmp_path):
    torch.manual_seed(93106)
    t0, t0_sha, weight = _write_t0(tmp_path)
    runtime = build_hira_v1_s3_triadic_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_factors=True,
    )
    scorer = runtime.triadic_cp_scorer
    assert scorer is not None
    assert torch.equal(scorer.projection.weight, weight)

    trainable = {
        name: p.numel()
        for name, p in runtime.named_parameters()
        if p.requires_grad
    }
    assert sum(trainable.values()) == 12288
    assert set(trainable) == {
        "triadic_cp_scorer.state_factor.weight",
        "triadic_cp_scorer.question_factor.weight",
        "triadic_cp_scorer.option_factor.weight",
    }
    assert not any(p.requires_grad for p in runtime.encoder.parameters())
    assert not any(p.requires_grad for p in runtime.hira.parameters())


def test_s3_parameter_free_builder_is_fully_frozen(tmp_path):
    torch.manual_seed(93107)
    t0, t0_sha, _ = _write_t0(tmp_path)
    runtime = build_hira_v1_s3_parameter_free_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
    )
    assert runtime.parameter_free_triadic_scorer is not None
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 0


def test_s3_factor_checkpoint_load_and_freeze():
    scorer = TriadicCPSemanticScorer(d_model=256, d_rel=128)
    state = {
        "state_factor.weight": torch.randn(32, 128),
        "question_factor.weight": torch.randn(32, 128),
        "option_factor.weight": torch.randn(32, 128),
    }
    scorer.load_factor_state_dict(state, freeze=True)
    own = scorer.state_dict()
    for key, value in state.items():
        assert torch.equal(own[key], value)
    assert scorer.factor_trainable_parameter_count == 0


def test_s3_checkpoint_roundtrip_freezes_replay(tmp_path):
    from nmd.v1_s3_checkpoint import (
        build_frozen_hira_v1_s3_candidate,
        load_hira_v1_s3_triadic_checkpoint,
    )

    torch.manual_seed(93108)
    t0, t0_sha, _ = _write_t0(tmp_path)
    checkpoint = tmp_path / "triadic.pt"
    factors = {
        "state_factor.weight": torch.randn(32, 128),
        "question_factor.weight": torch.randn(32, 128),
        "option_factor.weight": torch.randn(32, 128),
    }
    torch.save(
        {
            "schema_version": "hira-v1-s3-triadic-checkpoint-v1",
            "kind": "triadic-cp-semantic",
            "factor_parameter_count": 12288,
            "selected_dev_epoch": 3,
            "t0_checkpoint_sha256": t0_sha,
            "factor_state_dict": factors,
        },
        checkpoint,
    )
    checkpoint_sha = file_sha256(checkpoint)

    loaded, metadata = load_hira_v1_s3_triadic_checkpoint(
        checkpoint,
        expected_sha256=checkpoint_sha,
        expected_t0_sha256=t0_sha,
    )
    assert metadata["selected_dev_epoch"] == 3
    for key, value in factors.items():
        assert torch.equal(loaded[key], value.float())

    runtime, replay = build_frozen_hira_v1_s3_candidate(
        _encoder(),
        t0,
        checkpoint,
        expected_t0_sha256=t0_sha,
        expected_triadic_sha256=checkpoint_sha,
    )
    assert replay["sha256"] == checkpoint_sha
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 0
    scorer = runtime.triadic_cp_scorer
    assert scorer is not None
    own = scorer.state_dict()
    for key, value in factors.items():
        assert torch.equal(own[key], value.float())


def test_s3_checkpoint_rejects_wrong_t0_identity(tmp_path):
    from nmd.v1_s3_checkpoint import load_hira_v1_s3_triadic_checkpoint

    checkpoint = tmp_path / "bad-triadic.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s3-triadic-checkpoint-v1",
            "kind": "triadic-cp-semantic",
            "factor_parameter_count": 12288,
            "selected_dev_epoch": 1,
            "t0_checkpoint_sha256": "wrong",
            "factor_state_dict": {
                "state_factor.weight": torch.randn(32, 128),
                "question_factor.weight": torch.randn(32, 128),
                "option_factor.weight": torch.randn(32, 128),
            },
        },
        checkpoint,
    )
    try:
        load_hira_v1_s3_triadic_checkpoint(
            checkpoint,
            expected_t0_sha256="expected",
        )
    except RuntimeError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("wrong T0 identity must fail closed")
