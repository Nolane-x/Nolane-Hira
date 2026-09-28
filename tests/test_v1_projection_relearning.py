import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.v1_projection_relearning import TrainableProjectionTriadicScorer
from nmd.v1_triadic_semantic import ParameterFreeTriadicScorer


def _inputs():
    torch.manual_seed(75001)
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


def test_s5_projection_exact_parameter_count():
    scorer = TrainableProjectionTriadicScorer(d_model=256, d_rel=128)
    assert scorer.projection.weight.shape == (128, 256)
    assert scorer.projection_parameter_count == 32768
    assert scorer.projection_trainable_parameter_count == 32768
    assert scorer.trainable_parameter_count == 32768


def test_s5_w28_initialization_is_exact_parameter_free_identity():
    torch.manual_seed(75002)
    weight = torch.randn(128, 256)

    base = ParameterFreeTriadicScorer(d_model=256, d_rel=128)
    base.load_projection_weight(weight, freeze=True)

    learned = TrainableProjectionTriadicScorer(d_model=256, d_rel=128)
    learned.load_projection_weight(weight, freeze=False)

    args = _inputs()
    assert torch.equal(learned(**args), base(**args))


def test_s5_alignment_loss_is_finite_and_backpropagates_to_projection():
    torch.manual_seed(75003)
    scorer = TrainableProjectionTriadicScorer(d_model=256, d_rel=128)
    args = _inputs()

    loss = scorer.option_view_alignment_loss(
        option_view_tokens=args["option_view_tokens"],
        option_view_token_mask=args["option_view_token_mask"],
        option_view_mask=args["option_view_mask"],
        temperature=0.10,
    )
    assert bool(torch.isfinite(loss))
    assert float(loss) > 0.0

    loss.backward()
    grad = scorer.projection.weight.grad
    assert grad is not None
    assert bool(torch.isfinite(grad).all())
    assert float(grad.abs().sum()) > 0.0


def test_s5_option_permutation_equivariance():
    torch.manual_seed(75004)
    scorer = TrainableProjectionTriadicScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=False)

    args = _inputs()
    direct = scorer(**args)

    permutation = torch.tensor([3, 1, 0, 2])
    changed = dict(args)
    changed["option_view_tokens"] = args["option_view_tokens"][:, permutation]
    changed["option_view_token_mask"] = args["option_view_token_mask"][:, permutation]
    changed["option_view_mask"] = args["option_view_mask"][:, permutation]
    permuted = scorer(**changed)

    assert torch.allclose(
        permuted,
        direct[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )


def test_s5_alignment_requires_two_active_views():
    scorer = TrainableProjectionTriadicScorer(d_model=256, d_rel=128)
    args = _inputs()
    args["option_view_mask"] = args["option_view_mask"].clone()
    args["option_view_mask"][:, :, 1] = False
    try:
        scorer.option_view_alignment_loss(
            option_view_tokens=args["option_view_tokens"],
            option_view_token_mask=args["option_view_token_mask"],
            option_view_mask=args["option_view_mask"],
        )
    except ValueError as exc:
        assert "two active views" in str(exc)
    else:
        raise AssertionError("S5 alignment must fail closed without two views")


def _options():
    return (
        LogicalOption(
            "red",
            "the requested answer is red",
            aliases=("red is the answer value",),
            value=0.0,
        ),
        LogicalOption(
            "tuesday",
            "the requested answer is Tuesday",
            aliases=("Tuesday is the answer value",),
            value=1.0,
        ),
        LogicalOption(
            "large",
            "the requested answer is large",
            aliases=("large is the answer value",),
            value=2.0,
        ),
        LogicalOption(
            "unknown",
            "the requested fact is unknown",
            aliases=("the answer cannot be determined",),
            value=3.0,
        ),
    )


def _runtime():
    torch.manual_seed(75005)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = TrainableProjectionTriadicScorer(d_model=256, d_rel=128)
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=False)
    return NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        projection_triadic_scorer=scorer,
    ).eval()


def test_s5_runtime_state_once_full_k_relation_free():
    runtime = _runtime()
    before = runtime.state_encode_calls
    memory = runtime.compile_state(
        "Mira bought a red bicycle on Tuesday and selected a large frame."
    )
    assert runtime.state_encode_calls - before == 1

    for question in (
        "What color was the bicycle?",
        "On which day was the bicycle bought?",
    ):
        schema, _ = runtime.compile_schema(
            primitive="choice",
            question_text=question,
            options=_options(),
            include_token_artifacts=True,
            use_cache=False,
        )
        assert bool(schema.option_view_mask[:, :2].all())
        out = runtime.forward_compiled(
            memory,
            schema,
            coarse_mode="projection_triadic",
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


def test_s5_builder_exposes_only_projection(tmp_path):
    from nmd.v1_s5_semantic_core import (
        HIRA_V1_S5_PROJECTION_PARAMETER_COUNT,
        build_hira_v1_s5_projection_core,
    )

    t0, t0_sha, weight = _write_t0(tmp_path)
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    runtime = build_hira_v1_s5_projection_core(
        encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_projection=True,
    )
    scorer = runtime.projection_triadic_scorer
    assert scorer is not None
    assert torch.equal(scorer.projection.weight, weight)

    trainable = {
        name: p.numel()
        for name, p in runtime.named_parameters()
        if p.requires_grad
    }
    assert sum(trainable.values()) == HIRA_V1_S5_PROJECTION_PARAMETER_COUNT
    assert trainable == {
        "projection_triadic_scorer.projection.weight": 32768,
    }
    assert not any(p.requires_grad for p in runtime.encoder.parameters())
    assert not any(p.requires_grad for p in runtime.hira.parameters())


def test_s5_builder_can_freeze_complete_candidate(tmp_path):
    from nmd.v1_s5_semantic_core import build_hira_v1_s5_projection_core

    t0, t0_sha, _ = _write_t0(tmp_path)
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    runtime = build_hira_v1_s5_projection_core(
        encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_projection=False,
    )
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 0
