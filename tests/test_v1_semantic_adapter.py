import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.v1_semantic_adapter import AdaptedTriadicSemanticScorer
from nmd.v1_triadic_semantic import ParameterFreeTriadicScorer


def _inputs():
    torch.manual_seed(64001)
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


def test_s4_adapter_exact_parameter_count_and_frozen_projection():
    scorer = AdaptedTriadicSemanticScorer(
        d_model=256,
        d_rel=128,
        bottleneck=32,
    )
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    assert scorer.adapter.down.weight.shape == (32, 256)
    assert scorer.adapter.up.weight.shape == (256, 32)
    assert scorer.adapter_parameter_count == 16384
    assert scorer.adapter_trainable_parameter_count == 16384
    assert scorer.trainable_parameter_count == 16384
    assert scorer.projection.weight.requires_grad is False


def test_s4_zero_init_is_exact_parameter_free_triadic_identity():
    torch.manual_seed(64002)
    projection = torch.randn(128, 256)

    base = ParameterFreeTriadicScorer(d_model=256, d_rel=128)
    base.load_projection_weight(projection, freeze=True)

    adapted = AdaptedTriadicSemanticScorer(
        d_model=256,
        d_rel=128,
        bottleneck=32,
    )
    adapted.load_projection_weight(projection, freeze=True)

    args = _inputs()
    expected = base(**args)
    observed = adapted(**args)

    assert torch.equal(observed, expected)
    assert torch.equal(
        adapted.adapter.up.weight,
        torch.zeros_like(adapted.adapter.up.weight),
    )


def test_s4_adapter_can_change_question_conditioned_logits():
    torch.manual_seed(64003)
    scorer = AdaptedTriadicSemanticScorer(
        d_model=256,
        d_rel=128,
        bottleneck=32,
    )
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    with torch.no_grad():
        scorer.adapter.up.weight.normal_(mean=0.0, std=0.02)

    args = _inputs()
    first = scorer(**args)

    changed = dict(args)
    changed["question_tokens"] = args["question_tokens"].clone()
    changed["question_tokens"][:, 0] += 1.75
    second = scorer(**changed)

    assert not torch.equal(first, second)
    assert float((first - second).abs().max()) > 1e-7


def test_s4_option_permutation_equivariance():
    torch.manual_seed(64004)
    scorer = AdaptedTriadicSemanticScorer(
        d_model=256,
        d_rel=128,
        bottleneck=32,
    )
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)
    with torch.no_grad():
        scorer.adapter.up.weight.normal_(mean=0.0, std=0.01)

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


def _options():
    return (
        LogicalOption("red", "the requested answer is red", value=0.0),
        LogicalOption("tuesday", "the requested answer is Tuesday", value=1.0),
        LogicalOption("large", "the requested answer is large", value=2.0),
        LogicalOption("unknown", "the requested fact is unknown", value=3.0),
    )


def _runtime():
    torch.manual_seed(64005)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = AdaptedTriadicSemanticScorer(
        d_model=256,
        d_rel=128,
        bottleneck=32,
    )
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    with torch.no_grad():
        scorer.adapter.up.weight.normal_(mean=0.0, std=0.01)
    return NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        adapted_triadic_scorer=scorer,
    ).eval()


def test_s4_runtime_state_once_full_k_relation_free():
    runtime = _runtime()
    before = runtime.state_encode_calls
    memory = runtime.compile_state(
        "Mira bought a red bicycle on Tuesday and selected a large frame."
    )
    assert runtime.state_encode_calls - before == 1

    outputs = []
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
        out = runtime.forward_compiled(
            memory,
            schema,
            coarse_mode="adapted_triadic",
        )
        outputs.append(out)
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
    assert not torch.equal(outputs[0].logits, outputs[1].logits)


def test_s4_adapter_state_dict_boundary():
    scorer = AdaptedTriadicSemanticScorer(
        d_model=256,
        d_rel=128,
        bottleneck=32,
    )
    state = {
        "adapter.down.weight": torch.randn(32, 256),
        "adapter.up.weight": torch.randn(256, 32),
    }
    scorer.load_adapter_state_dict(state, freeze=True)

    assert scorer.adapter_trainable_parameter_count == 0
    assert scorer.trainable_parameter_count == scorer.projection.weight.numel()
    own = scorer.state_dict()
    for key, value in state.items():
        assert torch.equal(own[key], value)


def _write_t0(tmp_path):
    from nmd.typed_competitive_cache import file_sha256

    path = tmp_path / "t0.pt"
    torch.save(
        {
            "schema_version": "r8-w28-candidate-checkpoint-v1",
            "candidate": "T0",
            "kind": "projection",
            "projection_weight": torch.randn(128, 256),
        },
        path,
    )
    return path, file_sha256(path)


def test_s4_builder_exposes_only_adapter_params(tmp_path):
    from nmd.v1_s4_semantic_core import (
        HIRA_V1_S4_ADAPTER_PARAMETER_COUNT,
        build_hira_v1_s4_adapter_core,
    )

    t0, t0_sha = _write_t0(tmp_path)
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    runtime = build_hira_v1_s4_adapter_core(
        encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_adapter=True,
    )

    trainable = {
        name: p.numel()
        for name, p in runtime.named_parameters()
        if p.requires_grad
    }
    assert sum(trainable.values()) == HIRA_V1_S4_ADAPTER_PARAMETER_COUNT
    assert set(trainable) == {
        "adapted_triadic_scorer.adapter.down.weight",
        "adapted_triadic_scorer.adapter.up.weight",
    }
    assert not any(p.requires_grad for p in runtime.encoder.parameters())
    assert not any(p.requires_grad for p in runtime.hira.parameters())
    assert runtime.adapted_triadic_scorer.projection.weight.requires_grad is False


def test_s4_builder_can_freeze_complete_candidate(tmp_path):
    from nmd.v1_s4_semantic_core import build_hira_v1_s4_adapter_core

    t0, t0_sha = _write_t0(tmp_path)
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    runtime = build_hira_v1_s4_adapter_core(
        encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_adapter=False,
    )
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 0


def test_s4_checkpoint_roundtrip_freezes_replay(tmp_path):
    from nmd.typed_competitive_cache import file_sha256
    from nmd.v1_s4_checkpoint import (
        build_frozen_hira_v1_s4_candidate,
        load_hira_v1_s4_adapter_checkpoint,
    )

    torch.manual_seed(64006)
    t0, t0_sha = _write_t0(tmp_path)
    checkpoint = tmp_path / "adapter.pt"
    adapter_state = {
        "adapter.down.weight": torch.randn(32, 256),
        "adapter.up.weight": torch.randn(256, 32),
    }
    torch.save(
        {
            "schema_version": "hira-v1-s4-adapter-checkpoint-v1",
            "kind": "shared-residual-semantic-adapter",
            "bottleneck": 32,
            "adapter_parameter_count": 16384,
            "selected_dev_epoch": 5,
            "t0_checkpoint_sha256": t0_sha,
            "adapter_state_dict": adapter_state,
        },
        checkpoint,
    )
    adapter_sha = file_sha256(checkpoint)

    loaded, metadata = load_hira_v1_s4_adapter_checkpoint(
        checkpoint,
        expected_sha256=adapter_sha,
        expected_t0_sha256=t0_sha,
    )
    assert metadata["selected_dev_epoch"] == 5
    for key, value in adapter_state.items():
        assert torch.equal(loaded[key], value.float())

    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    runtime, replay = build_frozen_hira_v1_s4_candidate(
        encoder,
        t0,
        checkpoint,
        expected_t0_sha256=t0_sha,
        expected_adapter_sha256=adapter_sha,
    )
    assert replay["sha256"] == adapter_sha
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 0
    scorer = runtime.adapted_triadic_scorer
    assert scorer is not None
    own = scorer.state_dict()
    for key, value in adapter_state.items():
        assert torch.equal(own[key], value.float())


def test_s4_checkpoint_rejects_wrong_t0_identity(tmp_path):
    from nmd.v1_s4_checkpoint import load_hira_v1_s4_adapter_checkpoint

    checkpoint = tmp_path / "bad-adapter.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s4-adapter-checkpoint-v1",
            "kind": "shared-residual-semantic-adapter",
            "bottleneck": 32,
            "adapter_parameter_count": 16384,
            "selected_dev_epoch": 1,
            "t0_checkpoint_sha256": "wrong",
            "adapter_state_dict": {
                "adapter.down.weight": torch.randn(32, 256),
                "adapter.up.weight": torch.randn(256, 32),
            },
        },
        checkpoint,
    )
    try:
        load_hira_v1_s4_adapter_checkpoint(
            checkpoint,
            expected_t0_sha256="expected-t0",
        )
    except RuntimeError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("wrong S4 T0 identity must fail closed")
