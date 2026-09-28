import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256
from nmd.v1_query_keyed_evidence import (
    ParameterFreeQueryEvidenceScorer,
    QueryKeyedEvidenceScorer,
)
from nmd.v1_s1_semantic_core import (
    HIRA_V1_S1_CANDIDATE_PARAMETER_COUNT,
    HIRA_V1_S1_EXTRACTOR_PARAMETER_COUNT,
    build_hira_v1_s1_parameter_free_core,
    build_hira_v1_s1_query_keyed_core,
)


def _random_inputs():
    torch.manual_seed(51101)
    return dict(
        state_tokens=torch.randn(2, 7, 256),
        state_mask=torch.tensor([
            [True, True, True, True, True, False, False],
            [True, True, True, True, True, True, False],
        ]),
        question_tokens=torch.randn(2, 5, 256),
        question_mask=torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, False],
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


def _projection():
    weight = torch.zeros(128, 256)
    weight[:, :128] = torch.eye(128)
    return weight


def test_s1_parameter_free_top2_evidence_is_question_keyed_and_zero_param():
    scorer = ParameterFreeQueryEvidenceScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(_projection(), freeze=True)
    scorer.freeze_candidate()

    assert scorer.added_parameter_count == 0
    assert scorer.candidate_parameter_count == 8192
    assert scorer.trainable_parameter_count == 0

    state = torch.zeros(1, 4, 256)
    state[0, 0, 0] = 1.0
    state[0, 1, 1] = 1.0
    state[0, 2, 2] = 1.0
    state[0, 3, 3] = 1.0
    state_mask = torch.ones(1, 4, dtype=torch.bool)

    qa = torch.zeros(1, 1, 256)
    qa[0, 0, 0] = 1.0
    qb = torch.zeros(1, 1, 256)
    qb[0, 0, 3] = 1.0
    qmask = torch.ones(1, 1, dtype=torch.bool)

    ea, _, = scorer.extract_evidence(
        state_tokens=state,
        state_mask=state_mask,
        question_tokens=qa,
        question_mask=qmask,
    )
    eb, _, = scorer.extract_evidence(
        state_tokens=state,
        state_mask=state_mask,
        question_tokens=qb,
        question_mask=qmask,
    )

    assert not torch.equal(ea, eb)
    assert torch.equal(ea[:, 0], state[:, 0])
    assert torch.equal(eb[:, 0], state[:, 3])


def test_s1_qkee_has_exact_8192_extractor_params():
    scorer = QueryKeyedEvidenceScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    assert scorer.question_heads.weight.shape == (32, 128)
    assert scorer.state_keys.weight.shape == (32, 128)
    assert scorer.extractor_parameter_count == 8192
    assert scorer.candidate_parameter_count == 16384
    assert scorer.candidate_parameter_count == HIRA_V1_S1_CANDIDATE_PARAMETER_COUNT

    # Freeze only W34 base; extractor remains the sole trainable surface.
    base = {
        "state_adapter.down.weight": torch.randn(8, 128),
        "state_adapter.up.weight": torch.randn(128, 8),
        "schema_adapter.down.weight": torch.randn(8, 128),
        "schema_adapter.up.weight": torch.randn(128, 8),
        "interaction_state.weight": torch.randn(8, 128),
        "interaction_schema.weight": torch.randn(8, 128),
        "composition_state.weight": torch.randn(8, 128),
        "composition_schema.weight": torch.randn(8, 128),
    }
    scorer.load_w34_base_state_dict(base, freeze_base=True)

    assert scorer.extractor_trainable_parameter_count == 8192
    assert scorer.candidate_trainable_parameter_count == 8192
    assert scorer.trainable_parameter_count == 8192
    assert scorer.projection.weight.requires_grad is False


def test_s1_qkee_question_changes_attention_and_evidence():
    torch.manual_seed(51102)
    scorer = QueryKeyedEvidenceScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    args = _random_inputs()
    ea, ma, aa = scorer.extract_evidence(
        state_tokens=args["state_tokens"],
        state_mask=args["state_mask"],
        question_tokens=args["question_tokens"],
        question_mask=args["question_mask"],
    )
    changed_question = args["question_tokens"].clone()
    changed_question[:, 0] += 2.0
    eb, mb, ab = scorer.extract_evidence(
        state_tokens=args["state_tokens"],
        state_mask=args["state_mask"],
        question_tokens=changed_question,
        question_mask=args["question_mask"],
    )

    assert ea.shape == (2, 2, 256)
    assert aa.shape == (2, 2, 7)
    assert torch.equal(ma, mb)
    assert not torch.equal(aa, ab)
    assert not torch.equal(ea, eb)
    assert torch.allclose(aa.sum(-1), torch.ones_like(aa.sum(-1)), atol=1e-6)


def test_s1_qkee_option_permutation_equivariance():
    torch.manual_seed(51103)
    scorer = QueryKeyedEvidenceScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256), freeze=True)

    args = _random_inputs()
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
        LogicalOption("red", "the requested answer is red", value=0.0),
        LogicalOption("tuesday", "the requested answer is Tuesday", value=1.0),
        LogicalOption("large", "the requested answer is large", value=2.0),
        LogicalOption("unknown", "the requested fact is unknown", value=3.0),
    )


def _runtime():
    torch.manual_seed(51104)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = QueryKeyedEvidenceScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    return NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        query_keyed_evidence_scorer=scorer,
    ).eval()


def test_s1_runtime_is_state_once_full_k_relation_free():
    model = _runtime()
    before = model.state_encode_calls
    memory = model.compile_state(
        "Mira bought a red bicycle on Tuesday and chose the large frame."
    )
    assert model.state_encode_calls - before == 1

    for question in (
        "What color was the bicycle?",
        "On which day was the bicycle bought?",
    ):
        schema, _ = model.compile_schema(
            primitive="choice",
            question_text=question,
            options=_options(),
            include_token_artifacts=True,
            use_cache=False,
        )
        out = model.forward_compiled(
            memory,
            schema,
            coarse_mode="query_keyed_evidence",
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

    assert model.state_encode_calls - before == 1


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
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )


def test_s1_builder_exposes_only_8192_trainable_extractor_params(tmp_path):
    torch.manual_seed(51105)
    t0, t0_sha, w34, w34_sha, state = _write_checkpoints(tmp_path)
    runtime = build_hira_v1_s1_query_keyed_core(
        _encoder(),
        t0,
        w34,
        expected_t0_sha256=t0_sha,
        expected_w34_sha256=w34_sha,
        train_extractor=True,
    )
    scorer = runtime.query_keyed_evidence_scorer
    assert scorer is not None
    assert scorer.extractor_parameter_count == HIRA_V1_S1_EXTRACTOR_PARAMETER_COUNT

    trainable = {
        name: p.numel()
        for name, p in runtime.named_parameters()
        if p.requires_grad
    }
    assert sum(trainable.values()) == 8192
    assert set(trainable) == {
        "query_keyed_evidence_scorer.question_heads.weight",
        "query_keyed_evidence_scorer.state_keys.weight",
    }
    own = scorer.state_dict()
    for key, value in state.items():
        assert torch.equal(own[key], value)


def test_s1_parameter_free_builder_is_fully_frozen(tmp_path):
    torch.manual_seed(51106)
    t0, t0_sha, w34, w34_sha, _ = _write_checkpoints(tmp_path)
    runtime = build_hira_v1_s1_parameter_free_core(
        _encoder(),
        t0,
        w34,
        expected_t0_sha256=t0_sha,
        expected_w34_sha256=w34_sha,
    )
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 0
    assert runtime.parameter_free_query_evidence_scorer is not None
