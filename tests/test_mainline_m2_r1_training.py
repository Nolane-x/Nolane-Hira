import copy

import torch

from nmd.hira import HIRACore
from nmd.mainline import HiraV0Mainline, HiraV0Manifest
from nmd.mainline_m2_r1_training import (
    M2_R1_CANDIDATES,
    M2_R1_EPOCHS,
    M2_R1_GRAD_CLIP,
    M2_R1_LR,
    M2_R1_PAIR_MARGIN,
    M2_R1_PAIR_MARGIN_WEIGHT,
    M2_R1_PRIMARY_GATES,
    M2_R1_REPLICA_GATES,
    M2_R1_SEEDS,
    M2_R1_TEMPERATURE,
    M2_R1_WEIGHT_DECAY,
    candidate_family,
    clone_w34_for_rescue,
    install_m2_r1_candidate,
    m2_r1_runtime_gate,
    select_m2_r1_family,
    train_m2_r1_candidate,
    validate_m2_r1_cache,
)
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.w33_coevidence_semantic import CoEvidenceSemanticScorer


def _frozen_scorer():
    torch.manual_seed(23001)
    scorer = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    scorer.freeze_candidate()
    return scorer


def _cache(split):
    torch.manual_seed(23011 if split == "train" else 23013)
    cases = []
    for k in (64, 128):
        for index in range(2):
            state = torch.randn(5, 256)
            options = torch.randn(k, 1, 3, 256)
            token_mask = torch.ones(k, 1, 3, dtype=torch.bool)
            view_mask = torch.ones(k, 1, dtype=torch.bool)
            cases.append({
                "case_id": f"{split}-k{k}-{index}",
                "split": split,
                "domain_id": f"{split}-{k}",
                "k": k,
                "gold_index": (index * 7 + 3) % k,
                "option_ids": tuple(f"o-{i}" for i in range(k)),
                "state_tokens": state,
                "option_view_tokens": options,
                "option_view_token_mask": token_mask,
                "option_view_mask": view_mask,
            })
    return {
        "metadata": {
            "schema_version": "hira-v0-mainline-m2-r1-cache-v1",
            "split": split,
            "case_count": len(cases),
            "state_encode_count": len(cases),
            "state_encodes_per_case": 1.0,
            "decision_core_frozen": True,
            "encoder_gradient_updates": False,
            "projection_gradient_updates": False,
        },
        "cases": cases,
    }


def _metrics(k, top1, top5, mrr):
    return {
        "k": k,
        "case_count": 24,
        "top1": top1,
        "top5": top5,
        "mrr": mrr,
        "mean_gold_rank": 2.0,
        "mean_gold_probability": 0.2,
        "mean_confidence": 0.3,
        "probability_mass_max_error": 1e-7,
        "permutation_max_error": 1e-9,
        "selected_option_invariant_rate": 1.0,
        "full_k_rate": 1.0,
        "state_once_rate": 1.0,
        "relation_delta_max": 0.0,
        "finite_rate": 1.0,
        "mean_case_ms": 1.0,
    }


def test_m2_r1_frozen_hyperparameter_surface():
    assert M2_R1_CANDIDATES == (
        "ce-primary",
        "ce-replica",
        "ce-margin-primary",
        "ce-margin-replica",
    )
    assert M2_R1_SEEDS == {
        "ce-primary": 22031,
        "ce-replica": 22037,
        "ce-margin-primary": 22043,
        "ce-margin-replica": 22051,
    }
    assert M2_R1_EPOCHS == 8
    assert M2_R1_LR == 2e-4
    assert M2_R1_WEIGHT_DECAY == 0.01
    assert M2_R1_GRAD_CLIP == 1.0
    assert M2_R1_TEMPERATURE == 0.07
    assert M2_R1_PAIR_MARGIN == 0.06
    assert M2_R1_PAIR_MARGIN_WEIGHT == 0.25
    assert candidate_family("ce-primary") == "ce"
    assert candidate_family("ce-margin-primary") == "ce-margin"


def test_m2_r1_clone_trains_exactly_w34_candidate_surface():
    frozen = _frozen_scorer()
    candidate = clone_w34_for_rescue(frozen)
    assert candidate.projection.weight.requires_grad is False
    assert candidate.candidate_parameter_count == 8192
    assert candidate.candidate_trainable_parameter_count == 8192
    assert candidate.trainable_parameter_count == 8192


def test_m2_r1_cache_validation_rejects_split_and_state_once_drift():
    cache = _cache("train")
    validate_m2_r1_cache(cache, expected_split="train")

    broken = copy.deepcopy(cache)
    broken["metadata"]["state_encodes_per_case"] = 2.0
    try:
        validate_m2_r1_cache(broken, expected_split="train")
        assert False, "expected state-once validation failure"
    except ValueError as exc:
        assert "state-once" in str(exc)


def test_m2_r1_one_epoch_training_keeps_projection_frozen():
    frozen = _frozen_scorer()
    projection_before = frozen.projection.weight.detach().clone()
    result = train_m2_r1_candidate(
        "ce-primary",
        frozen,
        _cache("train"),
        _cache("dev"),
        epochs=1,
    )
    assert result["parameter_count"] == 8192
    assert result["selected_epoch"] == 1
    assert set(result["selected_dev"]["per_k"]) == {"64", "128"}
    assert torch.equal(frozen.projection.weight, projection_before)


def test_m2_r1_family_selection_binds_corresponding_replica():
    def row(name, key):
        return {
            "candidate": name,
            "family": candidate_family(name),
            "seed": M2_R1_SEEDS[name],
            "parameter_count": 8192,
            "selected_epoch": 1,
            "selected_dev": {
                "case_count": 48,
                "per_k": {
                    "64": {"case_count": 24.0, "top1": key[1], "top5": key[3], "mrr": key[5]},
                    "128": {"case_count": 24.0, "top1": key[0], "top5": key[2], "mrr": key[4]},
                },
            },
            "selected_state_dict": {},
            "selection_key": list(key) + [-1.0],
            "history": [],
        }

    ce = (0.60, 0.70, 0.85, 0.90, 0.65, 0.75)
    margin = (0.70, 0.80, 0.90, 0.95, 0.75, 0.85)
    rows = [
        row("ce-primary", ce),
        row("ce-replica", ce),
        row("ce-margin-primary", margin),
        row("ce-margin-replica", margin),
    ]
    selected = select_m2_r1_family(rows)
    assert selected["family"] == "ce-margin"
    assert selected["primary_candidate"] == "ce-margin-primary"
    assert selected["replica_candidate"] == "ce-margin-replica"
    assert selected["cached_dev_qualified"] is True


def test_m2_r1_runtime_gates_require_semantics_and_mechanics():
    strong = {
        "per_k": {
            "64": _metrics(64, 0.75, 0.95, 0.80),
            "128": _metrics(128, 0.65, 0.90, 0.70),
        }
    }
    assert m2_r1_runtime_gate(strong)["pass"] is True

    replica = {
        "per_k": {
            "64": _metrics(64, 0.68, 0.88, 0.72),
            "128": _metrics(128, 0.58, 0.82, 0.62),
        }
    }
    assert m2_r1_runtime_gate(replica, replica=True)["pass"] is True

    broken = copy.deepcopy(strong)
    broken["per_k"]["128"]["full_k_rate"] = 0.99
    gate = m2_r1_runtime_gate(broken)
    assert gate["pass"] is False
    assert gate["per_k"]["128"]["mechanics"]["full_k"] is False

    assert M2_R1_PRIMARY_GATES[128]["top1"] == 0.60
    assert M2_R1_REPLICA_GATES[128]["top1"] == 0.55


def test_install_m2_r1_candidate_freezes_runtime_scorer():
    encoder = TrainableSemanticEncoder(
        vocab_size=512,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=32,
    )
    scorer = _frozen_scorer()
    runtime = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        coevidence_symmetric_semantic_scorer=scorer,
    )
    model = HiraV0Mainline(
        runtime,
        manifest=HiraV0Manifest.m2_mechanics_available(),
    )
    trainable = clone_w34_for_rescue(scorer)
    state = {
        name: value
        for name, value in trainable.state_dict().items()
        if (
            name.startswith("state_adapter.")
            or name.startswith("schema_adapter.")
            or name.startswith("interaction_state.")
            or name.startswith("interaction_schema.")
            or name.startswith("composition_state.")
            or name.startswith("composition_schema.")
        )
    }
    install_m2_r1_candidate(model, state)
    assert model.runtime.coevidence_symmetric_semantic_scorer.trainable_parameter_count == 0
