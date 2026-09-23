import torch
from torch import nn

from nmd.semantic_encoder_adaptation import (
    all_w5d_vocab,
    competence_gates,
    confirm_verdict,
    discover_transformer_layers,
    generate_adaptation_authority,
)
from nmd.semantic_alignment_probes import all_w5c_vocab
from nmd import semantic_routing_curriculum as w5a
from nmd import semantic_token_curriculum as w5b


def prior_vocab():
    groups = (
        w5a.TRAIN_MATERIALS,
        w5a.TRAIN_ENTITIES,
        w5a.TRAIN_ACTIONS,
        w5a.TRAIN_LOCATIONS,
        w5a.CONFIRM_SYMBOLS,
        w5a.CONFIRM_ITEMS,
        w5a.CONFIRM_ACTIONS,
        w5a.CONFIRM_LOCATIONS,
        w5b.TRAIN_TAGS,
        w5b.TRAIN_ROLES,
        w5b.TRAIN_ACTIONS,
        w5b.TRAIN_SITES,
        w5b.CONFIRM_TAGS,
        w5b.CONFIRM_ROLES,
        w5b.CONFIRM_ACTIONS,
        w5b.CONFIRM_SITES,
    )
    return {item for group in groups for item in group} | all_w5c_vocab()


def test_w5d_vocab_is_disjoint_from_w5a_w5b_w5c():
    assert all_w5d_vocab().isdisjoint(prior_vocab())


def test_w5d_authority_counts_and_confirm_isolation():
    train = generate_adaptation_authority("train")
    dev = generate_adaptation_authority("dev")
    confirm = generate_adaptation_authority("confirm")

    assert len(train) == 416
    assert len(dev) == 144
    assert len(confirm) == 192
    assert sum(row.k == 128 for row in train) == 48
    assert sum(row.k == 255 for row in train) == 32
    assert sum(row.k == 128 for row in dev) == 24
    assert sum(row.k == 255 for row in dev) == 24
    assert sum(row.k == 128 for row in confirm) == 48
    assert sum(row.k == 255 for row in confirm) == 48
    assert {row.split for row in train} == {"train"}
    assert {row.split for row in dev} == {"dev"}
    assert {row.split for row in confirm} == {"confirm"}


class FakeBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(6)]
        )


class FakeDistil(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.layer = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(6)]
        )


def test_transformer_layer_discovery_supports_bert_and_distil_shapes():
    assert len(discover_transformer_layers(FakeBert())) == 6
    assert len(discover_transformer_layers(FakeDistil())) == 6


def metrics(overall, k128, k255, top5=0.8, mass=1e-7):
    return {
        "accuracy": overall,
        "top5_recall": top5,
        "mrr": 0.5,
        "hard_brier": 0.3,
        "probability_mass_max_error": mass,
        "per_k": {
            "32": {"accuracy": overall, "top5_recall": top5},
            "64": {"accuracy": overall, "top5_recall": top5},
            "128": {"accuracy": k128, "top5_recall": top5},
            "255": {"accuracy": k255, "top5_recall": top5},
        },
    }


def test_confirm_verdict_requires_competence_and_gain():
    frozen = metrics(0.05, 0.0, 0.0, top5=0.1)
    adapted = metrics(0.70, 0.50, 0.40, top5=0.80)
    verdict, gates = confirm_verdict(adapted, frozen)
    assert verdict == "A13_ADAPTATION_PASS"
    assert all(gates.values())

    weak = metrics(0.30, 0.10, 0.05, top5=0.2)
    verdict, gates = confirm_verdict(weak, frozen)
    assert verdict == "A13_ADAPTATION_FAIL_CAPACITY_TRIGGER"
    assert not all(gates.values())


def test_gain_gate_prevents_false_pass_when_frozen_is_already_strong():
    frozen = metrics(0.55, 0.45, 0.35, top5=0.8)
    adapted = metrics(0.65, 0.50, 0.40, top5=0.8)
    gates = competence_gates(adapted, frozen)
    assert gates["overall_accuracy"] is True
    assert gates["k128_accuracy"] is True
    assert gates["k255_accuracy"] is True
    assert gates["overall_gain_vs_frozen"] is False
