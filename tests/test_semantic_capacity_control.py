from nmd.semantic_capacity_control import (
    all_w5e_vocab,
    capacity_verdict,
    generate_capacity_authority,
)
from nmd.semantic_encoder_adaptation import all_w5d_vocab
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
    return (
        {item for group in groups for item in group}
        | all_w5c_vocab()
        | all_w5d_vocab()
    )


def test_w5e_vocab_is_disjoint_from_w5a_through_w5d():
    assert all_w5e_vocab().isdisjoint(prior_vocab())


def test_w5e_authority_counts_and_confirm_isolation():
    train = generate_capacity_authority("train")
    dev = generate_capacity_authority("dev")
    confirm = generate_capacity_authority("confirm")

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


def metrics(overall, k128, k255, mrr, top5=0.8, mass=1e-7):
    return {
        "accuracy": overall,
        "top5_recall": top5,
        "mrr": mrr,
        "hard_brier": 0.3,
        "probability_mass_max_error": mass,
        "per_k": {
            "32": {"accuracy": overall, "top5_recall": top5},
            "64": {"accuracy": overall, "top5_recall": top5},
            "128": {"accuracy": k128, "top5_recall": top5},
            "255": {"accuracy": k255, "top5_recall": top5},
        },
    }


def test_capacity_rescue_requires_competence_and_delta():
    a13 = metrics(0.45, 0.25, 0.15, 0.40)
    a22 = metrics(0.70, 0.50, 0.40, 0.60, top5=0.80)
    verdict, gates = capacity_verdict(a22, a13)
    assert verdict == "A22_CAPACITY_RESCUE"
    assert all(gates.values())


def test_capacity_no_rescue_when_a22_still_misses_competence():
    a13 = metrics(0.02, 0.0, 0.0, 0.04, top5=0.02)
    a22 = metrics(0.10, 0.05, 0.02, 0.10, top5=0.10)
    verdict, gates = capacity_verdict(a22, a13)
    assert verdict == "A22_CAPACITY_NO_RESCUE"
    assert gates["overall_accuracy"] is False


def test_capacity_ambiguous_when_a22_passes_but_does_not_gain_enough():
    a13 = metrics(0.58, 0.45, 0.35, 0.56, top5=0.75)
    a22 = metrics(0.62, 0.46, 0.36, 0.59, top5=0.75)
    verdict, gates = capacity_verdict(a22, a13)
    assert verdict == "A22_CAPACITY_AMBIGUOUS"
    assert gates["overall_accuracy"] is True
    assert gates["overall_capacity_gain"] is False
