from copy import deepcopy

import torch

from nmd.mainline_m3_r1_training import (
    M3_R1_ENGLISH_ANCHOR_COEFF,
    M3_R1_EPOCHS,
    M3_R1_GRAD_CLIP,
    M3_R1_LR,
    M3_R1_PAIR_COEFF,
    M3_R1_PRIMARY_SEED,
    M3_R1_REPLICA_SEED,
    M3_R1_WEIGHT_DECAY,
    _selection_key,
    m3_r1_joint_qualification,
)


def _report(
    en_top1,
    vi_top1,
    agreement,
    top1_ratio,
    mrr_ratio,
    en_mrr,
    vi_mrr,
):
    return {
        "en": {"top1": en_top1, "mrr": en_mrr},
        "vi": {"top1": vi_top1, "mrr": vi_mrr},
        "paired_prediction_agreement": agreement,
        "vi_en_top1_ratio": top1_ratio,
        "vi_en_mrr_ratio": mrr_ratio,
    }


def test_m3_r1_hyperparameters_are_frozen():
    assert M3_R1_EPOCHS == 8
    assert M3_R1_LR == 2e-4
    assert M3_R1_WEIGHT_DECAY == 0.01
    assert M3_R1_GRAD_CLIP == 1.0
    assert M3_R1_PAIR_COEFF == 0.35
    assert M3_R1_ENGLISH_ANCHOR_COEFF == 0.10
    assert M3_R1_PRIMARY_SEED == 23031
    assert M3_R1_REPLICA_SEED == 23037


def test_m3_r1_selection_prioritizes_worst_language_before_agreement():
    stronger_worst = _report(
        0.70, 0.69, 0.86, 0.986, 0.97, 0.80, 0.78
    )
    prettier_agreement = _report(
        0.90, 0.60, 0.99, 0.667, 0.95, 0.91, 0.86
    )
    assert _selection_key(stronger_worst, 8) > _selection_key(
        prettier_agreement,
        1,
    )


def test_m3_r1_selection_breaks_equal_quality_ties_with_earlier_epoch():
    report = _report(0.70, 0.70, 0.90, 1.0, 1.0, 0.82, 0.82)
    assert _selection_key(report, 2) > _selection_key(report, 7)


def _qualification(passed: bool):
    gates = {
        "en_top1": passed,
        "vi_top1": passed,
        "paired_prediction_agreement": passed,
    }
    return {"pass": passed, "gates": gates}


def test_m3_r1_joint_qualification_requires_primary_and_replica():
    primary = {"dev_qualification": _qualification(True)}
    replica = {"dev_qualification": _qualification(True)}
    result = m3_r1_joint_qualification(primary, replica)
    assert result["pass"] is True

    broken = deepcopy(replica)
    broken["dev_qualification"] = _qualification(False)
    failed = m3_r1_joint_qualification(primary, broken)
    assert failed["pass"] is False
    assert failed["primary"]["pass"] is True
    assert failed["replica"]["pass"] is False
