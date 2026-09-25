from __future__ import annotations

from nmd.semantic_alignment_verdict import semantic_alignment_verdict


def _result(
    *,
    k4: float,
    k16: float,
    coarse4: float | None = None,
    label4: float = 0.50,
    alignment: float,
):
    coarse = k4 if coarse4 is None else coarse4
    domains = {}
    align = {}
    for domain in ("BD", "BE"):
        domains[domain] = {
            "views": {
                "definition": {
                    "4": {
                        "final_top1": k4,
                        "coarse_top1": coarse,
                    },
                    "16": {
                        "final_top1": k16,
                    },
                },
                "label": {
                    "4": {
                        "final_top1": label4,
                    },
                },
            },
        }
        align[domain] = {"top1": alignment}
    return {
        "per_domain": domains,
        "alignment": {"per_domain": align},
        "probability_mass_max_error": 1e-7,
        "state_encodes_per_base": 1.0,
    }


def _candidate_set(
    frozen,
    projection,
    shared,
    primary,
    replica,
):
    return {
        "frozen-w6e-control": frozen,
        "projection-semantic-control": projection,
        "shared-bridge-semantic-control": shared,
        "asymmetric-bridge-primary": primary,
        "asymmetric-bridge-replica": replica,
    }


def test_full_asymmetric_rescue_verdict():
    rows = _candidate_set(
        _result(k4=.40, k16=.20, alignment=.40),
        _result(k4=.70, k16=.52, alignment=.64),
        _result(k4=.70, k16=.52, alignment=.64),
        _result(
            k4=.80,
            k16=.60,
            coarse4=.76,
            label4=.50,
            alignment=.75,
        ),
        _result(k4=.72, k16=.52, alignment=.68),
    )
    verdict, details = semantic_alignment_verdict(rows)
    assert verdict == "ASYMMETRIC_SEMANTIC_BRIDGE_RESCUE"
    assert details["full_primary"] is True
    assert details["full_replica"] is True
    assert details["full_causal"] is True


def test_alignment_rescue_without_asymmetry():
    rows = _candidate_set(
        _result(k4=.40, k16=.20, alignment=.40),
        _result(k4=.80, k16=.60, alignment=.75),
        _result(k4=.72, k16=.53, alignment=.66),
        _result(
            k4=.78,
            k16=.58,
            coarse4=.75,
            label4=.50,
            alignment=.73,
        ),
        _result(k4=.72, k16=.52, alignment=.68),
    )
    verdict, details = semantic_alignment_verdict(rows)
    assert verdict == "SEMANTIC_ALIGNMENT_RESCUE_NO_ASYMMETRY"
    assert "projection-semantic-control" in details["rescuing_controls"]


def test_partial_requires_both_domains():
    rows = _candidate_set(
        _result(k4=.40, k16=.20, alignment=.40),
        _result(k4=.45, k16=.22, alignment=.42),
        _result(k4=.44, k16=.23, alignment=.43),
        _result(k4=.55, k16=.30, alignment=.52),
        _result(k4=.48, k16=.24, alignment=.46),
    )
    verdict, _ = semantic_alignment_verdict(rows)
    assert verdict == "SEMANTIC_ALIGNMENT_PARTIAL"


def test_fail_when_material_gain_is_absent():
    rows = _candidate_set(
        _result(k4=.40, k16=.20, alignment=.40),
        _result(k4=.43, k16=.21, alignment=.41),
        _result(k4=.42, k16=.21, alignment=.41),
        _result(k4=.46, k16=.24, alignment=.44),
        _result(k4=.44, k16=.22, alignment=.43),
    )
    verdict, _ = semantic_alignment_verdict(rows)
    assert verdict == "SEMANTIC_ALIGNMENT_FAIL"
