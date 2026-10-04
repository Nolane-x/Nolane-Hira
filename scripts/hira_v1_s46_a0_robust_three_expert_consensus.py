from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.contracts import LogicalOption
from nmd.v1_evidence_fusion import standardize_full_k_evidence
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_robust_three_expert_consensus import RobustThreeExpertMedianFusion
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import HIRA_V1_S17_TOTAL_PARAMETER_COUNT
import hira_v1_s35_train_dev as s35
from hira_v1_s45_a0_cross_view_consistent_private_correction import (
    _native_outputs,
    _ownership_warmstart_court,
    _runtime,
)

SCHEMA_VERSION = "hira-v1-s46-a0-robust-three-expert-consensus-v1"
OUTCOME = "HIRA_V1_S46_A0_ROBUST_THREE_EXPERT_CONSENSUS_READY"

NATIVE_TRAINABLE = 49_152
PRIVATE_ADAPTER = 49_152
BILINEAR = 65_536
CORRECTION = 114_688
TREATMENT_TOTAL = 163_840
FUSION_PARAMETER_COUNT = 0
SEED = 67_001


@dataclass(frozen=True)
class Case:
    case_id: str
    noun: str
    field_a: str
    first: str
    field_b: str
    second: str
    wrong_a: str
    wrong_b: str
    seed: int

    @property
    def state_a(self):
        return (
            f"S46-A0 {self.noun} record {self.case_id}: "
            f"{self.field_a} {self.first}; {self.field_b} {self.second}."
        )

    @property
    def state_b(self):
        return (
            f"Audit {self.case_id} stores {self.second} for {self.field_b}; "
            f"the same S46-A0 {self.noun} stores {self.first} for {self.field_a}."
        )

    @property
    def qa1(self):
        return f"For S46-A0 {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which entry is tagged {self.field_a} in S46-A0 {self.case_id}?"

    @property
    def qb1(self):
        return f"For S46-A0 {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which entry is tagged {self.field_b} in S46-A0 {self.case_id}?"

    def option_pack(self):
        rows = [
            ("a", self.field_a, self.first),
            ("b", self.field_b, self.second),
            ("x", self.field_a, self.wrong_a),
            ("y", self.field_b, self.wrong_b),
        ]
        random.Random(self.seed).shuffle(rows)
        options = tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=f"for this {self.noun}, {field} is {value}",
                aliases=(
                    f"{value} is the S46 consensus-audit {field} entry for this {self.noun}",
                ),
            )
            for i, (_kind, field, value) in enumerate(rows)
        )
        ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
        gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
        return options, ga, gb


def cases():
    return (
        Case("RM11","quantum vortex camera","sensor","vector NV tile","noise","3.2 nT","Hall bar","12.8 nT",99101),
        Case("RM22","attosecond phase ruler","optic","chirped multilayer","jitter","6 as","glass plate","24 as",99102),
        Case("RM33","phonon spin compass","guide","chiral SiC ridge","angle floor","0.09 deg","Al bar","0.36 deg",99103),
        Case("RM44","Rydberg field scanner","cell","cryogenic Cs lattice","field floor","7 nV/cm","metal loop","28 nV/cm",99104),
        Case("RM55","exciton recoil mapper","sample","moire MoSe2 pair","momentum blur","0.12 nm-1","bulk Si","0.48 nm-1",99105),
        Case("RM66","muon timing lens","target","high-purity Ag foil","spread","5 ns","plastic tile","20 ns",99106),
        Case("RM77","skyrmion torque camera","film","DMI PtCo stack","floor","4 zNm","permalloy film","16 zNm",99107),
        Case("RM88","Casimir phase balance","surface","annealed Au sphere","drift","0.37 pN/m","rough Cu plate","1.48 pN/m",99108),
        Case("RN11","molecular parity compass","species","oriented HfF+ packet","drift","1.9e-16","thermal NO beam","7.6e-16",99109),
        Case("RN22","topological acoustic radar","waveguide","helical AlN ridge","backscatter","-46 dB","bulk ceramic","-18 dB",99110),
        Case("RN33","polariton coherence camera","cavity","high-Q perovskite","linewidth","0.38 meV","polymer cavity","1.52 meV",99111),
        Case("RN44","ion mobility mapper","species","clocked Yb+ chain","drift","1.4e-7","thermal Ba+","5.6e-7",99112),
        Case("RN55","nanostrain holograph","probe","SiC divacancy","floor","6 neps","foil gauge","24 neps",99113),
        Case("RN66","terahertz vector camera","detector","graphene FET array","noise","0.72 uV/cm","Si diode","2.88 uV/cm",99114),
        Case("RN77","superconducting flux radar","pickup","MoRe nanoloop","floor","15 nPhi0","Al loop","60 nPhi0",99115),
        Case("RN88","Raman rotation clock","medium","buffer-gas Rb cell","rotation floor","3 nrad/s","air cell","12 nrad/s",99116),
    )


def _rows(suite):
    out = []
    for case in suite:
        options, ga, gb = case.option_pack()
        out.append(
            S17FusionCase(
                case_id=f"a0-s46-{case.case_id}",
                split="train",
                domain="s46_a0_only",
                language="en",
                state_a=case.state_a,
                state_b=case.state_b,
                question_a1=case.qa1,
                question_a2=case.qa2,
                question_b1=case.qb1,
                question_b2=case.qb2,
                option_texts=tuple(x.criterion_text for x in options),
                option_aliases=tuple(x.aliases[0] for x in options),
                option_ids=tuple(x.option_id for x in options),
                gold_a=ga,
                gold_b=gb,
            )
        )
    return out


def _mechanics_court():
    op = RobustThreeExpertMedianFusion(epsilon=1e-6)
    if op.parameter_count != 0:
        raise RuntimeError("S46-A0 fusion gained parameters")

    g = torch.Generator().manual_seed(67460)
    arbitrary = {}
    max_mass_error = 0.0
    for k in (3, 7, 255):
        p = torch.randn(4, k, generator=g)
        n = torch.randn(4, k, generator=g)
        c = torch.randn(4, k, generator=g)
        fused, _ = op(p, n, c)
        if tuple(fused.shape) != (4, k):
            raise RuntimeError("S46-A0 arbitrary-K shape changed")
        if not bool(torch.isfinite(fused).all()):
            raise RuntimeError("S46-A0 arbitrary-K non-finite")
        mass = float((torch.softmax(fused, dim=-1).sum(-1) - 1.0).abs().max())
        max_mass_error = max(max_mass_error, mass)
        arbitrary[f"k{k}_pass"] = True

    p = torch.randn(3, 7, generator=g)
    n = torch.randn(3, 7, generator=g)
    c = torch.randn(3, 7, generator=g)
    base, _ = op(p, n, c)
    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    moved, _ = op(p[:, perm], n[:, perm], c[:, perm])
    permutation_error = float((moved - base[:, perm]).abs().max())

    shifted, _ = op(
        3.25 * p + 17.0,
        0.75 * n - 9.0,
        8.5 * c + 101.0,
    )
    scale_shift_error = float((shifted - base).abs().max())

    identical, _ = op(p, p, p)
    standardized, _ = standardize_full_k_evidence(p, epsilon=1e-6)
    identical_error = float((identical - standardized).abs().max())

    agreeing = torch.randn(5, 7, generator=g)
    agreeing2 = 5.0 * agreeing + 73.0
    outlier = torch.randn(5, 7, generator=g) * 1e9
    robust, _ = op(agreeing, agreeing2, outlier)
    expected, _ = standardize_full_k_evidence(agreeing, epsilon=1e-6)
    outlier_error = float((robust - expected).abs().max())

    experts = []
    for x in (p, n, c):
        sx, _ = standardize_full_k_evidence(x, epsilon=1e-6)
        experts.append(sx)
    stack = torch.stack(experts, dim=1)
    lower_violation = float((stack.amin(dim=1) - base).clamp_min(0).max())
    upper_violation = float((base - stack.amax(dim=1)).clamp_min(0).max())

    flat = torch.full_like(p, 3.0)
    flat_fused, _ = op(p, n, flat)
    flat_finite = bool(torch.isfinite(flat_fused).all())

    if permutation_error > 3e-6:
        raise RuntimeError("S46-A0 option permutation equivariance failed")
    if scale_shift_error > 3e-6:
        raise RuntimeError("S46-A0 shift/scale invariance failed")
    if identical_error != 0.0:
        raise RuntimeError("S46-A0 identical expert identity failed")
    if outlier_error > 3e-6:
        raise RuntimeError("S46-A0 outlier containment failed")
    if lower_violation != 0.0 or upper_violation != 0.0:
        raise RuntimeError("S46-A0 median envelope failed")
    if not flat_finite:
        raise RuntimeError("S46-A0 flat expert became non-finite")
    if max_mass_error > 1e-6:
        raise RuntimeError("S46-A0 probability mass failed")

    return {
        "fusion_parameter_count": op.parameter_count,
        "arbitrary_k3_pass": arbitrary["k3_pass"],
        "arbitrary_k7_pass": arbitrary["k7_pass"],
        "arbitrary_k255_pass": arbitrary["k255_pass"],
        "logical_option_permutation_max_abs_error": permutation_error,
        "independent_shift_positive_scale_max_abs_error": scale_shift_error,
        "all_three_identical_max_abs_error": identical_error,
        "one_extreme_outlier_max_abs_error": outlier_error,
        "median_lower_envelope_violation": lower_violation,
        "median_upper_envelope_violation": upper_violation,
        "flat_expert_finite": flat_finite,
        "max_probability_mass_error": max_mass_error,
    }


def _actual_shell_court(bundle, manifest, rows):
    runtime = _runtime(bundle=bundle, manifest=manifest, seed=SEED, train=False)
    raw_c, raw_p, native_c, native_p, sig_c, sig_p, encoded = _native_outputs(
        runtime, rows
    )

    correction = PrivateCorrectionRepresentationFork(train_correction=True)
    g = torch.Generator().manual_seed(67461)
    with torch.no_grad():
        correction.adapter_b.copy_(torch.randn(256, 64, generator=g) * 0.01)
        correction.bilinear_weight.copy_(torch.randn(256, 256, generator=g) * 0.01)

    corrected_c = correction.correction_logits(
        native_logits=native_c,
        signatures=sig_c,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    corrected_p = correction.correction_logits(
        native_logits=native_p,
        signatures=sig_p,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )

    legacy = GradientIsolatedFullKEvidenceFusion(epsilon=s35.FUSION_EPSILON)
    robust = RobustThreeExpertMedianFusion(epsilon=s35.FUSION_EPSILON)

    legacy_c, _ = legacy(raw_c, corrected_c)
    legacy_p, _ = legacy(raw_p, corrected_p)
    robust_c, diag_c = robust(raw_c, native_c, corrected_c)
    robust_p, diag_p = robust(raw_p, native_p, corrected_p)

    shell_delta = max(
        float((robust_c - legacy_c).abs().max().detach().cpu()),
        float((robust_p - legacy_p).abs().max().detach().cpu()),
    )
    if shell_delta <= 1e-7:
        raise RuntimeError("S46-A0 robust shell collapsed to legacy shell")

    mass = max(
        float((torch.softmax(robust_c, dim=-1).sum(-1) - 1.0).abs().max().cpu()),
        float((torch.softmax(robust_p, dim=-1).sum(-1) - 1.0).abs().max().cpu()),
    )
    if mass > 1e-6:
        raise RuntimeError("S46-A0 actual shell probability mass failed")

    if robust.parameter_count != 0:
        raise RuntimeError("S46-A0 actual shell gained parameters")
    if correction.correction_parameter_count != CORRECTION:
        raise RuntimeError("S46-A0 correction capacity changed")
    if HIRA_V1_S17_TOTAL_PARAMETER_COUNT != NATIVE_TRAINABLE:
        raise RuntimeError("S46-A0 native parameter surface changed")

    return {
        "actual_shell_legacy_vs_robust_max_abs": shell_delta,
        "actual_shell_probability_mass_error": mass,
        "actual_shell_state_view_encodes": 2 * len(rows),
        "actual_shell_one_encoder_batch": True,
        "actual_shell_fusion_parameter_count": robust.parameter_count,
        "actual_shell_canonical_diagnostics": diag_c.to_dict(),
        "actual_shell_paraphrase_diagnostics": diag_p.to_dict(),
        "correction_parameter_count": correction.correction_parameter_count,
        "treatment_total_trainable_parameter_count": (
            HIRA_V1_S17_TOTAL_PARAMETER_COUNT + correction.correction_parameter_count
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S46-A0 suite size changed")
    rows = _rows(suite)

    bundle = args.bundle.resolve()
    from nmd.local_runtime import read_runtime_bundle_manifest
    manifest = read_runtime_bundle_manifest(bundle)

    mechanics = _mechanics_court()
    shell = _actual_shell_court(bundle, manifest, rows)
    ownership = _ownership_warmstart_court(bundle, manifest, rows)

    if ownership["js_only_native_runtime_gradient_l1"] != 0.0:
        raise RuntimeError("S46-A0 inherited correction JS leaked into native runtime")
    if ownership["native_objective_correction_gradient_l1"] != 0.0:
        raise RuntimeError("S46-A0 inherited native objective leaked into correction")
    if ownership["matched_native_one_step_parameter_max_abs"] != 0.0:
        raise RuntimeError("S46-A0 inherited native trajectory identity changed")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S46_A0_ROBUST_THREE_EXPERT_CONSENSUS_ONLY",
        "semantic_case_count": len(suite),
        "state_view_count": 2 * len(suite),
        "k": 4,
        "views_per_option": 2,
        "seed": SEED,
        "native_trainable_parameter_count": NATIVE_TRAINABLE,
        "private_adapter_parameter_count": PRIVATE_ADAPTER,
        "bilinear_parameter_count": BILINEAR,
        "correction_parameter_count": CORRECTION,
        "treatment_total_trainable_parameter_count": TREATMENT_TOTAL,
        "fusion_family": "coordinatewise_median_of_three_independently_standardized_full_k_experts",
        "fusion_epsilon": 1e-6,
        "training_mechanics": "exact_s45",
        "correction_objective": "0.10_ce_plus_0.25_cross_view_js",
        "second_encoder_pass": False,
        **mechanics,
        **shell,
        "ownership": ownership,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "HIRA_V1_S46_A0_ROBUST_CONSENSUS_RECEIPT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
