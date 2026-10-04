from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_invariance import selected_choice_agreement, symmetric_js_divergence
from nmd.v1_ordinal_pairwise_consensus import OrdinalPairwiseConsensus
from nmd.v1_s46_authority import generate_s46_cases
from nmd.v1_s47_authority import generate_s47_cases, validate_s47_partitions
import hira_v1_s17_train_dev as s17mod
import hira_v1_s35_train_dev as s35
import hira_v1_s45_train_dev as s45
import hira_v1_s46_train_dev as s46
from hira_v1_s47_a0_ordinal_pairwise_consensus import cases as s47_a0_cases

SCHEMA_VERSION="hira-v1-s47-matched-ordinal-pairwise-consensus-train-dev-v1"
SEED=68_001

_SHELL_KEYS=s46._SHELL_KEYS


def _sha256(path: Path) -> str:
    digest=sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_texts(row):
    return (row.state_a,row.state_b)


def _question_texts(row):
    return (row.question_a1,row.question_a2,row.question_b1,row.question_b2)


def _option_texts(row):
    return (*row.option_texts,*row.option_aliases)


def _assert_s47_fresh(train_rows,dev_rows):
    s45._assert_s45_fresh(train_rows,dev_rows)

    current=(*train_rows,*dev_rows)
    states={x for r in current for x in _state_texts(r)}
    questions={x for r in current for x in _question_texts(r)}
    options={x for r in current for x in _option_texts(r)}

    prior46=(*generate_s46_cases("train"),*generate_s46_cases("dev"))
    prior_states={x for r in prior46 for x in _state_texts(r)}
    prior_questions={x for r in prior46 for x in _question_texts(r)}
    prior_options={x for r in prior46 for x in _option_texts(r)}
    if states&prior_states:
        raise RuntimeError("S47 exact state overlap with exposed S46 rows")
    if questions&prior_questions:
        raise RuntimeError("S47 exact question overlap with exposed S46 rows")
    if options&prior_options:
        raise RuntimeError("S47 exact option overlap with exposed S46 rows")

    a0=s47_a0_cases()
    a0_states={x for r in a0 for x in (r.state_a,r.state_b)}
    a0_questions={x for r in a0 for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    a0_options=set()
    for case in a0:
        opts,_ga,_gb=case.option_pack()
        for option in opts:
            a0_options.add(option.criterion_text)
            a0_options.update(option.aliases)
    if states&a0_states:
        raise RuntimeError("S47 TRAIN DEV state overlap with S47-A0")
    if questions&a0_questions:
        raise RuntimeError("S47 TRAIN DEV question overlap with S47-A0")
    if options&a0_options:
        raise RuntimeError("S47 TRAIN DEV option overlap with S47-A0")


def _shell_delta(reference,treatment):
    return {key:float(treatment[key])-float(reference[key]) for key in _SHELL_KEYS}


@torch.inference_mode()
def _same_checkpoint_shells(runtime,correction,rows):
    legacy_op=GradientIsolatedFullKEvidenceFusion(epsilon=s35.FUSION_EPSILON)
    ordinal_op=OrdinalPairwiseConsensus()

    names=("legacy_s45","ordinal_s47")
    acc={
        name:{
            "canonical_correct":0,
            "paraphrase_correct":0,
            "paired_both":0,
            "question_swap":0,
            "agreement_sum":0.0,
            "js_sum":0.0,
            "canonical_margin_sum":0.0,
            "paraphrase_margin_sum":0.0,
            "order_flips":0,
            "max_mass_error":0.0,
        }
        for name in names
    }
    expert={
        "raw_c_correct":0,"raw_p_correct":0,
        "native_c_correct":0,"native_p_correct":0,
        "corrected_c_correct":0,"corrected_p_correct":0,
        "corrected_agreement_sum":0.0,"corrected_js_sum":0.0,
        "native_agreement_sum":0.0,"native_js_sum":0.0,
        "signature_cosine_sum":0.0,"signature_margin_sum":0.0,"signature_count":0,
    }
    ordinal_diag={
        "canonical_unanimous_sum":0.0,
        "canonical_majority_sum":0.0,
        "canonical_tied_sum":0.0,
        "canonical_private_tiebreak_sum":0.0,
        "paraphrase_unanimous_sum":0.0,
        "paraphrase_majority_sum":0.0,
        "paraphrase_tied_sum":0.0,
        "paraphrase_private_tiebreak_sum":0.0,
        "query_count":0,
    }
    semantic_cases=0
    state_view_encodes=0
    full_k=True

    s45._set_relation_arm("reference",None)

    for start in range(0,len(rows),s35.BATCH_SIZE):
        batch_rows=list(rows[start:start+s35.BATCH_SIZE])
        (
            _total,_primary,_relation,_pieces,
            _native_fused_c,_native_fused_p,
            raw_c,raw_p,native_c,native_p,
            signature_c,signature_p,encoded,
        )=s45._native_losses(runtime,batch_rows)

        corrected_c=correction.correction_logits(
            native_logits=native_c,
            signatures=signature_c,
            question_tokens=encoded["question_canonical_tokens"],
            question_mask=encoded["question_canonical_mask"],
        )
        corrected_p=correction.correction_logits(
            native_logits=native_p,
            signatures=signature_p,
            question_tokens=encoded["question_paraphrase_tokens"],
            question_mask=encoded["question_paraphrase_mask"],
        )

        legacy_c,_=legacy_op(raw_c,corrected_c)
        legacy_p,_=legacy_op(raw_p,corrected_p)
        ordinal_c,diag_c=ordinal_op(raw_c,native_c,corrected_c)
        ordinal_p,diag_p=ordinal_op(raw_p,native_p,corrected_p)

        n=len(batch_rows)
        queries=2*n
        gold,_other=s35._gold_tensors(batch_rows,device=raw_c.device)

        shell_values={
            "legacy_s45":(legacy_c,legacy_p),
            "ordinal_s47":(ordinal_c,ordinal_p),
        }
        perm=torch.tensor([3,2,1,0],device=raw_c.device)

        for name,(fc_logits,fp_logits) in shell_values.items():
            fc=fc_logits.argmax(-1)
            fp=fp_logits.argmax(-1)
            acc[name]["canonical_correct"]+=int((fc==gold).sum().item())
            acc[name]["paraphrase_correct"]+=int((fp==gold).sum().item())

            pairs=fc.reshape(n,2)
            gpairs=gold.reshape(n,2)
            acc[name]["paired_both"]+=int(((pairs==gpairs).all(-1)).sum().item())
            acc[name]["question_swap"]+=int((pairs[:,0]!=pairs[:,1]).sum().item())
            acc[name]["agreement_sum"]+=float(selected_choice_agreement(fc_logits,fp_logits))*queries
            acc[name]["js_sum"]+=float(symmetric_js_divergence(fc_logits,fp_logits))*queries
            acc[name]["canonical_margin_sum"]+=s46._gold_margin_sum(fc_logits,gold)
            acc[name]["paraphrase_margin_sum"]+=s46._gold_margin_sum(fp_logits,gold)

            if name=="legacy_s45":
                permuted,_=legacy_op(raw_c[:,perm],corrected_c[:,perm])
            else:
                permuted,_=ordinal_op(
                    raw_c[:,perm],native_c[:,perm],corrected_c[:,perm]
                )
            mapped=perm[permuted.argmax(-1)]
            acc[name]["order_flips"]+=int((mapped!=fc).sum().item())

            for logits in (fc_logits,fp_logits):
                probs=torch.softmax(logits,dim=-1)
                acc[name]["max_mass_error"]=max(
                    acc[name]["max_mass_error"],
                    float((probs.sum(-1)-1.0).abs().max().cpu()),
                )

        dc=diag_c.to_dict()
        dp=diag_p.to_dict()
        ordinal_diag["canonical_unanimous_sum"]+=dc["unanimous_pair_fraction"]*queries
        ordinal_diag["canonical_majority_sum"]+=dc["majority_pair_fraction"]*queries
        ordinal_diag["canonical_tied_sum"]+=dc["tied_pair_fraction"]*queries
        ordinal_diag["canonical_private_tiebreak_sum"]+=dc["private_tiebreak_topset_fraction"]*queries
        ordinal_diag["paraphrase_unanimous_sum"]+=dp["unanimous_pair_fraction"]*queries
        ordinal_diag["paraphrase_majority_sum"]+=dp["majority_pair_fraction"]*queries
        ordinal_diag["paraphrase_tied_sum"]+=dp["tied_pair_fraction"]*queries
        ordinal_diag["paraphrase_private_tiebreak_sum"]+=dp["private_tiebreak_topset_fraction"]*queries
        ordinal_diag["query_count"]+=queries

        expert["raw_c_correct"]+=int((raw_c.argmax(-1)==gold).sum().item())
        expert["raw_p_correct"]+=int((raw_p.argmax(-1)==gold).sum().item())
        expert["native_c_correct"]+=int((native_c.argmax(-1)==gold).sum().item())
        expert["native_p_correct"]+=int((native_p.argmax(-1)==gold).sum().item())
        expert["corrected_c_correct"]+=int((corrected_c.argmax(-1)==gold).sum().item())
        expert["corrected_p_correct"]+=int((corrected_p.argmax(-1)==gold).sum().item())
        expert["corrected_agreement_sum"]+=float(
            selected_choice_agreement(corrected_c,corrected_p)
        )*queries
        expert["corrected_js_sum"]+=float(
            symmetric_js_divergence(corrected_c,corrected_p)
        )*queries
        expert["native_agreement_sum"]+=float(
            selected_choice_agreement(native_c,native_p)
        )*queries
        expert["native_js_sum"]+=float(
            symmetric_js_divergence(native_c,native_p)
        )*queries

        same=s17mod.relation_signature_same_option_cosine(signature_c,signature_p)
        c_norm=F.normalize(signature_c,dim=-1)
        p_norm=F.normalize(signature_p,dim=-1)
        cross=torch.einsum("nkd,njd->nkj",c_norm,p_norm)
        k=cross.shape[-1]
        eye=torch.eye(k,dtype=torch.bool,device=cross.device)[None]
        wrong=cross.masked_fill(eye,float("-inf")).amax(-1)
        expert["signature_cosine_sum"]+=float(same.sum().cpu())
        expert["signature_margin_sum"]+=float((same-wrong).sum().cpu())
        expert["signature_count"]+=int(same.numel())

        semantic_cases+=n
        state_view_encodes+=2*n
        full_k=full_k and all(
            x.shape[-1]==4
            for x in (
                raw_c,raw_p,native_c,native_p,corrected_c,corrected_p,
                legacy_c,legacy_p,ordinal_c,ordinal_p,
            )
        )

    queries_total=len(rows)*2
    shells={}
    for name in names:
        a=acc[name]
        shells[name]={
            "semantic_cases":semantic_cases,
            "canonical_queries":queries_total,
            "paraphrase_queries":queries_total,
            "fused_canonical_accuracy":a["canonical_correct"]/queries_total,
            "fused_paraphrase_accuracy":a["paraphrase_correct"]/queries_total,
            "fused_canonical_paired_both_correct_rate":a["paired_both"]/len(rows),
            "fused_question_swap_choice_change_rate":a["question_swap"]/len(rows),
            "fused_cross_view_selected_choice_agreement":a["agreement_sum"]/queries_total,
            "fused_cross_view_mean_js":a["js_sum"]/queries_total,
            "fused_canonical_mean_gold_margin":a["canonical_margin_sum"]/queries_total,
            "fused_paraphrase_mean_gold_margin":a["paraphrase_margin_sum"]/queries_total,
            "fused_option_order_flip_rate":a["order_flips"]/queries_total,
            "fused_max_probability_mass_error":a["max_mass_error"],
            "full_k":bool(full_k),
            "state_view_encodes":state_view_encodes,
        }

    expert_metrics={
        "raw_triadic_canonical_accuracy":expert["raw_c_correct"]/queries_total,
        "raw_triadic_paraphrase_accuracy":expert["raw_p_correct"]/queries_total,
        "native_relation_canonical_accuracy":expert["native_c_correct"]/queries_total,
        "native_relation_paraphrase_accuracy":expert["native_p_correct"]/queries_total,
        "native_relation_cross_view_agreement":expert["native_agreement_sum"]/queries_total,
        "native_relation_cross_view_mean_js":expert["native_js_sum"]/queries_total,
        "corrected_relation_canonical_accuracy":expert["corrected_c_correct"]/queries_total,
        "corrected_relation_paraphrase_accuracy":expert["corrected_p_correct"]/queries_total,
        "corrected_relation_cross_view_agreement":expert["corrected_agreement_sum"]/queries_total,
        "corrected_relation_cross_view_mean_js":expert["corrected_js_sum"]/queries_total,
        "mean_same_option_signature_cosine":expert["signature_cosine_sum"]/expert["signature_count"],
        "mean_signature_same_vs_strongest_wrong_margin":expert["signature_margin_sum"]/expert["signature_count"],
    }

    q=ordinal_diag["query_count"]
    diagnostics={
        "canonical_unanimous_pair_fraction":ordinal_diag["canonical_unanimous_sum"]/q,
        "canonical_majority_pair_fraction":ordinal_diag["canonical_majority_sum"]/q,
        "canonical_tied_pair_fraction":ordinal_diag["canonical_tied_sum"]/q,
        "canonical_private_tiebreak_topset_fraction":ordinal_diag["canonical_private_tiebreak_sum"]/q,
        "paraphrase_unanimous_pair_fraction":ordinal_diag["paraphrase_unanimous_sum"]/q,
        "paraphrase_majority_pair_fraction":ordinal_diag["paraphrase_majority_sum"]/q,
        "paraphrase_tied_pair_fraction":ordinal_diag["paraphrase_tied_sum"]/q,
        "paraphrase_private_tiebreak_topset_fraction":ordinal_diag["paraphrase_private_tiebreak_sum"]/q,
    }
    return shells,expert_metrics,diagnostics


def _ordinal_gates(s45_treatment,ordinal):
    gates=dict(s45_treatment["gates"])
    gates.update({
        "fused_canonical_accuracy_gte_0_85":ordinal["fused_canonical_accuracy"]>=0.85,
        "fused_paired_both_correct_gte_0_75":ordinal["fused_canonical_paired_both_correct_rate"]>=0.75,
        "fused_question_swap_change_gte_0_80":ordinal["fused_question_swap_choice_change_rate"]>=0.80,
        "fused_cross_view_choice_agreement_gte_0_95":ordinal["fused_cross_view_selected_choice_agreement"]>=0.95,
        "fused_cross_view_mean_js_lte_0_05":ordinal["fused_cross_view_mean_js"]<=0.05,
        "fused_canonical_margin_gte_0_15":ordinal["fused_canonical_mean_gold_margin"]>=0.15,
        "fused_option_order_flip_lte_0_02":ordinal["fused_option_order_flip_rate"]<=0.02,
        "fused_probability_mass_error_lte_1e_6":ordinal["fused_max_probability_mass_error"]<=1e-6,
        "full_k":bool(ordinal["full_k"]),
        "decision_added_parameters_zero":OrdinalPairwiseConsensus().parameter_count==0,
    })
    return gates


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--a0-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    a0=json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome")!="HIRA_V1_S47_A0_ORDINAL_PAIRWISE_CONSENSUS_READY":
        raise RuntimeError("S47-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S47-A0 unexpectedly used for model selection")
    expected={
        "native_trainable_parameter_count":49_152,
        "correction_parameter_count":114_688,
        "treatment_total_trainable_parameter_count":163_840,
        "decision_parameter_count":0,
    }
    for key,value in expected.items():
        if int(a0.get(key,-1))!=value:
            raise RuntimeError(f"S47-A0 capacity changed: {key}")
    if bool(a0.get("second_encoder_pass",True)):
        raise RuntimeError("S47-A0 opened a second encoder pass")
    for key in ("arbitrary_k3_pass","arbitrary_k7_pass","arbitrary_k255_pass","actual_shell_one_encoder_batch"):
        if not bool(a0.get(key,False)):
            raise RuntimeError(f"S47-A0 mechanics failed: {key}")
    for key in (
        "logical_option_permutation_max_abs_error",
        "independent_positive_affine_max_abs_error",
        "strictly_increasing_nonlinear_max_abs_error",
        "extreme_magnitude_blowup_max_abs_error",
    ):
        if float(a0.get(key,-1.0))!=0.0:
            raise RuntimeError(f"S47-A0 ordinal invariant failed: {key}")
    if int(a0.get("strict_copeland_dominance_violations",-1))!=0:
        raise RuntimeError("S47-A0 Copeland dominance failed")
    if a0.get("cycle_private_tiebreak_matches") is not True:
        raise RuntimeError("S47-A0 private tie-break failed")

    ownership=a0.get("ownership",{})
    for key in (
        "matched_native_one_step_parameter_max_abs",
        "matched_native_one_step_output_max_abs",
        "zero_init_correction_native_runtime_gradient_l1",
        "native_objective_correction_gradient_l1",
        "js_only_native_runtime_gradient_l1",
    ):
        if float(ownership.get(key,-1.0))!=0.0:
            raise RuntimeError(f"S47-A0 ownership failed: {key}")

    train_rows=generate_s47_cases("train")
    dev_rows=generate_s47_cases("dev")
    validate_s47_partitions(train_rows,dev_rows)
    _assert_s47_fresh(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S47 semantic revision changed")

    args.out.mkdir(parents=True,exist_ok=True)

    s45.SEED=SEED
    reference=s45._train_arm("reference",bundle,manifest,train_rows,dev_rows,args.out)
    treatment=s45._train_arm("treatment",bundle,manifest,train_rows,dev_rows,args.out)

    if reference["runtime_trajectory_sha256"]!=treatment["runtime_trajectory_sha256"]:
        raise RuntimeError("S47 matched native runtime trajectory diverged")
    treatment["gates"]["runtime_trajectory_identity"]=True
    treatment["dev_ready"]=all(treatment["gates"].values())

    runtime,correction,checkpoint_payload=s46._load_treatment_checkpoint(
        bundle,manifest,args.out/"treatment-private-correction-candidate.pt"
    )
    shells,experts,ordinal_diagnostics=_same_checkpoint_shells(runtime,correction,dev_rows)
    legacy=shells["legacy_s45"]
    ordinal=shells["ordinal_s47"]

    if int(legacy["state_view_encodes"])!=2*len(dev_rows):
        raise RuntimeError("S47 legacy shell state-once changed")
    if int(ordinal["state_view_encodes"])!=2*len(dev_rows):
        raise RuntimeError("S47 ordinal shell state-once changed")

    ordinal_gates=_ordinal_gates(treatment,ordinal)
    ordinal_dev_ready=all(ordinal_gates.values())
    outcome=(
        "HIRA_V1_S47_ORDINAL_PAIRWISE_CONSENSUS_DEV_READY"
        if ordinal_dev_ready
        else "HIRA_V1_S47_ORDINAL_PAIRWISE_CONSENSUS_DEV_COMPLETE"
    )

    train_manifest=args.out/"train-manifest.json"
    dev_manifest=args.out/"dev-manifest.json"
    train_manifest.write_text(
        json.dumps([r.to_dict() for r in train_rows],indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    dev_manifest.write_text(
        json.dumps([r.to_dict() for r in dev_rows],indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S47_FRESH_SAME_CHECKPOINT_LEGACY_VS_ORDINAL_PAIRWISE_CONSENSUS",
        "seed":SEED,
        "training_mechanics":{
            "family":"exact_s45_cross_view_consistent_private_correction",
            "checkpoint_selection":"exact_s45_selection_rule_before_ordinal_shell_evaluation",
            "epochs":s35.EPOCHS,
            "batch_size_semantic_cases":s35.BATCH_SIZE,
            "lr":s35.LR,
            "weight_decay":s35.WEIGHT_DECAY,
            "grad_clip":s35.GRAD_CLIP,
            "correction_ce_coefficient":s35.BINDING_COEFFICIENT,
            "correction_cross_view_js_coefficient":s35.INVARIANCE_COEFFICIENT,
            "second_encoder_pass":False,
        },
        "decision_shell":{
            "baseline":"S45_equal_mean_standardized_primary_plus_corrected_relation",
            "treatment":"S47_pairwise_majority_copeland_private_ordinal_tiebreak",
            "lexicographic_base_rule":"2K-1",
            "treatment_trainable_parameters":0,
            "uses_raw_magnitude_after_ranking":False,
            "learned_gate":False,
            "temperature":None,
            "threshold":None,
        },
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "k":4,
            "views_per_option":2,
            "identical_rows_across_training_arms":True,
            "prior_track_exact_rows_used":False,
            "s46_train_dev_rows_used":False,
            "s47_a0_rows_used":False,
        },
        "reference_arm":reference,
        "treatment_training_arm":treatment,
        "selected_treatment_checkpoint":{
            "epoch":int(checkpoint_payload["selected_dev_epoch"]),
            "total_parameter_count":int(checkpoint_payload["total_parameter_count"]),
            "correction_parameter_count":int(checkpoint_payload["correction_parameter_count"]),
        },
        "same_checkpoint":{
            "legacy_s45_shell":legacy,
            "ordinal_s47_shell":ordinal,
            "delta_ordinal_minus_legacy":_shell_delta(legacy,ordinal),
            "expert_metrics":experts,
            "ordinal_diagnostics":ordinal_diagnostics,
            "ordinal_gates":ordinal_gates,
            "ordinal_dev_ready":ordinal_dev_ready,
        },
        "trajectory_invariant":{
            "all_epoch_runtime_state_sha256_equal":True,
            "reference_runtime_trajectory_sha256":reference["runtime_trajectory_sha256"],
            "treatment_runtime_trajectory_sha256":treatment["runtime_trajectory_sha256"],
            "epoch_count":len(reference["history"]),
        },
        "post_dev_tuning_performed":False,
        "second_dev_run_performed":False,
        "sealed_confirm_opened":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
        "train_manifest_sha256":_sha256(train_manifest),
        "dev_manifest_sha256":_sha256(dev_manifest),
    }

    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S47_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
