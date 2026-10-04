from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import (
    make_native_authority_payload,
    load_native_authority,
    runtime_state_sha256,
    save_native_authority,
    verify_file_sha256,
)
from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import (
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)
from nmd.v1_shared_native_private_readouts import (
    correction_initialization_exact,
    reference_private_logits,
    treatment_private_logits,
)
from hira_v1_s50_train_dev import _materialize_cache


SCHEMA_VERSION="hira-v1-s51-a0-persisted-native-authority-v1"
OUTCOME="HIRA_V1_S51_A0_PERSISTED_NATIVE_AUTHORITY_READY"
SEED=72_001


@dataclass(frozen=True)
class Case:
    case_id:str
    noun:str
    field_a:str
    first:str
    field_b:str
    second:str
    wrong_a:str
    wrong_b:str
    seed:int

    @property
    def state_a(self):
        return f"S51-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Persist audit {self.case_id}: {self.second} belongs to {self.field_b}; {self.first} belongs to {self.field_a}."

    @property
    def qa1(self): return f"For persisted probe {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which persisted entry is tagged {self.field_a} for {self.case_id}?"
    @property
    def qb1(self): return f"For persisted probe {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which persisted entry is tagged {self.field_b} for {self.case_id}?"

    def option_pack(self):
        rows=[
            ("a",self.field_a,self.first),
            ("b",self.field_b,self.second),
            ("x",self.field_a,self.wrong_a),
            ("y",self.field_b,self.wrong_b),
        ]
        random.Random(self.seed).shuffle(rows)
        options=tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=f"for this {self.noun}, {field} is {value}",
                aliases=(f"{value} is the persisted A0 {field} entry for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("PA11","muon phase compass","target","silica aerogel","phase floor","1.3 mrad","plastic slab","5.2 mrad",14101),
        Case("PA22","quantum vortex camera","fluid","He3-B film","vortex blur","0.08 um","water film","0.32 um",14102),
        Case("PA33","spin caloric ruler","strip","Pt/YIG bilayer","thermal floor","4 nK","Cu strip","16 nK",14103),
        Case("PA44","Rydberg recoil scope","ensemble","cold Cs cloud","recoil spread","3 mm/s","hot vapor","12 mm/s",14104),
        Case("PA55","phonon parity clock","mode","edge-localized A0","parity drift","0.06 rad","bulk mode","0.24 rad",14105),
        Case("PA66","topological torque map","lattice","breathing kagome","torque leak","-47 dB","square lattice","-19 dB",14106),
        Case("PA77","nuclear phase lens","sample","enriched Ge crystal","phase noise","0.15 mrad","polymer tile","0.60 mrad",14107),
        Case("PA88","vacuum stress camera","surface","template Au plate","stress floor","0.7 mPa","rough steel","2.8 mPa",14108),
        Case("PB11","magnon chirality clock","guide","low-loss YIG","chirality drift","0.09 rad","Ni guide","0.36 rad",14109),
        Case("PB22","molecular rotation scope","beam","state-selected OCS","linewidth","11 kHz","thermal SO2","44 kHz",14110),
        Case("PB33","quantum charge compass","island","graphene dot","charge floor","0.05 e","metal island","0.20 e",14111),
        Case("PB44","atomic curvature map","species","lattice Sr","curvature floor","2 nrad/mm","thermal K","8 nrad/mm",14112),
        Case("PB55","optical spin lens","interface","SiV cavity","phase loss","0.07 rad","free-space link","0.28 rad",14113),
        Case("PB66","superfluid pressure clock","fluid","He4 film","pressure drift","5 nPa","oil film","20 nPa",14114),
        Case("PB77","neutron recoil compass","target","perfect Si","recoil blur","6 eV","plastic tile","24 eV",14115),
        Case("PB88","ferroelectric phase radar","crystal","strained BaTiO3","domain jitter","0.9 nm","ceramic slab","3.6 nm",14116),
    )


def _rows():
    out=[]
    for case in cases():
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s51-{case.case_id}",
            split="train",
            domain="s51_a0_only",
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
        ))
    return out


def _normal_cache_assert(pairs):
    digest_parts=[]
    tensor_count=0
    for canonical,paraphrase,_n in pairs:
        for evidence in (canonical,paraphrase):
            for tensor in evidence.tensors():
                tensor_count+=1
                if tensor.requires_grad:
                    raise RuntimeError("S51-A0 cache tensor requires grad")
                if torch.is_inference(tensor):
                    raise RuntimeError("S51-A0 cache retained inference tensor flag")
                if not tensor.is_contiguous():
                    raise RuntimeError("S51-A0 cache tensor not contiguous")
            digest_parts.append(evidence.digest())
    return tensor_count,digest_parts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    rows=_rows()
    if len(rows)!=16:
        raise RuntimeError("S51-A0 case count changed")

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)

    frozen=load_hira_v0_m4_bundle(bundle)
    runtime=build_hira_v1_s17_norm_balanced_core(
        frozen.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,
        train_projection=True,
    )
    del frozen
    runtime.clear_schema_cache()
    enforce_s17_eval(runtime)

    original_hash=runtime_state_sha256(runtime)
    payload=make_native_authority_payload(
        runtime,
        seed=SEED,
        epoch=24,
        semantic_revision=str(manifest["semantic_revision"]),
        initialization_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_manifest_sha256="a"*64,
    )

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/"a0-native-authority.pt"
    file_sha=save_native_authority(checkpoint,payload)

    loaded_a,payload_a=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=checkpoint,
        expected_file_sha256=file_sha,
        expected_seed=SEED,
    )
    loaded_b,payload_b=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=checkpoint,
        expected_file_sha256=file_sha,
        expected_seed=SEED,
    )

    loaded_hash_a=runtime_state_sha256(loaded_a)
    loaded_hash_b=runtime_state_sha256(loaded_b)
    if not (original_hash==loaded_hash_a==loaded_hash_b==payload["runtime_state_sha256"]):
        raise RuntimeError("S51-A0 authority save/load runtime hash changed")
    if payload_a["native_tensor_digest"]!=payload_b["native_tensor_digest"]:
        raise RuntimeError("S51-A0 load-order logical digest changed")
    if any(p.requires_grad for p in loaded_a.parameters()):
        raise RuntimeError("S51-A0 loaded native remained trainable")
    if sum(p.numel() for p in loaded_a.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S51-A0 loaded native optimizer surface nonzero")

    # File-level tamper rejection.
    tampered=out/"a0-native-authority-tampered.pt"
    shutil.copy2(checkpoint,tampered)
    data=bytearray(tampered.read_bytes())
    data[len(data)//2]^=0x01
    tampered.write_bytes(bytes(data))
    tamper_rejected=False
    try:
        verify_file_sha256(tampered,file_sha)
    except RuntimeError:
        tamper_rejected=True
    if not tamper_rejected:
        raise RuntimeError("S51-A0 tampered authority was accepted")

    cache_pairs,cache_digest=_materialize_cache(loaded_a,rows)
    tensor_count,cache_part_digests=_normal_cache_assert(cache_pairs)

    reference=PrivateCorrectionRepresentationFork(train_correction=True)
    treatment=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    if not correction_initialization_exact(reference,treatment):
        raise RuntimeError("S51-A0 correction initialization differs")
    if reference.correction_parameter_count!=114688:
        raise RuntimeError("S51-A0 reference correction capacity changed")
    if treatment.correction_parameter_count!=114688:
        raise RuntimeError("S51-A0 treatment correction capacity changed")
    if treatment.identity_parameter_count!=0:
        raise RuntimeError("S51-A0 identity gained parameters")

    # Branch-order exact replay using the same persisted-native cache.
    canonical=cache_pairs[0][0]
    r1=reference_private_logits(reference,canonical)
    t1,i1=treatment_private_logits(treatment,canonical)
    t2,i2=treatment_private_logits(treatment,canonical)
    r2=reference_private_logits(reference,canonical)
    replay_error=max(
        float((r1-r2).abs().max()),
        float((t1-t2).abs().max()),
        float((i1-i2).abs().max()),
    )
    if replay_error!=0.0:
        raise RuntimeError("S51-A0 branch-order replay changed")

    phase_a=Path("scripts/hira_v1_s51_native_authority.py").read_text(encoding="utf-8")
    phase_b=Path("scripts/hira_v1_s51_private_court.py").read_text(encoding="utf-8")
    phase_a_dev_absent=("generate_s51_dev_cases" not in phase_a)
    phase_a_private_absent=(
        "PrivateCorrectionRepresentationFork" not in phase_a
        and "QueryFreeIdentityPrivateCorrectionFork" not in phase_a
    )
    phase_b_native_train_absent=(
        "_train_shared_native" not in phase_b
        and "norm_balanced_gradient_update" not in phase_b
        and "build_hira_v1_s17_norm_balanced_core" not in phase_b
        and "load_hira_v0_m4_bundle" not in phase_b
    )
    if not phase_a_dev_absent or not phase_a_private_absent or not phase_b_native_train_absent:
        raise RuntimeError("S51-A0 phase isolation failed")

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S51_A0_PERSIST_LOAD_CACHE_ISOLATION_ONLY",
        "seed":SEED,
        "semantic_case_count":len(rows),
        "original_runtime_state_sha256":original_hash,
        "loaded_runtime_state_sha256":loaded_hash_a,
        "second_load_runtime_state_sha256":loaded_hash_b,
        "logical_native_tensor_digest":payload["native_tensor_digest"],
        "checkpoint_file_sha256":file_sha,
        "tamper_rejected":tamper_rejected,
        "loaded_native_trainable_parameter_count":0,
        "loaded_native_optimizer_parameter_count":0,
        "cache_digest":cache_digest,
        "cache_tensor_count":tensor_count,
        "cache_part_digest_count":len(cache_part_digests),
        "cache_tensors_require_grad":False,
        "cache_tensors_are_inference":False,
        "branch_order_replay_max_abs_error":replay_error,
        "correction_initialization_bit_identical":True,
        "reference_correction_parameter_count":reference.correction_parameter_count,
        "treatment_correction_parameter_count":treatment.correction_parameter_count,
        "identity_parameter_count":treatment.identity_parameter_count,
        "phase_a_dev_generation_absent":phase_a_dev_absent,
        "phase_a_private_construction_absent":phase_a_private_absent,
        "phase_b_native_training_call_absent":phase_b_native_train_absent,
        "second_encoder_pass":False,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(
        "HIRA_V1_S51_A0_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
