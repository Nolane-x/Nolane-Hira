from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import iter_a13_lora_modules
from nmd.v1_s3_semantic_core import build_hira_v1_s3_parameter_free_core
from nmd.v1_s7_semantic_core import (
    HIRA_V1_S7_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S7_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s7_coadapt_core,
)


SCHEMA_VERSION = "hira-v1-s7-a0-identity-v1"
OUTCOME = "HIRA_V1_S7_A0_IDENTITY_READY"


@dataclass(frozen=True)
class Case:
    case_id: str
    state: str
    question_a: str
    question_b: str
    answers: tuple[str, str, str, str]
    aliases: tuple[str, str, str, str]
    gold_a: int
    gold_b: int

    def options(self):
        return tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=self.answers[i],
                aliases=(self.aliases[i],),
            )
            for i in range(4)
        )


def _case(cid, state, qa, qb, a, b, x, y):
    values = (a, b, x, y)
    return Case(
        cid,
        state,
        qa,
        qb,
        tuple(f"the requested answer is {v}" for v in values),
        tuple(f"{v} is the corresponding value" for v in values),
        0,
        1,
    )


def cases() -> tuple[Case, ...]:
    return (
        _case("metrology", "Gauge ticket MG-14 records probe sapphire and tolerance 0.08 mm.", "Which probe is recorded on MG-14?", "What tolerance is recorded on MG-14?", "sapphire", "0.08 mm", "carbide", "0.21 mm"),
        _case("apiary", "Hive log AH-27 lists queen line Carniolan and frame count 18.", "Which queen line is listed in AH-27?", "What frame count is listed in AH-27?", "Carniolan", "18 frames", "Buckfast", "11 frames"),
        _case("cablecar", "Cable-car sheet CS-35 assigns cabin Orion and departure 07:42.", "Which cabin is assigned on CS-35?", "What departure is assigned on CS-35?", "cabin Orion", "07:42", "cabin Vega", "09:18"),
        _case("chromatography", "Chromatography run CH-48 uses column C18 and flow 1.2 mL/min.", "Which column is used in CH-48?", "What flow is used in CH-48?", "column C18", "1.2 mL/min", "column C8", "0.6 mL/min"),
        _case("dam", "Dam inspection DI-56 names spillway East and gate opening 34 percent.", "Which spillway is named in DI-56?", "What gate opening is named in DI-56?", "spillway East", "34 percent", "spillway West", "19 percent"),
        _case("filmstock", "Film archive FA-62 records stock acetate and reel length 820 meters.", "Which stock is recorded in FA-62?", "What reel length is recorded in FA-62?", "acetate", "820 meters", "polyester", "510 meters"),
        _case("greenhouse2", "Nursery card NC-73 lists substrate coco coir and irrigation 06:20.", "Which substrate is listed in NC-73?", "What irrigation time is listed in NC-73?", "coco coir", "06:20", "rockwool", "08:45"),
        _case("hydraulics", "Hydraulic test HT-81 uses fluid ester and pressure 185 bar.", "Which fluid is used in HT-81?", "What pressure is used in HT-81?", "ester", "185 bar", "glycol", "92 bar"),
        _case("icecore", "Ice-core record IC-24 identifies layer volcanic ash and depth 312 meters.", "Which layer is identified in IC-24?", "What depth is identified in IC-24?", "volcanic ash", "312 meters", "dust band", "146 meters"),
        _case("jewelry", "Jewelry assay JA-39 names alloy electrum and purity 91 percent.", "Which alloy is named in JA-39?", "What purity is named in JA-39?", "electrum", "91 percent", "sterling silver", "76 percent"),
        _case("kilnbrick", "Refractory batch RB-45 uses brick magnesia and rating 1650 C.", "Which brick is used in RB-45?", "What rating is listed in RB-45?", "magnesia", "1650 C", "fireclay", "1280 C"),
        _case("lighthouse", "Lighthouse service LS-58 assigns lens Fresnel and flash period 12 seconds.", "Which lens is assigned in LS-58?", "What flash period is assigned in LS-58?", "Fresnel", "12 seconds", "catadioptric", "7 seconds"),
        _case("milling", "Milling sheet MS-66 specifies cutter carbide and feed 420 mm/min.", "Which cutter is specified in MS-66?", "What feed is specified in MS-66?", "carbide", "420 mm/min", "ceramic", "260 mm/min"),
        _case("nurseryfish", "Hatchery card HC-79 lists species arctic char and tank T14.", "Which species is listed in HC-79?", "Which tank is listed in HC-79?", "arctic char", "tank T14", "rainbow trout", "tank T07"),
        _case("planetarium", "Planetarium cue PC-87 names constellation Lyra and start 19:36.", "Which constellation is named in PC-87?", "What start time is named in PC-87?", "Lyra", "19:36", "Cygnus", "20:18"),
        _case("recycling", "Recycling batch RC-93 identifies polymer PETG and bale mass 146 kg.", "Which polymer is identified in RC-93?", "What bale mass is identified in RC-93?", "PETG", "146 kg", "HDPE", "88 kg"),
    )


def _bank(suite):
    texts = []
    for c in suite:
        texts.extend((c.state, c.question_a, c.question_b))
        texts.extend(c.answers)
        texts.extend(c.aliases)
    return tuple(texts)


@torch.inference_mode()
def _decisions(runtime, suite, mode):
    rows = {}
    before = runtime.state_encode_calls
    correct = both = flips = 0
    max_mass = 0.0

    for case in suite:
        options = case.options()
        memory = runtime.compile_state(case.state)
        pair = []
        for label, question, gold in (
            ("a", case.question_a, case.gold_a),
            ("b", case.question_b, case.gold_b),
        ):
            schema, _ = runtime.compile_schema(
                primitive="choice",
                question_text=question,
                options=options,
                include_token_artifacts=True,
                use_cache=False,
            )
            out = runtime.forward_compiled(
                memory,
                schema,
                coarse_mode=mode,
                relation_refinement=False,
            )
            idx = int(out.probabilities.argmax())
            ok = idx == gold
            correct += int(ok)
            pair.append(ok)
            max_mass = max(max_mass, abs(float(out.probabilities.sum()) - 1.0))
            rows[(case.case_id, label)] = (
                out.logits.detach().cpu().clone(),
                out.selected_option_id,
                ok,
            )
            reverse_schema, _ = runtime.compile_schema(
                primitive="choice",
                question_text=question,
                options=tuple(reversed(options)),
                include_token_artifacts=True,
                use_cache=False,
            )
            reverse = runtime.forward_compiled(
                memory,
                reverse_schema,
                coarse_mode=mode,
                relation_refinement=False,
            )
            flips += int(reverse.selected_option_id != out.selected_option_id)
            if int(out.hira.candidate_budget.item()) != 4:
                raise RuntimeError("S7-A0 full-K changed")
            if not torch.equal(
                out.hira.relation_delta,
                torch.zeros_like(out.hira.relation_delta),
            ):
                raise RuntimeError("S7-A0 relation delta changed")
        both += int(pair[0] and pair[1])

    return {
        "rows": rows,
        "accuracy": correct / (len(suite) * 2),
        "paired_both_correct_rate": both / len(suite),
        "option_order_flip_rate": flips / (len(suite) * 2),
        "state_encode_calls": runtime.state_encode_calls - before,
        "max_probability_mass_error": max_mass,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S7-A0 suite size changed")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder
    encoder.eval()

    bank = _bank(suite)
    before_batch = encoder.encode_texts(bank)
    token_before = before_batch.token_embeddings.detach().cpu().clone()
    pooled_before = before_batch.pooled_embeddings.detach().cpu().clone()

    baseline = build_hira_v1_s3_parameter_free_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
    )
    base = _decisions(baseline, suite, "parameter_free_triadic")

    runtime = build_hira_v1_s7_coadapt_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,
        train_projection=False,
    )
    runtime.encoder.eval()

    after_batch = runtime.encoder.encode_texts(bank)
    token_identity = torch.equal(
        after_batch.token_embeddings.detach().cpu(),
        token_before,
    )
    pooled_identity = torch.equal(
        after_batch.pooled_embeddings.detach().cpu(),
        pooled_before,
    )
    if not token_identity or not pooled_identity:
        raise RuntimeError("S7-A0 zero-LoRA A13 identity changed")

    observed = _decisions(runtime, suite, "projection_triadic")

    exact_logits = exact_choices = 0
    per_case = []
    for case in suite:
        row = {"case_id": case.case_id}
        for label in ("a", "b"):
            b_logits, b_choice, _ = base["rows"][(case.case_id, label)]
            o_logits, o_choice, ok = observed["rows"][(case.case_id, label)]
            same_logits = torch.equal(b_logits, o_logits)
            same_choice = b_choice == o_choice
            if not same_logits or not same_choice:
                raise RuntimeError(f"{case.case_id}/{label}: S7 identity changed")
            exact_logits += int(same_logits)
            exact_choices += int(same_choice)
            row[f"{label}_correct"] = bool(ok)
        per_case.append(row)

    modules = iter_a13_lora_modules(runtime.encoder)
    lora_params = sum(
        p.numel() for m in modules for p in (m.lora_a, m.lora_b)
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S7 projection scorer missing")

    total_trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    original_a13_trainable = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )

    if lora_params != 16384:
        raise RuntimeError("S7-A0 LoRA parameter count changed")
    if scorer.projection_parameter_count != HIRA_V1_S7_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S7-A0 projection count changed")
    if lora_params + scorer.projection_parameter_count != HIRA_V1_S7_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S7-A0 total candidate capacity changed")
    if total_trainable != 0 or original_a13_trainable != 0:
        raise RuntimeError("S7-A0 must be completely frozen")
    if observed["state_encode_calls"] != len(suite):
        raise RuntimeError("S7-A0 state-once changed")
    if observed["option_order_flip_rate"] != 0.0:
        raise RuntimeError("S7-A0 option-order invariance changed")
    if observed["max_probability_mass_error"] > 1e-6:
        raise RuntimeError("S7-A0 probability mass changed")

    queries = len(suite) * 2
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S7_A0_IDENTITY_LOCALIZATION_ONLY",
        "case_count": len(suite),
        "query_count": queries,
        "language": "en",
        "k": 4,
        "views_per_option": 2,
        "a13_token_output_identity": token_identity,
        "a13_pooled_output_identity": pooled_identity,
        "exact_logit_identity_rate": exact_logits / queries,
        "exact_choice_identity_rate": exact_choices / queries,
        "accuracy": observed["accuracy"],
        "paired_both_correct_rate": observed["paired_both_correct_rate"],
        "option_order_flip_rate": observed["option_order_flip_rate"],
        "state_encode_calls": observed["state_encode_calls"],
        "max_probability_mass_error": observed["max_probability_mass_error"],
        "lora_parameter_count": lora_params,
        "projection_parameter_count": scorer.projection_parameter_count,
        "candidate_parameter_count": lora_params + scorer.projection_parameter_count,
        "runtime_trainable_parameter_count": total_trainable,
        "original_a13_trainable_parameter_count": original_a13_trainable,
        "full_k": True,
        "relation_refinement": False,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out / "per-case.jsonl").open("w", encoding="utf-8") as h:
        for row in per_case:
            h.write(json.dumps(row, sort_keys=True) + "\n")

    print("HIRA_V1_S7_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
