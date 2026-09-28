from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_s5_semantic_core import build_hira_v1_s5_projection_core
from nmd.v1_triadic_semantic import ParameterFreeTriadicScorer


SCHEMA_VERSION = "hira-v1-s5-a0-identity-v1"
OUTCOME = "HIRA_V1_S5_A0_IDENTITY_READY"


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

    def options(self) -> tuple[LogicalOption, ...]:
        return tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=self.answers[i],
                aliases=(self.aliases[i],),
            )
            for i in range(4)
        )


def cases() -> tuple[Case, ...]:
    return (
        Case("bridge", "Inspection BI-42 records deck material basalt fiber and span class S7.", "What deck material is recorded for BI-42?", "What span class is recorded for BI-42?", ("the answer is basalt fiber", "the answer is class S7", "the answer is carbon steel", "the answer is class T3"), ("basalt fiber is the recorded value", "class S7 is the recorded value", "carbon steel is the recorded value", "class T3 is the recorded value"), 0, 1),
        Case("orchard", "Crop log OL-55 lists cultivar Juniper and irrigation cycle 36 hours.", "Which cultivar is listed in crop log OL-55?", "What irrigation cycle is listed in crop log OL-55?", ("the answer is Juniper", "the answer is 36 hours", "the answer is Meridian", "the answer is 20 hours"), ("Juniper is the requested value", "36 hours is the requested value", "Meridian is the requested value", "20 hours is the requested value"), 0, 1),
        Case("freight", "Freight card FR-63 assigns container type Arctic and route code RQ-17.", "Which container type is assigned on FR-63?", "What route code is assigned on FR-63?", ("the answer is Arctic", "the answer is RQ-17", "the answer is Coastal", "the answer is LM-48"), ("Arctic is the assigned value", "RQ-17 is the assigned value", "Coastal is the assigned value", "LM-48 is the assigned value"), 0, 1),
        Case("reactor", "Reactor sheet RX-28 specifies catalyst cobalt and pressure 14 bar.", "Which catalyst is specified on RX-28?", "What pressure is specified on RX-28?", ("the answer is cobalt", "the answer is 14 bar", "the answer is nickel", "the answer is 9 bar"), ("cobalt is the specified value", "14 bar is the specified value", "nickel is the specified value", "9 bar is the specified value"), 0, 1),
        Case("school", "Course record CR-74 names room Atlas and session length 85 minutes.", "Which room is named in course record CR-74?", "What session length is named in CR-74?", ("the answer is room Atlas", "the answer is 85 minutes", "the answer is room Vega", "the answer is 50 minutes"), ("room Atlas is the recorded value", "85 minutes is the recorded value", "room Vega is the recorded value", "50 minutes is the recorded value"), 0, 1),
        Case("marine", "Buoy record BR-31 identifies sensor type fluorometer and channel C8.", "Which sensor type is identified on BR-31?", "Which channel is identified on BR-31?", ("the answer is fluorometer", "the answer is channel C8", "the answer is sonar", "the answer is channel A2"), ("fluorometer is the requested value", "channel C8 is the requested value", "sonar is the requested value", "channel A2 is the requested value"), 0, 1),
        Case("hotel", "Reservation note RN-92 assigns suite Amber and checkout 11:40.", "Which suite is assigned on RN-92?", "What checkout time is assigned on RN-92?", ("the answer is suite Amber", "the answer is 11:40", "the answer is suite Indigo", "the answer is 09:20"), ("suite Amber is the assigned value", "11:40 is the assigned value", "suite Indigo is the assigned value", "09:20 is the assigned value"), 0, 1),
        Case("energy", "Grid ticket GT-46 lists feeder Quartz and reserve level 18 percent.", "Which feeder is listed on GT-46?", "What reserve level is listed on GT-46?", ("the answer is feeder Quartz", "the answer is 18 percent", "the answer is feeder Cedar", "the answer is 31 percent"), ("feeder Quartz is the recorded value", "18 percent is the recorded value", "feeder Cedar is the recorded value", "31 percent is the recorded value"), 0, 1),
        Case("aviation", "Maintenance log ML-57 specifies fluid Skydrol and inspection interval 240 cycles.", "Which fluid is specified in ML-57?", "What inspection interval is specified in ML-57?", ("the answer is Skydrol", "the answer is 240 cycles", "the answer is glycol", "the answer is 120 cycles"), ("Skydrol is the specified value", "240 cycles is the specified value", "glycol is the specified value", "120 cycles is the specified value"), 0, 1),
        Case("market", "Vendor sheet VS-38 lists stall Orion and permit class P6.", "Which stall is listed on VS-38?", "What permit class is listed on VS-38?", ("the answer is stall Orion", "the answer is class P6", "the answer is stall Sol", "the answer is class K2"), ("stall Orion is the recorded value", "class P6 is the recorded value", "stall Sol is the recorded value", "class K2 is the recorded value"), 0, 1),
        Case("telemetry", "Telemetry packet TP-61 names codec Delta and frame rate 48 fps.", "Which codec is named in TP-61?", "What frame rate is named in TP-61?", ("the answer is codec Delta", "the answer is 48 fps", "the answer is codec Sigma", "the answer is 24 fps"), ("codec Delta is the requested value", "48 fps is the requested value", "codec Sigma is the requested value", "24 fps is the requested value"), 0, 1),
        Case("forest", "Survey FS-24 identifies species larch and plot number 73.", "Which species is identified in FS-24?", "What plot number is identified in FS-24?", ("the answer is larch", "the answer is plot 73", "the answer is spruce", "the answer is plot 41"), ("larch is the identified value", "plot 73 is the identified value", "spruce is the identified value", "plot 41 is the identified value"), 0, 1),
        Case("foundry", "Casting note CN-89 records alloy bronze and mold temperature 680 C.", "Which alloy is recorded in CN-89?", "What mold temperature is recorded in CN-89?", ("the answer is bronze", "the answer is 680 C", "the answer is brass", "the answer is 540 C"), ("bronze is the recorded value", "680 C is the recorded value", "brass is the recorded value", "540 C is the recorded value"), 0, 1),
        Case("biobank", "Sample note SN-67 lists tissue cortex and freezer slot F12.", "Which tissue is listed in SN-67?", "Which freezer slot is listed in SN-67?", ("the answer is cortex", "the answer is slot F12", "the answer is marrow", "the answer is slot B7"), ("cortex is the listed value", "slot F12 is the listed value", "marrow is the listed value", "slot B7 is the listed value"), 0, 1),
        Case("rail", "Rail manifest RM-45 assigns locomotive Echo and siding 22.", "Which locomotive is assigned on RM-45?", "Which siding is assigned on RM-45?", ("the answer is locomotive Echo", "the answer is siding 22", "the answer is locomotive Kilo", "the answer is siding 11"), ("locomotive Echo is the assigned value", "siding 22 is the assigned value", "locomotive Kilo is the assigned value", "siding 11 is the assigned value"), 0, 1),
        Case("cinema", "Projection card PC-76 lists lens anamorphic and audio format Atmos.", "Which lens is listed on PC-76?", "Which audio format is listed on PC-76?", ("the answer is anamorphic", "the answer is Atmos", "the answer is spherical", "the answer is stereo"), ("anamorphic is the listed value", "Atmos is the listed value", "spherical is the listed value", "stereo is the listed value"), 0, 1),
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S5-A0 suite size changed")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder

    runtime = build_hira_v1_s5_projection_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_projection=False,
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S5 projection scorer missing")

    base = ParameterFreeTriadicScorer(d_model=256, d_rel=128)
    base.load_projection_weight(
        scorer.projection.weight.detach().cpu(),
        freeze=True,
    )
    base.eval()

    if scorer.projection_parameter_count != 32768:
        raise RuntimeError("S5 projection parameter count changed")
    if scorer.projection_trainable_parameter_count != 0:
        raise RuntimeError("S5-A0 projection must be frozen")

    before = runtime.state_encode_calls
    exact_logits = 0
    exact_choices = 0
    correct = 0
    both = 0
    order_flips = 0
    max_mass_error = 0.0
    alignment_losses = []
    rows = []

    print("HIRA_V1_S5_A0_EXPOSURE_BEGIN", flush=True)

    for case in suite:
        options = case.options()
        memory = runtime.compile_state(case.state)
        state = memory.content_token_embeddings
        if state is None:
            raise RuntimeError(f"{case.case_id}: missing state tokens")
        state = state.unsqueeze(0)
        state_mask = torch.ones(
            1,
            state.shape[1],
            dtype=torch.bool,
            device=state.device,
        )

        pair = []
        for question, gold_index in (
            (case.question_a, case.gold_a),
            (case.question_b, case.gold_b),
        ):
            schema, _ = runtime.compile_schema(
                primitive="choice",
                question_text=question,
                options=options,
                include_token_artifacts=True,
                use_cache=False,
            )
            q = schema.question_token_embeddings
            qm = schema.question_content_token_mask
            ov = schema.option_view_token_embeddings
            ovtm = schema.option_view_token_mask
            ovm = schema.option_view_mask
            if any(x is None for x in (q, qm, ov, ovtm, ovm)):
                raise RuntimeError(f"{case.case_id}: token artifacts missing")
            if not bool(ovm[:, :2].all()):
                raise RuntimeError(f"{case.case_id}: two semantic views missing")

            kwargs = dict(
                state_tokens=state,
                state_mask=state_mask,
                question_tokens=q.unsqueeze(0),
                question_mask=qm.unsqueeze(0),
                option_view_tokens=ov.unsqueeze(0),
                option_view_token_mask=ovtm.unsqueeze(0),
                option_view_mask=ovm.unsqueeze(0),
            )
            expected = base(**kwargs)[0]
            observed = scorer(**kwargs)[0]
            same_logits = torch.equal(expected, observed)
            exact_logits += int(same_logits)
            if not same_logits:
                raise RuntimeError(f"{case.case_id}: W28 identity changed")

            probs = torch.softmax(observed, dim=-1)
            mass_error = abs(float(probs.sum()) - 1.0)
            max_mass_error = max(max_mass_error, mass_error)
            index = int(probs.argmax())
            base_index = int(torch.softmax(expected, dim=-1).argmax())
            same_choice = index == base_index
            exact_choices += int(same_choice)
            if not same_choice:
                raise RuntimeError(f"{case.case_id}: W28 choice identity changed")

            alignment = scorer.option_view_alignment_loss(
                option_view_tokens=ov.unsqueeze(0),
                option_view_token_mask=ovtm.unsqueeze(0),
                option_view_mask=ovm.unsqueeze(0),
                temperature=0.10,
            )
            alignment_losses.append(float(alignment))

            ok = index == gold_index
            correct += int(ok)
            pair.append(ok)

            reverse_schema, _ = runtime.compile_schema(
                primitive="choice",
                question_text=question,
                options=tuple(reversed(options)),
                include_token_artifacts=True,
                use_cache=False,
            )
            reverse_out = runtime.forward_compiled(
                memory,
                reverse_schema,
                coarse_mode="projection_triadic",
                relation_refinement=False,
            )
            order_flips += int(
                reverse_out.selected_option_id != options[index].option_id
            )

        both += int(pair[0] and pair[1])
        rows.append(
            {
                "case_id": case.case_id,
                "a_correct": pair[0],
                "b_correct": pair[1],
            }
        )

    state_calls = runtime.state_encode_calls - before
    if state_calls != len(suite):
        raise RuntimeError("S5-A0 state-once changed")

    queries = len(suite) * 2
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S5_A0_IDENTITY_LOCALIZATION_ONLY",
        "case_count": len(suite),
        "query_count": queries,
        "language": "en",
        "k": 4,
        "views_per_option": 2,
        "projection_parameter_count": scorer.projection_parameter_count,
        "projection_trainable_parameter_count": scorer.projection_trainable_parameter_count,
        "exact_logit_identity_rate": exact_logits / queries,
        "exact_choice_identity_rate": exact_choices / queries,
        "accuracy": correct / queries,
        "paired_both_correct_rate": both / len(suite),
        "option_order_flip_rate": order_flips / queries,
        "mean_frozen_view_infonce": sum(alignment_losses) / len(alignment_losses),
        "state_encode_calls": state_calls,
        "max_probability_mass_error": max_mass_error,
        "full_k": True,
        "relation_refinement": False,
        "m4_manifest": frozen.manifest.to_dict(),
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out / "per-case.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print("HIRA_V1_S5_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
