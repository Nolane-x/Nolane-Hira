from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd import typed_reliability_authority as w6c_authority
from nmd.typed_competitive_authority import all_w6b_values
from nmd.typed_reliability_authority import (
    CONFIRM_SEED,
    DEV_SEED,
    TRAIN_SEED,
    all_w6c_values,
    generate_w6c_authority,
)
from nmd.typed_reliability_cache import (
    W6B_HIRA_SHA256,
    W6B_SCORER_SHA256,
    compile_w6c_logit_cache,
    file_sha256,
    save_w6c_logit_cache,
)


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256


def list_hash(values) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return sha256(payload).hexdigest()


def prior_w5_values() -> set[str]:
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_cross_candidate_binding import all_w5i_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab
    from nmd import semantic_alignment_probes as w5c
    from nmd import semantic_capacity_control as w5e
    from nmd import semantic_encoder_adaptation as w5d
    from nmd import semantic_routing_curriculum as w5a
    from nmd import semantic_token_curriculum as w5b

    prior = (
        set(all_w5f_vocab())
        | set(all_w5g_vocab())
        | set(all_w5h_vocab())
        | set(all_w5i_vocab())
    )
    for module in (w5a, w5b, w5c, w5d, w5e):
        for name, value in vars(module).items():
            if not (
                name.startswith("TRAIN_")
                or name.startswith("CONFIRM_")
            ):
                continue
            if isinstance(value, tuple) and all(
                isinstance(item, str) for item in value
            ):
                prior.update(value)
    return prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w6b-hira", type=Path, required=True)
    parser.add_argument("--w6b-scorer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.w6b_hira) != W6B_HIRA_SHA256:
        raise RuntimeError("W6c base HIRA SHA mismatch")
    if file_sha256(args.w6b_scorer) != W6B_SCORER_SHA256:
        raise RuntimeError("W6c base competitive scorer SHA mismatch")

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=A13_MODEL,
            revision=A13_REVISION,
        )
    )
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=MAX_LENGTH,
    )
    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(
            args.w6b_hira,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    scorer = CompetitiveCoarseScorer(
        d_model=256,
        d_rel=128,
    )
    scorer.load_state_dict(
        __import__("torch").load(
            args.w6b_scorer,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)

    model = NolaneHira(
        encoder,
        hira,
        coarse_scorer=scorer,
    )
    model.eval()

    values = all_w6c_values()
    prior = set(all_w6b_values()) | prior_w5_values()
    overlap = sorted(values & prior)
    if overlap:
        raise RuntimeError(
            f"W6c value vocabulary overlaps prior W6b/W5 authorities: {overlap}"
        )

    train_dev_values = set().union(
        w6c_authority.TRAIN_COMPONENTS,
        w6c_authority.TRAIN_ZONES,
        w6c_authority.TRAIN_ANOMALIES,
        w6c_authority.TRAIN_CHANNELS,
    )
    reserved_confirm_values = set().union(
        w6c_authority.CONFIRM_COMPONENTS,
        w6c_authority.CONFIRM_ZONES,
        w6c_authority.CONFIRM_ANOMALIES,
        w6c_authority.CONFIRM_CHANNELS,
    )
    if train_dev_values & reserved_confirm_values:
        raise RuntimeError(
            "W6c reserved CONFIRM value vocabulary is not disjoint"
        )
    template_ids = (
        *w6c_authority.TRAIN_TEMPLATES,
        *w6c_authority.DEV_TEMPLATES,
        *w6c_authority.CONFIRM_TEMPLATES,
    )

    train_cases = generate_w6c_authority("train")
    dev_cases = generate_w6c_authority("dev")
    train_cache = compile_w6c_logit_cache(
        model,
        train_cases,
        expected_split="train",
    )
    dev_cache = compile_w6c_logit_cache(
        model,
        dev_cases,
        expected_split="dev",
    )

    train_path = save_w6c_logit_cache(
        train_cache,
        args.out / "train.pt",
    )
    dev_path = save_w6c_logit_cache(
        dev_cache,
        args.out / "dev.pt",
    )
    receipt = {
        "schema_version": "r8-w6c-cache-receipt-v1",
        "status": "PASS",
        "scope": "fresh W6c TRAIN+DEV raw production logits; CONFIRM sealed",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "base_w6b_hira_sha256": W6B_HIRA_SHA256,
        "base_w6b_scorer_sha256": W6B_SCORER_SHA256,
        "train_seed": TRAIN_SEED,
        "dev_seed": DEV_SEED,
        "reserved_confirm_seed": CONFIRM_SEED,
        "train_case_count": len(train_cases),
        "dev_case_count": len(dev_cases),
        "train_decision_count": 5 * len(train_cases),
        "dev_decision_count": 5 * len(dev_cases),
        "train_case_id_sha256": train_cache["metadata"]["case_id_sha256"],
        "dev_case_id_sha256": dev_cache["metadata"]["case_id_sha256"],
        "value_lexicon_sha256": list_hash(values),
        "value_count": len(values),
        "train_dev_value_lexicon_sha256": list_hash(train_dev_values),
        "reserved_confirm_value_lexicon_sha256": list_hash(
            reserved_confirm_values
        ),
        "train_template_ids": list(w6c_authority.TRAIN_TEMPLATES),
        "dev_template_ids": list(w6c_authority.DEV_TEMPLATES),
        "reserved_confirm_template_ids": list(
            w6c_authority.CONFIRM_TEMPLATES
        ),
        "template_ids_sha256": list_hash(template_ids),
        "prior_w6b_w5_value_overlap": [],
        "joint_strata_balanced": True,
        "state_encodes_per_case_train": train_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "state_encodes_per_case_dev": dev_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "train_cache_sha256": file_sha256(train_path),
        "dev_cache_sha256": file_sha256(dev_path),
        "cached_production_logits": True,
        "hira_frozen": True,
        "competitive_scorer_frozen": True,
        "confirm_exposed": False,
        "w6b_confirm_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
