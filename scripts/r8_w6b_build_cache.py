from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd import typed_competitive_authority as w6b_authority
from nmd.typed_competitive_authority import (
    CONFIRM_SEED,
    DEV_SEED,
    TRAIN_SEED,
    all_w6b_values,
    generate_w6b_authority,
)
from nmd.typed_competitive_cache import (
    compile_w6b_cache,
    file_sha256,
    save_w6b_cache,
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

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
    model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    model.eval()

    values = all_w6b_values()
    train_dev_values = set().union(
        w6b_authority.TRAIN_EQUIPMENT,
        w6b_authority.TRAIN_ZONES,
        w6b_authority.TRAIN_ANOMALIES,
        w6b_authority.TRAIN_CHANNELS,
    )
    reserved_confirm_values = set().union(
        w6b_authority.CONFIRM_EQUIPMENT,
        w6b_authority.CONFIRM_ZONES,
        w6b_authority.CONFIRM_ANOMALIES,
        w6b_authority.CONFIRM_CHANNELS,
    )
    template_ids = (
        *w6b_authority.TRAIN_TEMPLATES,
        *w6b_authority.DEV_TEMPLATES,
        *w6b_authority.CONFIRM_TEMPLATES,
    )
    overlap = sorted(values & prior_w5_values())
    if overlap:
        raise RuntimeError(
            f"W6b value vocabulary overlaps prior W5 authorities: {overlap}"
        )

    train_cases = generate_w6b_authority("train")
    dev_cases = generate_w6b_authority("dev")
    train_cache = compile_w6b_cache(
        model,
        train_cases,
        expected_split="train",
    )
    dev_cache = compile_w6b_cache(
        model,
        dev_cases,
        expected_split="dev",
    )

    train_path = save_w6b_cache(
        train_cache,
        args.out / "train.pt",
    )
    dev_path = save_w6b_cache(
        dev_cache,
        args.out / "dev.pt",
    )
    receipt = {
        "schema_version": "r8-w6b-cache-receipt-v1",
        "status": "PASS",
        "scope": "fresh W6b TRAIN+DEV only; CONFIRM sealed",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
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
        "train_template_ids": list(w6b_authority.TRAIN_TEMPLATES),
        "dev_template_ids": list(w6b_authority.DEV_TEMPLATES),
        "reserved_confirm_template_ids": list(
            w6b_authority.CONFIRM_TEMPLATES
        ),
        "template_ids_sha256": list_hash(template_ids),
        "prior_w5_value_overlap": [],
        "state_encodes_per_case_train": train_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "state_encodes_per_case_dev": dev_cache[
            "metadata"
        ]["state_encode_calls_per_case"],
        "train_cache_sha256": file_sha256(train_path),
        "dev_cache_sha256": file_sha256(dev_path),
        "confirm_exposed": False,
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
