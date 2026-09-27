from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from nmd.compositional_projection_authority import all_w28_text_atoms
from nmd.compositional_projection_reference import evaluate_reference_panel
from nmd.hira_v0_authority import all_w29_text_atoms
from nmd.hira_v0_eval import reference_domain_pass_w29
from nmd.typed_competitive_cache import file_sha256
from nmd.w30_transfer_authority import all_w30_text_atoms
from nmd.w31_transfer_authority import all_w31_text_atoms
from nmd.w32_transfer_authority import (
    PARTITION_DOMAINS,
    REFERENCE_HYPOTHESES,
    all_w32_text_atoms,
    generate_w32_partition,
)

REFERENCE_PANEL = {
    "deberta_nli": {
        "repo": "cross-encoder/nli-deberta-v3-base",
        "revision": "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
        "weight_sha256": "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa",
    },
    "roberta_nli": {
        "repo": "cross-encoder/nli-roberta-base",
        "revision": "1be0567456f0543475805e758725f151f283705a",
        "weight_sha256": "efc90996d2ed80123c26c9091c91385ffddc6d2fd0b2bacf3187fbd6c5b87953",
    },
}
REFERENCE_NAMES = tuple(REFERENCE_PANEL)
REFERENCE_TASKS = ("F0", "F1", "U", "C", "F2")


def _entailment_index(config) -> int:
    for key, value in (getattr(config, "id2label", {}) or {}).items():
        if "entail" in str(value).lower():
            return int(key)
    for key, value in (getattr(config, "label2id", {}) or {}).items():
        if "entail" in str(key).lower():
            return int(value)
    raise RuntimeError("W32 reference config has no entailment label")


def _load_reference(name: str, spec_data: dict[str, str]):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = Path(
        snapshot_download(
            repo_id=spec_data["repo"],
            revision=spec_data["revision"],
        )
    )
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != spec_data["weight_sha256"]:
        raise RuntimeError(f"W32 {name} reference SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, actual, _entailment_index(model.config)


@torch.inference_mode()
def _pair_scores(model, tokenizer, pairs, entailment_index: int):
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(pairs), 64):
        batch = pairs[start : start + 64]
        encoded = tokenizer(
            [left for left, _ in batch],
            [right for _, right in batch],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }
        logits = model(**encoded, return_dict=True).logits
        probs = torch.softmax(
            logits.float(),
            dim=-1,
        )[:, entailment_index]
        values.extend(float(value) for value in probs.cpu())
    return values


def _reference_scores(rows, model, tokenizer, entailment_index: int):
    lookup = {
        row.case_id: {
            task: [0.0, 0.0]
            for task in REFERENCE_TASKS
        }
        for row in rows
    }
    requests = []
    for row in rows:
        for task in REFERENCE_TASKS:
            for value, hypothesis in enumerate(REFERENCE_HYPOTHESES[task]):
                requests.append(
                    (
                        row.case_id,
                        task,
                        value,
                        row.state_text,
                        hypothesis,
                    )
                )

    scores = _pair_scores(
        model,
        tokenizer,
        [
            (state_text, hypothesis)
            for _, _, _, state_text, hypothesis in requests
        ],
        entailment_index,
    )
    for request, score in zip(requests, scores):
        case_id, task, value, _, _ = request
        lookup[case_id][task][value] = score
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prior = (
        set(all_w28_text_atoms())
        | set(all_w29_text_atoms())
        | set(all_w30_text_atoms())
        | set(all_w31_text_atoms())
    )
    overlap = sorted(all_w32_text_atoms() & prior)
    if overlap:
        raise RuntimeError(
            f"W32 exact text overlaps exposed authorities: {overlap}"
        )

    rows = generate_w32_partition("qualification")
    model_scores = {}
    hashes = {}
    entailment_indices = {}

    for name in REFERENCE_NAMES:
        model, tokenizer, actual, entailment_index = _load_reference(
            name,
            REFERENCE_PANEL[name],
        )
        model_scores[name] = _reference_scores(
            rows,
            model,
            tokenizer,
            entailment_index,
        )
        hashes[name] = actual
        entailment_indices[name] = entailment_index
        del model
        gc.collect()

    panel = evaluate_reference_panel(
        [row.__dict__ for row in rows],
        model_scores,
    )
    per_domain = {
        domain: reference_domain_pass_w29(panel, domain)
        for domain in PARTITION_DOMAINS["qualification"]
    }
    passed = all(per_domain.values())
    result = {
        "schema_version": "r8-w32-reference-qualification-v1",
        "status": "PASS",
        "outcome": (
            "W32_REFERENCE_QUALIFIED"
            if passed
            else "W32_REFERENCE_QUALIFICATION_FAIL"
        ),
        "domains": list(PARTITION_DOMAINS["qualification"]),
        "case_count": len(rows),
        "per_domain": per_domain,
        "reference": panel,
        "reference_weight_sha256": hashes,
        "reference_entailment_label_index": entailment_indices,
        "hira_candidate_evaluated": False,
        "a13_loaded": False,
        "w31_rows_used": False,
        "w30_rows_used": False,
        "w29_rows_used": False,
        "older_authority_rows_used": False,
        "exact_text_overlap": [],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "qualification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "W32_QUALIFICATION="
        + json.dumps(
            {
                "outcome": result["outcome"],
                "per_domain": per_domain,
                "case_count": len(rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
