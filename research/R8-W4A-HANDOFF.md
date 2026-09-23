# R8-W4a handoff — held-out Banking77 full-K direct authority

Status: **COMPLETE; authoritative held-out Banking77 one-shot executed and preserved as a negative result.**

## Frozen purpose

Test HIRA's high-cardinality architecture directly against the Laya application target without any Banking77 task training, calibration, checkpoint selection or retrieval.

Primary headline:
- `laya.app.banking77_full`;
- target 0.492;
- higher is better.

## Frozen model

Selected W3a head:
- candidate `r15-balanced-lr1e3`;
- head SHA-256 `2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c`;
- 422,159 HIRA-head parameters;
- W3a run `35748778854`;
- artifact `10706135649`;
- artifact digest `sha256:8a2efd31a6ccea3b987957d328c4b73267ab00ff9024c469bcb7f3cd578d5d92`.

The head is frozen before Banking77 exposure.

## Frozen encoder

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- state segment size 32;
- encoder frozen.

## Frozen Banking77 source/protocol

- dataset `mteb/banking77`;
- revision `18072d2685ea682290f7b8924d94c62acc19c0b2`;
- split `test`;
- exact Laya application source: `bench_apps.py` blob `c0b255ba89b2ca5ac6674638cfe2ab9f8078ab7b`;
- first 400 examples;
- label space = sorted unique `label_text` over the frozen test authority;
- expected 77 labels;
- semantic options = `label_text.replace("_", " ")`;
- opaque routing IDs = `intent-000..intent-076`;
- question = `Which banking intent does \`message\` express?`;
- state = canonical JSON `{"message": row["text"]}`.

No Banking77 train split or labeled retrieval is allowed.

## Frozen execution

- one schema compile for all 400 examples;
- one state encode per example;
- forced budget 255, clamped to K=77;
- adaptive budget false;
- full-K only;
- no label-dependent preprocessing;
- no candidate pruning.

## Metrics

Primary:
- accuracy.

Diagnostics:
- macro F1;
- hard multiclass Brier;
- NLL;
- raw 15-bin ECE;
- mean confidence;
- AURC;
- descriptive CI CPU p50/p95 latency;
- probability-mass max error;
- state encode calls;
- candidate budget;
- tail mass.

## Reporting discipline

W4a may populate only:
- `laya.app.banking77_full`.

It must not populate any Jev Banking77 cells.

The final workflow also builds a cumulative campaign scorecard by preserving the eight frozen W3 typed values and adding exactly this one W4a cell.

Expected populated campaign cells after W4a:
- 8 frozen Laya typed cells from W3;
- 1 Banking77 application cell;
- 44 remaining cells MISSING.

## One-shot discipline

Implementation/unit tests run before final exposure.

The Banking77 final workflow triggers only when:
- branch = `exec/r8-w4a-banking77-direct`;
- marker path = `research/R8-W4A-BANKING77-FINAL-AUTHORITY.json`.

The marker must exactly match `EXPECTED_MARKER` in `src/nmd/banking77_direct.py`.

The execution workflow verifies:
1. marker;
2. selected artifact metadata/digest;
3. exact selected head SHA;
4. W3a selection receipt;
5. evaluator compile.

Only then does it print:
`R8_W4A_BANKING77_EXPOSURE_BEGIN`

and request the frozen Banking77 test source.

If the one-shot result is weak, preserve it unchanged and continue to W4b K=128/K=255 synthetic stress. Do not tune on exposed first-400 rows.


## Authoritative W4a result

Pure pre-exposure technical failure:
- run `35821668904`;
- exposure boundary not reached;
- Banking77 not requested;
- no scientific result.

Valid retry under the frozen issue #54 rule:
- workflow run `35822128041`;
- head `6e27b724196bed3cf82976610967f1a7a1122fb7`;
- artifact `10733603675`;
- artifact digest `sha256:6bf0aabeabaceda1bdaa570414d47077a647e717d06b6359b0e56182a5bc72ac`.

The explicit boundary `R8_W4A_BANKING77_EXPOSURE_BEGIN` was reached only in the valid authority run.

### Exact held-out outcome

Frozen source/protocol:
- 400 examples;
- 77 labels;
- full-K all 77 options;
- no Banking77 task training/retrieval/calibration;
- one state encode per case;
- forced budget 255 -> K=77;
- adaptive budget false.

Metrics:
- accuracy: **0.000000 = 0/400**;
- macro F1: **0.000000**;
- hard Brier: **0.9904985925**;
- NLL: **4.3209583304**;
- raw ECE: **0.0523322652**;
- mean confidence: **0.0523322652**;
- AURC: **1.000000**;
- CI-hosted CPU p50: **9.321545 ms**;
- CI-hosted CPU p95: **10.563081 ms**;
- probability-mass max error: **1.6578e-7**;
- tail mass max: **0.0**;
- state encode calls: **400**;
- state encode calls/case: **1.0**;
- candidate budget min/max: **77/77**.

Headline:
- `laya.app.banking77_full`;
- HIRA = **0.000**;
- Laya frozen target = **0.492**;
- status = **LOSS**.

Campaign after W4a:
- **1 WIN / 0 TIE / 8 LOSS / 44 MISSING**.

The only existing WIN remains the W3 raw ECE typed cell.

Machine-readable evidence:
- `artifacts/r8-w4a-banking77/summary.json`.

## Scientific interpretation

W4a is a clean negative result for held-out Banking77 semantic generalization of the frozen W3 specialist head.

The mechanics themselves did execute correctly:
- all-K K=77;
- one-state-encode;
- probability mass preserved;
- no tail truncation;
- exact frozen head;
- exact frozen dataset/protocol.

Therefore the 0/400 result must not be hidden by changing the head, prompt, semantic option text, dataset window, or evaluation mapping after exposure.

W4a does **not** prove that HIRA's high-cardinality mechanism is intrinsically broken. It confounds:
1. large-K mechanics; and
2. semantic zero-shot/generalization quality of the current head/encoder.

The next lane is W4b fresh synthetic K=128/K=255 stress, designed to isolate large-K mechanics without using the now-exposed Banking77 rows.

Do not rerun or tune W4a against the first-400 authority.
