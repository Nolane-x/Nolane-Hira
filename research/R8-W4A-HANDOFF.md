# R8-W4a handoff — held-out Banking77 full-K direct authority

Status: **PRE-EXPOSURE implementation complete on feature branch; Banking77 one-shot final has not yet executed.**

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
