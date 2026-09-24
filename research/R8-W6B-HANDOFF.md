# R8-W6b handoff — fresh typed/reliability production authority

Status: **PRE-AUTHORITY FROZEN. No W6b CONFIRM result is valid yet.**

Issue: #79
PR: #80

## Scientific purpose

W6 Phase A integrated the promoted forward competitive binder into HIRA's real state-once coarse path without changing legacy defaults.

W6b is the first fresh typed/reliability authority that tests whether the integrated production competitive path improves over a same-authority legacy HIRA control.

This lane is no longer a synthetic semantic-scorer search.

## Frozen upstream identities

A13:
- model `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`.

W3 HIRA head:
- run `35748778854`;
- artifact `10706135649`;
- artifact name `r8-w3-selected-head`;
- artifact digest `sha256:8a2efd31a6ccea3b987957d328c4b73267ab00ff9024c469bcb7f3cd578d5d92`;
- head SHA-256 `2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c`;
- HIRACore parameters 422,159.

W5i promoted forward scorer:
- run `35884130186`;
- artifact `10764240783`;
- artifact name `r8-w5i-candidate-idf-competitive-forward-proj128`;
- artifact digest `sha256:f3c0ff451c976caff10b8098b010abc7b8758d4c4797110c4cce18943edb4f74`;
- candidate `idf-competitive-forward-proj128`;
- matcher SHA-256 `5ce9cdceb5a87c8b18b8d50e68396f23184e1dfb9acad338f25f24277d438a0d`;
- scorer parameters 32,769.

The W5i reverse/listwise scorer is not promoted.

## Fresh authority

Every case has exactly five decisions:
1. diagnosis — choice;
2. response — choice;
3. needs_review — noul;
4. risk — score;
5. urgency — score.

Counts:
- TRAIN: 320 cases / 1,600 decisions;
- DEV: 128 cases / 640 decisions;
- CONFIRM: 160 cases / 800 decisions.

Seeds:
- TRAIN 161127;
- DEV 162229;
- CONFIRM 163331;
- training/global 809.

CONFIRM remains sealed until DEV selection and all selected checkpoint hashes are frozen.

Forbidden:
- typed-decisions final/test rows;
- prior W5 semantic CONFIRM rows;
- public campaign cells.

## Exactly three candidates

1. `legacy-w3-joint`
   - legacy coarse mode;
   - W3 init;
   - 422,159 trainable/total parameters.

2. `competitive-w5i-joint`
   - W3 + W5i forward scorer init;
   - competitive coarse mode;
   - 454,928 trainable/total parameters.

3. `competitive-w5i-scorer-only`
   - frozen W3;
   - train competitive scorer only;
   - 32,769 trainable / 454,928 total parameters.

All use the same frozen A13 cached features, six epochs and frozen loss/optimizer contract from issue #79.

## DEV freeze

Each candidate freezes its own epoch by:
1. higher overall accuracy;
2. lower hard Brier;
3. higher soft accuracy;
4. higher diagnosis K64 accuracy;
5. lower score MAE;
6. lower soft-target ECE;
7. earlier epoch.

Select one competitive candidate between joint and scorer-only.
Freeze legacy independently.
Only then may CONFIRM be generated.

## Frozen production gates

Absolute selected gates:
- overall accuracy >= 0.65;
- choice accuracy >= 0.65;
- noul accuracy >= 0.70;
- score accuracy >= 0.60;
- diagnosis K64 accuracy >= 0.55;
- hard Brier <= 0.50;
- soft-target ECE <= 0.15;
- score MAE <= 0.55;
- probability error <= 1e-6;
- source state encodes/case = 1.0.

Mechanism gates vs legacy:
- overall accuracy gain >= +0.03;
- diagnosis K64 gain >= +0.05;
- hard Brier no worse by > 0.02;
- soft-target ECE no worse by > 0.02;
- score MAE no worse by > 0.05.

Valid verdicts:
- `PRODUCTION_COMPETITIVE_RESCUE`;
- `PRODUCTION_LEGACY_ALREADY_STRONG`;
- `PRODUCTION_COMPETITIVE_PARTIAL`;
- `PRODUCTION_COMPETITIVE_FAIL`.

## Pre-authority evidence

Head `29ebe5a1971ed8214bc1593007aa7934fa4a1ee0`:
- W6b push unit workflow PASS;
- W6b PR unit workflow PASS;
- **14/14 pre-authority contracts PASS**;
- repo CI Python 3.10 PASS;
- repo CI Python 3.12 PASS;
- preflight PASS.

No W6b CONFIRM has been generated or exposed by this boundary.

## Authority chain

The first eligible authority workflow must gate:

`unit -> upstream provenance -> fresh A13 TRAIN/DEV cache -> 3 candidates -> DEV selection freeze -> untouched CONFIRM`

No branch mutation is allowed while an eligible exact-head authority run is active.


## Pre-data calibration amendment

Before any W6b TRAIN/DEV cache or CONFIRM was generated, audit found that hard ECE conflicts with the declared soft reliability targets.

A model matching a target confidence of 0.90/0.75/0.60 should be considered reliability-calibrated. Hard ECE compares those confidences with binary correctness and can penalize an otherwise perfect soft-target match for not being overconfident.

Therefore:
- raw hard ECE remains recorded as a diagnostic;
- soft-target ECE compares model confidence with the frozen teacher probability assigned to the predicted class;
- DEV tie-breaking uses soft-target ECE;
- the absolute 0.15 calibration gate uses soft-target ECE;
- the +0.02 non-regression mechanism gate uses soft-target ECE.

Regression coverage explicitly proves that [0.90, 0.75, 0.60] against the same soft targets yields approximately zero soft-target ECE while hard ECE exceeds 0.15.

This amendment occurred before empirical authority execution and does not change authority rows, model architecture, optimizer, loss weights, checkpoint identities, accuracy gates or verdict taxonomy.

## Pre-data integrity strengthening

Also before empirical authority execution:
- cache receipts were extended with TRAIN/DEV/CONFIRM template IDs and template hash;
- TRAIN/DEV and reserved-CONFIRM value lexicons receive separate hashes;
- candidate receipts bind to exact TRAIN/DEV cache SHA-256;
- selector rejects candidates from mismatched caches;
- confirm receipt carries cache provenance and explicit parameter counts;
- a regression contract proves TRAIN and DEV gold diagnosis semantic signatures are internally unique and cross-split disjoint.

Earlier queued authority heads are superseded; only the final post-amendment exact head may be authoritative.
