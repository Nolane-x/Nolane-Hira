# R8-W5e handoff — A22 paired semantic capacity control

Status: **COMPLETE / NEGATIVE. W5e merged; authoritative verdict `A22_CAPACITY_NO_RESCUE`.**

## Authorization

W5d verdict:
`A13_ADAPTATION_FAIL_CAPACITY_TRIGGER`.

A22 control is explicitly authorized.

## Frozen models

A13:
- microsoft/xtremedistil-l6-h256-uncased;
- revision 4226d9e4d2c08703e5cb0491b479bfc6a1607181;
- weight SHA-256 5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880.

A22:
- microsoft/xtremedistil-l6-h384-uncased;
- immutable revision 359df7d52613d4edc15647e6d65e0d87200eb747;
- pytorch_model.bin SHA-256 38bd5f8a7d1b7045de8fee25bfac1777edf5a2ec8cd3399b21bde917b0278e23;
- hidden 384;
- 6 layers;
- 22,565,376 encoder params excluding pooler;
- 22,713,216 total params.

Pin authority:
- run 35849268011 PASS;
- artifact 10745261136;
- no semantic case generated/evaluated.

## Paired design

Same fresh W5e TRAIN/DEV/CONFIRM for both models.

Tracks:
- A13 frozen;
- A22 frozen;
- A13 top2-lr1e5;
- A22 top2-lr1e5.

Adaptation recipe copied from the selected W5d recipe:
- top 2 layers;
- lr 1e-5;
- AdamW wd 0.01;
- 3 epochs;
- seed 271;
- no scheduler/warmup/clipping/AMP/accumulation.

No HIRA scorer.

Both adapted tracks select their own epoch on DEV only.
CONFIRM is generated only after both checkpoint SHAs are frozen.

## Interpretation authority

Possible verdicts:
- A22_CAPACITY_RESCUE;
- A22_CAPACITY_AMBIGUOUS;
- A22_CAPACITY_NO_RESCUE.

No public campaign cell is populated by W5e.


## Authoritative completion

Merged to `main` as `c7dd96f0e854ea3ebf14c466ff3574945fa8ebee`.

Authoritative run:
- `35850323054`: PASS technical;
- A13 track PASS;
- A22 track PASS;
- paired untouched CONFIRM PASS;
- verdict: **`A22_CAPACITY_NO_RESCUE`**.

Untouched CONFIRM, 192 cases:
- A13 frozen: accuracy 0.0000, top-5 0.0104, MRR 0.04069;
- A13 adapted: accuracy 0.0000, top-5 0.0781, MRR 0.06068;
- A22 frozen: accuracy 0.0000, top-5 0.0365, MRR 0.04866;
- A22 adapted: accuracy **1/192 = 0.00521**, top-5 0.08854, MRR 0.07065.

High-K:
- A22 adapted K=128 top-1/top-5: 0 / 0;
- A22 adapted K=255 top-1: 0;
- A22 adapted K=255 top-5: 0.0625.

All capacity/competence gates fail except probability-mass integrity.

Machine-readable authority:
`artifacts/r8-w5e-capacity-control/summary.json`.

Interpretation:
- capacity increase from A13 to A22 gives only small ranking gains;
- encoder size alone is not the dominant blocker;
- do not automatically escalate to a larger backbone;
- next work must change semantic learning/representation mechanism.
