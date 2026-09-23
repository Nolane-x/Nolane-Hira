# R8-W5e handoff — A22 paired semantic capacity control

Status: **implementation active under issue #67; A22 immutable pin complete; no semantic result yet.**

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
