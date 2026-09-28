# HIRA V1 S7 handoff — to S8 Paraphrase-Invariant Semantic Binding

S7 is frozen as:

`HIRA_V1_S7_COADAPT_DEV_FAIL`

## Canonical evidence

A0:
- run `36411385031`
- artifact `10964497583`
- digest `sha256:38c9a23606053c4131143817a555677bdbe0e09dac645cbd725572a1d6cd4bdc`

TRAIN/DEV:
- run `36411829108`
- artifact `10966140203`
- digest `sha256:4ba4cf0b0e3a95c85994df3d329ddb66a0039bea6d6e0b895ed034e266fa1c3a`
- selected epoch 5
- checkpoint SHA256 `5aa9e5b9cdb90d8dac1d76cdd46021085011177ff9fa010add8437e68c46dbac`

Selected DEV:
- accuracy 0.3463541667
- paired both-correct 0.125
- question-swap choice-change 0.578125
- option-order flip 0.0
- full-K/state-once/relation-delta-zero PASS.

## Key result

Joint A13-W28 optimization learns strongly on TRAIN, but fresh DEV peaks early and then regresses while TRAIN losses continue improving.

This reduces the plausibility of "insufficient trainable capacity" as the immediate bottleneck.

## Do not do

Do not:
- retune S7 against exposed DEV;
- invent early stopping thresholds from S7 DEV;
- widen LoRA/projection solely from S7;
- reuse S7 rows for S8 fitting/selection;
- reopen M5/Laya/Jev from S7.

## S8 target

Keep S7 architecture unchanged.

Test explicit paraphrase/template invariance:
- multiple state views per underlying fact pair;
- multiple question views per semantic query;
- consistency loss across semantically equivalent views;
- fresh authority only.

The key question is whether explicit invariance prevents semantic binding from overfitting wording/templates.
