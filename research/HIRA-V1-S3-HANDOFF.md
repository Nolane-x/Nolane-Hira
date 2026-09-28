# HIRA V1 S3 handoff — CLOSED DEV FAIL -> S4

Canonical status:

`HIRA_V1_S3_TRIADIC_DEV_FAIL`

S3 must not be retuned against exposed DEV.

## Frozen S3 evidence

A0:
- run `36393304057`
- artifact `10956484697`
- digest `sha256:614c1b48421392e53906075c34670b879fc2033141f7e824c31cc848bf41e0e6`
- accuracy 0.3125
- question changes logits 1.0
- question changes choice 0.25
- frozen-v0 control accuracy 0.21875

Triadic TRAIN/DEV:
- run `36393893293`
- artifact `10957677382`
- digest `sha256:6ef1e9dc5188814ea77a2b3ecfbc66787e93cf8b2ca754ce91a186e42d5f9863`
- selected epoch 1
- checkpoint SHA256 `ddcb2e99049aa3c321030cd93b499d4dcbb2c6f1fd3550de8e07710f67d3efd5`
- DEV accuracy 0.3125
- paired both-correct 0.09375
- question-swap choice-change 0.25
- option-order flip 0.0
- full-K/state-once/relation-delta-zero PASS
- W34 excluded from semantic path

## What S3 established

1. Direct triadic state-question-option geometry has some fresh zero-param signal.
2. A 12,288-param learned triadic scorer can rapidly fit TRAIN.
3. As TRAIN CE falls from ~1.36 to ~0.71, DEV accuracy collapses toward chance and DEV loss rises.
4. Removing frozen W34 does not restore transferable semantic quality.
5. The strongest common frozen bottleneck across S0–S3 is now the A13/W28 semantic representation substrate.

## Forbidden reuse

Never use S3 A0 or DEV rows for:
- fitting;
- hyperparameter tuning;
- architecture ranking;
- threshold selection;
- candidate selection.

Do not reopen S3 sealed or multilingual lanes.

## S4 target

Start a separately versioned track:

**HIRA V1 S4 — Compact Semantic Representation Adaptation**

Fresh hypothesis:
- keep A13 backbone frozen at first;
- learn one shared low-rank token adapter before state/question/option comparison;
- the same adapter must transform all semantic roles;
- no domain/task-specific head;
- state encoder still exactly once;
- dynamic full-K and opaque IDs preserved;
- strict compact parameter budget;
- wholly fresh TRAIN/DEV.

The key question is:

> Can Hira learn a transferable shared semantic geometry, rather than merely fitting a decision surface over a frozen geometry?

Production readiness remains false. Laya/Jev parity remains unestablished.
