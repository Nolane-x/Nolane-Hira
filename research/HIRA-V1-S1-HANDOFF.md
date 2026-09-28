# HIRA V1 S1 handoff — to S2

Status: **S1 CLOSED / S2 AUTHORIZED AS A NEW TRACK**

Canonical S1 outcome:

`HIRA_V1_S1_QKEE_DEV_FAIL`

## Frozen evidence

A0:
- run `36387872121`
- artifact `10955282264`
- digest `sha256:98428470ca03e5de015517c1355b4d16d2dc1aa6bf8688e6ae2f05476cf5da3a`
- 0 new params
- evidence-change 0.625
- choice-change 0.0
- accuracy 0.21875

QKEE TRAIN/DEV:
- run `36388286637`
- artifact `10955218190`
- digest `sha256:51b7a808531922f1e3c73c4d01d931562b7f0dc2032e39af7d9d68d5a575468d`
- selected epoch 7
- checkpoint SHA256 `50232f45f0e7b6123c84f9b5378473f8b3028978044e6e0163f93e5cfdc1a4c7`
- DEV accuracy 0.265625
- paired both-correct 0.0
- question-swap choice-change 0.0
- order flip 0.0
- state-once/full-K PASS

## Forbidden for S2 fitting/selection

- S0 localization/TRAIN/DEV;
- S1-A0 rows;
- S1 TRAIN/DEV rows;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

Only aggregate conclusions may motivate S2.

## S2 hypothesis

Do **not** retry S1 with different temperature/head count on exposed S1 DEV.

Use a distinct architecture:

`Query-Token Cross-Attention Residual Fusion`

Preserve all state tokens. Compute low-rank cross-attention between frozen W28 state/question relation vectors and fuse question context residually into each state relation token before W34 option scoring.

Key design goals:
- no two-slot state compression;
- query must alter state representation;
- state encoder remains once;
- W28/W34 may remain frozen for first isolation experiment;
- <= ~16k new trainable parameters preferred;
- option permutation equivariance;
- full-K;
- fresh English TRAIN/DEV first;
- multilingual separately later.

## S2 first gate

Before any external benchmark:
1. preregister architecture and parameter budget;
2. unit-test exact frozen-base isolation;
3. use fresh paired-query TRAIN/DEV;
4. require strong question-swap behavior plus accuracy;
5. only a DEV-ready candidate may enter a one-shot sealed confirm.

No production/parity claim is authorized by this handoff.
