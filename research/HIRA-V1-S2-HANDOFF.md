# HIRA V1 S2 handoff — CLOSED DEV FAIL -> S3

Canonical status:

`HIRA_V1_S2_QTRF_DEV_FAIL`

S2 must not be retuned against exposed DEV.

## Frozen S2 evidence

A0:
- run `36390086785`
- artifact `10955893093`
- digest `sha256:e17ccf529e503450686895c983d77df945b21e90570a8945da8c48e95c57c471`
- accuracy 0.25
- question changes logits 1.0
- question changes choice 0.0625

Qualified QTRF TRAIN/DEV:
- run `36391742553`
- artifact `10957020098`
- digest `sha256:0e665e01074eada113ef481fe980fd78076692337344f6312d7022ef4d170472`
- selected epoch 2
- checkpoint SHA256 `dc2cff637771aea70ab7f7011d108cfc08a63048e8ef2502767099126eea4af2`
- accuracy 0.2578125
- paired both-correct 0.09375
- question-swap choice-change 0.546875
- option-order flip 0.0
- full-K PASS
- state-once PASS
- relation delta 0

First attempted TRAIN/DEV run `36390503180` is pre-exposure harness provenance only. It stopped before any optimizer step or DEV exposure because a short freshness sentinel matched an unrelated substring.

## What S2 established

1. Appending question tokens directly changes W34 logits but barely changes decisions.
2. Learned token-level residual fusion materially changes decisions according to the question.
3. That question sensitivity does not translate into correct option selection.
4. Mechanical runtime invariants remain healthy.
5. The next bottleneck is no longer well described as question blindness alone.

## Forbidden reuse

Never use S2 DEV rows for:
- fitting;
- hyperparameter tuning;
- architecture ranking;
- threshold selection;
- candidate selection.

Do not reopen S2 sealed or multilingual lanes.

## S3 research target

Start a separate versioned track:

**HIRA V1 S3 — Triadic State–Question–Option Semantic Scoring**

Fresh hypothesis:
- state tokens remain encoded once;
- question tokens and option views remain dynamic schema artifacts;
- learn a shared compact triadic interaction over state × question × option;
- do not force the final semantic decision through a frozen W34 state-option scorer;
- retain opaque option IDs and option permutation equivariance;
- retain full-K;
- no domain/task-specific heads;
- keep a strict compact parameter budget.

The key scientific question for S3 is:

> If question routing is already active, can a compact jointly trainable option-grounding scorer convert that routing into correct fresh decisions?

S3 requires entirely fresh TRAIN/DEV data and preregistration before exposure.

Production readiness remains false. Laya/Jev parity remains unestablished.
