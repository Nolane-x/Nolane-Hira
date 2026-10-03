# HIRA V1 S40 closure — Reference-Anchored Projected Joint Bilinear Co-Adaptation

Status: **CLOSED PRE-DEV — OPTIMIZER-FAITHFULNESS INVALIDATION / NO S40 TRAIN-DEV EXPOSURE**

Issue: #263
PR: #264

## What completed

S40 preregistered and implemented:
- matched native reference runtime;
- exact S38 full bilinear treatment capacity;
- parameter-free native-signature anchor;
- raw-runtime-gradient projection against the anchor;
- wholly fresh S40 authority;
- A0 mechanics court;
- matched TRAIN/DEV trainer/workflow staging.

Qualified A0:
- run `37095924764`
- artifact `11264530873`
- digest `sha256:38a3179453383491f8640c9e44dd4d2be33bf16730b81725f072e016cf67c3b8`
- authority head `87082ac11e7ec655b62fee6b91b1768b2e3236a6`
- outcome `HIRA_V1_S40_A0_REFERENCE_ANCHORED_PROJECTED_COADAPTATION_READY`

Matched trainer/workflow staged head:
`6aedd1bdb8644bcf9f366f1c120dff973eff8330`

Exact staged-head generic CI:
- run `37096245409`
- Python 3.10 PASS
- Python 3.12 PASS
- preflight PASS

## No fresh DEV was opened

There is no:
`research/HIRA-V1-S40-ENABLE-TRAIN-DEV`

Therefore:
- no S40 fresh TRAIN/DEV workflow ran;
- no S40 DEV row was evaluated;
- no selected S40 DEV checkpoint exists;
- no S40 DEV metric was exposed;
- no S40 one-DEV budget was consumed.

## Mechanistic invalidation found before DEV

The frozen S40 projection acts on the **raw treatment runtime gradient**.

For anchor gradient `a` and raw treatment gradient `g`, S40 enforces:
- if conflicting, `a dot g' = 0`.

That statement is valid for a plain infinitesimal gradient-descent step `-g'`.

However the actual frozen optimizer is AdamW:
- moment accumulation;
- coordinate-wise second-moment preconditioning;
- bias correction;
- decoupled weight decay;
- then the parameter update.

The actual parameter delta is therefore not generally a scalar multiple of `-g'`.

Consequently:

> `a dot g' = 0` does not imply `a dot delta_AdamW <= 0`.

A raw-gradient projection can be rotated/rescaled by AdamW and can still move parameters in a first-order anchor-increasing direction.

The A0 synthetic projected-step check does not close this gap because it validates a direct synthetic projected gradient step, not the full stateful AdamW parameter delta used in the matched trainer.

## Scientific consequence

Running S40 DEV would not cleanly test the preregistered scientific question.

The intended claim was:
- preserve joint correctness co-adaptation;
- prevent first-order native-signature drift increase.

The staged implementation only guarantees the second condition **before optimizer transformation**, not on the actual parameter movement.

This is a mechanistic contract failure, not a negative model result.

No S40 A/B/C/D DEV interpretation is made.

## Why no in-stage repair

The qualified S40 A0 receipt explicitly freezes:
- no replacement A0 after the qualified receipt.

Changing projection from raw-gradient space to actual AdamW-step space changes the tested mechanism and requires a new mechanical court.

Therefore S40 is closed pre-DEV rather than silently mutating the mechanism after A0 exposure.

## Next controlled direction

**S41 — Optimizer-Step-Anchored Joint Bilinear Co-Adaptation**

Keep:
- exact S38 full bilinear capacity;
- matched S35/S17 reference trajectory;
- same native-signature anchor;
- no anchor coefficient;
- no partial detach;
- no new learned parameters.

Change:
- constrain the **actual proposed AdamW runtime parameter delta**, after moment/preconditioning/weight-decay computation;
- leave W proposed AdamW delta unchanged;
- update AdamW state deterministically from the frozen raw clipped gradients;
- project only the actual runtime delta if it would increase the anchor to first order.

For actual runtime parameter delta `delta`:
- if `a dot delta <= 0`: identity;
- if `a dot delta > 0`: `delta' = delta - ((a dot delta)/(||a||^2 + 1e-12)) a`.

Then:
- `a dot delta' = 0`;
- actual first-order anchor change is non-positive;
- W delta remains the exact AdamW candidate delta.

S41 must preregister this optimizer-state semantics and qualify a new A0 before any fresh DEV.

Production-ready remains false.
Laya/Jev parity remains unestablished.
