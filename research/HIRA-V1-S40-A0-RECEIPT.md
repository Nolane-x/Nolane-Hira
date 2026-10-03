# HIRA V1 S40 A0 receipt — Reference-Anchored Projected Joint Bilinear Co-Adaptation

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #263
PR: #264

## Canonical authority

- run `37095924764`
- artifact `11264530873`
- digest `sha256:38a3179453383491f8640c9e44dd4d2be33bf16730b81725f072e016cf67c3b8`
- authority head `87082ac11e7ec655b62fee6b91b1768b2e3236a6`
- outcome `HIRA_V1_S40_A0_REFERENCE_ANCHORED_PROJECTED_COADAPTATION_READY`

A prior attempted A0 `37095497741` was a pre-qualified mechanical abort:
- no receipt
- no semantic accuracy
- no artifact
- synthetic drift discriminator only

No replacement A0 is authorized after this qualified receipt.

## Frozen capacity

- reference trainable surface: **49,152**
- treatment trainable surface: **114,688**
- W: **256x256 / 65,536 params**
- residual scale: **1.0**
- query normalization epsilon: **1e-12**
- projection epsilon: **1e-12**
- anchor coefficient: **none**

## Zero-init identity

- treatment/base relation logit max abs: **0**
- treatment/base signature max abs: **0**
- treatment/base fused logit max abs: **0**
- selected-choice identity: **1.0**
- matched reference/treatment native relation max abs: **0**
- matched reference/treatment native signature max abs: **0**

## Reference anchor mechanics

Anchor:
`mean(1 - cosine(treatment_signature, stopgrad(reference_signature)))`

Synthetic drift:
- native signature max abs: **0.0084399059**
- anchor value: **0.0006158862**

Anchor gradients:
- treatment runtime L1: **0.8388252266**
- treatment LoRA L1: **0.1350415703**
- reference runtime L1: **0**
- W L1: **0**

The reference path is therefore detached and W is not regularized by the anchor.

## Joint correctness path

Exact S38-style joint co-adaptive correctness gradient:
- W L1: **161.3534240723**
- W off-diagonal L1: **160.6846160889**
- treatment LoRA L1: **90.7297470570**

Therefore S40 preserves the co-adaptive gradient path that S39 intentionally removed.

## Parameter-free projection

Synthetic conflicting runtime gradient:
- pre-projection anchor dot: **-0.0003868329**
- projected: **true**
- post-projection anchor dot: **0**

Synthetic non-conflicting gradient:
- projected: **false**
- identity max abs: **0**

W preservation:
- W projection max abs difference: **0**

Small projected synthetic step:
- anchor before: **0.0006158862**
- anchor after: **0.0006158862**

The projection removes only the first-order anchor-conflicting runtime component and does not modify W.

## Invariance / mechanics

- query intervention residual change: **0.0210716370**
- signature intervention residual change: **0.0132539095**
- question-token permutation logit error: **2.9802322e-8**
- question-token permutation signature error: **0**
- masked query-padding logit/signature error: **0 / 0**
- logical-option permutation logit error: **5.9604645e-8**
- logical-option permutation signature error: **0**
- arbitrary K=3/K=7 PASS
- checkpoint key count **1**
- checkpoint roundtrip PASS
- native probability-mass error **1.1920929e-7**
- full-K PASS

Projection independence:
- native relation-logit perturbation from shared projection: **0**
- native signature perturbation from shared projection: **0**
- primary projection perturbation: **0.0019234901**

## A0 semantic diagnostics

Diagnostic only:
- native relation accuracy: **0.421875**
- fused accuracy: **0.390625**

These values MUST NOT tune:
- anchor target
- projection equation/epsilon
- W capacity
- residual scale
- optimizer
- selector/gates
- TRAIN/DEV seed/LR/epochs/batch

## Consequence

S40-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. this receipt is frozen;
2. matched trainer/workflow are frozen;
3. exact staged-head generic CI passes;
4. a separate one-shot TRAIN/DEV marker is committed.

No external Laya/Jev benchmark is authorized from A0.
