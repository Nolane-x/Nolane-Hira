# HIRA V1 S41 A0 receipt — Optimizer-Step-Anchored Joint Bilinear Co-Adaptation

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #265
PR: #266

## Canonical authority

- run `37101438505`
- artifact `11266051980`
- digest `sha256:b4312c61d6f1afa107192c004909579fd2b2677e651c73c2f82ce67fb4063af3`
- authority head `1b6ad4ef353a3d194b436d645fbe9c1a158e0f55`
- outcome `HIRA_V1_S41_A0_OPTIMIZER_STEP_ANCHORED_COADAPTATION_READY`

Prior failed A0 attempts were pre-qualified mechanical aborts:
- no qualified receipt
- no artifact accepted as authority
- no TRAIN/DEV authorization

No replacement S41-A0 is authorized after this qualified receipt.

## Frozen capacity

- reference trainable surface: **49,152**
- treatment trainable surface: **114,688**
- W: **256x256 / 65,536 params**
- residual scale: **1.0**
- query normalization epsilon: **1e-12**
- projection epsilon: **1e-12**
- anchor coefficient: **none**

## Zero-init identity

- relation-logit max abs: **0**
- native-signature max abs: **0**
- fused-logit max abs: **0**
- selected-choice identity: **1.0**
- matched reference/treatment native relation max abs: **0**
- matched reference/treatment native signature max abs: **0**

## Optimizer-faithful AdamW parity

Configured optimizer:
- AdamW
- lr **2e-4**
- betas **(0.9, 0.999)**
- eps **1e-8**
- weight decay **0.01**
- grad clip **1.0**
- foreach **false**
- fused **false**

Candidate/state parity against standard PyTorch AdamW:
- candidate parameter max abs error: **0**
- first-moment max abs error: **0**
- second-moment max abs error: **0**
- step-counter max abs error: **0**
- candidate reconstruction max abs error: **1.4551915228e-11**

The S41 engine therefore derives candidate movement from the same stateful AdamW transition being constrained.

## Anchor ownership

Anchor:
`mean(1 - cosine(treatment_signature, stopgrad(reference_signature)))`

Gradients:
- anchor -> treatment runtime L1: **0.5171466316**
- anchor -> treatment LoRA L1: **0.0962870780**
- anchor -> reference runtime L1: **0**
- anchor -> W L1: **0**

The reference path is detached and W is not an anchor target.

## Joint correctness path

Exact S38-style joint correctness gradient remains live:
- W L1: **304.9225463867**
- off-diagonal W L1: **303.6598205566**
- treatment LoRA L1: **183.6386508942**

Live candidate:
- runtime-anchor dot: **-4.5057109674e-5**
- projected: **false**
- W candidate delta L1: **13.1045475006**

So non-conflicting live co-adaptation is left untouched.

## Actual-step conflict projection

Synthetic actual AdamW conflicting step:
- pre-dot: **+0.0001034269953**
- projected: **true**
- post-dot: **3.6379788071e-12**
- next AdamW step counter: **1**

Safe actual step:
- projected: **false**
- identity max abs: **0**

W projection max abs difference:
- **0**

## Float-precision applied movement

Runtime:
- applied-vs-target max abs error: **2.7503119782e-9**
- explicit float-rounding bound: **1.4901189616e-8**
- applied runtime anchor dot: **-4.0927261580e-11**
- anchor-dot rounding bound: **2.9907049059e-10**

W:
- applied delta max abs error: **0**
- rounding bound: **2.8025969286e-45**

Thus the actually representable parameter movement, not only the ideal real-valued delta, is verified within frozen precision bounds.

## Synthetic anchor check

- synthetic signature drift max abs: **0.0049407389**
- anchor before applied optimizer-faithful projected step: **0.0002271612175**
- anchor after: **0.0002273174468**
- observed increase: **1.5622936e-7**

The preregistered quantized application tolerance is **2e-7**, so the applied step remains within the frozen first-order/float-precision guard.

## Invariance / mechanics

- query intervention residual change: **0.0210716370**
- signature intervention residual change: **0.0132539095**
- question-token permutation logit error: **2.9802322e-8**
- masked query-padding logit/signature error: **0 / 0**
- logical-option permutation logit error: **5.9604645e-8**
- arbitrary K=3/K=7 PASS
- checkpoint key count **1**
- checkpoint roundtrip PASS
- native probability-mass error **1.1920929e-7**
- full-K PASS
- native projection perturbation relation/signature **0 / 0**
- primary projection perturbation **0.0021007801**

## A0 semantic diagnostics

Diagnostic only:
- native relation accuracy: **0.359375**
- fused accuracy: **0.328125**

These MUST NOT tune:
- anchor target
- projection equation/epsilon
- optimizer state semantics
- rounding bounds
- W capacity
- residual scale
- selector/gates
- TRAIN/DEV seed/LR/epochs/batch

## Consequence

S41-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. this receipt is frozen;
2. matched trainer/workflow are frozen;
3. exact staged-head generic CI passes;
4. a separate one-shot TRAIN/DEV marker is committed.

No external Laya/Jev benchmark is authorized from A0.
