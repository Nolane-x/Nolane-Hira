# HIRA V1 S42 A0 receipt — Cross-View Relational Signature-Geometry Anchoring

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #267
PR: #268

## Canonical authority

- run `37105795175`
- artifact `11267578716`
- digest `sha256:d87c03d7ec07c98cdf1922f6fa7eb210559a4929d962243848cf7bac65f12542`
- authority head `6f12bdef22d09fa205e09dbb10f9f577be5b400d`
- outcome `HIRA_V1_S42_A0_CROSS_VIEW_RELATIONAL_GEOMETRY_READY`

No replacement A0 is authorized after this qualified receipt.

## Capacity / identity

- W: **65,536**
- reference trainable: **49,152**
- treatment trainable: **114,688**
- zero-init relation/signature/fused identity: exact
- selected-choice identity: **1.0**

## Full KxK relational geometry

Diagnostic geometry shape:
**[3,4,4]**

Sensitivity:
- diagonal-only perturbation loss: **0.0009765625**
- off-diagonal-only perturbation loss: **0.0013281251303851604**
- shared logical-option permutation scalar error: **1.4901161193847656e-8**

Therefore the qualified S42 anchor:
- uses the full KxK matrix;
- responds to off-diagonal wrong-option geometry;
- remains option-permutation invariant to numerical tolerance.

## Anchor ownership / liveness

Synthetic:
- signature drift max abs **0.0068934150**
- relational anchor **2.1487843e-7**

Gradients:
- anchor -> treatment runtime L1 **0.0016135182**
- anchor -> treatment LoRA L1 **0.0011998600**
- anchor -> reference runtime **0**
- anchor -> W **0**

Correctness:
- joint W gradient L1 **233.0650024**
- off-diagonal W gradient L1 **232.0627136**
- treatment LoRA correctness gradient L1 **110.0632954**

The relational constraint and correctness co-adaptation are simultaneously live.

## AdamW candidate/state parity

- candidate movement error **0**
- first moment error **0**
- second moment error **0**
- step counter error **0**
- reconstruction-only rounding error **1.4551915e-11**

S42 retains the qualified S41 optimizer-faithful candidate engine.

## Actual-step relational projection

Synthetic conflict:
- pre-dot **+3.2266917e-7**
- projected **true**
- post-dot **3.6513370e-10**

Safe actual step:
- projected **false**
- identity max abs **0**

W projection difference:
- **0**

## Float-representable applied movement

Runtime:
- movement error **3.5070116e-9**
- rounding bound **1.4901190e-8**

W:
- movement error **0**
- rounding bound **2.8025969e-45**

Actual relational-anchor dot:
- **3.6512660e-10**
- permitted rounding bound **3.6566567e-10**

The actually representable update remains inside the frozen optimizer-step relational guard.

## Mechanics

- query intervention residual **0.0210716370**
- signature intervention residual **0.0132539095**
- question permutation logit error **2.9802322e-8**
- padding logit error **0**
- logical-option permutation logit error **5.9604645e-8**
- arbitrary K=3 / K=7 PASS
- checkpoint roundtrip PASS
- probability mass error **1.1920929e-7**
- full-K PASS

## A0 semantic diagnostics

Diagnostic only:
- relation accuracy **0.375**
- fused accuracy **0.3125**
- used for model selection: **false**

These values MUST NOT tune:
- anchor definition
- diagonal/off-diagonal weighting
- matrix norm
- optimizer semantics
- projection epsilon
- W capacity
- selector/gates
- fresh TRAIN/DEV authority.

## Consequence

S42-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. wholly fresh S42 authority is frozen;
2. matched trainer/workflow are frozen;
3. exact staged-head generic CI PASS;
4. separate one-shot TRAIN/DEV authorization marker.

No external Laya/Jev benchmark is authorized from A0.
