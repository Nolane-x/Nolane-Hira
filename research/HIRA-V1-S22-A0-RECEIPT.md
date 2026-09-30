# HIRA V1 S22 A0 receipt — Primary-Priority Norm-Balanced Optimization

Status: **QUALIFIED / DIAGNOSTIC ONLY**

Issue: #227  
PR: #228

## Authority

- workflow run: `36709243180`
- artifact: `11093551516`
- artifact digest: `sha256:217b60410f9dc3e63d0c5d82dc75af6dadccdb2dc181ae7fbe2885f0eb6ac638`
- exact head: `592ae0a70c1085c79b8f8d85c6f0b2be632da2dd`
- outcome: `HIRA_V1_S22_A0_PRIMARY_PRIORITY_READY`

## Optimizer-only identity

S22 must not alter S21 inference.

Observed:
- S21 logit identity rate: **1.0**
- S21 choice identity rate: **1.0**
- A13 token output identity: PASS
- A13 pooled output identity: PASS
- role/content factorization added params: **0**
- total physical surface: **49,152**
- runtime trainable during A0: **0**
- original A13 trainable: **0**

## Primary-priority conflict court

Controlled conflict:
- normalized pre-dot: **-0.7999999523**
- normalized post-dot: **+4.470348358e-08**
- projection coefficient: **-0.8000000119**
- analytic expected-direction max abs: **5.960464478e-08**
- relation direction change L1: **1.073312521**
- protected primary direction identity max abs: **0.0**

No-conflict:
- S22 vs S17 output max abs: **0.0**
- S22 conflict flag: false
- S17 conflict flag: false

Joint positive scale equivariance:
- max abs error: **0.0**

This proves the controlled optimizer change before DEV.

## Preserved S21 structural mechanism

- synthetic role/content court: PASS
- question role A selected: **0**
- question role B selected: **2**
- correct vs same-role/wrong-content margin: **+0.2051138282**
- correct vs wrong-role/same-content margin: **+0.5093060732**
- factorized option permutation max abs: **0.0**
- degenerate geometry finite: PASS
- role temperature: **0.10**
- role/content weights: **0.50 / 0.50**

## Fusion / mechanics

- fusion expert-swap max abs: **0.0**
- fusion vs S14 forward max abs: **0.0**
- fused option-order flip: **0.0**
- primary option-order flip: **0.0**
- fused mass error: **1.192092896e-07**
- primary mass error: **1.192092896e-07**
- state encode calls: **32**
- full-K: PASS
- relation refinement: off

## Real shared-gradient diagnostic

Fresh A0 semantic batch:
- primary norm: **2.999655485**
- relation norm: **0.2558864057**
- raw norm ratio: **11.72260588**
- normalized pre-dot: **+0.1188236549**
- conflict: false
- normalized post-dot: **+0.1188236549**
- combined norm: **1.627770901**
- physical surface: **49,152**

This particular A0 batch is a no-conflict real-gradient case; the separate controlled tensor court proves conflict behavior.

## Diagnostic semantic metrics

A0 is not model-selection evidence.

- primary canonical accuracy: **0.34375**
- primary paraphrase accuracy: **0.25**
- primary cross-view agreement: **0.8125**
- relation canonical accuracy: **0.3125**
- relation paraphrase accuracy: **0.4375**
- fused canonical accuracy: **0.3125**
- fused paraphrase accuracy: **0.34375**
- fused paired both-correct: **0.0625**
- fused cross-view agreement: **0.75**
- fused mean JS: **0.0371042937**

No tuning may use these diagnostics.

## Authorization consequence

S22-A0 is QUALIFIED.

TRAIN/DEV may open only after:
1. interpretation plan is frozen;
2. TRAIN/DEV workflow is staged;
3. exact-head CI passes.

No optimizer-priority retry or role/content tuning is authorized after fresh DEV exposure.
