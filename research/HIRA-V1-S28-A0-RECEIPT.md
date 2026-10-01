# HIRA V1 S28 A0 receipt — Anchor-Level Cross-View Factor Transport

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #239  
PR: #240

## Canonical authority

- run `36867793857`
- artifact `11164614334`
- artifact digest `sha256:b3033c40357ba4b9b8cca378377a6d7e87f27f7e8449489e0beef20eb7fdec22`
- authority head `73cc786ee59f74098665024648955704718aa8a0`
- outcome `HIRA_V1_S28_A0_ANCHOR_FACTOR_TRANSPORT_READY`

No replacement A0 is authorized.

## Frozen inherited surface

- shared A13 LoRA **16,384**
- primary-private projection **32,768**
- relation-private projection **32,768**
- exact physical trainable surface **81,920**
- anchor transport added params **0**
- A0 runtime trainable params **0**
- original A13 trainable params **0**
- HIRACore trainable params **0**
- factorized signature **256D = 128D role + 128D value**

## Production inference identity

S28 training-only intervention leaves S27/S26 production inference exactly unchanged:

- primary inference max abs **0**
- relation inference max abs **0**
- factorized signature max abs **0**
- fused inference max abs **0**
- shared encoder identity PASS

## Anchor extractor identity

The independent S28 extractor reproduces the preregistered S26 state-side latent construction:

- role anchor reference max abs **0**
- value anchor reference max abs **0**
- role norm error **1.1920928955e-7**
- value norm error **1.1920928955e-7**

No learned state is introduced by the extractor/objective.

## Anti-collapse / localization court

Identical well-separated views:
- total loss **0**
- role total **0**
- value total **0**

Role-only mismatch:
- role total **0.3499999940**
- value total **0**

Value-only mismatch:
- role total **0**
- value total **0.3499999940**

Constant-anchor collapse:
- role separation **0.2000000030**
- value separation **0.2000000030**
- total **0.2000000626**

Batch permutation:
- total abs **0**
- role abs **0**
- value abs **0**

Thus the cross-query negative court explicitly penalizes constant-anchor collapse while localizing role/value perturbations to the intended factor.

## Gradient ownership

- primary → relation-private max abs **0**
- relation → primary-private max abs **0**
- primary-private gradient L1 **124.0352783203**
- relation-private gradient L1 **64.9374542236**
- primary shared-LoRA gradient L1 **4.5386595726**
- relation shared-LoRA gradient L1 **2.5749397278**
- shared neutral-bisector projection coefficient **0**

S25 ownership semantics remain intact.

## Full-K / permutation / numerical court

- primary option permutation max abs **0**
- relation option permutation max abs **0**
- factorized signature option permutation max abs **0**
- fused option-order flip rate **0**
- option permutation effect on state-anchor objective **0**
- max probability-mass error **1.1920928955e-7**
- full-K PASS
- state-once views **32**

## A0 semantic diagnostics

Diagnostic only; not selection/tuning authority:

- primary canonical accuracy **37.5%**
- relation canonical accuracy **28.125%**
- fused canonical accuracy **34.375%**
- real anchor transport total **0.6269187331**
- role transport total **0.4248559475**
- value transport total **0.8289815187**

These values MUST NOT tune anchor weights, margin, coefficient, formula, seed, selector or DEV gates.

## Consequence

S28-A0 is QUALIFIED.

Fresh S28 TRAIN/DEV may open only after:
1. this receipt is frozen;
2. the already-frozen S28 interpretation plan remains unchanged;
3. fresh S28 TRAIN/DEV authority generator remains frozen;
4. TRAIN/DEV workflow is staged;
5. exact-head generic CI passes;
6. a separate one-shot TRAIN/DEV authorization is committed.

No additional S28-A0 run is authorized.
