# HIRA V1 S44 A0 receipt — Private Correction Representation Fork

Status: **QUALIFIED / DIAGNOSTIC ONLY**

Issue: #271
PR: #272

## Canonical authority

- run `37120084360`
- artifact `11272917607`
- digest `sha256:90d77927a932c374e5cb2aadc64ce250827e1032c15a1dae36d8e06699f47deb`
- authority head `9f7602d5c6456b43438d3c2f933d40f4db02bc4f`
- outcome `HIRA_V1_S44_A0_PRIVATE_CORRECTION_REPRESENTATION_READY`

No replacement S44-A0 is authorized after this qualified receipt.

## Frozen capacity

Native trainable:
- **49,152**

Private adapter:
- A: **32,768**
- B: **16,384**
- total: **49,152**

Full bilinear W:
- **65,536**

Correction-only:
- **114,688**

Treatment total:
- **163,840**

Hidden width:
- **64**

Adapter seed:
- **65044**

## Frozen correction objective

Exact S39 objective:
- `CE_corr = 0.5 * (CE(canonical) + CE(paraphrase))`
- `L_corr = 0.10 * CE_corr`

No fused-specific correction loss.
No new coefficient.

## Zero-init identity / ownership

- corrected/native relation max abs: **0**
- private residual max abs: **0**
- private/native signature max abs: **2.9802322387695312e-8**
- fused-logit max abs: **0**
- selected-choice identity: **1.0**

Native treatment/reference:
- parameter max abs: **0**
- relation max abs: **0**
- signature max abs: **0**
- primary max abs: **0**

Gradient ownership:
- correction -> treatment native runtime: **0**
- correction -> reference runtime: **0**
- native objective -> A/B/W: **0**
- warm correction -> native runtime: **0**
- warm correction -> LoRA: **0**
- warm correction -> projection: **0**

Matched native mechanics:
- native gradient max abs: **0**
- one-step parameter max abs: **0**
- one-step output max abs: **0**

## Deterministic warm-start

Step 1:
- W gradient L1 **0.4508321881**
- B gradient L1 **0**
- A gradient L1 **0**
- W norm after step **0.0506699048**

Step 2:
- W gradient L1 **0.4469819069**
- B gradient L1 **0.00343205384**
- A gradient L1 **0**
- B norm **0.0167770460**

Step 3:
- W gradient L1 **0.4455194771**
- B gradient L1 **0.00682977308**
- A gradient L1 **0.000197497313**
- W norm **0.151803121**

Therefore the intended causal activation order is qualified:
**W -> B -> A**.

## New representation family

After deterministic warm-start:
- private-vs-W-only residual max abs **0.000398762524**

The nonlinear private branch is observably distinct from the S39 W-only family.

## Mechanics

- arbitrary K=3 PASS
- arbitrary K=7 PASS
- logical-option permutation error **0**
- question permutation error **1.1920929e-7**
- masked padding error **0**
- checkpoint roundtrip exact
- native probability-mass error **1.1920929e-7**
- full-K PASS
- native projection perturb relation/signature **0 / 0**
- primary projection perturb **0.0018739020**
- state-view count **32 / expected 32**

## A0 semantic diagnostics

Diagnostic only:
- relation accuracy **0.328125**
- fused accuracy **0.28125**

These MUST NOT tune:
- hidden width
- activation
- adapter seed/init scale
- correction coefficient
- optimizer
- capacity
- selector/gates

## Consequence

S44-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. this receipt is frozen;
2. wholly fresh S44 authority is frozen;
3. matched trainer/workflow are frozen;
4. exact staged-head generic CI passes;
5. a separate one-shot TRAIN/DEV marker is committed.

No Laya/Jev evaluation is authorized from A0.
