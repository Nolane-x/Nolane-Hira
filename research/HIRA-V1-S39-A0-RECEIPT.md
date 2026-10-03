# HIRA V1 S39 A0 receipt — Gradient-Isolated Full Bilinear Correctness Readout

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #261
PR: #262

## Canonical authority

- run `37089648986`
- artifact `11261657743`
- artifact digest `sha256:afc20e60cabac8de5db33eeee15574207d7e0592547a1787e4980cc0b81f6e20`
- authority head `6f1eb465cfcefc7bc368218378fc144ea3f6c347`
- outcome `HIRA_V1_S39_A0_GRADIENT_ISOLATED_BILINEAR_READY`

No replacement S39-A0 is authorized.

## Frozen capacity

- W shape: **256 x 256**
- added W params: **65,536**
- control trainable surface: **49,152**
- treatment trainable surface: **114,688**
- LoRA: **16,384**
- shared primary projection: **32,768**
- original A13 trainable: **0**
- HIRACore trainable: **0**
- residual scale: **1.0**
- query normalization epsilon: **1e-12**

## Exact zero-init identity

- relation-logit max abs difference: **0**
- signature max abs difference: **0**
- primary-logit max abs difference: **0**
- fused-logit max abs difference: **0**
- selected-choice identity: **1.0**

## Hard gradient isolation

Correction route:
- correction CE: **1.3823730946**
- correction block: **0.1382373124**
- correction -> W gradient L1: **0.3294754326**
- correction -> off-diagonal W gradient L1: **0.3281979561**
- correction -> LoRA gradient L1: **0**
- correction -> shared projection gradient L1: **0**
- correction -> HIRACore gradient L1: **0**

Runtime route:
- primary -> W gradient L1: **0**
- native relation -> W gradient L1: **0**
- native relation -> LoRA gradient L1: **0.1888366621**
- primary -> projection gradient L1: **886.7951660156**

Matched synthetic update:
- runtime gradient max abs control-treatment: **0**
- runtime update max abs control-treatment: **0**
- W update max abs: **0.0001999941**
- runtime clip: **1.0**
- W clip: **1.0**

This proves the correction head can learn while the matched native runtime update remains unchanged.

## Intervention / invariance evidence

- query intervention residual change: **0.0183070228**
- signature intervention residual change: **0.0088960072**
- question-token permutation logit error: **7.4505806e-9**
- question-token permutation signature error: **0**
- masked query-padding logit error: **0**
- masked query-padding signature error: **0**
- logical-option permutation logit error: **5.9604645e-8**
- logical-option permutation signature error: **0**

## Projection independence / arbitrary K / mechanics

- native projection perturbation relation-logit change: **0**
- native projection perturbation signature change: **0**
- primary projection perturbation change: **0.0018783105**
- K=3: logits `[3,3]`, signatures `[3,3,256]`
- K=7: logits `[3,7]`, signatures `[3,7,256]`
- checkpoint key count: **1**
- checkpoint roundtrip: PASS
- probability-mass error: **1.1920929e-7**
- full-K: PASS
- expected state views: **32**

## A0 semantic diagnostics

Diagnostic only:
- native relation accuracy: **31.25%**
- fused accuracy: **20.3125%**

These values MUST NOT tune:
- detach/gradient mixing
- W LR/scheduler
- rank/factorization
- residual scale/bias/nonlinearity
- native geometry
- projected/native mixing
- optimizer
- selector/gates
- seed/LR/epochs/batch

## Consequence

S39-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. this receipt is frozen;
2. trainer/workflow are frozen;
3. exact-head generic CI passes;
4. a separate one-shot TRAIN/DEV marker is committed.

No second S39-A0 run is authorized.
