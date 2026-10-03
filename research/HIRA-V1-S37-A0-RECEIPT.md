# HIRA V1 S37 A0 receipt — Query-Gated Native Signature Correctness Readout

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #257
PR: #258

## Canonical authority

- run `37078578256`
- artifact `11257573101`
- artifact digest `sha256:adbff1f2af80978615f437d2350f4dd132905a42aeb18b39225c7d50fc0c31e6`
- authority head `00d166ca9950359aeada6b2f18e94a5471bfb102`
- outcome `HIRA_V1_S37_A0_QUERY_GATED_READOUT_READY`

No replacement S37-A0 is authorized.

## Exact zero-init identity

Treatment added learned params:
- shared query-gated readout vector: **256**
- query summary: masked arithmetic mean -> L2 normalize
- query normalization epsilon: **1e-12**
- residual scale: **1.0**
- bias/nonlinearity/rank expansion: **none**

Capacity:
- control trainable surface: **49,152**
- treatment trainable surface: **49,408**
- LoRA: **16,384**
- shared primary projection: **32,768**
- original A13 trainable: **0**
- HIRACore trainable: **0**

At zero initialization:
- relation-logit max abs difference: **0**
- native signature max abs difference: **0**
- primary-logit max abs difference: **0**
- fused-logit max abs difference: **0**
- selected-choice identity rate: **1.0**

## Query-conditioning mechanism

With a fixed nonzero readout vector:
- query-content intervention residual max abs change: **0.0220453572**
- question-token permutation logit error: **7.4505806e-9**
- question-token permutation signature error: **0**
- masked query-padding logit error: **0**
- masked query-padding signature error: **0**

The treatment therefore depends on query content while remaining insensitive to token order under the frozen masked-mean reducer and to masked padding.

## Gradient ownership

Treatment:
- relation block -> readout L1: **0.0012450286**
- primary block -> readout L1: **0**
- native relation block -> shared projection direct L1: **0**
- primary block -> shared projection L1: **969.4917602539**
- relation block -> A13 LoRA L1: **0.2209856110**

This matches the preregistered ownership split.

## Projection independence

After strong shared-projection perturbation:
- native relation-logit max abs change: **0**
- native signature max abs change: **0**
- primary-logit max abs change: **0.0017415156**

The query-gated readout consumes native signatures/question tokens and does not introduce a hidden shared-projection dependency.

## Equivariance / arbitrary K / mechanics

- logical-option permutation logit error: **5.9604645e-8**
- logical-option permutation signature error: **0**
- K=3 output shapes: logits `[3,3]`, signatures `[3,3,256]`
- K=7 output shapes: logits `[3,7]`, signatures `[3,7,256]`
- checkpoint key count: **1**
- readout checkpoint roundtrip: PASS
- probability-mass error: **1.1920929e-7**
- full-K: PASS
- expected state views: **32**

## A0 semantic diagnostics

Diagnostic only:
- native relation accuracy: **37.5%**
- fused accuracy: **34.375%**
- relation CE: **1.3688333035**
- canonicalization: **0.3336372077**
- relation block: **0.1869289130**

These values MUST NOT tune:
- query-summary reducer
- query normalization epsilon
- residual scale
- bias/nonlinearity/rank
- readout-only LR
- native geometry
- projected/native mixing
- optimizer
- selector/gates
- seed/LR/epochs/batch

## Consequence

S37-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. this receipt is frozen;
2. trainer/workflow are frozen;
3. exact-head generic CI passes;
4. a separate one-shot TRAIN/DEV marker is committed.

No second S37-A0 run is authorized.
