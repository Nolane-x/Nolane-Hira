# HIRA V1 S38 A0 receipt — Full Bilinear Native Query-Signature Correctness Readout

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #259
PR: #260

## Canonical authority

- run `37084720616`
- artifact `11259982918`
- artifact digest `sha256:d6e6cfc05cdc65ab5486979a4bd69775a41fd6ff4c6debabdc35c0d0d528d03b`
- authority head `2d0fc9bc26fe3610e9c40bc55dc2986f91a207a2`
- outcome `HIRA_V1_S38_A0_FULL_BILINEAR_READOUT_READY`

No replacement S38-A0 is authorized.

## Exact zero-init identity

Treatment:
- one shared full bilinear matrix `W in R^(256x256)`
- added params **65,536**
- query summary: masked arithmetic mean -> L2 normalize
- query normalization epsilon **1e-12**
- residual scale **1.0**
- bias/nonlinearity/factorization: **none**

Capacity:
- control trainable surface: **49,152**
- treatment trainable surface: **114,688**
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

## Bilinear gradient evidence

At exact zero W:
- relation block -> full W gradient L1: **0.3336779475**
- relation block -> off-diagonal W gradient L1: **0.3323763907**
- primary block -> W gradient L1: **0**

The off-diagonal fraction dominates the gradient mass, confirming that S38 is testing a genuinely broader interaction family than S37's diagonal map.

Gradient ownership:
- native relation block -> shared projection direct L1: **0**
- primary block -> shared projection L1: **1517.2321777344**
- relation block -> A13 LoRA L1: **0.2464779774**

## Bilinear intervention evidence

With a fixed nonzero off-diagonal W:
- query-content intervention residual max abs change: **0.0210716370**
- signature-content intervention residual max abs change: **0.0132539095**

Invariance:
- question-token permutation logit error: **2.9802322e-8**
- question-token permutation signature error: **0**
- masked query-padding logit error: **0**
- masked query-padding signature error: **0**
- logical-option permutation logit error: **5.9604645e-8**
- logical-option permutation signature error: **0**

## Projection independence / mechanics

After strong shared-projection perturbation:
- native relation-logit max abs change: **0**
- native signature max abs change: **0**
- primary-logit max abs change: **0.0018599442**

Other mechanics:
- K=3 output shapes: logits `[3,3]`, signatures `[3,3,256]`
- K=7 output shapes: logits `[3,7]`, signatures `[3,7,256]`
- checkpoint key count: **1**
- checkpoint roundtrip: PASS
- probability-mass error: **1.1920929e-7**
- full-K: PASS
- expected state views: **32**

## A0 semantic diagnostics

Diagnostic only:
- native relation accuracy: **20.3125%**
- fused accuracy: **25.0%**
- relation CE: **1.383131981**
- canonicalization: **0.3406974673**
- relation block: **0.1894178241**

These values MUST NOT tune:
- W shape/rank/factorization
- matrix regularization
- query-summary reducer
- query normalization epsilon
- residual scale
- bias/nonlinearity
- W-only LR/scheduler
- native geometry
- projected/native mixing
- optimizer
- selector/gates
- seed/LR/epochs/batch

## Consequence

S38-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. this receipt is frozen;
2. trainer/workflow are frozen;
3. exact-head generic CI passes;
4. a separate one-shot TRAIN/DEV marker is committed.

No second S38-A0 run is authorized.
